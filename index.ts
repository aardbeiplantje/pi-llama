/**
 * llama.cpp provider for pi.
 *
 * Auto-discovers models from a running `llama-server` and
 * registers them under the `llama-cpp` provider.
 *
 * Usage: `pi install github.com/huggingface/pi-llama`
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { Compile } from "typebox/compile";
import { Loader, truncateToWidth, visibleWidth } from "@earendil-works/pi-tui";

const PROVIDER_ID = "llama-cpp";
const DEFAULT_BASE_URL = "http://localhost:8080/v1";
// Fallback for /v1/models entries missing meta.n_ctx.
const DEFAULT_CONTEXT_WINDOW = 8192;
// llama.cpp has no output-token cap (no endpoint reports one; generation is only
// bounded by the context window), so use Pi's own default for models that omit
// maxTokens (see model-registry.ts parseModels).
const DEFAULT_MAX_TOKENS = 16384;
const PROPS_TIMEOUT_MS = 120_000;
// Default slot ID for llama.cpp — can be overridden via LLAMA_SLOT_ID env var.
const DEFAULT_SLOT_ID = 0;

// ---------------------------------------------------------------------------
// Slot pool allocator — parses LLAMA_SLOT_ID as a range (e.g. "0-3") and
// auto-assigns slots from the pool. Sub-agents get their own slot so the
// main agent's KV cache is never evicted.
// ---------------------------------------------------------------------------

interface SlotPool {
	slots: number[];
	allocated: Map<string, number>; // agentId → slot
	nextIndex: number;
}

/** Parse "0-3" → [0,1,2,3], "0" → [0], invalid → [0]. */
function parseSlotRange(raw: string | undefined): number[] {
	if (!raw || raw.trim() === "") return [DEFAULT_SLOT_ID];
	const trimmed = raw.trim();
	// Range format: "0-3"
	const rangeMatch = trimmed.match(/^(\d+)\s*-\s*(\d+)$/);
	if (rangeMatch) {
		const start = parseInt(rangeMatch[1], 10);
		const end = parseInt(rangeMatch[2], 10);
		if (start <= end && end < 100) {
			const slots: number[] = [];
			for (let i = start; i <= end; i++) slots.push(i);
			return slots;
		}
	}
	// Single value
	const parsed = parseInt(trimmed, 10);
	if (!isNaN(parsed) && parsed >= 0) return [parsed];
	// Fallback
	console.warn(`[llama-cpp] invalid LLAMA_SLOT_ID="${raw}", using default [${DEFAULT_SLOT_ID}]`);
	return [DEFAULT_SLOT_ID];
}

function createSlotPool(slots: number[]): SlotPool {
	return { slots, allocated: new Map(), nextIndex: 0 };
}

/** Get next available slot from pool, wrapping around. */
function allocateSlot(pool: SlotPool, agentId: string): number {
	if (pool.allocated.has(agentId)) return pool.allocated.get(agentId)!;
	// Find first unallocated slot
	for (let i = 0; i < pool.slots.length; i++) {
		const slot = pool.slots[(pool.nextIndex + i) % pool.slots.length];
		if (!pool.allocated.has(String(slot))) {
			pool.allocated.set(agentId, slot);
			pool.allocated.set(String(slot), slot); // track by slot value too
			pool.nextIndex = (i + 1) % pool.slots.length;
			return slot;
		}
	}
	// All slots allocated — reuse first slot (shouldn't happen with maxConcurrent=1)
	const firstSlot = pool.slots[0];
	pool.allocated.set(agentId, firstSlot);
	return firstSlot;
}

/** Release a slot back to the pool. */
function releaseSlot(pool: SlotPool, agentId: string): void {
	const slot = pool.allocated.get(agentId);
	if (slot !== undefined) {
		pool.allocated.delete(agentId);
		pool.allocated.delete(String(slot));
	}
}

const ModelsResponseSchema = Type.Object({
	data: Type.Optional(
		Type.Array(
			Type.Object({
				id: Type.String(),
				aliases: Type.Optional(Type.Array(Type.String())),
				status: Type.Optional(
					Type.Object({
						value: Type.Optional(
							Type.Union([
								Type.Literal("unloaded"),
								Type.Literal("loading"),
								Type.Literal("loaded"),
								Type.Literal("sleeping"),
								Type.Literal("unknown"),
							]),
						),
					}),
				),
				architecture: Type.Optional(
					Type.Object({
						input_modalities: Type.Optional(Type.Array(Type.String())),
					}),
				),
				meta: Type.Optional(
					Type.Object({
						n_ctx: Type.Optional(Type.Number()),
						n_params: Type.Optional(Type.Number()),
					}),
				),
			}),
		),
	),
});

const validateModelsResponse = Compile(ModelsResponseSchema);

const PropsResponseSchema = Type.Object({
	default_generation_settings: Type.Optional(
		Type.Object({
			n_ctx: Type.Optional(Type.Number()),
		}),
	),
	chat_template: Type.Optional(Type.String()),
	build_info: Type.Optional(Type.String()),
});

const validatePropsResponse = Compile(PropsResponseSchema);

// SSE event types for model loading progress
type ApiModelLoadStage = "text_model" | "spec_model" | "mmproj_model";

type ApiModelsSseProgress = {
	stages: ApiModelLoadStage[];
	current: ApiModelLoadStage;
	value: number;
};

type ApiModelsSseData = {
	status: string;
	progress?: ApiModelsSseProgress;
	exit_code?: number;
};

type ApiModelsSseEvent = {
	model: string;
	event: string;
	data: ApiModelsSseData;
};

const MODEL_LOAD_STAGE_LABELS: Record<ApiModelLoadStage, string> = {
	text_model: "Loading weights",
	spec_model: "Loading draft",
	mmproj_model: "Loading projector",
};

type LlamaModel = NonNullable<Parameters<ExtensionAPI["registerProvider"]>[1]["models"]>[number];
type ExtensionCtx = Parameters<Parameters<ExtensionAPI["on"]>[1]>[1];

// llama.cpp template thinking is boolean, so expose Pi's default off/medium toggle only.
const TEMPLATE_THINKING_LEVEL_MAP = {
	minimal: null,
	low: null,
	high: null,
	xhigh: null,
} satisfies NonNullable<LlamaModel["thinkingLevelMap"]>;

// Minimal shape needed to update both registered models and Pi's active model snapshot.
type MutableModelMetadata = {
	reasoning: boolean;
	thinkingLevelMap?: LlamaModel["thinkingLevelMap"];
	compat?: LlamaModel["compat"];
	contextWindow: number;
	maxTokens: number;
};

// Mark a model as using llama.cpp's chat_template_kwargs.enable_thinking control.
function applyTemplateThinkingSupport(model: MutableModelMetadata): void {
	model.reasoning = true;
	model.thinkingLevelMap = TEMPLATE_THINKING_LEVEL_MAP;
	model.compat = {
		...model.compat,
		// Despite the Pi enum name, this sends llama.cpp's generic
		// chat_template_kwargs.enable_thinking payload, not a Qwen-only option.
		thinkingFormat: "qwen-chat-template",
	};
}

// Pi invalidates a captured ctx when the session is replaced (e.g. new_session in
// RPC mode). Any later ctx access then throws this error. Background work started
// before the replacement should treat it as "session gone" and stop quietly.
function isStaleContextError(error: unknown): boolean {
	return error instanceof Error && error.message.includes("stale after session replacement");
}

export default async function (pi: ExtensionAPI) {
	let currentModels: LlamaModel[] = [];

	pi.registerCommand("llama-version", {
		description: "Get build info of llama.cpp server",
		handler: async (_args, ctx) => {
			const response = await fetch(`${baseUrl.replace(/\/v1$/, "")}/props`);
			if (!response.ok) {
				ctx.ui.notify(`[llama-cpp] /props returned ${response.status}`, "error");
				return;
			}

			const data: unknown = await response.json();
			if (!validatePropsResponse.Check(data)) {
				const errors = [...validatePropsResponse.Errors(data)]
					.map((e) => `${"path" in e ? e.path : ""} ${e.message}`)
					.join("; ");
				ctx.ui.notify(`[llama-cpp] invalid /props response: ${errors}`, "error");
				return;
			}

			const match = data.build_info?.match(/^b([a-zA-Z0-9]+)-([a-zA-Z0-9]+)$/);

			if (match && match.length === 3) {
				ctx.ui.notify(`Build number: ${match[1]}, Commit hash: ${match[2]}`, "info");
			} else {
				ctx.ui.notify(`Malformed build info: ${data.build_info}`, "warning");
			}
		},
	});

	const baseUrl = (process.env.LLAMA_BASE_URL ?? DEFAULT_BASE_URL).replace(/\/+$/, "");
	const apiKey = process.env.LLAMA_API_KEY ?? "no-key";

	// -----------------------------------------------------------------------
	// Slot pool — parse LLAMA_SLOT_ID as a range and auto-assign slots
	// Slot 0 is reserved for the main agent IF it's in the pool.
	// Otherwise the main agent uses the first available slot.
	// Sub-agents get other slots from the pool.
	// If only 1 slot is available (e.g. LLAMA_SLOT_ID=0 or LLAMA_SLOT_ID=3),
	// it's shared and llama.cpp will do prompt prefill.
	// -----------------------------------------------------------------------
	const allSlots = parseSlotRange(process.env.LLAMA_SLOT_ID);
	const mainAgentId = "main";
	// Reserve slot 0 for main agent if present in the pool; otherwise use first slot
	const mainAgentSlot = allSlots.includes(0) ? 0 : allSlots[0];
	const sas = allSlots.filter(s => s !== mainAgentSlot);
	const slotPool = createSlotPool(sas);
	let currentSlotId = mainAgentSlot;

	console.log(`[llama-cpp] slot pool: [${allSlots.join(",")}] → main agent uses slot ${mainAgentSlot}, sub-agents use [${sas.join(",") || "none"}]`);

	async function refreshProvider(): Promise<void> {
		try {
			const response = await fetch(`${baseUrl}/models`);
			if (!response.ok) {
				console.warn(`[llama-cpp] ${baseUrl}/models returned ${response.status}`);
				return;
			}

			const payload: unknown = await response.json();
			if (!validateModelsResponse.Check(payload)) {
				const errors = [...validateModelsResponse.Errors(payload)]
					.map((e) => `${"path" in e ? e.path : ""} ${e.message}`)
					.join("; ");
				console.warn(`[llama-cpp] invalid /models response: ${errors}`);
				return;
			}

			const previousById = new Map(currentModels.map((m) => [m.id, m]));

			currentModels = (payload.data ?? []).map((model) => {
				const previous = previousById.get(model.id);
				const isLoaded = model.status?.value === "loaded";
				const modalities = model.architecture?.input_modalities ?? ["text"];
				const input = modalities.filter(
					(m): m is "text" | "image" => m === "text" || m === "image",
				);
				const suffixes: string[] = [];
				if (input.includes("image")) {
					suffixes.push("(image)");
				}
				if (isLoaded) {
					suffixes.push("(loaded)");
				}
				const contextWindow =
					model.meta?.n_ctx ?? previous?.contextWindow ?? DEFAULT_CONTEXT_WINDOW;
				const displayName = model.aliases?.[0] || model.id;
				return {
					id: model.id,
					name: suffixes.length > 0 ? `${displayName} ${suffixes.join(" ")}` : displayName,
					// /v1/models does not include /props-discovered capabilities, so preserve
					// template thinking metadata across refreshes.
					reasoning: previous?.reasoning ?? false,
					thinkingLevelMap: previous?.thinkingLevelMap,
					input,
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
					contextWindow,
					maxTokens: Math.min(DEFAULT_MAX_TOKENS, contextWindow),
					compat: previous?.compat,
					status: model.status,
				} as LlamaModel;
			});

			if (currentModels.length === 0) {
				console.warn(`[llama-cpp] no models returned from ${baseUrl}/models`);
				return;
			}

			// Track which model is currently loaded on the server
			const loadedModel = currentModels.find((m) => m.status?.value === "loaded");
			currentlyLoadedModel = loadedModel?.id ?? null;

			pi.registerProvider(PROVIDER_ID, {
				name: "llama.cpp",
				baseUrl,
				apiKey,
				api: "openai-completions",
				models: currentModels,
			});
		} catch (error) {
			console.warn(`[llama-cpp] failed to reach ${baseUrl}/models: ${(error as Error).message}`);
		}
	}

	const discoveredMetadata = new Set<string>();
	const pendingMetadata = new Set<string>();
	let currentlyLoadedModel: string | null = null;
	let statusTimeout: ReturnType<typeof setTimeout> | undefined;
	let sseAbortController: AbortController | null = null;
	let propsAbortController: AbortController | null = null;

	function clearFooterStatusTimeout(): void {
		if (statusTimeout !== undefined) {
			clearTimeout(statusTimeout);
			statusTimeout = undefined;
		}
	}

	// Connect to SSE stream for model loading progress
	async function connectToLoadingProgress(
		modelId: string,
		ctx: ExtensionCtx,
		loader: Loader,
	): Promise<void> {
		// Close any existing SSE connection
		if (sseAbortController) {
			sseAbortController.abort();
			sseAbortController = null;
		}

		sseAbortController = new AbortController();
		const signal = sseAbortController.signal;

		try {
			const response = await fetch(`${baseUrl.replace(/\/v1$/, "")}/models/sse`, { signal });

			if (!response.ok) {
				if (response.status !== 404) {
					ctx?.ui.notify(`[llama-cpp] loading progress ${response.status})`, "warning");
				}
				return;
			}

			const reader = response.body?.getReader();
			if (!reader) {
				return;
			}

			const decoder = new TextDecoder();
			let buffer = "";

			while (!signal.aborted) {
				const { done, value } = await reader.read();
				if (done) {
					break;
				}

				buffer += decoder.decode(value, { stream: true });
				const events = buffer.split("\n\n");
				buffer = events.pop() || "";

				for (const event of events) {
					if (!event) {
						continue;
					}

					// Parse SSE record: extract data lines
					const dataLines = event
						.split("\n")
						.filter((line) => line.startsWith("data:"))
						.map((line) => line.slice(5).trim())
						.join("\n");

					if (!dataLines) {
						continue;
					}

					try {
						const sseEvent: ApiModelsSseEvent = JSON.parse(dataLines);

						// Process status events for all models to keep discoveredMetadata and
						// currentlyLoadedModel in sync.
						if (
							sseEvent.event === "model_status" ||
							sseEvent.event === "status_change" ||
							sseEvent.event === "status_update"
						) {
							const status = sseEvent.data.status;

							if (status === "unloaded") {
								discoveredMetadata.delete(sseEvent.model);
								if (currentlyLoadedModel === sseEvent.model) {
									currentlyLoadedModel = null;
								}
							}
							if (status === "loaded") {
								currentlyLoadedModel = sseEvent.model;
							}
						}

						// Progress UI is only for the model we're actively loading
						if (sseEvent.model === modelId) {
							const currentModel = currentModels.find((m) => m.id === sseEvent.model);
							const displayName = currentModel?.name.split(" ")[0] || sseEvent.model;
							const progress = sseEvent.data.progress;

							if (sseEvent.data.exit_code && sseEvent.data.exit_code !== 0) {
								ctx?.ui.setWidget(PROVIDER_ID, [
									ctx?.ui.theme.fg("error", "[llama.cpp] ") +
										ctx?.ui.theme.fg(
											"text",
											`${displayName}:  failed (exit ${sseEvent.data.exit_code})`,
										),
								]);
								sseAbortController?.abort();
								return;
							}

							if (sseEvent.data.status === "loading" && progress) {
								const stageLabel = progress.current
									? MODEL_LOAD_STAGE_LABELS[progress.current] || progress.current
									: "Loading";
								const progressPercent = Math.round(progress.value * 100);
								loader?.setMessage(`${displayName}: ${stageLabel} (${progressPercent}%)`);
							}
						}
					} catch {
						// Ignore parse errors
					}
				}
			}
		} catch (error) {
			// Suppress errors from intentionally-aborted SSE connections and from a
			// stale ctx (session was replaced while streaming).
			const msg = (error as Error).message;
			if (
				isStaleContextError(error) ||
				(signal.aborted && (error instanceof DOMException || msg === "terminated"))
			) {
				return;
			}
			ctx?.ui.notify(`[llama-cpp] SSE error: ${msg}`, "warning");
		} finally {
			sseAbortController = null;
		}
	}

	async function discoverModelMetadata(
		modelId: string,
		ctx?: ExtensionCtx,
		autoload = true,
		timeoutMs = PROPS_TIMEOUT_MS,
		selectedModel?: MutableModelMetadata,
	): Promise<void> {
		const model = currentModels.find((m) => m.id === modelId);
		if (!model) {
			return;
		}
		const displayName = model.name.split(" ")[0];
		// Use tracked state instead of stale currentModels status.
		const isLoaded = currentlyLoadedModel === modelId;

		if (discoveredMetadata.has(modelId)) {
			// If discovered but no longer loaded, clear cache and fall through to reload.
			if (!isLoaded) {
				discoveredMetadata.delete(modelId);
			} else {
				// Copy cached metadata into the selected model snapshot.
				if (selectedModel) {
					selectedModel.contextWindow = model.contextWindow;
					selectedModel.maxTokens = model.maxTokens;
					if (model.reasoning) {
						selectedModel.reasoning = model.reasoning;
						selectedModel.thinkingLevelMap = model.thinkingLevelMap;
						selectedModel.compat = model.compat;
					}
				}
				return;
			}
		}
		if (pendingMetadata.has(modelId)) {
			return;
		}

		pendingMetadata.add(modelId);
		// Cancel any pending clear timeout from a previous model load.
		clearFooterStatusTimeout();
		// Abort any in-flight /props request from a previous model.
		if (propsAbortController) {
			propsAbortController.abort();
		}
		propsAbortController = new AbortController();
		const timer = setTimeout(() => propsAbortController.abort(), timeoutMs);
		const shouldAutoload = autoload && !isLoaded;
		const propsUrl = `${baseUrl.replace(/\/v1$/, "")}/props?model=${encodeURIComponent(modelId)}&autoload=${shouldAutoload}`;
		const clearFooterStatusLater = () => {
			clearFooterStatusTimeout();
			statusTimeout = setTimeout(() => {
				statusTimeout = undefined;
				ctx?.ui.setWidget(PROVIDER_ID, undefined);
			}, 8000);
		};

		try {
			if (shouldAutoload && ctx) {
				let loader = null;
				ctx.ui.setWidget(PROVIDER_ID, (ui, theme) => {
					const prefix = theme.fg("accent", " [llama.cpp]");
					const prefixWidth = visibleWidth(" [llama.cpp]");
					loader = new Loader(
						ui,
						(s) => theme.fg("accent", s),
						(t) => theme.fg("text", t),
						`${displayName}: Loading...`,
					);
					return {
						dispose: () => loader?.stop(),
						render: (width: number) => {
							const [_, line] = loader.render(width - prefixWidth);
							return [prefix + truncateToWidth(line, width - prefixWidth)];
						},
					};
				});
				// Start SSE connection to monitor loading progress
				void connectToLoadingProgress(modelId, ctx, loader);
			}

			const response = await fetch(propsUrl, { signal: propsAbortController.signal });
			if (!response.ok) {
				// 500 during autoload is expected when the server cancels a load to start
				// another model. Suppress the notification for that case.
				if (!(shouldAutoload && response.status === 500)) {
					ctx?.ui.notify(`[llama-cpp] /props for ${modelId} returned ${response.status}`, "error");
				}
				return;
			}
			const data: unknown = await response.json();
			if (!validatePropsResponse.Check(data)) {
				const errors = [...validatePropsResponse.Errors(data)]
					.map((e) => `${"path" in e ? e.path : ""} ${e.message}`)
					.join("; ");
				ctx?.ui.notify(`[llama-cpp] invalid /props response for ${modelId}: ${errors}`, "error");
				return;
			}
			const nCtx = data.default_generation_settings?.n_ctx;
			let updated = false;
			let loadedFooterStatus = shouldAutoload ? `[llama.cpp] ${displayName} loaded` : undefined;
			if (typeof nCtx === "number" && nCtx > 0) {
				model.contextWindow = nCtx;
				model.maxTokens = Math.min(DEFAULT_MAX_TOKENS, nCtx);
				loadedFooterStatus = `[llama.cpp] ${displayName} loaded with ctx ${nCtx} tokens`;
				updated = true;
			}
			if (selectedModel) {
				selectedModel.contextWindow = model.contextWindow;
				selectedModel.maxTokens = model.maxTokens;
			}
			if (data.chat_template?.includes("enable_thinking") === true) {
				applyTemplateThinkingSupport(model);
				if (selectedModel) {
					applyTemplateThinkingSupport(selectedModel);
					if (pi.getThinkingLevel() === "off") {
						pi.setThinkingLevel("medium");
					}
				}
				updated = true;
			}
			discoveredMetadata.add(modelId);
			if (shouldAutoload) {
				currentlyLoadedModel = modelId;
			}
			if (loadedFooterStatus && ctx && !isLoaded) {
				const prefix = ctx.ui.theme.fg("success", "[llama.cpp] ✓");
				ctx.ui.setWidget(PROVIDER_ID, [
					prefix +
						ctx.ui.theme.fg(
							"text",
							` ${displayName}: Loaded` + (nCtx ? ` with context ${nCtx} tokens` : ""),
						),
				]);
				clearFooterStatusLater();
			}
			if (!updated) {
				return;
			}
			pi.registerProvider(PROVIDER_ID, {
				name: "llama.cpp",
				baseUrl,
				apiKey,
				api: "openai-completions",
				models: currentModels,
			});
		} catch (error) {
			const err = error as Error;
			// Suppress notification for aborted requests (model was switched) and for a
			// stale ctx (session was replaced while awaiting) — both are expected.
			if (err.name !== "AbortError" && !isStaleContextError(err)) {
				ctx?.ui.notify(`[llama-cpp] /props for ${modelId} failed: ${err.message}`, "error");
			}
		} finally {
			clearTimeout(timer);
			pendingMetadata.delete(modelId);
			propsAbortController = null;
			// Stop SSE connection when done
			if (sseAbortController) {
				sseAbortController.abort();
				sseAbortController = null;
			}
		}
	}

	await refreshProvider();

	pi.on("input", async (event) => {
		const trimmed = event.text.trim().toLowerCase();
		if (trimmed === "/model") {
			await refreshProvider();
		}
	});

	pi.on("model_select", (event, ctx) => {
		if (event.model.provider !== PROVIDER_ID) {
			return;
		}
		void discoverModelMetadata(event.model.id, ctx, true, PROPS_TIMEOUT_MS, event.model);
	});

	// -----------------------------------------------------------------------
	// Sub-agent slot tracking via pi.events
	// -----------------------------------------------------------------------
	// Sub-agents emit events that we listen to for slot assignment/release.
	const ssubAgentSlots = new Map<string, number>(); // agentId → slot

	// Listen for sub-agent lifecycle events from pi-subagents extension
	pi.on("subagents:started", (event: { agentId: string; slotId?: number }) => {
		const agentId = event.agentId;
		if (event.slotId !== undefined) {
			ssubAgentSlots.set(agentId, event.slotId);
			console.log(`[llama-cpp] sub-agent ${agentId} assigned slot ${event.slotId}`);
		} else {
			// Auto-assign from pool
			const slot = allocateSlot(slotPool, agentId);
			ssubAgentSlots.set(agentId, slot);
			console.log(`[llama-cpp] sub-agent ${agentId} auto-assigned slot ${slot}`);
		}
		// Also store by short ID (first 8 chars) to match session name format
		// used by agent-runner.ts: `${baseSessionName}#${agentId.slice(0, 8)}`
		const shortId = agentId.slice(0, 8);
		if (shortId !== agentId) {
			ssubAgentSlots.set(shortId, ssubAgentSlots.get(agentId)!);
		}
	});

	pi.on("subagents:completed", (event: { agentId: string }) => {
		const agentId = event.agentId;
		ssubAgentSlots.delete(agentId);
		console.log(`[llama-cpp] sub-agent ${agentId} completed, slot released`);
	});

	// Discover /props for already-active models because re-selecting them does not emit model_select.
	// Inject slot_id into every provider request so llama.cpp reuses the same KV cache slot.
	pi.on("before_provider_request", (event, ctx) => {
		try {
			const modelId = (event.payload as { model?: unknown })?.model;
			if (typeof modelId === "string") {
				const activeModel =
					ctx.model?.provider === PROVIDER_ID && ctx.model.id === modelId ? ctx.model : undefined;
				void discoverModelMetadata(modelId, ctx, true, PROPS_TIMEOUT_MS, activeModel);
			}
		} catch (error) {
			// Session was replaced as the request fired; nothing to discover.
			if (!isStaleContextError(error)) {
				throw error;
			}
		}

		// Determine which slot to use: sub-agent slot if available, otherwise main agent slot
		let requestSlotId = currentSlotId;
		// Check if we can identify the current agent from the session context
		// Sub-agent sessions have names like "Explore#a1b2c3d4"
		try {
			const sessionName = ctx.sessionManager?.getSessionName?.() ?? "";
			console.log(`[llama-cpp] before_provider_request: sessionName="${sessionName}" currentSlotId=${currentSlotId}`);
			if (sessionName && sessionName.includes("#")) {
				// Extract agent ID from session name (format: "Type#agentId")
				const parts = sessionName.split("#");
				const agentId = parts.length > 1 ? parts[1] : sessionName;
				if (ssubAgentSlots.has(agentId)) {
					requestSlotId = ssubAgentSlots.get(agentId)!;
					console.log(`[llama-cpp] sub-agent ${agentId} using slot ${requestSlotId}`);
				}
			}
		} catch (err) {
			console.log(`[llama-cpp] session name access failed: ${(err as Error).message}`);
			// Session name access failed — use main agent slot
		}

		// Inject id_slot into the request payload (llama.cpp expects "id_slot", not "slot_id")
		const payload = event.payload as { [key: string]: unknown } | undefined;
		if (payload && typeof payload === "object") {
			(payload as Record<string, unknown>).id_slot = requestSlotId;
			console.log(`[llama-cpp] injected id_slot=${requestSlotId} into request`);
		} else {
			console.log(`[llama-cpp] WARNING: payload is null/undefined, id_slot NOT injected`);
		}
	});

	pi.on("session_shutdown", () => {
		clearFooterStatusTimeout();
		// Stop in-flight /props and SSE so they don't resume against a stale ctx.
		propsAbortController?.abort();
		sseAbortController?.abort();
	});
}
