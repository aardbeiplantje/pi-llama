/**
 * llama.cpp provider for pi.
 *
 * Auto-discovers models from a running `llama-server` and
 * registers them under the `llama-cpp` provider.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { Compile } from "typebox/compile";
import { Loader, truncateToWidth, visibleWidth } from "@earendil-works/pi-tui";

const PROVIDER_ID = "llama-cpp";
const DEFAULT_BASE_URL = "http://localhost:8080/v1";
// Fallback for /v1/models entries missing meta.n_ctx.
const DEFAULT_CONTEXT_WINDOW = 8192;
const DEFAULT_MAX_TOKENS = 16384;
const PROPS_TIMEOUT_MS = 120_000;

// ---------------------------------------------------------------------------
// Slot tracking — manual save/restore
// ---------------------------------------------------------------------------
// llama.cpp exposes POST /slots/<id>?action=save|restore. The server does NOT
// auto-save — we call the API manually via /slot commands.
//
// The slot ID is injected into every llama.cpp provider request so the server
// reuses the same KV cache slot for the current session.

interface SlotCheckpoint {
	slotId: number;
	modelName: string;
	modelProvider: string;
	filename: string;
	timestamp: number;
	sessionId: string;
}

interface SlotSaveRequestBody {
	filename: string;
	model: string;
}

// Global slot state
let currentSlotId: number | null = null;
let activeModelName: string | null = null;
let slotPersistFn: ((customType: string, data?: unknown) => void) | null = null;
let _baseUrl: string = DEFAULT_BASE_URL;

// Track whether slot save/restore is supported (set by saveSlot on first call)
let slotSaveSupported: boolean | null = null;

async function saveSlot(slotId: number): Promise<boolean> {
	const serverUrl = _baseUrl.replace(/\/v1$/, "");
	if (!activeModelName) {
		console.warn(`[llama-cpp] saveSlot(${slotId}) — no active model name`);
		return false;
	}
	const slotName = activeModelName.split("/").join("_").replace(/[^a-zA-Z0-9_]/g, "_");
	const filename = `${slotName}_${slotId}.kv`;
	const body: SlotSaveRequestBody = { filename, model: activeModelName };
	console.log(`[llama-cpp] saveSlot(${slotId}): url=${serverUrl}/slots/${slotId}?action=save body=${JSON.stringify(body)}`);
	try {
		const res = await fetch(`${serverUrl}/slots/${slotId}?action=save`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify(body),
		});
		const resText = await res.text();
		if (!res.ok) {
			// First failure marks slot save as unsupported to avoid repeated noisy errors
			if (slotSaveSupported === null) {
				slotSaveSupported = false;
				console.warn(`[llama-cpp] saveSlot(${slotId}) ${res.status}: ${resText}`);
				console.warn(`[llama-cpp] slot save/restore not supported — server may need --slot-save-path`);
			} else {
				console.warn(`[llama-cpp] saveSlot(${slotId}) ${res.status}: ${resText}`);
			}
			return false;
		}
		// First successful save marks slot support as enabled
		if (slotSaveSupported === null) {
			slotSaveSupported = true;
		}
		const result = JSON.parse(resText);
		const nWritten = (result as { n_written?: number })?.n_written;
		console.log(`[llama-cpp] slot ${slotId} saved ${nWritten ?? "??"} tokens to ${filename}`);
		return true;
	} catch (error) {
		// Connection errors also indicate slot saving is not supported
		if (slotSaveSupported === null) {
			slotSaveSupported = false;
			console.warn(`[llama-cpp] saveSlot(${slotId}) failed: ${(error as Error).message}`);
		}
		console.warn(`[llama-cpp] saveSlot(${slotId}) failed: ${(error as Error).message}`);
		return false;
	}
}

async function restoreSlot(slotId: number, filename: string): Promise<boolean> {
	const serverUrl = _baseUrl.replace(/\/v1$/, "");
	if (!activeModelName) {
		console.warn(`[llama-cpp] restoreSlot(${slotId}) — no active model name`);
		return false;
	}
	const body: SlotSaveRequestBody = { filename, model: activeModelName };
	try {
		const res = await fetch(`${serverUrl}/slots/${slotId}?action=restore`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify(body),
		});
		if (!res.ok) {
			const errText = await res.text();
			console.warn(`[llama-cpp] restoreSlot(${slotId}) ${res.status}: ${errText}`);
			return false;
		}
		const result = await res.json();
		const nWritten = (result as { n_written?: number })?.n_written;
		console.log(`[llama-cpp] slot ${slotId} restored ${nWritten ?? "??"} tokens from ${filename}`);
		return true;
	} catch (error) {
		console.warn(`[llama-cpp] restoreSlot(${slotId}) failed: ${(error as Error).message}`);
		return false;
	}
}

async function discoverSlots(): Promise<void> {
	try {
		const modelsUrl = `${_baseUrl}/models`;
		console.log(`[llama-cpp] discoverSlots: modelsUrl=${modelsUrl}`);
		const res = await fetch(modelsUrl);
		if (!res.ok) {
			console.warn(`[llama-cpp] discoverSlots: /models returned ${res.status}`);
			return;
		}
		const payload = await res.json();
		if (!payload.data) {
			console.warn(`[llama-cpp] discoverSlots: no data in /models response`);
			return;
		}
		// Find currently loaded model name(s)
		for (const model of payload.data as Array<{ id: string; status?: { value?: string } }>) {
			if (model.status?.value === "loaded") {
				activeModelName = model.id;
			}
		}
		if (!activeModelName) {
			console.warn(`[llama-cpp] discoverSlots: no loaded model found (ids: ${(payload.data as any[]).map((m: any) => m.id).join(", ")})`);
			return;
		}

		// Query /slots endpoint to get numeric slot IDs
		const slotsRes = await fetch(
			`${_baseUrl.replace(/\/v1$/, "")}/slots?model=${encodeURIComponent(activeModelName)}`,
		);
		if (!slotsRes.ok) return;
		const slotsData = await slotsRes.json();
		if (!Array.isArray(slotsData)) return;

		const idleSlot = slotsData.find((s: { id: number; is_processing?: boolean }) => !s.is_processing);
		if (idleSlot) {
			currentSlotId = idleSlot.id;
			console.log(`[llama-cpp] selected slot ${currentSlotId} (idle) for ${activeModelName}`);
		}
	} catch {
		// Non-fatal
	}
}

function persistSlotCheckpointToSession(slotId: number, modelName: string): void {
	const slotName = modelName.split("/").join("_").replace(/[^a-zA-Z0-9_]/g, "_");
	const filename = `${slotName}_${slotId}.kv`;
	try {
		slotPersistFn?.("llama-cpp-slot", {
			type: "slot_checkpoint",
			slotId,
			modelName,
			modelProvider: PROVIDER_ID,
			filename,
			timestamp: Date.now(),
		});
	} catch {
		// Ignore persist errors
	}
}

// Attempt to restore from a saved checkpoint. Returns false if slot saving
// is not supported by the llama.cpp server, to avoid breaking the flow.
async function restoreCheckpoint(slotId: number, modelName: string, filename: string): Promise<boolean> {
	if (slotSaveSupported === false) {
		return false;
	}
	return restoreSlot(slotId, filename);
}

// ---------------------------------------------------------------------------
// Provider / models
// ---------------------------------------------------------------------------

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
	let currentlyLoadedModel: string | null = null;
	let currentSessionFile: string | null = null;

	pi.registerCommand("llama-version", {
		description: "Get build info of llama.cpp server",
		handler: async (_args, ctx) => {
			const response = await fetch(`${_baseUrl.replace(/\/v1$/, "")}/props`);
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

	// -----------------------------------------------------------------------
	// Slots commands
	// -----------------------------------------------------------------------

	pi.registerCommand("slots", {
		description: "Manage llama.cpp KV cache slots (manual save/restore)",
		handler: async (args, ctx) => {
			const sub = args[0]?.toString().trim().toLowerCase() || "";
			const slotIdArg = args[1];

			switch (sub) {
				case "save": {
					await discoverSlots();
					if (currentSlotId === null || activeModelName === null) {
						ctx.ui.notify(
							"[llama-cpp] No loaded model or slot found. Select a model first.",
							"error",
						);
						return;
					}
					ctx.ui.notify(
						`[llama-cpp] Saving slot ${currentSlotId} for ${activeModelName}...`,
						"info",
					);
					const ok = await saveSlot(currentSlotId);
					if (ok) {
						persistSlotCheckpointToSession(currentSlotId, activeModelName);
						ctx.ui.notify(
							`[llama-cpp] Slot ${currentSlotId} saved. Session file updated.`,
							"success",
						);
					} else {
						const hint = slotSaveSupported === false
							? " Server may be missing --slot-save-path."
							: "";
						ctx.ui.notify(
							`[llama-cpp] Failed to save slot ${currentSlotId}.${hint}`,
							"error",
						);
					}
					break;
				}
				case "restore": {
					const slotId = typeof slotIdArg === "number" ? slotIdArg : parseInt(slotIdArg?.toString() ?? "");
					if (isNaN(slotId)) {
						ctx.ui.notify(
							"[llama-cpp] Usage: /slots restore <slot_id>",
							"error",
						);
						return;
					}
					await discoverSlots();
					if (activeModelName === null) {
						ctx.ui.notify(
							"[llama-cpp] No loaded model found. Select a model first.",
							"error",
						);
						return;
					}
					const slotName = activeModelName.split("/").join("_").replace(/[^a-zA-Z0-9_]/g, "_");
					const filename = `${slotName}_${slotId}.kv`;
					ctx.ui.notify(
						`[llama-cpp] Restoring slot ${slotId} from ${filename} for ${activeModelName}...`,
						"info",
					);
					const ok = await restoreSlot(slotId, filename);
					if (ok) {
						currentSlotId = slotId;
						ctx.ui.notify(
							`[llama-cpp] Slot ${slotId} restored. Will be used for all requests.`,
							"success",
						);
					} else {
						const hint = slotSaveSupported === false
						? " Server may be missing --slot-save-path."
						: "";
					ctx.ui.notify(
						`[llama-cpp] Failed to restore slot ${slotId}.${hint}`,
						"error",
					);
					}
					break;
				}
				case "status":
				default: {
					await discoverSlots();
					const lines: string[] = [];
					lines.push(`[llama-cpp] Slot status:`);
					lines.push(`  Model: ${activeModelName ?? "(none)"} `);
					lines.push(`  Current slot: ${currentSlotId ?? "(none)"}`);
					lines.push(`  Usage:`);
					lines.push(`    /slots save          — save current slot to disk`);
					lines.push(`    /slots restore <id>  — restore slot <id> from disk`);
					ctx.ui.notify(lines.join("\n"), "info");
					break;
				}
			}
		},
	});

	const baseUrl = (process.env.LLAMA_BASE_URL ?? process.env.LLAMA_SERVER_URL ?? DEFAULT_BASE_URL).replace(/\/+$/, "");
	const apiKey = process.env.LLAMA_API_KEY ?? "no-key";
	_baseUrl = baseUrl;
	slotPersistFn = pi.appendEntry.bind(pi);

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
	let statusTimeout: ReturnType<typeof setTimeout> | undefined;
	let sseAbortController: AbortController | null = null;
	let propsAbortController: AbortController | null = null;

	function clearFooterStatusTimeout(): void {
		if (statusTimeout !== undefined) {
			clearTimeout(statusTimeout);
			statusTimeout = undefined;
		}
	}

	async function connectToLoadingProgress(
		modelId: string,
		ctx: ExtensionCtx,
		loader: Loader,
	): Promise<void> {
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
		const isLoaded = currentlyLoadedModel === modelId;

		if (discoveredMetadata.has(modelId)) {
			if (!isLoaded) {
				discoveredMetadata.delete(modelId);
			} else {
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
		clearFooterStatusTimeout();
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
				void connectToLoadingProgress(modelId, ctx, loader);
			}

			const response = await fetch(propsUrl, { signal: propsAbortController.signal });
			if (!response.ok) {
				// /props failure is non-blocking — slot saving is best-effort,
				// prompt processing will take over if the slot is not available.
				if (shouldAutoload && response.status === 500) {
					// autoload returned 500 — the model is loading, just skip metadata
					console.warn(`[llama-cpp] /props for ${modelId} returned ${response.status} (model is loading, skipping metadata)`);
				} else {
					console.warn(`[llama-cpp] /props for ${modelId} returned ${response.status}`);
				}
			} else {
				const data: unknown = await response.json();
				if (validatePropsResponse.Check(data)) {
					const nCtx = data.default_generation_settings?.n_ctx;
					let loadedFooterStatus = shouldAutoload ? `[llama.cpp] ${displayName} loaded` : undefined;
					if (typeof nCtx === "number" && nCtx > 0) {
						model.contextWindow = nCtx;
						model.maxTokens = Math.min(DEFAULT_MAX_TOKENS, nCtx);
						loadedFooterStatus = `[llama.cpp] ${displayName} loaded with ctx ${nCtx} tokens`;
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
					}
					if (loadedFooterStatus && ctx && !isLoaded) {
						const nCtxVal = data.default_generation_settings?.n_ctx;
						const prefix = ctx.ui.theme.fg("success", "[llama.cpp] ✓");
						ctx.ui.setWidget(PROVIDER_ID, [
							prefix +
								ctx.ui.theme.fg(
									"text",
									` ${displayName}: Loaded` + (nCtxVal ? ` with context ${nCtxVal} tokens` : ""),
								),
						]);
						clearFooterStatusLater();
					}
				} else {
					const errors = [...validatePropsResponse.Errors(data)]
						.map((e) => `${"path" in e ? e.path : ""} ${e.message}`)
						.join("; ");
					console.warn(`[llama-cpp] invalid /props response for ${modelId}: ${errors} (non-blocking, slot saving is best-effort)`);
				}
			}
			// Always register the provider so the model is selectable even without /props metadata
			pi.registerProvider(PROVIDER_ID, {
				name: "llama.cpp",
				baseUrl,
				apiKey,
				api: "openai-completions",
				models: currentModels,
			});
		} catch (error) {
			const err = error as Error;
			if (err.name !== "AbortError" && !isStaleContextError(err)) {
				console.warn(`[llama-cpp] /props for ${modelId} failed: ${err.message} (non-blocking, slot saving is best-effort)`);
			}
		} finally {
			// Always mark metadata as discovered and model as loaded — /props is best-effort
			discoveredMetadata.add(modelId);
			if (shouldAutoload) {
				currentlyLoadedModel = modelId;
			}
			clearTimeout(timer);
			pendingMetadata.delete(modelId);
			propsAbortController = null;
			if (sseAbortController) {
				sseAbortController.abort();
				sseAbortController = null;
			}
		}
	}

	await refreshProvider();

	// -----------------------------------------------------------------------
	// Auto-save / auto-restore event handlers (fail-safe: no-op if slots unsupported)
	// -----------------------------------------------------------------------

	// Save active slot before session shutdown
	const saveActiveSlot = async (): Promise<void> => {
		if (currentSlotId === null || activeModelName === null || slotSaveSupported === false) {
			return;
		}
		try {
			const ok = await saveSlot(currentSlotId);
			if (ok) {
				persistSlotCheckpointToSession(currentSlotId, activeModelName);
			}
		} catch {
			// Non-fatal — slot saving is best-effort
		}
	};

	// Try to restore slot on session start (resume from previous session)
	const restoreActiveSlot = async (): Promise<void> => {
		if (currentSlotId === null || activeModelName === null || slotSaveSupported === false) {
			return;
		}
		try {
			const slotName = activeModelName.split("/").join("_").replace(/[^a-zA-Z0-9_]/g, "_");
			const filename = `${slotName}_${currentSlotId}.kv`;
			const ok = await restoreSlot(currentSlotId, filename);
			if (ok) {
				console.log(`[llama-cpp] auto-restored slot ${currentSlotId} from ${filename}`);
			}
		} catch {
			// Non-fatal — slot restore is best-effort
		}
	};

	// Save slot on model switch (save old model, try restore new model)
	const switchModelAndSlots = async (modelId: string, ctx?: ExtensionCtx): Promise<void> => {
		if (slotSaveSupported === false || currentSlotId === null || activeModelName === null) {
			return;
		}
		try {
			// Save the previous model's slot before switching
			await saveSlot(currentSlotId);
			persistSlotCheckpointToSession(currentSlotId, activeModelName);
		} catch {
			// Non-fatal
		}

		// After the new model loads, try to restore its checkpoint
		setTimeout(async () => {
			if (slotSaveSupported === false || currentSlotId === null || activeModelName === null) {
				return;
			}
			try {
				const slotName = activeModelName.split("/").join("_").replace(/[^a-zA-Z0-9_]/g, "_");
				const filename = `${slotName}_${currentSlotId}.kv`;
				const ok = await restoreSlot(currentSlotId, filename);
				if (ok) {
					console.log(`[llama-cpp] model_select auto-restored slot ${currentSlotId} from ${filename}`);
				}
			} catch {
				// Non-fatal
			}
		}, 2000); // Wait for model autoload
	};

	pi.on("session_shutdown", async (event, ctx) => {
		await saveActiveSlot();
	});

	pi.on("session_start", async (event, ctx) => {
		currentSessionFile = ctx.sessionManager.getSessionFile();
		await discoverSlots();
		await restoreActiveSlot();
	});

	pi.on("model_select", async (event, ctx) => {
		if (event.model.provider !== PROVIDER_ID) {
			return;
		}
		void discoverModelMetadata(event.model.id, ctx, true, PROPS_TIMEOUT_MS, event.model);
		discoverSlotForModel(event.model.id, ctx);
		void switchModelAndSlots(event.model.id, ctx);
	});

	pi.on("input", async (event) => {
		const trimmed = event.text.trim().toLowerCase();
		if (trimmed === "/model") {
			await refreshProvider();
			await discoverSlots();
		}
	});

	const pendingSlotDiscoveries = new Set<string>();

	function discoverSlotForModel(modelId: string, ctx?: ExtensionCtx): void {
		if (pendingSlotDiscoveries.has(modelId)) {
			return;
		}
		pendingSlotDiscoveries.add(modelId);

		void (async () => {
			try {
				const response = await fetch(`${baseUrl}/models`);
				if (!response.ok) return;
				const payload: unknown = await response.json();
				if (!validateModelsResponse.Check(payload)) return;

				const models = payload.data as Array<{
					id: string;
					status?: { value?: string };
				}>;

				const loaded = models.find(
					(m) => m.id === modelId && m.status?.value === "loaded",
				);

				// Even if not loaded, record this model as known so
				// discoverSlots can pick it up after autoload.
				discoveredMetadata.add(modelId);

				if (loaded) {
					await discoverSlots();
					const slotName = modelId
						.split("/")
						.join("_")
						.replace(/[^a-zA-Z0-9_]/g, "_");
					ctx?.ui.notify(
						`[llama-cpp] Slot ${currentSlotId ?? "(none)"} active for ${modelId}`,
						"info",
					);
				}
			} catch {
				// Non-fatal
			} finally {
				pendingSlotDiscoveries.delete(modelId);
			}
		})();
	}



	pi.on("before_provider_request", (event, ctx) => {
		try {
			const modelId = (event.payload as { model?: unknown })?.model;
			if (typeof modelId === "string") {
				const activeModel =
					ctx.model?.provider === PROVIDER_ID && ctx.model.id === modelId ? ctx.model : undefined;
				void discoverModelMetadata(modelId, ctx, true, PROPS_TIMEOUT_MS, activeModel);
			}
		} catch (error) {
			if (!isStaleContextError(error)) {
				throw error;
			}
		}

		// Inject slot_id into the request payload if we have one
		const payload = event.payload as { [key: string]: unknown } | undefined;
		if (payload && typeof payload === "object" && currentSlotId !== null) {
			(payload as Record<string, unknown>).slot_id = currentSlotId;
		}
	});

	pi.on("session_start", async (event, ctx) => {
		currentSessionFile = ctx.sessionManager.getSessionFile();
	});
}
