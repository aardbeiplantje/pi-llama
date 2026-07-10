# Slot Save/Restore Investigation

## Background — The Reality

**llama.cpp does NOT auto-save/restore slot KV cache to disk.** The `--slot-save-path`
flag only enables REST API endpoints; it does not trigger automatic persistence.

Key sources:
- Feature request #17107 (automatic disk persistence) was **closed as "not planned"**
- Community confirmation: "llama.cpp exposes `POST /slots/<id>?action=save|restore`
  so you can save/restore a KV slot to disk by hand. It won't do it for you"
- The `/llama.cpp/slots/` directory on the server is **empty** — no files have been
  written

## How It Actually Works

### llama.cpp REST API

When `--slot-save-path /llama.cpp/slots` is set, the server exposes:

```
POST /slots/<id>?action=save     → saves slot <id>'s KV cache to disk
POST /slots/<id>?action=restore  → restores slot <id>'s KV cache from disk
```

Example:
```bash
# Save slot 0's KV cache to disk
curl -X POST "http://[2a02:a03f:8789:e700::20]:8000/slots/0?action=save"

# Restore slot 0's KV cache from disk
curl -X POST "http://[2a02:a03f:8789:e700::20]:8000/slots/0?action=restore"
```

Files are written to `--slot-save-path` with names like:
```
/llama.cpp/slots/0-kv-cache.bin
```
(Exact naming convention may vary by version)

### Measured Performance (from community)

On a ~4K token slot:
- **Save**: 211ms for 219MB file
- **Restore**: 87ms
- **Cold re-prefill** for a 5K chat: ~9.9s
- **Restore** for a 5K chat: ~1.4s (≈7× faster)
- File size grows linearly with conversation length (50K tokens ≈ 2.7GB)

## What We NEED To Do

Since the server won't auto-save, **pi-llama must drive the save/restore API calls**
at the right moments:

### Save Triggers

| Trigger | Slot ID | What to save |
|---------|---------|--------------|
| `model_select` (switching away) | active slot(s) | KV cache before model unloads |
| `session_shutdown` | active slot(s) | KV cache before session ends |
| `session_before_switch` (new/resume) | active slot(s) | KV cache before switching |

### Restore Triggers

| Trigger | Slot ID | What to restore |
|---------|---------|------------------|
| `session_start` (reason: "resume") | preferred slot | KV cache from last session |
| `model_select` (same model, different slot) | same slot ID | KV cache from previous conversation |

## Implementation Architecture

```typescript
// State tracking
let activeSlotId: string | null = null;
let lastSavedSlotId: string | null = null;
let slotCheckpoints: Record<string, { modelId: string; timestamp: number }> = {};

// REST client to llama.cpp server
const LLAMA_BASE_URL = process.env.LEMONADE_URL || "http://[2a02:a03f:8789:e700::20]:8000";

async function saveSlot(slotId: string): Promise<void> {
  await fetch(`${LLAMA_BASE_URL}/slots/${slotId}?action=save`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });
}

async function restoreSlot(slotId: string): Promise<void> {
  await fetch(`${LLAMA_BASE_URL}/slots/${slotId}?action=restore`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });
}

// Event handlers
pi.on("model_select", async (event, ctx) => {
  // Save old model's slot before it unloads
  if (activeSlotId && event.previousModel?.provider === PROVIDER_ID) {
    try {
      await saveSlot(activeSlotId);
      slotCheckpoints[activeSlotId] = {
        modelId: event.previousModel.id,
        timestamp: Date.now(),
      };
    } catch (e) {
      pi.log.warn("Failed to save slot:", e);
    }
  }

  // Restore slot when switching to a model we've seen before
  if (event.model.provider === PROVIDER_ID && event.model.id) {
    // Find a checkpoint for this model and restore
    const matchingSlotId = Object.keys(slotCheckpoints).find(
      (id) => slotCheckpoints[id].modelId === event.model.id
    );
    if (matchingSlotId) {
      try {
        await restoreSlot(matchingSlotId);
        activeSlotId = matchingSlotId;
      } catch (e) {
        pi.log.warn("Failed to restore slot:", e);
      }
    }
  }
});

pi.on("session_shutdown", async (event, ctx) => {
  if (activeSlotId) {
    try {
      await saveSlot(activeSlotId);
    } catch (e) {
      pi.log.warn("Failed to save slot on shutdown:", e);
    }
  }
});

pi.on("session_start", async (event, ctx) => {
  // On resume, try to restore the previous session's slot
  if (event.reason === "resume" && lastSavedSlotId) {
    try {
      await restoreSlot(lastSavedSlotId);
      activeSlotId = lastSavedSlotId;
    } catch (e) {
      pi.log.warn("Failed to restore slot on resume:", e);
    }
  }
});
```

## Key Findings

1. **We must call the API ourselves** — no automatic slot save/restore exists in
   llama.cpp. This is the critical difference from what I initially assumed.

2. **The REST API is simple** — just `POST /slots/<id>?action=save|restore`. The
   implementation is primarily about *when* to call them.

3. **Session file persistence still applies** — use `pi.appendEntry()` to record
   which slot ID was active in each session, so cross-session restore knows which
   slot to target.

4. **Performance gain is real** — restore is ~7× faster than cold re-prefill for
   typical conversations. The 200-300ms save/restore overhead is well worth it for
   conversations longer than a few hundred tokens.

5. **Slot ID mapping is the challenge** — when a model is unloaded/reloaded, does
   the slot ID change? If the server re-assigns slot IDs, we need to track the
   mapping between "slot for model X" and the current slot ID. We may need to
   query the `/slots` or `/v1/models` endpoint to discover active slot IDs.

6. **Error handling is critical** — the server may not respond to save/restore
   calls if the model has already fully unloaded. Wrap in try/catch.

## Updated Recommended Scope (Phase 1)

Given the manual API requirement:

1. **model_select** — Call `POST /slots/<id>?action=save` for the old model's slot,
   then call `POST /slots/<id>?action=restore` for a previously-saved slot of the new
   model (if one exists). Track slot ID ↔ model ID mapping.

2. **session_shutdown** — Save the active slot. Record which slot ID belongs to this
   session in the session file via `pi.appendEntry()`.

3. **session_start** (reason: "resume") — Look up the previous session's slot ID
   from the session file, restore it.

4. **Monitor `/v1/models`** — Query periodically to discover current slot IDs. The
   response includes `status.value` (loading/loaded/sleeping) which tells us when
   a slot is available for save/restore.

## Remaining Questions

1. **Does the slot ID survive model reload?** Need to test: unload a model, reload
   it, check if the slot ID stays the same.
2. **What happens with multiple slots per model?** The server supports multiple slots
   per model — we need to know which slot corresponds to our conversation.
3. **What if save/restore fails mid-conversation?** The KV cache is partial — is it
   still usable? Probably yes, since the server only loads what was saved.
4. **Is there a way to list saved slot files?** We may need to parse the slot-save-path
   directory or use the server's API to discover which slot files exist.
