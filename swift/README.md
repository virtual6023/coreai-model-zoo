# CoreAIRunner — Swift package

A small, self-contained Swift package that drives Core AI `.aimodel` LLM bundles — including the
**non-standard architectures** Apple's high-level `CoreAILM` pipeline can't (hybrid SSM states,
dual-KV + per-layer-embedding decoders). Built on the low-level `CoreAI` system framework only.

- `Sources/CoreAIRunner/HybridCoreAIEngine.swift` — generic **N-state** engine (Apple's
  `CoreAISequentialEngine` is hard-coded to 2 states; Qwen3.5/Gemma 4 need 4), fixed-capacity,
  greedy generate. Drives the all-in-one stateful `.aimodel` (`input_ids,position_ids → logits` +
  N in-place states).
- `Sources/CoreAIRunner/NDArrayHelpers.swift` — self-contained NDArray fill/read (no dependency
  on Apple's CoreAILanguageModels module).
- `Sources/coreai-run/` — a minimal CLI to validate the engine on macOS (feeds raw token ids,
  greedy-decodes, prints) before the iOS app.

## Status

⚠️ **DRAFT — written on macOS 26.6, not yet compiled** (the Core AI Swift runtime needs macOS 27).
Authored against the exact API used by Apple's `CoreAISequentialEngine.swift`. Build + fix on
macOS 27:

```bash
export DEVELOPER_DIR=/path/to/Xcode-beta.app/Contents/Developer   # Xcode 27 beta, no sudo needed
swift build
swift run coreai-run --model <bundle.aimodel> --vocab <N> --prompt "id,id,..." --max 5
```

See [`../knowledge/swift-runtime.md`](../knowledge/swift-runtime.md) for the API + per-model
runtime contracts, and [`../apps/CoreAIChat/`](../apps/CoreAIChat/) for the iOS chat app that
embeds this package.
