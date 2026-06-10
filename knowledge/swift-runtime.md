# Core AI Swift runtime (on-device)

How to drive a `.aimodel` LLM bundle from Swift on iOS/macOS 27 — including non-standard
architectures Apple's high-level `CoreAILM` pipeline can't express.

## Toolchain setup (macOS 26.4+ is enough for Xcode 27)

Xcode 27 beta requires only **macOS 26.4+** (NOT macOS 27). You can build + deploy an iOS-27 app
from macOS 26.x. (Running the Core AI Swift runtime *as a macOS CLI* does need macOS 27, since the
package declares `.macOS("27.0")`.) Use the beta without moving it to /Applications or sudo:

```bash
export DEVELOPER_DIR=/path/to/Xcode-beta.app/Contents/Developer
xcodebuild -version            # Xcode 27.x
xcrun coreai-build --help      # the AOT CLI (the verb): compile/package/inspect/metadata
xcrun --find aimodelc          # the underlying compiler binary (+ the .aimodelc output extension)
xcrun devicectl list devices   # connected iPhone (iOS 27)
```

`CoreAI.framework` lives in the iOS 27 / macOS 27 SDKs. AOT-compile (now VERIFIED working on the beta,
2026-06-10): `xcrun coreai-build compile <m>.aimodel --platform iOS --preferred-compute neural-engine`
→ per-arch `.aimodelc`. (Earlier note here said "NOT coreai-build" — that was wrong: `coreai-build` is the
command, `aimodelc` the binary/extension.) For specialization, `AIModelCache` / `AIModel.specialize()` and
the full flag list: see [`aot-and-specialization.md`](aot-and-specialization.md).

## Pushing model files to the device — verify the copy BEFORE the first load

Push bundles into the app sandbox with `xcrun devicectl device copy to --domain-type
appDataContainer --domain-identifier <bundle-id> --source <m>.aimodel --destination
Documents/models/<m>.aimodel`. **Burned-in gotcha (2026-06-10): a load attempt against a
partially-copied `.aimodel` permanently poisons the on-device specialization cache** — the cache
(`Library/Caches/coreai-cache/<os-build>/<bundle-id>/<content-hash>/`) is keyed by content hash, so
once a half-pushed file's load fails mid-specialize, every later load of that model errors
`NSPOSIXErrorDomain Code=2` (ENOENT) *even after the copy completes or the file is renamed*.
Recovery is painful (deleting the live cache dir from app code at startup hangs; uninstalling the
app wipes `Documents/`). So: after every multi-GB push, list the destination
(`xcrun devicectl device info files … --subdirectory Documents/models/<m>.aimodel`) and confirm
`main.mlirb` is full-size before launching anything that loads it. AOT'd `.aimodelc` bundles skip
the heavy on-device specialize step entirely (warm load 0.0 s) — see
[`aot-and-specialization.md`](aot-and-specialization.md), including the ⚠️ architecture-naming rule
(`iPhone18,1` → `h18p`, not the marketing name).

## Apple's high-level pipeline is standard-only

`coreai-models/swift` (`CoreAILM` library) assumes a STANDARD model: `input_ids → logits`, single
KV cache, `ModelShapeConfig` = (entrypoint, ctx, query_size). It can't express:
- hybrid SSM states (Qwen3.5: 4 states — keyCache, valueCache, convState, recState),
- dual-KV + per-layer-embedding front-end (Gemma 4).

So for those you write a **thin custom runner** on the low-level `CoreAI` framework, reusing the
tokenizer (swift-transformers) + samplers. Apple's `CoreAISequentialEngine.swift` (2-state KV) is
the perfect template — generalize it to N states.

## The low-level API (verified, from `CoreAISequentialEngine`)

```swift
import CoreAI
let prepared = try await PreparedModel.prepare(at: url)
let model = prepared.model
let desc = model.functionDescriptor(for: "main")!          // .inputNames/.outputNames/.stateNames
guard case .ndArray(let d) = desc.inputDescriptor(of: name) else { ... }   // also output/stateDescriptor(of:)
let resolved = d.resolvingDynamicDimensions(d.shape.map { $0 < 0 ? cap : $0 })  // dynamic dims are < 0
var arr = NDArray(descriptor: resolved)                    // fill via mutableView(as:).withUnsafeMutablePointer
let fn = try model.loadFunction(named: "main")!

var states = InferenceFunction.MutableViews()
states.insert(&keyCache, for: keyCacheName)                // ... one insert per state (in-place, persist)
var outputs = InferenceFunction.MutableViews()
outputs.insert(&logits, for: logitsName)
_ = try await fn.run(inputs: [inputIdsName: inputIds, positionIdsName: positionIds],
                     states: consume states, outputViews: consume outputs)
```

- States are **mutated in place** across calls — reuse the same buffers to persist KV/SSM state.
- One dynamic graph does **prefill + decode**: `offset = position_ids.len − query_len`. Prefill =
  full positions + zero state; decode = 1 query token + persisted state. Position ids are the full
  `[0..total)` each call.
- Logits are typically fp16 (`LogitsScalarType = Float16`); the bundle may be `last_token_only`
  (output `[1,1,vocab]`).
- Module-internal helpers in `CoreAILanguageModels`: `fillNDArray`, `readNDArray`, `lastTokenLogits`.

A generic N-state runner built on this is in [`../swift/Sources/CoreAIRunner/`](../swift/Sources/CoreAIRunner/).

## Model-specific notes

- **Qwen3.5**: all-in-one bundle `input_ids,position_ids → logits` + 4 states. No logit softcap.
  Feed fp16 state. (0.8B/2B differ only in width.)
- **Gemma 4 E2B**: 3 stages — front-end gather (functions `gather_embeds`/`gather_per_layer`,
  holds the big int8 embedding tables) → core (`inputs_embeds, per_layer_inputs, position_ids` + 4
  dual-KV states `slidingKeyCache/slidingValueCache/fullKeyCache/fullValueCache` → `hidden`) →
  head (`hidden → logits`, tied lm_head + `tanh(z/30)·30` softcap). Core is fp16 (cast `hidden` to
  fp32 before the head); CLI-exported core's dynamic seq starts at 16 (pad short prompts).
