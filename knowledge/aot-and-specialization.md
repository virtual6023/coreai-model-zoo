# Specialization, AIModelCache & AOT compilation (the ANE-later track)

> Foundation note for the **ANE-later / first-run-latency** track. Everything here is the official
> Core AI mechanism for getting a model from `.aimodel` to fast on-device execution, and for moving the
> one-time cost off the interactive path.
> Sources: WWDC **324 "Meet Core AI"** (`XJFfCVW1UZ0`), **326 "Integrate on-device AI models"** (`gl5lD2gEhb0`)
> — verbatim in `ondevice/_wwdc{324,326}_transcript.txt`; `coreai-models/models/README.md`,
> `swift/.../CoreAIShared/Bundle/ModelBundle.swift`, `skills/.../model-authoring/references/common_issues.md`,
> Apple docs `developer.apple.com/core-ai/` + `/documentation/coreai/compiling-core-ai-models-ahead-of-time`.

## What "specialization" is
A shipped `.aimodel` is a **source/device-agnostic** representation. To run it, the OS **specializes** it for
the *specific device + OS version*. Two transforms (324/326 verbatim):
1. a **core set of compilation steps** that segment, plan, and optimize compute — **this is where most of the latency is**;
2. **executable-artifact generation** for the compute units used — these artifacts are **tied to the device + OS version**.

The result is **cached**: first load pays the cost, later loads are fast. *"This process can take a significant
amount of time for very large models… avoid having model specialization occur within user-interactive flows."*

This is exactly this project's **re-specialization** finding: a *dynamic*-shape core re-specializes on every new
sequence length (~60–80× per-shape compile tax). (Project memories: `project_macos_speed_state`,
`reference_wwdc_coreai_sessions`; verbatim talks in `ondevice/_wwdc{324,326}_transcript.txt`.)

## Moving the cost off the interactive path (Swift API, 324 verbatim)
```swift
// 1) Check the cache; nil => not specialized yet => gate the feature / show "preparing…"
let cache = AIModelCache.default
guard let model = try cache.model(for: modelURL, options: .default) else {
    informUser("Preparing AI features. This may take a while…"); return
}

// 2) Or specialize explicitly, ahead of first use (after asset download / on opt-in)
try await AIModel.specialize(contentsOf: modelURL)
```
`AIModelCache` also: delete unused entries, control retention policy, and **share a cache across apps in one
app group**. `SpecializationOptions` configures how the model is optimized for inference (and, on macOS,
the preferred compute unit — see `runtime/_specialization_options.py`: `cpu_only()`, `default()`,
`from_preferred_compute_unit_kind(ComputeUnitKind.gpu()/.ane()/...)`). Article: *"Managing model
specialization and caching"*.

## Ahead-of-time (AOT) compilation — shift the compile to your dev machine
The expensive **compilation** step can be done ahead of time on the dev machine, producing a **compiled
model**; the device then only finishes the (much smaller) device-specific specialization. 326 verbatim:
*"…do some of that compilation ahead-of-time on my development machine… there is now much less work to do
and finishes significantly faster… generates one or more compiled models targeting specific device
architectures… a background asset for each compiled model."*

### Tool naming — RESOLVED (corrects the earlier "aimodelc not coreai-build" note)
- **CLI command you invoke = `xcrun coreai-build compile`.** Confirmed: `models/README.md:157`
  (*"Run `xcrun coreai-build compile --help` for usage"*), `ModelBundle.swift:101`, WWDC 326 verbatim
  (*"done with the coreai-build command"*), Apple docs.
- **Output artifact + underlying binary = `aimodelc` / `.aimodelc`.** The compiled bundle is
  `modelName.architectureName.aimodelc` (`ModelBundle.swift:103`); the runner accepts `.aimodel` or
  `.aimodelc` (`LLMRunnerMain.swift:719-722`); and `aimodelc` exists as a binary in Xcode's toolchain
  (`Xcode-beta.app/.../usr/bin/aimodelc`).
- So both names are real: **`coreai-build` = the verb, `aimodelc` = the compiler binary / compiled extension.**

### Flags (full surface, from `xcrun coreai-build compile --help`, verified 2026-06-10)
```
coreai-build compile <input.aimodel> [--output <dir>] [--platform iOS|macOS|watchOS|visionOS|tvOS ...]
    [--min-deployment-version 27.0] [--preferred-compute gpu|neural-engine|none]
    [--architecture <arch> ...] [--expect-frequent-reshapes]
```
- `--preferred-compute neural-engine|gpu` — pin the target compute unit. `neural-engine` is the concrete
  ANE-later lever (also the documented *fix* for "compiles but runs on CPU", `common_issues.md:112`); `gpu`
  pins the GPU `.aimodel` for the GPU track.
- `--expect-frequent-reshapes` — the CLI twin of `SpecializationOptions.expectFrequentReshapes` (the flag
  gemma4's 3-stage pipeline already needs); hint for dynamic/bucketed cores.
- Output: **one `.aimodelc` per device architecture**, named `base.<arch>.aimodelc`, each containing
  `main-<arch>.mlirb` + `main-<arch>-delegates`. Ship as **Background Assets**; app detects the device arch
  and requests the matching one (326). See "Compiling Core AI models ahead of time" + "Discover Apple-Hosted
  Background Assets".

### ⚠️ Architecture names track the DEVICE IDENTIFIER, not the marketing name (device-validated 2026-06-10)
The `--architecture` h-numbers follow the hardware **device-identifier major version** (`iPhone18,1`,
`Mac16,5`), NOT the marketing name ("iPhone 17 Pro", "M4 Max"):
- **iPhone 17 Pro = `iPhone18,1` → `h18p`.** An `h17p` `.aimodelc` pushed to it fails to load with
  `invalidCompiledModel`; the same model compiled `--architecture h18p` loads + runs (validated with the
  gemma4 int4km head, `AIModel(contentsOf:)` in CoreAIChat).
- **M4 Max Mac = `Mac16,x` → `h16c`.** Of all 20 macOS archs, only `h16c` loads in the Python runtime on an
  M4 Max (`ondevice/_aimodelc_head_check.py`); h17*/h16g/h16s all raise RuntimeError.
- **`coreai-build compile` EXITs 0 for ANY requested arch** — a successful compile does NOT validate the
  arch choice; only a device load does. (Earlier notes saying "h17p for iPhone 17 Pro" were name-matching,
  unvalidated — corrected here.)
- The same check also proved: a custom-Metal-kernel (`TorchMetalKernel`) model **survives AOT** — the
  `.aimodelc`'s `specialized_model_*.mpsgraph` contains the full `[[kernel]]` MSL signature + compiled MTLB
  in `resources.bin`, and the compiled asset's outputs are **bit-identical** to the source `.aimodel`.

### Deployment shape (326 demo)
Bundle the models out of the app download (they add >1 GB); gate the download behind a first-run feature
intro; **download assets → kick off specialization (with AOT already done) → fast first inference**, all off
the interactive flow. Subsequent inferences use the cached specialized asset.

## Status / caveats for this project (verified vs inferred)
- ✅ **AOT now WORKS on the beta — the toolchain-skew blocker (B2) is RESOLVED.** Verified end-to-end
  2026-06-10 (Xcode `27A5194q`, Metal Toolchain `v27.1.5194.15` / `metal 32023.917`, macOS 27.0 26A5353q):
  `xcrun coreai-build compile tiny.aimodel --platform macOS` → **EXIT 0**, 20 per-arch `.aimodelc`
  (`h13c…h17s`); `--platform iOS --preferred-compute neural-engine` → **EXIT 0**, 8 `.aimodelc`
  (`h13g h14g h15g h16g h16p h17g h17p h18p`). So AOT is testable on the real cores now — biggest open lever.
- 🟡 **Does AOT avoid the ANE first-run-compile OOM (that forces chunking)? — STRONG Mac-side evidence YES;
  device proof pending.** Verified 2026-06-10: AOT-compiling the un-chunked **35-layer monolith**
  (`gemma4_e2b_hostcache_L35_int8.aimodel`, 1.8 GB = the OOM size) for `--platform iOS
  --preferred-compute neural-engine --architecture h17p` → **EXIT 0, ~4.0 GB peak RSS**, and the `.aimodelc`
  embeds a **pre-compiled MPSGraph executable** (`main-h17p-delegates/MPSGraph/mpsExecutable.mpsgraphpackage`).
  The device OOM (jetsam ~6.4 GB) is during this very compile step → AOT moves it off-device and ships the
  pre-compiled artifact, so the residual on-device specialize is much lighter. **Still device-only to fully
  prove:** push the `.aimodelc` to the iPhone, load, confirm no jetsam (a gemma4-iOS device test). Caveat: I
  tested the host-cache monolith (MPSGraph delegate even with `neural-engine` preferred); re-run on the
  pure-ANE-primitive monolith for an exact result. If it holds, **gemma4-iOS can un-chunk**.
- **int4 head differs BY compute unit (resolved):** k-means palettization is `F.linear`-only, so the GPU
  int4-class head = a **fused-int8 Metal kernel** (GPU-only, [`custom-metal-kernels.md`](custom-metal-kernels.md)).
  The **ANE** can't run that MSL → its low-bit head path is **int4 per-output-channel *quantization* on a
  Conv2d head** (coreai-opt quant, not palettization) and/or **vocab pruning**, or split the head to the GPU.
- **The official iOS ANE stateful decode is blocked by the SAME KV-write bug** — Apple's own
  `KVCacheHandler` (`primitives/ios/cache.py`) uses the data-tensor `in_step` write that SIGSEGV/SIGTRAPs the
  beta (verified on GPU; device-ANE fails MLIR lowering). So ANE-later genuinely waits on the Apple fix
  (FB23024751); it is not a self-inflicted pattern. Apple's own skill even prescribes the **readonly-KV-I/O**
  (host-cache) pattern as the fix for stateful-reset (`common_issues.md:145-148`) — i.e. host-cache is an
  Apple-acknowledged workaround, not a hack.
- The ANE-later goal (~34 tok/s class) bundles three things: (1) **stateful KV** (blocked by the KV-write
  SIGSEGV — [`coreai-beta-mpsgraph-kvwrite-bug.md`](coreai-beta-mpsgraph-kvwrite-bug.md), memory
  `project_ane_vs_gpu_premise`), (2) **int4 / vocab-pruned head** (resolved above + [`compression-reference.md`](compression-reference.md)),
  (3) **AOT** (now unblocked ✅).
