# Core AI knowledge base

Hard-won, verified notes on Apple's Core AI (iOS/macOS 27) — what the docs don't spell out.

## Orientation
- [`coreai-overview.md`](coreai-overview.md) — what Core AI is, the 3 Apple repos, the `.aimodel`
  format, the PyTorch → `.aimodel` → Swift-runtime pipeline.
- [`conversion-guide.md`](conversion-guide.md) — converting a PyTorch model to `.aimodel`: the
  canonical `TorchConverter` API + the gotchas that cost real time.
- [`compute-units-and-authoring.md`](compute-units-and-authoring.md) — **ANE vs GPU vs CPU**: the
  static/BC1S/Conv2d/per-head/fp16 ANE rules vs the dynamic/fused/custom-kernel GPU rules, the
  macOS↔iOS export split, and the PSNR verification gates. Read this to choose a target.
- [`performance-ceiling.md`](performance-ceiling.md) — **reality check**: where Core AI LLM decode tops out
  (Mac GPU near its ceiling, MLX gap structural, fusion closed), what AOT does/doesn't do, and why ANE is an
  energy play not a speed one. Read before chasing a "dramatic speed" win.

## GPU-now track (speed)
- [`pipelined-engine.md`](pipelined-engine.md) — **read this first for decode speed**: riding Apple's
  `coreai-pipelined` engine = 3.5× over a hand-rolled per-token loop with ZERO custom kernels
  (qwen3.5: Mac 204 tok/s, iPhone 50.3–51.5). The decode-only loop-free export, the extra-states
  engine patch, the chunk=1 / warmup-256 traps, LUT-vs-linear int8, oracle gating, and what
  fits/doesn't (Gemma 4's PLE doesn't — yet).
- [`custom-metal-kernels.md`](custom-metal-kernels.md) — `TorchMetalKernel` (WWDC 325): the API, the
  register-then-add order, MSL embedded in the `.aimodel`, GPU-only, what to (and not to) kernelize.
  Still the tool when a model CAN'T ride the pipelined engine (e.g. Gemma 4).
- [`tensorops-quantized-kernels.md`](tensorops-quantized-kernels.md) — the layer UNDER custom kernels
  (WWDC 330): TensorOps quantized matmul (int4/int8 = OS 26; fp4/fp8/int2 + E8M0 scale planes = OS 27),
  cooperative tensors, the FlashAttention recipe, and the M5/A19 GPU **neural accelerator** — the
  compute-bound/prefill lever hand-rolled MSL can't reach.

## ANE-later track (when the beta KV-write bug lifts + int4 head + AOT)
- [`aot-and-specialization.md`](aot-and-specialization.md) — specialization, `AIModelCache` /
  `AIModel.specialize()`, and AOT compile (`xcrun coreai-build compile` → `.aimodelc`,
  `--preferred-compute neural-engine`). The first-run-latency mitigation path.
- [`compression-reference.md`](compression-reference.md) — `coreai-opt` quantization & palettization
  API reference (int4/int8, granularity, mixed-precision, joint); the LM-head/embedding lever.
- [`coreai-beta-mpsgraph-kvwrite-bug.md`](coreai-beta-mpsgraph-kvwrite-bug.md) — the data-indexed
  in-graph KV write SIGSEGV (FB23024751 / apple#5): platform-agnostic (GPU too), host-cache workaround.

## Runtime & decode internals
- [`compression.md`](compression.md) — this project's LLM-specific empirical compression notes
  (int8 floor, per-subsystem sensitivity); pairs with `compression-reference.md`.
- [`stateful-kv-cache.md`](stateful-kv-cache.md) — stateful decode export, dual/hybrid KV state,
  the sliding-window ring buffer, the dynamic prefill+decode graph.
- [`swift-runtime.md`](swift-runtime.md) — the Core AI Swift API, driving `.aimodel` from Swift,
  non-standard architectures, macOS/Xcode 27 setup (incl. running Xcode 27 beta without sudo).

Primary official sources behind these notes: the open repos (`coreai-torch`, `coreai-optimization`,
`coreai-models` incl. its agent skills), the WWDC26 talks **324 / 325 / 326 / 330** (verbatim transcripts in
`ondevice/_wwdc{324,325,326,330}_transcript.txt`), and `developer.apple.com/core-ai/`. Verified against
Hugging Face references (convert + numeric parity); on-device iOS 27 notes marked where still in bring-up.
