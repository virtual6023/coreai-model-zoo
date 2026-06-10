# Gemma 4 E2B (text decoder) — Core AI

Gemma 4 E2B multimodal model; this card is the **text decoder** (`model_type` `gemma4` /
`gemma4_text`). Source: `google/gemma-4-E2B-it`.

Signature features (all handled in the re-authored model): 35 layers, dual head_dim (sliding 256 /
full 512), attention **scale = 1.0** (QK-norm bounds magnitudes), per-head Q/K RMSNorm + scale-free
V RMSNorm, **KV-sharing** (last 20 layers reuse a producer's K/V), double-wide MLP on shared
layers, **Per-Layer Embeddings** (gated per-layer skip), dual RoPE (sliding θ=1e4 full / full θ=1e6
proportional), final logit softcap `tanh(z/30)·30`. RMSNorm multiplies by `weight` directly.

## On-device status (iOS/macOS 27 beta — measured, greedy, 8/8 top-1 vs HF)

<!-- Mac numbers RELEASE-VERIFIED by R2 2026-06-10 (ondevice/MACOS_RELEASE_README.md);
     iOS GPU 17.7 = R1 int4km kernel path 2026-06-10 (RELEASE_PLAN row #3); before publish:
     confirm build config (Release) + re-measure ANE (the ~6 is a suspected Debug-underestimate
     per the qwen-static session lesson). -->

| | macOS GPU (M4 Max) | iOS GPU (iPhone 17 Pro) | iOS ANE (iPhone 17 Pro) |
|---|---|---|---|
| Correctness | ✅ 8/8 exact | ✅ 8/8 exact | ✅ 8/8 exact |
| Decode | **56.6–59.0 tok/s** (Swift e2e; core ~70 tok/s) | **17.7 tok/s** | **~6 tok/s** |
| Path | host-cache fixed-shape + custom Metal kernels (fused-**int8** FFN + head+argmax) | same kernel family at **int4 k-means** (1.3 GB core), monolith + host KV | 6-chunk host-cache, fp32-safe ANE authoring, on-ANE argmax head |

- "What is the capital of France?" → "The capital of France is **Paris**." on every cell.
- macOS ANE is intentionally out of scope (GPU dominates on a Mac; the runtime auto-prefers GPU
  for this structure).
- The ANE number is **energy-play honest**: it is capped by the 262144-vocab head (→ int4/pruned
  head, separate lever) and the host-cache KV re-feed (→ the beta KV-write bug,
  [`../knowledge/coreai-beta-mpsgraph-kvwrite-bug.md`](../knowledge/coreai-beta-mpsgraph-kvwrite-bug.md);
  an in-graph escape — write-mask-as-input blend — is Mac-GPU-proven, device test pending) — see
  [`../knowledge/performance-ceiling.md`](../knowledge/performance-ceiling.md).

### What it took (the interesting parts)

1. **Fixed-shape host-cache decode** — the beta crashes on any data-indexed in-graph KV write
   (FB23024751 / [apple/coreai-models#5](https://github.com/apple/coreai-models/issues/5)), so KV
   caches are plain I/O: in-graph `cat`, masked SDPA, host writes the new column back. 8/8,
   and it unblocked Mac GPU + device GPU + device ANE with one core.
2. **Custom Metal kernels (GPU)** — fused int8 dequant-LUT matvec for the FFN and the
   262144-vocab head (+ two-level in-kernel argmax, no logit readback). MSL embedded in the
   `.aimodel` (WWDC 325), still 100% Core AI — and it **survives AOT** (`coreai-build` →
   `.aimodelc`, device output bit-identical). Mac decode 13 → 27 → ~57 tok/s. On the iPhone the
   same kernels at **int4 k-means** buy another 1.43× (device is bandwidth-bound where the Mac
   is ALU-bound) → 17.7 tok/s.
3. **ANE fp16 exactness** — two root causes, both fixed default-on (GPU-equiv ~2e-8):
   composite RMSNorm overflows fp16 `mean(x²)` on gemma4's large activations → `[x,-x]`
   LayerNorm trick; `nn.Linear` accumulates fp16 → Conv2d 1×1 (fp32 MAC on the conv engine).
   `.float()` casts are no-ops on the ANE.
4. **Chunking for the ANE** — the 35-layer monolith OOMs the on-device first-run compile →
   6 chunks (≤8 layers). AOT compilation (`xcrun coreai-build compile`) moves that compile
   off-device and likely un-chunks it; device proof pending
   ([`../knowledge/aot-and-specialization.md`](../knowledge/aot-and-specialization.md)).

## Conversion status (macOS, vs HF eager — 8-token canonical prompt)

- Eager parity: argmax exact; logits maxdiff 1.4e-4.
- Decode core: fp32 (7.0 GB) / **fp16 8/8 (3.5 GB)** / **int8 k-means 8/8 (1.9 GB)**.
- **Stateful** dual-KV core + **ring-buffer** sliding cache (constant ~6 MB sliding KV vs growing) —
  single-pass + incremental match stateless; ring is argmax-exact incl. wrap.
- Tied **head** + softcap as its own unit: fp16 (768 MB) / int8 (392 MB), 8/8.
- Full pipeline (front-end gather → core → head → argmax): **8/8 + top-5 vs HF**, fp16 and int8.
- int4 cannot reach exact (best 6/8); int8 is the floor. gate/up projections must be int8.

## On-device artifacts (int8, 3-stage)

| Stage | Bundle | I/O |
|---|---|---|
| Front-end gather | `gemma4_e2b_frontend_int8` (2.6 GB) | `input_ids → inputs_embeds, per_layer_inputs` — on device this becomes a Swift **mmap gather** (`Gemma4Gather`), the tables never enter process memory |
| Decode core | host-cache fixed-shape core, int8 (+ Metal-kernel FFN variant for GPU) ~1.8–2.0 GB | `inputs_embeds, per_layer_inputs, position_ids, masks, KV-cache I/O → hidden, new K/V columns` |
| Head | `gemma4_e2b_int8_head` (392 MB) / GPU: fused head+argmax kernel (388 MB) | `hidden → logits` (tied lm_head + softcap) / GPU: `hidden → (value,index) partials` |

Full int8 set ~4.9 GB on disk; runs in budget on an iPhone 17 Pro (mmap front-end keeps the
resident footprint flat). Dual-KV state names (when the stateful path returns post-beta):
`slidingKeyCache/slidingValueCache/fullKeyCache/fullValueCache`. Flow details:
[`../knowledge/swift-runtime.md`](../knowledge/swift-runtime.md).

## Reproduce

Re-authored decoder + stateful + ring + head + front-end gather in `conversion/`. CLI:
`coreai.llm.export gemma-4-e2b --compression int8` (core); `convert_head.py int8` (head);
`export_gemma4_frontend.py` (front-end gather). Fixed-shape host-cache core + chunked ANE export +
Metal-kernel variants ship alongside (`export_gemma4_hostcache*.py`, `export_gemma4_metal.py`,
`export_gemma4_head_kernel.py`). E4B (adds MoE) is the next variant on the same path.

Reference CoreML (NOT Core AI) throughput for scale: Gemma4-E2B ~34 tok/s on iPhone 17 Pro
(stateful KV + pruned head + AOT — the stack Core AI reaches once the beta KV-write bug lifts).
