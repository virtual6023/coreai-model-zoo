# Qwen3.5 (text decoder) — Core AI

Hybrid linear + full-attention decoder (Mamba/SSM-style linear mixer + periodic full attention,
gated-delta, partial mRoPE, MTP head). Source: `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-2B`
(image-text-to-text checkpoints; this card is the **text decoder**).

To our knowledge this is the first hybrid linear-attention (SSM) LLM running on Core AI —
including on the iPhone **Neural Engine**.

## On-device status (iOS/macOS 27 beta — measured, greedy)

<!-- Mac numbers RELEASE-VERIFIED by R2 2026-06-10 (ondevice/MACOS_RELEASE_README.md);
     dynamic device numbers 2026-06-09; static device numbers 2026-06-10
     (ondevice/_qwen_ios_fast_RESULTS.md §"更新 2026-06-10 (午後)"). All iOS numbers are
     RELEASE builds. -->

Two engines, both 100% Core AI:

| Engine | macOS GPU (M4 Max) | iOS GPU (iPhone 17 Pro) | iOS ANE (iPhone 17 Pro) |
|---|---|---|---|
| **dynamic** (int8, 969 MB — shipped app) | **58.5 tok/s** · 8/8 | 12.5 · 8/8 | **14.7 · 8/8** |
| **static fixed-shape** (fp16 host-cache monolith, **ctx 2048**) | — | **27.7 tok/s** · exact ← release config | ✗ numerics (this beta) |

- "The capital of France is" → " Paris." on every shipped cell (static: Mac-GPU 8/8 vs HF +
  on-device greedy correct).
- **The static fixed-shape engine is ~2× the dynamic path on the iPhone GPU** (27.7 vs 14.7
  best-dynamic) — the re-specialization tax gone, one dispatch per token (monolith).
- **Context capacity is nearly free on this architecture**: growing the fixed bucket 8×
  (256 → 2048) costs only ~9% (30.4 → 27.7 tok/s) — only the 6 full-attention layers carry
  growing KV; the 18 SSM layers' states are fixed-size. Hybrid models suit fixed-shape export
  unusually well. Dynamic stays as the unbounded-context fallback.
- Within the dynamic engine the **ANE beats the GPU** (14.7 vs 12.5) — a hybrid SSM decoding on a
  phone NPU, exactly. The *static* ANE variant is blocked **this beta**: the loop-free SSM
  recurrence needs fp32 accumulation, the ANE executes fp16-only, there is no blessed SSM
  composite to externalize, and custom kernels are GPU-only. Re-test each new beta.
- Static is bandwidth/compute-bound now (~32 ms/tok ≈ reading 1.5 GB fp16 weights per token) —
  the next lever to 35+ tok/s is int8/int4 weights (the gemma4 fused-int8/int4 kernel path,
  already proven on the iPhone GPU). Open: q>1 chunked prefill for long prompts.
- Prefill ≈ decode tok/s by construction (s=1 chunked prefill through the same graph).
- ⚠️ **Benchmark release builds only** (Mac AND iOS): a debug binary measures the Mac at ~19
  tok/s (vs 58.5) and the static iOS path at ~10 (vs ~30) — per-token host work dominates
  unoptimized Swift.

### The device unlock: loop-free decode

The gated-delta recurrence is a sequential scan (`torch.ops.higher_order.while_loop` via Apple's
`GatedDeltaUpdate` composite). It converts and runs on the **Python** runtime, but the `scf.while`
doesn't lower on either **Swift** delegate. The fix: at `query_len == 1` the scan is one step —
a loop-free single-step recurrence (`use_loopfree_step`) is **bit-identical** (maxdiff 0.0) to the
composite at s=1. Decode runs the loop-free graph; prefill runs as chunked s=1 steps (zero-state
OK). That makes the hybrid architecture device-deployable today, with no converter patch.

## Conversion status (macOS, vs HF eager)

- Prefill + stateful decode: **cosine 1.0 / top-1 100%** (fp32).
- **fp16**: top-1 exact; multi-token generation stable (the gated-delta recurrent state is a leaky
  integrator, so fp16 rounding decays — no mixed-precision SSM state needed; only fix was casting
  RoPE cos/sin to the query dtype).
- **int8 k-means palettization**: top-1 EXACT (prefill + decode). int4 degrades (not shippable).

## On-device artifacts (int8, all-in-one stateful bundle)

`input_ids, position_ids → logits` with **4 in-place states** `(keyCache, valueCache, convState,
recState)`; fp16 embed + tied lm_head in-graph, int8 transformer (conv1d full precision);
`last_token_only` → logits `[1,1,vocab]`.

| Size | Bundle | Parity |
|---|---|---|
| 0.8B | **969 MB** | prefill + decode top-1 exact (Mac + iPhone GPU/ANE) |
| 2B | **2.2 GB** | prefill + decode top-1 exact (Mac) |

The app pushes the bundle into its sandbox (`devicectl device copy to --domain-type
appDataContainer`) rather than bundling it. Runtime contract + state shapes:
[`../knowledge/swift-runtime.md`](../knowledge/swift-runtime.md) and
[`../knowledge/stateful-kv-cache.md`](../knowledge/stateful-kv-cache.md). Apple's Swift engine is
2-state; this model needs the generic N-state runner (`swift/`).

## Reproduce

Re-authored decoder + hybrid 4-state export live in `conversion/` (port of `coreai_models`'s macOS
authoring path + a `qwen3_5_text` registry entry). On-device bundle: `conversion/export_qwen3_5.py
[0.8b|2b]` (int8 k-means stateful); decode graph `export_qwen3_5_decode.py [--int8]`. HF reference
oracle generated separately (a transformers build with `qwen3_5`).

Reference CoreML (NOT Core AI) device throughput for scale: 0.8B ~48 tok/s, 2B ~27 tok/s on iPhone
17 Pro.
