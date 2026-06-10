# Qwen3.5 (text decoder) — Core AI

Hybrid linear + full-attention decoder (Mamba/SSM-style linear mixer + periodic full attention,
gated-delta, partial mRoPE, MTP head). Source: `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-2B`
(image-text-to-text checkpoints; this card is the **text decoder**).

**⬇️ Converted `.aimodel` bundles (ready to run):
[mlboydaisuke/qwen3.5-0.8B-CoreAI](https://huggingface.co/mlboydaisuke/qwen3.5-0.8B-CoreAI)** —
`ios-gpu/` (static ctx-2048: **int8 fused-kernel monolith 42.5–45.4 tok/s** + its q16 chunked-
prefill companion **147 tok/s prefill**, plus the previous fp16 monolith 27.7), `ios-ane/` +
`macos/` (dynamic int8, 14.7 / 58.5 tok/s).

To our knowledge this is the first hybrid linear-attention (SSM) LLM running on Core AI —
including on the iPhone **Neural Engine**.

## On-device status (iOS/macOS 27 beta — measured, greedy)

<!-- Mac numbers RELEASE-VERIFIED by R2 2026-06-10 (ondevice/MACOS_RELEASE_README.md);
     dynamic device numbers 2026-06-09; static fp16 numbers 2026-06-10 PM; int8-kernel +
     chunked-prefill numbers 2026-06-10 evening (ondevice/_qwen_ios_fast_RESULTS.md
     §Session Q — Q1/Q2). All iOS numbers are RELEASE builds. -->

Two engines, both 100% Core AI:

| Engine | macOS GPU (M4 Max) | iOS GPU (iPhone 17 Pro) | iOS ANE (iPhone 17 Pro) |
|---|---|---|---|
| **dynamic** (int8, 969 MB — shipped app) | **58.5 tok/s** · 8/8 | 12.5 · 8/8 | **14.7 · 8/8** |
| **static fixed-shape** (fp16 host-cache monolith, **ctx 2048**) | — | 27.7 tok/s · exact | ✗ numerics (this beta) |
| **static + fused int8 Metal kernels** (ctx 2048, GPU argmax head) | — | **42.5–45.4 tok/s** · exact ← release config | GPU-only (custom kernels) |
| **+ chunked prefill** (q16 blocks, int8 LUT companion graph) | — | **prefill 147 tok/s** (185-tok prompt: 4.2 s → 1.26 s; decode unchanged) | — |

- "The capital of France is" → " Paris." on every shipped cell (static: Mac-GPU 8/8 vs HF +
  on-device greedy correct; the int8-kernel monolith is ALSO Mac-GPU 8/8 EXACT and produced
  byte-identical device output to the fp16 path on a 185-token prompt).
- **The int8 fused-kernel monolith is ~3× the dynamic path and ~1.6× the fp16 static path**
  (42.5–45.4 vs 27.7): the device GPU is weight-bandwidth-bound (~1.5 GB of fp16 weights per
  token), so fused dequant-in-matvec Metal kernels (k-means LUT, fp32 accumulate — embedded in
  the `.aimodel`, 100% Core AI) halve the stream to ~760 MB/token. The 248320-token tied head
  runs as a fused matvec + two-level **GPU argmax** (greedy; the host reduces vocab/8 partials).
- **int4 does NOT survive on this model** (PyTorch gate, k-means group-32: head 3/8, MLP 0/8,
  SSM 2/8 vs the HF oracle) — unlike gemma4, whose FFN+head take int4 k-means at 8/8. int8 is
  the qwen sweet spot; 4-bit would need finer codebooks (kernel-level change).
- **Chunked prefill (q16 blocks)**: a prefill-only companion graph consumes the prompt 16 tokens
  per pass (SSM scan unrolled in-graph, fp32 recurrence; KV/conv/rec states handed off to the
  decode graph; FULL blocks only — the remainder uses q=1 decode). 185-token prompt: 4.2 s →
  **1.26 s (147 tok/s, 6.8 ms/token)**, decode untouched, output identical. q32 measured NOT
  faster (blocks become unroll/SDPA-bound, and the q=1 remainder grows). The companion is int8
  (MPSGraph LUT — fp16 prefill + the decode monolith together exceed the app memory budget).
- **Context capacity is nearly free on this architecture**: growing the fixed bucket 8×
  (256 → 2048) costs only ~9% (30.4 → 27.7 tok/s) — only the 6 full-attention layers carry
  growing KV; the 18 SSM layers' states are fixed-size. Hybrid models suit fixed-shape export
  unusually well. Dynamic stays as the unbounded-context fallback.
- Within the dynamic engine the **ANE beats the GPU** (14.7 vs 12.5) — a hybrid SSM decoding on a
  phone NPU, exactly. The *static* ANE variant is blocked **this beta**: the loop-free SSM
  recurrence needs fp32 accumulation, the ANE executes fp16-only, there is no blessed SSM
  composite to externalize, and custom kernels are GPU-only. Re-test each new beta.
- The int8-kernel monolith is still weight-bandwidth-bound (~21 ms/tok ≈ 760 MB int8 stream at
  ~36 GB/s — the same kernel bandwidth measured on gemma4); the rest is KV/SDPA I/O and
  dispatch. Both former "next levers" (int8 kernels, chunked prefill) are now shipped.
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

The static iOS exports (workspace scripts, to be consolidated here): host-cache monolith
`export_qwen3_5_hostcache.py --num-chunks 1 --max-ctx 2048 [--kind int8v3]` (fp16 or fused-kernel
+ GPU-argmax head; `--verify` = chained greedy 8/8 vs the HF oracle on the Mac GPU) and the
prefill companion `export_qwen3_5_prefill.py --chunk 16 --max-ctx 2048 --int8 --verify`
(oracle + chunked-vs-q=1 parity gates). Fused kernels are the generic gemma4 set
(`knowledge/custom-metal-kernels.md`) wrapped over qwen's Linears (MLP, attn q/o, SSM
in/z/out, tied head).

Reference CoreML (NOT Core AI) device throughput for scale: 0.8B ~48 tok/s, 2B ~27 tok/s on iPhone
17 Pro.
