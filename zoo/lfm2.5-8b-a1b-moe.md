# LFM2.5-8B-A1B (sparse-MoE text decoder) — Core AI

Conv + full-attention **MoE** hybrid decoder (LiquidAI): 24 layers = **18 short-conv mixers**
(depthwise causal conv, kernel 3, B/C/x gating) + **6 GQA attention layers** (32 q / 8 kv heads,
head_dim 64, per-head q/k RMSNorm, full-dim RoPE θ=5e6), hidden 2048, vocab 128 000, tied head.
The first two layers are dense (SwiGLU, intermediate 7168); **every layer after is a 32-expert
top-4 sparse MoE** (expert MLP intermediate 1792, sigmoid routing + selection-only expert bias,
no shared expert). 8.3B total / **~1.5B active per token**. Source: `LiquidAI/LFM2.5-8B-A1B`.

**This is the zoo's first MoE on the iPhone-class fast path** — and it only became practical
because of a custom Metal kernel that fixes how MoE decode reads memory.

## The `gather_qmm` kernel — the MoE-decode fix

MoE FFN decode normally lowers through the `SwitchGLU`/`GatherMM` composite, whose matmul — as
lowered — reads **all 32 experts' weights every token** even though only the top-4 are routed. So
MoE decode is over-read-bound, not active-param-bound: plain int8 here was bandwidth-saturated at
**39 tok/s** reading its full 8.8 GB/token.

The fix ([`models/macos/moe_metal.py`](../../coreai-models/python/src/coreai_models/models/macos/moe_metal.py),
a `coreai_torch.TorchMetalKernel`) takes the routed expert indices as a kernel **input** and reads
**only the 4 routed experts'** weight slabs (`QP[w, n, e]` with `e = IDX[slot]` — an indexed load,
so the other 28 experts are never fetched). `MetalSwitchGLU` is a drop-in for `SwitchGLU`;
`metalize_moe(model, nbits)` swaps every MoE layer. Weights are k-means palettized in-file
(int8km = 256-entry codebook; int4km = 16-entry, 8 nibbles/uint32). The kernel is GPU-only by
construction, so the old GatherMM→ANE raw-load abort can't occur, and it composes with the
stateful KV+conv decode graph.

## Measured (macOS + iOS 27 beta, release builds, p=128 g=256, `COREAI_CHUNK_THRESHOLD=1`)

| config | bundle | prefill tok/s | decode tok/s | vs baseline |
|---|---:|---:|---:|---|
| int8 linear (GatherMM dense over-read), M4 Max | 8.8 GB | 38.9 | 39.2 | baseline (BW-saturated) |
| **int8km-gather (Mac), M4 Max** | **8.4 GB** | **141.0** | **141.0** | **3.6× — reads only 4/32 experts** |
| int4 linear (GatherMM), M4 Max | 5.0 GB | 170.6 | 169.6 | quality-degraded (broken grammar) |
| **int4km-gather, M4 Max** | **4.7 GB** | **162.7** | **162.7** | k-means int4 (grammatical) |
| **int4km-gather, iPhone 17 Pro** (PipelinedBench) | **4.7 GB** | **32.0** | **31.3** | **the zoo's first iPhone MoE on hardware** (jetsam-safe; engine warm-load 7.6 s) |

iPhone reading: ~8B-quality MoE at ~32 tok/s decode — the same ballpark as the dense LFM2.5-1.2B
on the same phone (38–39), but at ~7× the model quality, in 4.7 GB (under the ~6.4 GB jetsam limit).

## Quality — fp32-oracle margin-rule gate (honest)

Teacher-forced over a 41-token factual paragraph (S=1, accumulating KV+conv = the decode path).
A top-1 disagreement vs the fp32 oracle counts as a **REAL flip** only if the oracle's logit gap
to the token the model picked is **≥ 0.1** (below = statistical tie). The kernel itself is
bit-exact vs the dequantized reference — the cost measured here is the **expert weight
quantization** (k-means palettization), and it is real:

| model | real flips / 41 | introduced beyond fp16 | worst fp32 margin |
|---|---:|---:|---:|
| fp16 (ceiling) | 1 | — | 0.11 |
| **int8km** | 6 | **5 (~12%)** | 1.18 |
| **int4km** | 13 | **12 (~29%)** | 7.81 |

Read honestly:
- The **fp16 base is essentially fp32-faithful** (1 borderline flip).
- **int8km introduces ~5 real flips / 41** with moderate margins (≤1.2) — a small but real step
  below fp16, *not* fp16-identical. Usable, noticeably above int4.
- **int4km introduces ~12 / 41** with several large margins (up to 7.8) — **clearly degraded vs
  fp32**; this is the size/speed/on-device tradeoff, not a quality-equivalent bundle.
- Both stay grammatical and on-topic; neither shows the broken grammar of plain-symmetric int4
  (k-means is the better 4-bit). On a bare prompt the model greedy-degenerates into repetition —
  that is the *base model's* behavior (present in fp16 too), not a kernel/quant artifact; use the
  chat template + sampling for real generation.
- A cleaner **affine/linear-int8 expert kernel variant** (likely closer to the shipped int8-linear
  quality) is a follow-up; the current kernel reads a k-means codebook.

## Bundles

**⬇️ [mlboydaisuke/LFM2.5-8B-A1B-CoreAI](https://huggingface.co/mlboydaisuke/LFM2.5-8B-A1B-CoreAI)** —
ships the upstream LFM Open License v1.0 LICENSE file. Convert locally with
[`conversion/export_lfm2_moe_metal_decode_pipelined.py`](../conversion/export_lfm2_moe_metal_decode_pipelined.py)
(`int8km` = Mac fp16-faithful; `int4km` = iPhone-jetsam-safe). Run with
`COREAI_CHUNK_THRESHOLD=1` (the decode graph's `input_ids` is static `[1,1]`; prefill runs as S=1
pipelined steps via `llm-benchmark`).
