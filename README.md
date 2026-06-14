# CoreAI-Model-Zoo

LLMs converted to Apple **Core AI** (`.aimodel`, iOS 27 / macOS 27) — downloadable, verified
on-device, with the conversion code and a knowledge base. Successor to
[`CoreML-Models`](https://github.com/john-rocky/CoreML-Models).

## Models

| Model | Download (`.aimodel`) | License |
|---|---|---|
| **Qwen3.5-0.8B** | [🤗 qwen3.5-0.8B-CoreAI](https://huggingface.co/mlboydaisuke/qwen3.5-0.8B-CoreAI) | Apache-2.0 |
| **Qwen3.5-2B** | [🤗 qwen3.5-2B-CoreAI](https://huggingface.co/mlboydaisuke/qwen3.5-2B-CoreAI) | Apache-2.0 |
| **Qwen3.6-35B-A3B** (MoE, Mac-only) | [🤗 Qwen3.6-35B-A3B-CoreAI](https://huggingface.co/mlboydaisuke/Qwen3.6-35B-A3B-CoreAI) | Apache-2.0 |
| **Qwen3.6-27B** (dense, Mac-only) | [🤗 Qwen3.6-27B-CoreAI](https://huggingface.co/mlboydaisuke/Qwen3.6-27B-CoreAI) | Apache-2.0 |
| **GLM-4.7-Flash** (MoE + MLA, Mac-only) | [🤗 GLM-4.7-Flash-CoreAI](https://huggingface.co/mlboydaisuke/GLM-4.7-Flash-CoreAI) | MIT |
| **Gemma 4 E2B** (text, incl. official-QAT int4) | [🤗 gemma-4-E2B-CoreAI](https://huggingface.co/mlboydaisuke/gemma-4-E2B-CoreAI) | Gemma |
| **Gemma 4 E4B** (text, official-QAT int4) | [🤗 gemma-4-E4B-CoreAI](https://huggingface.co/mlboydaisuke/gemma-4-E4B-CoreAI) | Gemma |
| **Gemma 4 12B** (dense, Mac-only — custom flash-decode kernel ‡) | [🤗 Gemma-4-12B-CoreAI](https://huggingface.co/mlboydaisuke/Gemma-4-12B-CoreAI) | Gemma |
| **Gemma 4 31B** (dense, Mac-only — custom flash-decode kernel ‡) | [🤗 Gemma-4-31B-CoreAI](https://huggingface.co/mlboydaisuke/Gemma-4-31B-CoreAI) | Gemma |
| **LFM2.5-1.2B-Instruct** | [🤗 LFM2.5-1.2B-CoreAI](https://huggingface.co/mlboydaisuke/LFM2.5-1.2B-CoreAI) | LFM Open License v1.0 |
| **LFM2.5-8B-A1B** (MoE, custom `gather_qmm` kernel — first iPhone MoE) | [🤗 LFM2.5-8B-A1B-CoreAI](https://huggingface.co/mlboydaisuke/LFM2.5-8B-A1B-CoreAI) | LFM Open License v1.0 |
| **Granite 4.0-H 1B / 350M** | [🤗 granite-4.0-h-CoreAI](https://huggingface.co/mlboydaisuke/granite-4.0-h-CoreAI) | Apache-2.0 |
| **Qwen3-VL** (vision-language) | [🤗 2B](https://huggingface.co/mlboydaisuke/Qwen3-VL-2B-CoreAI) · [4B](https://huggingface.co/mlboydaisuke/Qwen3-VL-4B-CoreAI) · [8B](https://huggingface.co/mlboydaisuke/Qwen3-VL-8B-CoreAI) | Apache-2.0 |
| **MiniCPM-V 4.6** (vision-language, sub-2B — strongest tiny VLM) | [🤗 MiniCPM-V-4.6-CoreAI](https://huggingface.co/mlboydaisuke/MiniCPM-V-4.6-CoreAI) | Apache-2.0 |
| **Gemma 4 E2B vision (VL)** (image+text) | `vl/` in [🤗 gemma-4-E2B-CoreAI](https://huggingface.co/mlboydaisuke/gemma-4-E2B-CoreAI) | Gemma |
| **EmbeddingGemma 300M** (text embeddings — on-device RAG / semantic search) | [🤗 embeddinggemma-300m-CoreAI](https://huggingface.co/mlboydaisuke/embeddinggemma-300m-CoreAI) | Gemma |
| **RF-DETR nano/small/medium/large** (object detection, no NMS) | [🤗 RF-DETR-CoreAI](https://huggingface.co/mlboydaisuke/RF-DETR-CoreAI) | Apache-2.0 |
| **RF-DETR-Seg nano→2xlarge** (instance segmentation, 6 sizes) | [🤗 RF-DETR-CoreAI](https://huggingface.co/mlboydaisuke/RF-DETR-CoreAI) | Apache-2.0 |

### Decode throughput (tok/s, greedy; output top-1 exact vs the Hugging Face reference)

| | iPhone 17 Pro · GPU | iPhone 17 Pro · ANE | M4 Max · GPU |
|---|---|---|---|
| **Qwen3.5-0.8B** | **71.9** | 14.7 | **210** |
| **Qwen3.5-2B** | **29** | — | **161** |
| **LFM2.5-1.2B** | **45.4** | — | **276.5** |
| **Granite 4.0-H 1B** | **36.3** | — | **136.5** |
| **Gemma 4 E2B** | **30.3** (QAT 30.7) | 6 | **77.0** (QAT 78.9) |
| **Gemma 4 E4B** (official QAT) | **15.1** | — | **55.8** |
| **Gemma 4 E2B VL** (image+text, official QAT) | **25.5** | — | **82.4** |
| **MiniCPM-V 4.6** (vision-language, sub-2B) | **53.4** | — | **224.3** |
| **Qwen3.6-35B-A3B** (MoE, 35B/~3B active, Mac-only) | — | — | **64.9** † |
| **Qwen3.6-27B** (dense, Mac-only) | — | — | **15.9** |
| **GLM-4.7-Flash** (MoE + MLA, 30B/~3B active, Mac-only) | — | — | **52.4** † |
| **Gemma 4 12B** (dense, Mac-only) | — | — | **23** int8 / **33** int4 ‡ |
| **Gemma 4 31B** (dense, Mac-only) | — | — | **17.2** int4 ‡ |

Measured on the iOS 27 / macOS 27 beta, Apple's `coreai-pipelined` GPU engine, zero custom
kernels (ANE column + **†**/**‡** excepted). **†** = MoE bundle using the custom
[`gather_qmm`](knowledge/compute-units-and-authoring.md) Metal kernel (reads only the routed
experts). **‡** = dense bundle whose full/global-attention SDPA is a custom flash-decode Metal
kernel — the stock MPSGraph SDPA crashes on the ≥16-head × 512 Q (a GPU scratch-heap overflow,
[apple/coreai-models#27](https://github.com/apple/coreai-models/issues/27)), so these models are
**unrunnable without it**. Prefill, sizes, per-model caveats: [`zoo/`](zoo/).

- **LFM2.5-8B-A1B** (MoE, 8.3B/~1.5B active) — a 32-expert MoE made practical by a custom
  [`gather_qmm`](knowledge/compute-units-and-authoring.md) Metal kernel that reads only the 4/32
  routed experts (fixes the GatherMM dense over-read), **39 → 141 tok/s (3.6×)**. Kept OUT of the
  table above (custom kernel). **Shipped Mac-only:** the `sym8` (linear int8) bundle is **clean**
  (fp32-oracle margin gate: +1 flip/41, at the fp16 ceiling) AND 3.6× faster. The int4 bundle that
  fits the iPhone was *validated to run on device* (first MoE on the phone) but **non-QAT int4 is a
  quality wall** (~12 flips/41, two schemes) so it is **not shipped**. Full numbers:
  [`zoo/lfm2.5-8b-a1b-moe.md`](zoo/lfm2.5-8b-a1b-moe.md)
- **Qwen3.6-35B-A3B** (MoE, 35B/~3B active) — the `gather_qmm` kernel takes decode **30.9 →
  64.9 tok/s (2.1×) at the SAME clean int8 quality** (0 introduced flips/18 vs fp16), closing the
  expert-gather half of the old ~4× MLX gap (the rest is int8-vs-int4 bytes, and int4 fails this
  model's numerics); [`zoo/qwen3.6.md`](zoo/qwen3.6.md)
- **Qwen3.6-27B** (dense) — the quality pick: int8 output == fp16; dense reads the whole
  model per token, hence slower than the ~3B-active MoE; [`zoo/qwen3.6-27b.md`](zoo/qwen3.6-27b.md)
- **Gemma 4 12B / 31B** (dense) — the first Core AI runtime for a ≥16-head × 512 full-attention
  model: the stock SDPA crashes on the full layers' Q (scratch-heap overflow, #27), so the full
  layers' SDPA is a **custom flash-decode Metal kernel** (block-GQA, higher-occupancy sequence-split
  for long context). 12B int8 == fp32 oracle; 31B is a frontier dense at int4 (4 global KV heads);
  [`zoo/gemma4-12b.md`](zoo/gemma4-12b.md) · [`zoo/gemma4-31b.md`](zoo/gemma4-31b.md)
- **GLM-4.7-Flash** (MoE + MLA, 30B/~3B active) — the zoo's first Multi-head Latent Attention
  model; full-MLA attention on all 47 layers (absorbed-MLA is the speed follow-up);
  [`zoo/glm-4.7-flash.md`](zoo/glm-4.7-flash.md)
- **RF-DETR / RF-DETR-Seg** — detection 33–39 FPS live on iPhone 17 Pro; instance
  segmentation in 6 sizes, masks gated IoU 1.000, 10.7–59.1 ms/frame on M4 Max;
  [`zoo/rf-detr.md`](zoo/rf-detr.md)
- **Gemma 4 E2B VL** — same text decoder + a 3-line image splice;
  [`zoo/gemma4-vl.md`](zoo/gemma4-vl.md)

<p align="center">
  <img width="380" alt="CoreAIChat screen recording" src="https://github.com/user-attachments/assets/999dbd95-45b5-468f-b1a8-34112ee3b74d" />
</p>
<p align="center"><i>CoreAIChat (<a href="apps/">apps/</a>) — the zoo's models running on-device on iPhone.</i></p>

## Repository layout

| Dir | What |
|---|---|
| [`zoo/`](zoo/) | Model cards — configurations, sizes, parity, measured throughput. |
| [`knowledge/`](knowledge/) | Verified notes on the framework: conversion, compression, stateful KV, custom Metal kernels, AOT, compute-unit rules, the Swift runtime. |
| [`conversion/`](conversion/) | Re-authored models + convert / verify / compress scripts (PyTorch → `.aimodel`). |
| [`swift/`](swift/) | `CoreAIRunner` — a Swift package that drives `.aimodel` LLM bundles, including architectures beyond the standard runtime. |
| [`apps/`](apps/) | SwiftUI on-device chat apps (iOS 27): CoreAIChat (Gemma 4 E2B GPU/ANE/⚡ + Qwen3.5 / Qwen3.5-2B / LFM2.5 / Granite ⚡pipelined, one picker) + QwenChatFast (Qwen3.5 static kernels) with in-app model download. |

## Start here

- **Run a model on device** → [`knowledge/swift-runtime.md`](knowledge/swift-runtime.md) + the model card
- **Convert a model** → [`knowledge/conversion-guide.md`](knowledge/conversion-guide.md)
- **Compress** → [`knowledge/compression.md`](knowledge/compression.md)
- **Make it fast** → [`knowledge/custom-metal-kernels.md`](knowledge/custom-metal-kernels.md) · [`knowledge/performance-ceiling.md`](knowledge/performance-ceiling.md)
- **Known beta issue** (in-graph KV-write crash; workarounds + the input-mask escape) → [`knowledge/coreai-beta-mpsgraph-kvwrite-bug.md`](knowledge/coreai-beta-mpsgraph-kvwrite-bug.md) — FB23024751 / [apple/coreai-models#5](https://github.com/apple/coreai-models/issues/5)

## License

BSD-3-Clause ([`LICENSE`](LICENSE)). Re-authored model code derives from Apple's BSD-3-Clause
`coreai_models` and retains its notices. Model weights follow their own licenses (see each
Hugging Face repo).
