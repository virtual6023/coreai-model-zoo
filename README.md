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
| **Gemma 4 E2B** (text, incl. official-QAT int4) | [🤗 gemma-4-E2B-CoreAI](https://huggingface.co/mlboydaisuke/gemma-4-E2B-CoreAI) | Gemma |
| **Gemma 4 E4B** (text, official-QAT int4) | [🤗 gemma-4-E4B-CoreAI](https://huggingface.co/mlboydaisuke/gemma-4-E4B-CoreAI) | Gemma |
| **LFM2.5-1.2B-Instruct** | [🤗 LFM2.5-1.2B-CoreAI](https://huggingface.co/mlboydaisuke/LFM2.5-1.2B-CoreAI) | LFM Open License v1.0 |
| **Granite 4.0-H 1B / 350M** | [🤗 granite-4.0-h-CoreAI](https://huggingface.co/mlboydaisuke/granite-4.0-h-CoreAI) | Apache-2.0 |
| **Qwen3-VL** (vision-language) | [🤗 2B](https://huggingface.co/mlboydaisuke/Qwen3-VL-2B-CoreAI) · [4B](https://huggingface.co/mlboydaisuke/Qwen3-VL-4B-CoreAI) · [8B](https://huggingface.co/mlboydaisuke/Qwen3-VL-8B-CoreAI) | Apache-2.0 |
| **Gemma 4 E2B vision (VL)** (image+text) | `vl/` in [🤗 gemma-4-E2B-CoreAI](https://huggingface.co/mlboydaisuke/gemma-4-E2B-CoreAI) | Gemma |
| **RF-DETR nano/small/medium/large** (object detection, no NMS) | [🤗 RF-DETR-CoreAI](https://huggingface.co/mlboydaisuke/RF-DETR-CoreAI) | Apache-2.0 |

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
| **Qwen3.6-35B-A3B** (MoE, 35B/~3B active, Mac-only) | — | — | **30.9** |

Measured on the iOS 27 / macOS 27 beta, Apple's `coreai-pipelined` GPU engine, zero custom
kernels (ANE column excepted). Prefill, sizes, per-model caveats: [`zoo/`](zoo/).

- **Qwen3.6-35B-A3B** (MoE, 35B/~3B active) — 30.9 tok/s is expert-gather-bound in the
  current beta; [`zoo/qwen3.6.md`](zoo/qwen3.6.md)
- **RF-DETR** — 33–39 FPS live on iPhone 17 Pro, 8.6–19.1 ms/frame on M4 Max;
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
