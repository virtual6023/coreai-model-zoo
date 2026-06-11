# CoreAI-Model-Zoo

LLMs converted to Apple **Core AI** (`.aimodel`, iOS 27 / macOS 27) — downloadable, verified
on-device, with the conversion code and a knowledge base. Successor to
[`CoreML-Models`](https://github.com/john-rocky/CoreML-Models).

## Models

| Model | Download (`.aimodel`) | License |
|---|---|---|
| **Qwen3.5-0.8B** | [🤗 qwen3.5-0.8B-CoreAI](https://huggingface.co/mlboydaisuke/qwen3.5-0.8B-CoreAI) | Apache-2.0 |
| **Qwen3.5-2B** | [🤗 qwen3.5-2B-CoreAI](https://huggingface.co/mlboydaisuke/qwen3.5-2B-CoreAI) | Apache-2.0 |
| **Gemma 4 E2B** (text, incl. official-QAT int4) | [🤗 gemma-4-E2B-CoreAI](https://huggingface.co/mlboydaisuke/gemma-4-E2B-CoreAI) | Gemma |
| **Gemma 4 E4B** (text, official-QAT int4) | [🤗 gemma-4-E4B-CoreAI](https://huggingface.co/mlboydaisuke/gemma-4-E4B-CoreAI) | Gemma |
| **LFM2.5-1.2B-Instruct** | [🤗 LFM2.5-1.2B-CoreAI](https://huggingface.co/mlboydaisuke/LFM2.5-1.2B-CoreAI) | LFM Open License v1.0 |
| **Granite 4.0-H 1B / 350M** | [🤗 granite-4.0-h-CoreAI](https://huggingface.co/mlboydaisuke/granite-4.0-h-CoreAI) | Apache-2.0 |
| **Qwen3-VL 2B** (vision-language) | [🤗 Qwen3-VL-2B-CoreAI](https://huggingface.co/mlboydaisuke/Qwen3-VL-2B-CoreAI) | Apache-2.0 |

### Decode throughput (tok/s, greedy; output top-1 exact vs the Hugging Face reference)

| | iPhone 17 Pro · GPU | iPhone 17 Pro · ANE | M4 Max · GPU |
|---|---|---|---|
| **Qwen3.5-0.8B** | **71.9** | 14.7 | **210** |
| **Qwen3.5-2B** | **29** | — | **161** |
| **LFM2.5-1.2B** | **45.4** | — | **276.5** |
| **Granite 4.0-H 1B** | **36.3** | — | **136.5** |
| **Gemma 4 E2B** | **30.3** (QAT 30.7) | 6 | **77.0** (QAT 78.9) |
| **Gemma 4 E4B** (official QAT) | **15.1** | — | **55.8** |

Measured on the iOS 27 / macOS 27 beta, all on Apple's `coreai-pipelined` GPU engine (zero
custom kernels) except the ANE column. Per-model configurations, prefill numbers, sizes, and
caveats: [`zoo/`](zoo/). The Gemma 4 QAT rows are re-exports of Google's official
QAT-q4_0 checkpoints — same speed, **int4 ≈ bf16 quality by design**
([`zoo/gemma4-e2b.md`](zoo/gemma4-e2b.md)).

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
