# CoreAI-Model-Zoo

LLMs converted to Apple **Core AI** (`.aimodel`, iOS 27 / macOS 27) — downloadable, verified
on-device, with the conversion code and a knowledge base. Successor to
[`CoreML-Models`](https://github.com/john-rocky/CoreML-Models).

## Models

| Model | Download (`.aimodel`) | License |
|---|---|---|
| **Qwen3.5-0.8B** | [🤗 qwen3.5-0.8B-CoreAI](https://huggingface.co/mlboydaisuke/qwen3.5-0.8B-CoreAI) | Apache-2.0 |
| **Qwen3.5-2B** | [🤗 qwen3.5-2B-CoreAI](https://huggingface.co/mlboydaisuke/qwen3.5-2B-CoreAI) | Apache-2.0 |
| **Gemma 4 E2B** (text) | [🤗 gemma-4-E2B-CoreAI](https://huggingface.co/mlboydaisuke/gemma-4-E2B-CoreAI) | Gemma |
| **LFM2.5-1.2B-Instruct** | [🤗 LFM2.5-1.2B-CoreAI](https://huggingface.co/mlboydaisuke/LFM2.5-1.2B-CoreAI) | LFM Open License v1.0 |
| **Granite 4.0-H 1B / 350M** | HF upload pending — convert locally via [`conversion/`](conversion/) | Apache-2.0 |

### Decode throughput (greedy; output top-1 exact vs the Hugging Face reference)

| | iPhone 17 Pro · GPU | iPhone 17 Pro · ANE | M4 Max · GPU |
|---|---|---|---|
| **Qwen3.5-0.8B** | **50.3–51.5 tok/s**<br><sub>int8 linear · loop-free × pipelined engine (shipped app: int8 fused kernels 42.5–45.4 · prefill 147 tok/s q16 chunks)</sub> | **14.7 tok/s**<br><sub>int8 · dynamic</sub> | **204 tok/s**<br><sub>int8 linear · loop-free × pipelined engine (custom-kernel CLI: 58.5)</sub> |
| **Qwen3.5-2B** | 19–21 tok/s<br><sub>runs (24/24 ≡ Mac-GPU; needs the increased-memory entitlement) — the CoreML-2B port (~27) is still faster on phone → Mac-recommended</sub> | — | **127 tok/s**<br><sub>int8 linear · loop-free × pipelined engine</sub> |
| **LFM2.5-1.2B** | **38.0–39.6 tok/s**<br><sub>int8 linear · pipelined engine (~87% of naive BW ceiling; 24/24 ≡ Mac-GPU)</sub> | — | **253 tok/s**<br><sub>int8 linear · pipelined engine — first non-Qwen rider (fp16: 162)</sub> |
| **Granite 4.0-H 1B** | —<br><sub>not yet measured (naive ceiling ~37 tok/s)</sub> | — | **136.5 tok/s**<br><sub>int8 linear · pipelined engine — first Mamba2 SSM rider (fp16: 103.6; 350m fp16: 191)</sub> |
| **Gemma 4 E2B** | **22 tok/s**<br><sub>int4 k-means kernels</sub> | **6 tok/s**<br><sub>int8 · 6 chunks</sub> | **56.6–59.0 tok/s**<br><sub>int8 kernels</sub> |

Measured on the iOS 27 / macOS 27 beta. Sizes, configurations, and caveats: [`zoo/`](zoo/).

Next up: Gemma 4 E4B · Qwen3-VL.

## Repository layout

| Dir | What |
|---|---|
| [`zoo/`](zoo/) | Model cards — configurations, sizes, parity, measured throughput. |
| [`knowledge/`](knowledge/) | Verified notes on the framework: conversion, compression, stateful KV, custom Metal kernels, AOT, compute-unit rules, the Swift runtime. |
| [`conversion/`](conversion/) | Re-authored models + convert / verify / compress scripts (PyTorch → `.aimodel`). |
| [`swift/`](swift/) | `CoreAIRunner` — a Swift package that drives `.aimodel` LLM bundles, including architectures beyond the standard runtime. |
| [`apps/`](apps/) | SwiftUI on-device chat apps (iOS 27): CoreAIChat (Gemma 4 E2B GPU/ANE + Qwen3.5 ⚡pipelined, one picker) + QwenChatFast (Qwen3.5 static kernels) with in-app model download. |

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
