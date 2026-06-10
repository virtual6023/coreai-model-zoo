# CoreAI-Model-Zoo

> The community resource for **Apple Core AI** (the Core ML successor, iOS/macOS 27) — the
> models, conversions, knowledge, and apps Apple's own `coreai-models` doesn't ship. Successor
> in spirit to the [`CoreML-Models`](https://github.com/john-rocky/CoreML-Models) zoo.

Apple's first-party `coreai-models` zoo stops ~one generation back (Qwen3 / Gemma 3, no VLM) and
its Swift runtime only drives standard architectures. This repo covers the gap: **newer models
converted end-to-end, the conversion tooling, the hard-won knowledge, a reusable Swift runner,
and working on-device chat apps.**

> ⚠️ **Known beta issue:** on the WWDC26 betas, the official fixed-shape KV write crashes at
> execute — see [`knowledge/coreai-beta-mpsgraph-kvwrite-bug.md`](knowledge/coreai-beta-mpsgraph-kvwrite-bug.md)
> (filed: FB23024751 / [apple/coreai-models#5](https://github.com/apple/coreai-models/issues/5))
> for the isolation, the host-cache workaround used throughout this repo, and the input-mask
> escape that restores stateful KV.

## What's inside

| Dir | What |
|---|---|
| [`zoo/`](zoo/) | Converted models — model cards, parity numbers, sizes, how to get/run. (Weights are not committed; reproduce via `conversion/`.) |
| [`conversion/`](conversion/) | Re-authored models + convert/verify/compress scripts (PyTorch → Core AI `.aimodel`). |
| [`knowledge/`](knowledge/) | The knowledge base — Core AI API, the `.aimodel` pipeline, conversion gotchas, compression (int8 vs int4), stateful/KV-cache, custom Metal kernels, AOT, the Swift runtime. |
| [`swift/`](swift/) | `CoreAIRunner` — a reusable Swift package that drives `.aimodel` LLM bundles (incl. non-standard architectures Apple's runtime can't). |
| [`apps/CoreAIChat/`](apps/CoreAIChat/) | SwiftUI on-device LLM chat (iOS 27) built on `CoreAIRunner`. |

## Models (status)

| Model | Type | Parity vs HF | Bundle | Mac GPU | iPhone 17 Pro |
|---|---|---|---|---|---|
| **Qwen3.5-0.8B** | hybrid linear+full attn | top-1 exact | 969 MB int8 (dyn) / fp16 static ctx-2048 | **58.5 tok/s** | **GPU 27.7 (static, ctx 2048) · ANE 14.7 (dynamic)** |
| **Qwen3.5-2B** | hybrid linear+full attn | top-1 exact | 2.2 GB | converts (same path) | — |
| **Gemma 4 E2B** | multimodal (text decoder) | 8/8 exact | ~4.2 GB (3-stage, int4 core) | **56.6–59.0 tok/s** (custom Metal kernels) | **GPU 17.7 / ANE ~6 tok/s** |
| Qwen3-VL 2B | VLM | ⏳ later | — | — | — |
| Gemma 4 E4B | multimodal (+MoE) | 🔜 same authoring path | — | — | — |

All cells greedy, top-1 vs the HF eager reference; device cells measured on iOS 27 beta.
Full matrix + caveats: [`zoo/README.md`](zoo/README.md).

Verified end-to-end: conversion + numeric parity on macOS, then re-verified on-device
(iPhone 17 Pro, iOS 27 beta — GPU and Neural Engine). Both models run in working SwiftUI chat
apps (`apps/`).

## Quick links

- New to Core AI? → [`knowledge/coreai-overview.md`](knowledge/coreai-overview.md)
- Converting a model? → [`knowledge/conversion-guide.md`](knowledge/conversion-guide.md) (+ the burned-in gotchas)
- Compressing? → [`knowledge/compression.md`](knowledge/compression.md) (TL;DR: **int8 k-means is the exactness floor; 4-bit only via the k-means/grouped form + re-verification**)
- Need speed? → [`knowledge/custom-metal-kernels.md`](knowledge/custom-metal-kernels.md) + [`knowledge/performance-ceiling.md`](knowledge/performance-ceiling.md) (honest ceilings)
- Running on device? → [`knowledge/swift-runtime.md`](knowledge/swift-runtime.md) + [`apps/CoreAIChat/`](apps/CoreAIChat/)

## Status & license

Early / actively built (2026). Licensed **BSD-3-Clause** (see [`LICENSE`](LICENSE)); the
re-authored model code derives from Apple's BSD-3-clause `coreai_models` and retains its notices.
