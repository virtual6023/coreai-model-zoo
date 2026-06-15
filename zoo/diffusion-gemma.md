---
license: apache-2.0
base_model: google/diffusiongemma-26B-A4B-it
language:
- en
tags:
- core-ai
- apple-silicon
- diffusion-llm
- block-diffusion
- mixture-of-experts
- text-generation
pipeline_tag: text-generation
library_name: core-ai
---

# DiffusionGemma 26B-A4B → Core AI (Apple silicon, macOS)

A port of [`google/diffusiongemma-26B-A4B-it`](https://huggingface.co/google/diffusiongemma-26B-A4B-it)
— an experimental **block-diffusion text LLM** (25.2B total / ~3.8B active, 128-expert top-8 MoE on a
Gemma-4 backbone) — to **Apple's Core AI runtime** (the stock MPSGraph GPU engine, no framework fork).
The zoo's first diffusion-LM (dLLM) on Core AI: it denoises a 64-token **canvas** in parallel over a
short schedule (≤48 steps, early-stop) rather than emitting one token at a time.

## Honest summary (read this first)

- **macOS only.** ~51 GB (int8) / smaller (int4); needs a ≥64 GB Apple-silicon Mac. Not iOS (too large).
- **Speed: ~1.5–2× slower than MLX — NOT faster.** On an M4 Max, warm (after a one-time ~60 s graph
  compile that is then cached):
  - short prompts ≈ **1.0–2.7× MLX** (avg ~1.5×; "capital of Japan" ≈ parity, "capital of France" ≈ 2.7×),
  - longer prompts (≤512-token bucket) ≈ **2–4× MLX** (heavier prefill).
  - **If you want maximum speed or unlimited input length, use MLX**
    ([`mlx-community/diffusiongemma-26B-A4B-it-4bit`](https://huggingface.co/mlx-community/diffusiongemma-26B-A4B-it-4bit)):
    it is int4 with a **dynamic encoder** (encodes exactly the prompt, no padding) — two edges this port
    cannot fully match on the stock engine (int4 is only a marginal gain on the q=64 grouped-MoE kernel
    here; a dynamic encoder deadlocks MPSGraph's dynamic-shape path, so a fixed-length encoder is forced).
  - **What this port is for:** running the model on Apple's **stock Core AI stack** (drop-in for the
    zoo's Swift apps; no MLX dependency, no framework fork). Its output **matches MLX** on the prompts
    tested (France→Paris, Japan→Tokyo, 2+2=4, primary colors, a one-line photosynthesis definition).
- **Input:** free text, served from **bucketed static encoders {SP=128, SP=512}** (pick the smallest
  bucket ≥ prompt length) → up to **512 input tokens**. **Output:** a 64-token canvas — a **short-answer** model.

## Important: this model is QAT (quantization-aware-trained)

The released `google/diffusiongemma-26B-A4B-it` **bf16 weights are QAT master weights for the MoE
experts** — they *degenerate* in full precision (incoherent output, reproducing identically in HF
Transformers and unquantized MLX, so it is a property of the weights, not any port). Coherent generation
needs the **QAT int4 expert grid**, so this bundle is built from the published int4 expert values. Because
the experts are natively 4-bit, **both the int8 and int4 bundles are clean** (verified token-for-token vs
the MLX 4-bit release).

## How it was made fast (on a static-shape engine)

The naive Core AI diffusion path was ~74× MLX. The gap closed via, in order of impact:

1. **Grouped/sorted MoE Metal kernel** (the big one): a q=N `gather_qmm` custom kernel that sorts the
   canvas tokens by expert and reads each routed expert's weights once (the standard sort→grouped-GEMM
   MoE technique; the engine's default `GatherMM` did a per-token gather). Shipped through the stock
   custom-kernel externalize path — **no framework fork**. ~12× on the decode forward.
2. **Fixed-shape specialization** (`expectFrequentReshapes=false`): ~12× on prefill.
3. **GPU-fused sampler**: the last decoder chunk computes argmax + entropy + the self-conditioning
   soft-embeds on the GPU, so the 67 MB per-step logits never leave the device. Greedy denoiser
   (== MLX `temperature=0`). Per-step then ≈ pure forward (~0.22 s).
4. **Static-SP + right-pad + additive cross-attention pad-mask** for free variable-length input
   (a dynamic encoder deadlocks MPSGraph → a fixed SP with the pad masked out is the workaround → the
   bucketing above).

## Bundles

| variant | input buckets | note |
|---|---|---|
| **int8** | SP=128, SP=512 | validated default |
| **int4** | SP=128, SP=512 | QAT-native, same outputs, ~same speed — a smaller-size option |

Each bundle: `encoder.aimodel` (prompt → per-layer KV), 5× `decoder_chunk*.aimodel` (the 30-layer denoiser
with the fused GPU sampler in the last chunk), `decoder_chunks.json`. Conversion script + reference Swift
driver (`diffusion-lm-gate`) are in the [coreai-model-zoo](https://github.com/john-rocky/coreai-model-zoo).

## License

Apache-2.0, inherited from `google/diffusiongemma-26B-A4B-it`. A community port — not an Apple model.
