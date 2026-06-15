# DiffusionGemma 26B-A4B (block-diffusion dLLM) — Core AI

The **first diffusion language model (dLLM) on the zoo.** Source:
`google/diffusiongemma-26B-A4B-it` (Apache-2.0, `model_type` = a Gemma-4 backbone with a
block-diffusion head).

Unlike an autoregressive LLM that emits one token at a time, DiffusionGemma denoises a whole
**canvas** of tokens in parallel over a short schedule, conditioned on the prompt. Each step:
the decoder scores every canvas position, the host accepts the lowest-entropy tokens and
re-noises the rest, builds a self-conditioning soft-embedding from the scores, and feeds it back
— stopping early once the canvas is stable and confident.

Architecturally a **25.2B / ~3.8B-active Mixture-of-Experts** on a Gemma-4 backbone:

- **MoE FFN** — 128 routed experts, **top-8**, plus one shared dense expert; **softmax**
  router (not sigmoid). Experts ride Apple's `SwitchGLU` / `GatherMM` composite.
- **Gemma-4 backbone** — 30 layers, hidden 2816, vocab 262144, dual attention (sliding
  `head_dim` 256 / full 512 interleaved), **dual parallel FFN** (dense MLP ‖ MoE, summed ×
  per-layer scalar), 7 RMSNorms/layer, final logit softcap 30.
- **Self-conditioning** — a gated-MLP fuses the previous step's soft token-embeddings into the
  canvas embeddings (`soft_embeds = 0` on step 0 == the no-self-cond branch).

## Important: this model is QAT (the bf16 masters degenerate)

`google/diffusiongemma-26B-A4B-it` ships **bf16 master weights that are QAT
(quantization-aware-trained) for the MoE experts** — they *degenerate in full precision* (the
full-precision forward is incoherent). This is **not a port or backend bug**: the PyTorch port,
HF Transformers, and unquantized MLX all reproduce the same degenerate output bit-for-bit, while
the **published MLX 4-bit release generates coherently**. The published-int4 experts deviate from
the bf16 masters by cos 0.964 / maxdiff 0.318 (a naive 4-bit quant deviates only 0.026) — the
fingerprint of a QAT grid, not a quantization of the masters.

⇒ Convert from the **dequantized published weights** (`_diffgemma_pub_bf16`, the working QAT
grid), *not* google's masters. The int8 quantization here preserves that grid finely.

## ⬇️ Converted `.aimodel` bundle

🤗 **[DiffusionGemma-26B-A4B-CoreAI](https://huggingface.co/mlboydaisuke/DiffusionGemma-26B-A4B-CoreAI)**
(~51 GB int8, **macOS / Apple-silicon GPU only**). A host-driven 3-graph block-diffusion pipeline:

| graph | role |
|---|---|
| `encoder.aimodel` | prompt → 60 per-layer KV tensors (the conditioning; run once) |
| `decoder_chunk0..4.aimodel` | the 30-layer denoiser, split into 5×6-layer chunks chained host-side |
| `soft_proj.aimodel` | self-conditioning soft-embedding builder (`softmax(logits) @ embed`) |

- **Canvas 64**, static prompt **SP=26** (the graphs are traced static to dodge an MPSGraph
  dynamic-shape crash). Caps output at ~64 tokens — ample for short-answer / chat replies.
- int8 per-block-32 linear; experts 4-D int8 (the QAT grid); router + embedding fp16; `lm_head`
  absmax int8 (big-vocab-head rule).

Convert with [`conversion/export_diffusion_gemma.py`](../conversion/export_diffusion_gemma.py).

## Measured (macOS 27 beta, M-series Mac, release `diffusion-lm-gate`)

Prompt *"What is the capital of France? Answer in one short sentence."* (26 tok, chat-templated),
real 48-step schedule with early-stop, fast graph reuse (no per-step reload):

| phase | time |
|---|---:|
| load 7 graphs | 11.3 s |
| prefill (encode prompt → KV) | 34.8 s |
| **denoise step (fast reuse)** | **~2.9–3.1 s/step** |
| early-stop | **step 3** (mean-entropy 0.0086 → 0.0023 → 0.0014) |
| generation total | 8.9 s |

→ `The capital of France is Paris.` (`[818, 5279, 529, 7001, 563, 9079, 236761]` then `<eos>`
padding).

**The engineering result = fast graph REUSE.** A 30-layer q=C decoder is too heavy to bring up
as one graph (it overflows the GPU command queue) so it is split into 5×6-layer chunks chained
host-side. At **canvas 256** the chunk graph **deadlocks the GPU command queue when its loaded
function handle is reused across denoise steps** (the reuse-without-reload path the shipped LLM
engines decode with). Shrinking the **canvas to 64** makes the chunk graph ~4× lighter — it
reuses the handle like the q=1 LLM-decode regime — so the fast no-reload loop runs the real
denoise schedule at ~3 s/step (a per-step graph *reload* dodges the deadlock too, but costs
~30–47 s/step and leaks GPU memory — not shippable).

## Numerics / validation

- **QAT root cause proven** (port == HF == unquantized MLX, all degenerate full-precision;
  published-int4 → coherent; dequantized-published → coherent in both MLX and the PyTorch port).
  So the port is faithful and the working weights are established.
- **engine ≡ port.** The engine (GPU int8, fast-reuse loop) commits
  `[818, 5279, 529, 7001, 563, 9079, 236761]`; the PyTorch port (CPU bf16, the same greedy
  denoise loop, `_diffgemma_greedy_loop.py`) commits the **identical tokens** (early-stop step 45,
  ~48 s/step on CPU vs the engine's ~3 s/step on GPU = ~16× faster).

## Notes

- **Mac-only**: ~51 GB int8 (encoder graph + decoder graphs each carry the 30-layer backbone —
  a ~2× weight duplication; an encoder/decoder weight de-dupe → ~26 GB is an optional follow-up).
  Far past the iPhone jetsam limit. This is the big-Mac research/demo slot, like
  [Qwen3.6-35B-A3B](qwen3.6.md) / [GLM-4.7-Flash](glm-4.7-flash.md).
- **Static prompt length** (SP=26 for this bundle): the encoder is traced static (dynamic
  graphs hit an MPSGraph shape-bytecode crash). Re-trace for a different prompt length.
- The MoE at q=C (canvas) reads ~all experts (C×top-8 routing), so the decode-only `gather_qmm`
  kernel that accelerates q=1 MoE decode does **not** apply to the bidirectional canvas.
- The host computes the 262144-wide `softmax @ embed` self-conditioning on the CPU (Accelerate);
  running it on the engine right after a decode corrupts the next decode (the same giant op).

## How to reproduce

```bash
# 1) convert (CPU-side; ~48 GB load, mmap quantize; canvas=64 reuses the canvas-independent encoder)
cd coreai-models   # with the diffusion_gemma model overlay
.venv/bin/python ../coreai-models-community/conversion/export_diffusion_gemma.py \
    --mode int8 --static --trace-seq 26 --decoder-chunk 6 --canvas 64 \
    --model-dir ../_diffgemma_pub_bf16 --only decoder,soft_proj \
    --out ../_diffgemma_coreai_s26_qat_cl64
# reuse the existing encoder.aimodel (canvas-independent):
ln -s ../_diffgemma_coreai_s26_qat/encoder.aimodel ../_diffgemma_coreai_s26_qat_cl64/encoder.aimodel

# 2) generate (real schedule, fast reuse, no reload)
DG_GEN=1 DG_GEN_STEPS=48 .build/release/diffusion-lm-gate \
    ../_diffgemma_coreai_s26_qat_cl64 ../_diffgemma_gate_io_qat_cl64
# -> writes gate_io/gen_ids.i32 ; decode with the google/diffusiongemma-26B-A4B-it tokenizer
```

Model overlay: `models/macos/diffusion_gemma.py` (the block-diffusion port: dual attention /
dual parallel FFN / softmax-router MoE / self-conditioning, with `forward_encoder` /
`decode_with_soft` for the KV-handoff split). Driver: the Swift `diffusion-lm-gate` tool (the
host denoise loop + entropy-bound sampler + `StableAndConfidentStopping`).
