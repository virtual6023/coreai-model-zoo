# Compression (palettization & quantization)

**TL;DR for LLM decoders: int8 k-means palettization is the floor that stays exact when applied
across the whole transformer; whole-model int4 degrades. SELECTIVE 4-bit works: k-means int4 on
the FFN + lm_head only (attention/embeddings kept ≥int8/fp16) measured top-1 exact and is the
shipping iPhone-GPU config (via custom fused kernels — see
[`custom-metal-kernels.md`](custom-metal-kernels.md)).**

## int8 k-means > whole-model int4 (for these models)

Across Gemma 4 E2B and Qwen3.5, linear int4 and k-means int4 both flip next-token argmax vs the
HF reference; **int8 k-means palettization reproduces HF top-1 exactly** at ~half the fp16 size.

- k-means fits a per-group lookup table to the actual weight clusters → tracks non-uniform weight
  distributions far better than symmetric per-block int4 at the same bit width.
- **Finer groups are the main int4 lever** (group32 → group8 helps), but still don't reach exact.
  Per-channel scale is marginal or harmful.
- Sensitivity is broadly distributed; for Gemma 4 the **gate/up MLP projections must be int8** for
  exactness (keeping them at 4-bit caps accuracy regardless of other layers).
- k-means palettizes **`F.linear`/`F.conv` weights only**, so RMSNorm/RoPE params stay full
  precision automatically — exactly what you want given their wide range.

Recommended LLM recipe: **int8 k-means, group 32, all projections**; keep tied lm_head + 1-D conv
(SSM) full precision. Sizes seen: Gemma 4 E2B core 7.0 GB fp32 → 3.5 GB fp16 → **1.9 GB int8**;
Qwen3.5-0.8B **969 MB**, -2B **2.2 GB** (fp16 embed + int8 transformer, single bundle).

## Palettization × stateful export composes

Palettizing the **stateful** decode core (mutable KV/SSM state + dynamic prefill+decode graph)
works: read the export spec (reference inputs / dynamic shapes / state names) from the ORIGINAL
model first (the finalized palettized model loses that method), palettize, then drive
`export_to_coreai` with that spec. Verified top-1-exact for both Gemma 4 (dual-KV) and Qwen3.5
(hybrid 4-state).

## Embedding tables (the on-device memory problem)

Big-vocab models have huge embedding tables (Gemma 4's per-layer table is 9.4 GB fp32). For device:
- The decode **core** keeps these tables OUT of the graph (gathered on a front-end).
- The front-end gather table compresses with **plain int8 per-row dequant-gather**
  (`q_table[ids].to(fp16) * scale[ids]`) — k-means is `F.linear`-only so it doesn't apply to a
  gather, and the iOS palettized-embedding custom op doesn't lower on macOS. int4 gather has no
  clean path today; **int8 is the practical floor** for embedding gather too.

## Via the CLI

`coreai.llm.export <model> --compression int8` routes a new macOS int8 k-means preset through the
decode-core signature (palettizes the *extracted* core, not the `input_ids→logits` forward) for
models that expose `export_core()`. Other models keep the standard quantization path.
