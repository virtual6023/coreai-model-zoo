# coreai-optimization (`coreai-opt`) reference — quantization & palettization

> Foundation note (API reference). Complements [`compression.md`](compression.md) (this project's LLM-specific
> empirical notes). Relevant to the **int4 / vocab-pruned head** lever in the ANE-later plan and to shrinking
> deployable assets. Sources: `coreai-optimization/README.md`, `docs/src/{introduction,quantization,palettization,utils}/*`,
> `skills/.../model-compression-exploration/references/{compression_patterns,size_estimation}.md`.

## Where it plugs in
```
PyTorch model → coreai-opt (compress) → finalize() → torch.export(run_decompositions(get_decomp_table()))
              → cast_to_16_bit_precision → coreai_torch.TorchConverter → .optimize() → save_asset() → .aimodel
```
Every compressor output is itself a PyTorch model (validate/finetune/export it). Lifecycle:
`Quantizer/KMeansPalettizer(model, config)` → `prepare(example_inputs)` → optional `calibration_mode()` /
`training_mode()` (QAT) → `finalize(backend=ExportBackend.CoreAI)`.

## Quantization (weights ±activations)
- **dtypes**: INT2/4/8 (signed+unsigned), FP4_E2M1, FP8_E4M3FN/E5M2 (limited Core AI support).
- **granularity**: per-tensor / per-channel (axis 0 for Linear/Conv/Embedding) / **per-block** (`block_size`,
  e.g. 32 along the in-features axis). Finer = better quality, more scale overhead.
  ⚠️ per-channel (axis-0) int8 Linear weights are **broken on the macOS-27-beta MPSGraph GPU
  delegate** — torch-level numerics are clean but the lowered matmul returns garbage (minimal
  head-only repro 2026-06-11, multiple shapes, sym and clipping alike); use per-block-32 there
  (see [pipelined-engine.md](pipelined-engine.md)).
- **scheme**: symmetric vs asymmetric. At int8 the gap is small (~1.5 dB); at int4 asymmetric gains +3–5 dB,
  and `symmetric_with_clipping` can add +7 dB.
- **workflows**: data-free weight-only PTQ (seconds; good ≥8-bit, sometimes 4–6) → calibration (≈128 samples,
  needed for activation ranges) → QAT (full training; the only way to recover ≤4-bit).
- **modes**: graph (torchao PT2E, default; needs `torch.export`-able model; best for weight+activation) vs
  eager (`__torch_function__`; weight-only or when graph fails; supports dynamic control flow).
- **config**: `QuantizerConfig` → `module_type_configs`/`module_name_configs` override `global_config`
  (name > type > global). No-arg default = **W_INT8_A_INT8**. Presets: `QuantizerConfig.presets.w8()`,
  `.w4()` (int4 per-block 32). `.without(nn.LayerNorm, "model.lm_head")` to skip layers.

## Palettization (k-means LUT, weights only)
- **`n_bits ∈ {1,2,3,4,6,8}`**, LUT = `2^n_bits` centroids; each weight → index into LUT.
- **scalar** (1-D k-means, default) vs **vector** (`cluster_dim>1`; effective bpw = n_bits/cluster_dim).
- **granularity**: per-tensor vs **per-grouped-channel** (`group_size`). **Per-channel (group_size=1) basically
  always wins**; at per-channel, k-means beats quantization by ~15–19 dB at both 8-bit and 4-bit. Per-tensor
  palettization can be *worse* than per-channel quantization.
- **`lut_qspec`**: quantize the LUT centroids to int8 → enables W_INT8-A_INT8 execution (a fp LUT forces fp ops).
- **sensitivity-based k-means** (SqueezeLLM): weight clustering by per-weight importance from calibration grads.
- vector k-means is **non-deterministic** — seed numpy+torch before each `prepare()` (and `num_workers=1`).

## Mixed precision & joint compression
- **Mixed precision** (`utils/mixed_precision.md`): per-layer bit-widths from a layer-sensitivity sweep
  (compress one layer at a time, score by PSNR), then walk least-loss-first until a target avg-bitwidth is met.
- **Joint** (`utils/joint_compression.md`): palettize weights **first** (with int8 `lut_qspec`), then quantize
  activations on the palettized model. **Finalizable to the Core AI backend only.**
- ⚠️ When compressing a **stateful** decode core, read the export spec (reference inputs, dynamic_shapes,
  state_names) from the ORIGINAL model — the finalized model loses those methods.

## The LM head + embeddings (biggest tensors; the ANE-later lever)
- **Head** = vocab × hidden (e.g. 262144 × 1536) — largest single tensor, high sensitivity, needs **per-row
  (per-output-channel)** scales for matmul efficiency.
- **Embeddings** (and gemma4-style per-layer tables) can be multiple GB → gathered on a **front-end**, kept OUT
  of the decode-core graph.
- This project's measured floor: **int8 k-means (group 32, all projections) stays argmax-exact; int4 flips the
  next token** (both linear-int4 and k-means-int4). Gate/up MLP must be int8. Keep tied lm_head + 1-D conv (SSM)
  full precision for exactness. Embedding gather = **plain int8 per-row dequant-gather** (`q[ids].fp16*scale[ids]`);
  k-means is `F.linear`-only so it can't palettize a gather; **int4 gather has no clean macOS path → int8 is the
  embedding floor**. (So an int4 head needs a *kernel* path, not coreai-opt's F.linear quantizer — ties back to
  [`custom-metal-kernels.md`](custom-metal-kernels.md): the fused-int8 head+argmax kernel.)

## Pitfalls
- **Silent skips**: per-block quant / per-grouped-channel palettization silently skip layers whose dim isn't
  divisible by the block/group → those layers stay uncompressed. Check divisibility before trusting a size.
- **Boundary layers** (first/last) are high-error — skipping them can add up to +9 dB; always ablate.
- graph-mode export fails on dynamic control flow → fall back to eager for weight-only.

## Theoretical size
```
weight/index bytes = numel * n_bits/8           # int4 = 0.5 B/elem, int8 = 1 B/elem
scale bytes        = n_groups * 2 (fp16)         # n_groups per granularity (per-tensor=1, per-channel=shape[axis], per-block=ceil(dim/B)*…)
zero_point bytes   = n_groups * n_bits/8         # asymmetric only
lut bytes          = 2^n_bits * n_luts * 2       # palettization
total ≈ Σ(above) + uncompressed (biases, fp embeds, skipped layers)
avg_bitwidth = Σ(numel_i * bits_i) / Σ numel_i
```
Sizes hit in this project: gemma4 E2B core 7.0 GB fp32 → 3.5 GB fp16 → **1.9 GB int8**; qwen3.5-0.8B **969 MB**.
