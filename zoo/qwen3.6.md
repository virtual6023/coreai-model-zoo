# Qwen3.6-35B-A3B (text decoder) — Core AI

The **first MoE model on the zoo**, and the answer to Apple's most-requested model
(`coreai-models` issue #1). Source: `Qwen/Qwen3.6-35B-A3B` (an image-text-to-text
checkpoint; this card is the **text decoder**).

Architecturally this is **Qwen3.5's hybrid decoder + a sparse Mixture-of-Experts FFN**:
the token mixers are exactly [Qwen3.5](qwen3.5.md) — 40 layers on a 3:1 interleave of
**GatedDeltaNet** linear-attention mixers and **gated full attention** (head_dim 256, GQA
16/2, partial mRoPE θ=1e7) — but here the GatedDeltaNet runs **32 value heads over 16 key
heads** (GVA: each k/q head is shared across two value heads) and every FFN is a
**256-expert top-8 MoE** with a shared expert:

- router `Linear(2048, 256)` → fp32 softmax over all 256 → top-8 → renormalize by the top-8 sum,
- experts via Apple's `SwitchGLU` / `GatherMM` composite (the data-dependent expert gather),
- an always-on shared `MLP(512)` scaled by a per-token `sigmoid(shared_expert_gate(x))`,
- output = sparse-sum + gated-shared.

**35B parameters, ~3B active per token** → a frontier-quality model that decodes at
**Mac-class speed** because only 8 of 256 experts fire per token.

**⬇️ Converted `.aimodel` bundle:** `qwen3_6_35b_a3b_decode_int8hu_block32_sym/` (35 GB,
full LanguageBundle incl. tokenizer; decode-only loop-free for the
[pipelined engine](../knowledge/pipelined-engine.md)). *HF upload user-gated.*

## Measured (macOS 27 beta, M4 Max 128 GB, release `llm-benchmark`, `COREAI_CHUNK_THRESHOLD=1`)

| config | bundle | prefill tok/s | decode tok/s | numerics (teacher-forced vs bf16 HF oracle) |
|---|---:|---:|---:|---|
| **int8 linear per-block-32 + untied absmax int8 head (`int8hu --head-sym`) = SHIP** | **35 GB** | **30.8** | **30.9** | 14/16 — int8 reproduces full-precision argmax at every confident position but one (pos 6, margin 0.31, cos 0.991) |
| fp16 control (4 layers) | 8 GB | 167 | 175 | — (truncated; engine-path smoke) |

**Numerics in full** (the 35B is too large for an fp32 oracle on 128 GB, so the oracle is
the checkpoint's native **bf16**; gate = teacher-forced single-step argmax under the
oracle-margin≥0.1 rule):

- **fp16 full-scale eager tracks the bf16 oracle** at every confident position; its only
  argmax disagreements are sub-0.2-margin positions where bf16↔fp16 rounding itself flips
  the top-2 (pos 12, margin 0.188 — a three-way bf16/fp16/int8 disagreement, i.e. an
  oracle-resolution artifact, not a port defect). So the loader (packed-expert split,
  untied head, GVA) and the MoE math are correct.
- **int8hu adds exactly ONE flip** vs full precision (pos 6, margin 0.31, cos 0.991) —
  body-attributable: `int8lin` (fp16 head) gives the **byte-identical** pos-6 result, so
  the absmax int8 head is clean and `int8hu` ships with no fidelity cost over `int8lin`.
  This is the practical int8 ceiling for a 35B MoE; free-running engine greedy is NOT a
  valid gate here (the probe prompt has many sub-0.1-margin ties that drift under any
  rounding).

- **31 tok/s for a 35B-class model on a Mac** — this is the "frontier quality on your
  desk" slot. Comfortably above reading speed; the active-parameter count (≈3B), not the
  35B total, sets the decode rate.
- The MoE rides Apple's `SwitchGLU`/`GatherMM` exactly like Apple's own `gpt_oss`/`qwen3_moe`,
  so the **expert int8 quantization uses the documented 4-D SwitchLinear override**
  (`block_size [1,1,1,32]`, `axis: None` — Apple's `export/presets.py` MoE recipe, int4→int8);
  the **router and shared-expert scalar gate stay fp16** (quantizing the router can flip
  discrete expert selection for ~0.1 % of the bytes). The untied 248K-vocab head uses the
  **absmax `symmetric` per-block-32** rule (clipping corrupts big-vocab heads — see the
  qwen3.5 card).
- Prefill ≈ decode because prefill runs as pipelined S=1 steps (`COREAI_CHUNK_THRESHOLD=1`).
- Mac-only: at 35 GB int8 the bundle is far past the iPhone jetsam limit. This is the
  64/128 GB-Mac flagship; a smaller MoE (LFM2-8B-A1B class) is the iPhone-MoE story.

## Speed: an honest accounting (and why it will improve)

30.9 tok/s is real and usable, but it is **~4× slower than MLX 4-bit on the same M4 Max**
(`mlx-community/Qwen3.6-35B-A3B-4bit`: 125 tok/s, measured). We ship this honestly with
the gap fully characterized:

- The 4× decomposes into **~2× (int8 vs 4-bit bytes)** and **~2× (the MoE expert-gather)**.
- **int4 does not survive this model's numerics.** The attention/GatedDeltaNet body can't
  take 4-bit: full int4 teacher-forced gates **9/16 (symmetric) / 10/16 (asymmetric)** vs
  int8's 14/16 — asymmetric's usual +3-5 dB doesn't transfer to this LLM's mixer. Only a
  mixed recipe (experts int4, body int8) reaches a borderline 12/16. So the easy "4-bit 2×"
  is not free here, unlike Apple's QAT-int4 Gemma models.
- The other ~2× is structural: Apple's `GatherMM` composite **gathers then runs a DENSE
  matmul** with no active-experts-only read — it over-reads the 256-expert tensor, sitting
  at ~25 % of the memory-bandwidth ceiling, whereas MLX's custom-Metal `gather_qmm` reads
  only the 8 routed experts (~50 %). Apple's own MoE models use the same `GatherMM`, so
  this is not a porting mistake — it's the maturity of the Core AI GPU MoE path.
- **Core AI is not generally slow:** on a *dense* model the same pipelined engine is ~2×
  MLX (qwen3-0.6b-4bit: 1,150 tok/s). The gap is specific to the MoE expert-gather.

The real fix is a custom Metal gather-matmul kernel (Core AI exposes `TorchMetalKernel` →
`coreai.metal4_kernel`), but that API is beta-experimental and the integration with the
pipelined decode path is unverified — a high-variance multi-day spike we are **deliberately
deferring until the Core AI MoE path / kernel API matures**. So this card ships at frontier
*quality* with honest decode *speed*, and we expect the number to rise on OS/runtime
updates without any change to the bundle. For a "fast on Mac today" model, the dense Gemma /
Qwen ports are the better fit; this is the frontier-capability slot.

## A Core AI GPU bug this port surfaced (worked around at the engine surface)

A MoE decode graph raw-loaded via `AIModel.load(from_preferred_compute_unit_kind(.gpu))`
makes MPSGraph try to lower the 256-expert `GatherMM` to the **Neural Engine**, which
fails pathologically: a 2-layer slice aborts with
`MPSGraphANEUtils.mm:939: failed assertion 'ANE compilation writeToFile failed!'`, and a
4-layer slice **explodes the MPSGraph temp dir past 100 GB** (it crashed the host once).
Bisection (`_smoke/isolate_moe_ane_blowup.py`, single-layer random-weight probes under a
hard temp guard): a **dense** MLP layer loads in 0.2 s / 1 GB temp; swapping in the
**256-expert MoE** triggers the ANE assertion. So it's the `GatherMM`→ANE path, not the
GatedDeltaNet or the states.

**The real pipelined engine never hits it.** It loads `dynamic` models with
`SpecializationOptions(preferredComputeUnitKind: .gpu)` **plus `expectFrequentReshapes =
true`** (`ModelStructure.swift`), and that flag steers the graph off the ANE path. The
Python runtime binding doesn't expose `expectFrequentReshapes`, so raw Python load can't
replicate it — but the engine is the deployment path anyway, and Apple's own gpt_oss-20B
MoE runs the same way (78 tok/s). Proof: `llm-benchmark` on the real bundle is rc=0, peak
temp **1 GB**, 30.9 tok/s. Filed as Core AI feedback (ANE assertion / temp blowup on a
raw-loaded GPU-preferred GatherMM graph).

## How to reproduce

```bash
cd coreai-models   # with the qwen3_5 + qwen3_5_moe model overlay (see ../conversion)
# convert (CPU-side; ~70 GB fp16 load, mmap quantize keeps RAM in budget; ~35 GB bundle)
.venv/bin/python ../coreai-models-community/conversion/export_qwen3_6_decode_pipelined.py \
    int8hu --head-sym
# bench (needs the coreai-pipelined-extra-states engine patch + COREAI_CHUNK_THRESHOLD=1)
COREAI_CHUNK_THRESHOLD=1 .build/release/llm-benchmark \
    --model exports/qwen3_6_35b_a3b_decode_int8hu_block32_sym -p 64 -g 128 -n 3
```

Model overlay: `models/macos/qwen3_5_moe.py` (MoE FFN on `SwitchGLU` + the packed-expert
checkpoint loader) on top of the `qwen3_5.py` hybrid decoder (which gained the GVA
query/key head-repeat for this model). Same engine patch stack as the other pipelined
riders; **decode-only loop-free** because the GatedDeltaNet `while_loop` doesn't lower on
the GPU delegate.
