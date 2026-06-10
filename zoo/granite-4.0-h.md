# Granite 4.0-H 1B / 350M (text decoder) — Core AI

Mamba2 + attention hybrid decoder (IBM, dense "H" variants): the 1b is 40 layers =
**36 Mamba2 mixers** (selective-scan SSM: 48 heads × d_head 64, d_state 128, kernel-4
depthwise conv, gated output RMSNorm) + **4 GQA attention layers** (12 q / 4 kv heads,
**NoPE** — no positional embedding, fixed config scale, no q/k norm), hidden 1536,
shared SwiGLU MLP 4096, vocab 100 352, tied head, mup-style scalar multipliers
(embedding ×12, residual ×0.22, logits ÷6). The 350m: 32 layers = 28 + 4, hidden 768.
Source: `ibm-granite/granite-4.0-h-1b`, `ibm-granite/granite-4.0-h-350m`.

**The first SSM-scan architecture on the [pipelined-engine fast path](../knowledge/pipelined-engine.md).**
The enabler is the same observation that unlocked qwen3.5's GDN: **at S=1 the Mamba2
selective scan is a single recurrence step** (`state = state*dA + dt*B*x;
y = (state·C) + D*x` — the HF `use_precomputed_states` branch), so the decode-only
graph is loop-free and lowers on the MPSGraph GPU delegate. State = growing KV for the
4 attention layers + two fixed-shape stacks (conv columns `[36,1,conv_dim,3]`, SSM
state `[36,1,48,64,128]`) — exactly the extra-states patch budget (≤2), the same
`(convState, recState)` shape-class as qwen3.5. No engine changes; neither LFM2.5
GPU-delegate workaround was needed here (per-layer `SSMState` writes compile fine —
the multi-write drop is pattern-dependent — and NoPE attention without q/k-norm gains
shows no fp16-matmul amplification; the SSM step itself computes in fp32 in-graph).

## Measured (macOS 27 beta, M4 Max, release builds, p=128 g=256, `COREAI_CHUNK_THRESHOLD=1`)

| config | bundle | prefill tok/s | decode tok/s | numerics |
|---|---:|---:|---:|---|
| **1b int8 linear per-block-32 (ship)** | **1.6 GB** | **136.7** | **136.5** | **16/16 oracle gate + HF-seeded decode step — on a margin-clean oracle (min top-2 margin 0.137, decode 2.53)** |
| 1b fp16 (control) | 2.8 GB | 103.7 | 103.6 | 16/16 oracle gate + decode step |
| 350m fp16 (ship at this size) | 0.66 GB | 193.2 | 191.1 | 16/16 oracle gate + decode step; engine path ≡ torch 24/24 greedy ×2 prompts; `llm-runner` 217.9 tok/s short-run |
| 350m int8 (lin/b8/sel/mix) | — | — | 185.8–186.6 | **not shipped**: gate FAILs *and* no speed win (see below) |

- Prefill ≈ decode because prefill runs as pipelined S=1 steps (`COREAI_CHUNK_THRESHOLD=1`).
- int8lin buys the 1b **+32%** (BW-bound, as expected). The 350m is *overhead-bound*:
  int8 made it *slower* (185.8 vs 191.1) — at ~0.7 GB/token the GPU isn't waiting on
  weights, so dequant work is pure cost. Ship rule of thumb: quantize for speed only
  when the model is big enough to be bandwidth-bound on your device.
- iPhone: not yet measured. Naive ceiling for the 1b int8lin ≈ 60 GB/s ÷ 1.6 GB ≈ 37 tok/s
  (LFM2.5 realized ~87% of its ceiling on an iPhone 17 Pro, qwen-0.8B ~60%).

## Why the 350m int8 gate fails (and why that's two separate findings)

1. **Real block-32 sensitivity in the shared MLP.** Torch fake-quant attribution
   (`_smoke/probe_granite_quant_attrib.py`) shows per-block-32 int8 on
   `shared_mlp.output_linear` alone flips sweep positions whose oracle margins are
   *solid* (0.57, 0.17) — genuine quantization damage, not ties. Per-block-8 on the
   MLP (rest at 32) repairs the sweep.
2. **The decode-step check was a statistical tie on this prompt.** The 350m fp32
   oracle's decode top-2 margin is **+0.0102** (and one sweep position sits at 0.045)
   — below the **margin ≥ 0.1 oracle-validity rule** (see the LFM2.5 card): healthy
   int8 noise flips it regardless of recipe (block-8-all lands at +0.0093 — a coin
   toss). The same prompt on the 1b yields margins 0.137–2.95, which is why the 1b
   gates cleanly with the *standard* recipe, no rescues.

Since int8 is also no faster at 350m, the variants were dropped rather than re-gated
on a new prompt: **350m ships fp16, 1b ships int8lin.**

## int8lin recipe (1b)

Per-block-32 **linear** int8 (scale-multiply dequant, no LUT). Quantized: Mamba2
in/out projections, shared-MLP input/output, attention q/k/v/o (NoPE attention
quantizes cleanly here — no LFM-style exclusion needed). Excluded: embedding (tied,
fp16), depthwise conv1d, all norms (incl. the gated Mamba output norm), lm_head.

## Numerics gating

- Tier A parity ladder (authored fp32 vs HF eager oracle, `_smoke/test_granite_ladder.py`):
  Mamba2 mixer stepped S=1 (output + final conv/SSM state), attention mixer + KV
  content, full-model teacher-forced sweep — 16/16 + HF-seeded decode step, both sizes.
- GPU gate (`_smoke/test_granite_decode_oracle_gate.py`, the
  [pipelined-engine](../knowledge/pipelined-engine.md) shipping gate): teacher-forced
  S=1 sweep on a fresh state + HF-cache-seeded decode step vs the fp32-HF oracle.
  **16/16 + decode for 1b int8lin, 1b fp16, 350m fp16.**
- Engine-path agreement: 24-token greedy rollouts (fixed random + natural prompts),
  python-GPU vs torch reference — 24/24 on the 350m; 1b run + `llm-runner` warm-load
  numbers to follow.

## Convert it yourself

```bash
cd coreai-models   # with the granite4h model overlay (models/macos/granite4h.py) in place
.venv/bin/python ../coreai-models-community/conversion/export_granite4h_decode_pipelined.py int8lin
COREAI_CHUNK_THRESHOLD=1 ./.build/release/llm-benchmark \
    --model exports/granite_4_0_h_1b_decode_int8lin -p 128 -g 256 -n 3
```

`--hf-id ibm-granite/granite-4.0-h-350m` exports the 350m on the same path (use
`fp16` there). Run contract (same as qwen3.5): Swift package patches
[`../apps/coreai-shared-product.patch`](../apps/coreai-shared-product.patch) +
[`../apps/coreai-pipelined-extra-states.patch`](../apps/coreai-pipelined-extra-states.patch)
applied on a fresh `coreai-models` clone; `COREAI_CHUNK_THRESHOLD=1` before engine
creation; never call `engine.warmup()` on the S=1 bundle (warm with a 1-token
generate; `llm-runner` needs `--warmup exact --warmup-length 1`).

## License

Model weights: **Apache-2.0** (IBM Granite). The conversion *code* in this repo stays
BSD-3-Clause.
