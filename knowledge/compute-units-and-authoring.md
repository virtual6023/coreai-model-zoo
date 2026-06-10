# Compute units (ANE / GPU / CPU) & on-device authoring rules

> Foundation note: the empirical do's/don'ts for making a model run correctly + fast on each compute unit.
> The single most important framing: **iOS/ANE = static-shape, BC1S, Conv2d, per-head, fp16-only**;
> **macOS/GPU = dynamic-shape, standard layout, fused, custom kernels**. These are Apple's two first-class modes.
> Sources: `coreai-models/skills/.../model-authoring/references/{neural_engine_rules,gpu_rules,common_issues}.md`,
> `.../working-with-coreai/{SKILL.md,references/guidance.md}`, and the official primitives
> `coreai-models/python/src/coreai_models/primitives/{ios,macos}/` + `export/{ios,macos}.py`.

## The three compute units
| | ANE (Neural Engine) | GPU | CPU (BNNS) |
|---|---|---|---|
| Best for | energy-efficient inference, fixed shapes, iOS foreground | large models, dynamic shapes, batch, max throughput | validation, fallback |
| Shapes | **fully static** (one fn per shape config) | dynamic OK | any |
| Layout | **BC1S** `(B, C, 1, S)` | standard `(B,S,D)` / `(B,H,S,D)` | any |
| Projections | **1×1 Conv2d** (Conv engine accumulates fp32) | `nn.Linear`, fused QKV | any |
| Attention | **per-head, sequential** (no fused SDPA) | **fused native SDPA** (all heads) | either |
| KV cache | **readonly functional I/O** (host writes), seq on **dim 4** | **stateful** (`mutable_slice_update`), seq on **dim 3** | — |
| Custom MSL kernels | **NO** (fixed ops only) | **YES** (`TorchMetalKernel`) | no |
| Precision | **fp16 only** (no fp32 literals/intermediates) | fp16 weights, fp32 intermediates OK | fp32/fp16 |

> The "ANE can't run custom MSL" row is *why* the GPU speed track exists — see
> [`custom-metal-kernels.md`](custom-metal-kernels.md) (project memory: `project_ane_vs_gpu_premise`).

## ANE authoring rules (the high-leverage ones)
- **BC1S layout** `(B, C, 1, S)`; all matmuls as 1×1 Conv2d. `neural_engine_rules.md:43-65,92-109`.
- **Conv2d not nn.Linear** — Linear falls back off-ANE; Conv2d maps to the conv engine **and accumulates in
  fp32** (the fix for fp16 matmul drift over many layers). `neural_engine_rules.md:92-109`.
- **No fp32 anywhere** — a single Python float literal (`1.0`) creates an f32 buffer and breaks ANE residency.
  Use `torch.ones(1, dtype=x.dtype)`. `.float()` is a **no-op on the ANE** (MPSGraph drops the cast). To get
  fp32 accumulation you must use an op the hardware accumulates in fp32 (Conv engine, LayerNorm kernel).
  `neural_engine_rules.md:120-134`, `common_issues.md:49-52`.
- **RMSNorm trap**: composite RMSNorm computes `mean(x²)` in fp16 → **overflows** large activations. Use the
  `[x,-x]` LayerNorm trick (`LayerNorm([x,-x]) == RMSNorm`, and the ANE runs LayerNorm with an fp32-accumulating
  hardware kernel). (This project's gemma4 fix; same root cause as Conv2d.)
- **Per-head SDPA** via einsum `bchq,bkhc->bkhq` (no reshape copies). `primitives/ios/sdpa.py:35-80`.
- **Causal mask**: shape `(1, key, 1, query)` (transposed vs GPU), masked value **`-40000.0` not `-inf`**
  (ANE softmax mishandles IEEE −inf). `neural_engine_rules.md:357-372`, `common_issues.md:12-15`.
- **RoPE as input**: precompute cos/sin outside the graph, pass as 4D `(1, head_dim, 1, S)` (in-graph
  `gather_nd` makes rank-3 → ANE rejects). `neural_engine_rules.md:375-379`.
- **KV cache = readonly I/O**: model concats past+new, returns new K/V; host writes the cache. **Cache the
  post-RoPE key** (`key_rope`), not raw — else stale keys → PSNR ~20 dB. `neural_engine_rules.md:382-427`.
- **Last dim aligned to 64 B / power-of-2**; **rank ≤ 5**; strides/dilations factored into 2s and 3s; large
  kernels decomposed (`k = k1+k2-1`). `neural_engine_rules.md:19-40,147-239`.
- **Chunked prefill** (S_q=64) for long prompts — fp16 per-token decode drifts ~5–10 dB/50 tokens.
  `neural_engine_rules.md:451-465`.

## GPU authoring rules
- Standard layout, `nn.Linear`, **fused QKV** (`gpu_rules.md:132-154`).
- **Native fused SDPA** `F.scaled_dot_product_attention(...)` (`gpu_rules.md:50-65`, `primitives/macos/sdpa.py:13-28`).
- **Stateful KV** via `register_buffer` + `mutable_slice_update`, cache `[n,B,H_kv,max_S,D]` seq dim 3
  (`gpu_rules.md:189-258`, `primitives/macos/cache.py:12-54`). ⚠️ The **data-indexed** write SIGSEGVs the
  WWDC26 beta on GPU+ANE — use a shape-symint index or host-cache (see `coreai-beta-mpsgraph-kvwrite-bug.md`).
- IEEE `-inf` mask is fine on GPU; dynamic shapes + control flow OK; custom Metal kernels available.
- **MoE**: `SwitchLinear` + composite `GatherMM` (cast expert idx to uint16). `gpu_rules.md:262-276`.
- **Memory-efficient load** (7B+): meta-device init + `load_state_dict(assign=True)` + per-layer streaming. `gpu_rules.md:279-297`.

## macOS vs iOS export (the official split)
`export/pipeline.py` picks dynamic (macOS) vs static (iOS).
| | iOS (`export/ios.py`) | macOS (`export/macos.py`) |
|---|---|---|
| Shapes | static buckets (query `[8,16,64]` × cache `256,512,1024,…`) | dynamic `torch.export.Dim` |
| KV | `state_names` + `in_step` data-tensor write + IOSurface/interleave | `state_names` + shape-symint offset |
| Engine | `CoreAIStaticShapeEngine` (host owns KV NDArray, passes state views each step) | `CoreAIPipelinedEngine` (GPU) |
| Target | Neural Engine | GPU |

## Runtime compute-unit selection — auto-derived from STRUCTURE, preferred-not-forced, overridable
The official runtime does NOT hard-pin a compute unit; `export/ios.py`/`export/macos.py` bake none. Instead the
Swift runtime probes the model's **structure** and derives a *preference* (`CoreAIShared/Runtime/ModelStructure.swift:57-66`):
- **`chunkedStatic`** (the iOS recipe: chunked + static shapes) → `SpecializationOptions(preferredComputeUnitKind: .neuralEngine)`
- **`dynamic`** (single `main`, the macOS recipe) → `.gpu` + `expectFrequentReshapes`

`PreparedModel.prepare(at:)` → `probeStructure` → `AIModel(contentsOf: url, options:)` (`:137-141`). Notes:
- It's a **preference, not a lock** — the compiler places ops; AOT `--preferred-compute` **defaults to `none`**
  (compiler decides), and a "compiles but runs on CPU" case needs an explicit `--preferred-compute neural-engine`
  (`common_issues.md:109-112`). So "iOS ⇒ ANE" is the *default tendency*, not a guarantee.
- The axis is **structure, not literally iOS**: static/chunked ⇒ ANE-preferred, dynamic ⇒ GPU-preferred.
- **Overridable**: `EngineFactory` takes an `EngineOptions.variant` override; the low-level path accepts your own
  `SpecializationOptions`; AOT chooses with `--preferred-compute gpu|neural-engine|none`. So ANE is selectable, not forced.

## Verification gates (PSNR)
`working-with-coreai/SKILL.md:94-99`, `guidance.md:145-153`:
- re-authored vs source (fp16): **> 70 dB** (investigate < 60)
- compiled vs torch (fp16): **≥ 40–50 dB**
- 4-bit palettized: **~40 dB** (investigate < 30)

**Localize divergence with REAL inputs** — degenerate constant-input probes lie (they said an ANE chunk was
exact when real inputs showed it diverged from layer 1). This project's hardest-won ANE lesson.

## Decision guidance
- **ANE** when energy/battery + predictable shapes + model fits (iOS ~2 GB) + single-token latency matters.
- **GPU** when large (7B+), batch, dynamic shapes, or you need custom kernels / max throughput. macOS default.
- **CPU** for debugging/fallback only.
- This project's call: **GPU now** (custom kernels, beta-robust) **+ ANE later** (when the KV-write bug lifts +
  int4 head + AOT). (Project memory: `project_ane_vs_gpu_premise`.)
