# Riding Apple's pipelined GPU engine — 3.5× over a hand-rolled loop, zero custom kernels

> The single highest-leverage Core AI LLM finding in this project (2026-06-10/11, verified on
> M4 Max + iPhone 17 Pro): the same `.aimodel` weights decode **3.5× faster** when Apple's
> `coreai-pipelined` engine drives them instead of a hand-rolled per-token `fn.run()` loop —
> Qwen3.5-0.8B int8: **204 tok/s vs 58.5 on M4 Max, 50.3–51.5 vs 42.5–45.4 (fused-kernel
> monolith) on iPhone 17 Pro**. This page is how to put a model on that engine, every trap we
> hit, and what doesn't fit. Working artifacts: the qwen3.5 fast path in
> [`../conversion/export_qwen3_5_decode_pipelined.py`](../conversion/export_qwen3_5_decode_pipelined.py)
> + [`../apps/coreai-pipelined-extra-states.patch`](../apps/coreai-pipelined-extra-states.patch)
> + the [`gpu-pipelined/` HF bundle](https://huggingface.co/mlboydaisuke/qwen3.5-0.8B-CoreAI).

## Why hand-rolled per-token loops lose ~3×

The zoo's first engines (and most sample code) drive low-level `AIModel.load()` →
`await fn.run()` per token. Three structural costs, none of them kernel-fixable:

1. **Synchronous dispatch** — `await fn.run()` blocks on GPU completion every token; CPU prep
   for token N+1 never overlaps GPU work for N. The pipelined engine `encode()`s and returns,
   with a `PipelineGate(capacity: 3)` bounding in-flight steps; the sampler's Metal completion
   handler yields the token.
2. **CPU sampling = a full-vocab logits readback per token** — 250–600 K-vocab fp16 logits
   cross the GPU→CPU bus, then argmax on CPU. The engine samples **on-GPU** (MPSGraph
   argmax/topK); only the sampled int crosses.
3. **Host-cache KV** — re-feeding the whole KV per token (the workaround for the in-graph
   KV-write bug, see [`coreai-beta-mpsgraph-kvwrite-bug.md`](coreai-beta-mpsgraph-kvwrite-bug.md))
   plus multi-dispatch per token. The engine grows KV **on-device** (buffer-expand + async blit).

Correction this forced: [`performance-ceiling.md`](performance-ceiling.md)'s "the MLX gap is
structural" verdict was measured on the hand-rolled loop — it was that loop's ceiling, not Core
AI's. The official engine is ~2× **faster** than MLX on the same machine (qwen3-0.6B-4bit:
~1,150 tok/s vs MLX ~535).

## What a model needs to ride the engine

`EngineFactory.createEngine` auto-selects `coreai-pipelined` for **dynamic-shape** bundles
(`chunkedStatic` → the static/ANE engine instead). The checklist:

- **One `main` graph, `input_ids [1,S] → logits`**, embedding and lm_head **in-graph** (the
  engine feeds token ids and reads sampled ids — there is no hook for host-side gathers or a
  separate head dispatch). Embed+head tables must therefore fit in the graph: fine up to a
  few GB fp16, which covers 0.3–4B-class models.
- **A growing KV pair** (`keyCache`/`valueCache`, dynamic seq dim, layers stacked in one
  tensor). Sliding-window layers can ride as plain linear KV (the mask keeps the math right;
  you pay some memory).
- **At most 2 extra fixed-shape states** — with our engine patch
  ([`../apps/coreai-pipelined-extra-states.patch`](../apps/coreai-pipelined-extra-states.patch)).
  Stock, the engine hard-requires *exactly* the 2 KV states; the patch lets it carry e.g. a
  hybrid-SSM's `convState`/`recState`: allocated zero-filled, bound on every encode, zeroed on
  `reset()`. (Patch gotcha for future edits: `AsyncMutableViews.insert` is
  `@_lifetime(self: &value)` — the insert and the consuming `encode` must sit in ONE scope;
  hence the `switch extraStates.count` blocks instead of a loop.)
- **A full LanguageBundle directory** — `metadata.json` (kind `llm`, `assets.main`,
  `language.{tokenizer,vocab_size,max_context_length,function_map}`) + `tokenizer/` +
  the `.aimodel`. A bare `.aimodel` dir is not loadable by `LanguageBundle`/`EngineFactory`.
- **No giant per-token host tables.** This is what excludes Gemma 4 (3n-style): its 9.4 GB
  per-layer-embedding table is gathered by token id on the host every step, and the engine has
  no per-token host-input hook. (Feasible engine mod, untested: the sampled token is known in
  the sampler completion handler before the next encode — a "per-token extra inputs" provider
  could mmap-gather the PLE rows there. The gather itself is sub-ms.)

## The export trick: decode-only, loop-free

Recurrent scans (`torch.ops.higher_order.while_loop`, e.g. Qwen3.5's GatedDeltaUpdate) do
**not lower** on the MPSGraph GPU delegate (`'scf.while'` region type mismatch), and on the
macOS-27 beta the while_loop bundle fails even `cpu_only` (Compiler error 2) — so "it verified
on CPU earlier" proves nothing about the GPU graph. The escape that made the qwen3.5 ride
possible:

- **`input_ids` STATIC `[1,1]`** — at S=1 a scan is one step, so a loop-free single-step
  recurrence (`use_loopfree_step=True` on every linear-attention layer, set BEFORE
  quantization/tracing) is numerically identical and removes `scf.while` from the graph
  entirely.
- `position_ids` and the KV seq dims stay **dynamic** → the factory still classifies the
  bundle as dynamic → pipelined engine.
- **Prefill becomes pipelined S=1 steps**: run with `COREAI_CHUNK_THRESHOLD=1`. Prompt tok/s ≈
  decode tok/s (~51 on iPhone, ~200 on M4 Max). For long prompts a static chunked-prefill
  companion graph (q=16 unrolled scan, 147 tok/s on iPhone) still wins time-to-first-token;
  a chunkwise-parallel (no-scan) prefill formulation is the open fix.

## Run contract (every one of these bit us)

- `COREAI_CHUNK_THRESHOLD=1` **before engine creation** (`ModelConfig.chunkThreshold` reads the
  env var; `setenv` in app init is fine).
- **Never call `engine.warmup()`** on an S=1 bundle — it warms query length **256** and the
  static `[1,1]` graph rejects it (`NDArrayDescriptor` fatal). `llm-benchmark` is safe (warms
  via a real trial); `llm-runner` needs `--warmup exact --warmup-length 1`; in an app, a
  1-token generate after load IS the warmup.
- **Benchmark Release builds only** — a Debug engine measures ~3× slow (host-side per-token
  work dominates unoptimized Swift).
- Cold GPU specialization of the 0.8B bundle: ~4.8 s on iPhone, then ~0.2–1.0 s warm loads
  (content-keyed cache) — no AOT needed on this path.

## Quantization on the GPU delegate (measured, qwen3.5-0.8B, M4 Max p128/g256)

| config | size | decode tok/s | verdict |
|---|---:|---:|---|
| fp16 | 1.4 GB | 175.8 | baseline |
| int8 k-means g32 (256-entry LUT) | 1.0 GB | 113.2 | **LUT gather is the bottleneck** — slower than fp16 |
| **int8 linear per-block-32 (no LUT)** | 1.0 GB | **204.1** | ship config; ≡ fp16-GPU token-for-token |
| + untied int8 lm_head | 1.3 GB | 201.4 | **no win** — the head matvec isn't the critical path; naive bandwidth models can lie |
| int4 k-means g8 (+int8 rescue variants) | 0.75–0.88 GB | — | **fails the oracle gate** (12–16, 14/16, 12/16); per_tensor int8 rescue is *coarser* than int4-g8 LUT for SSM in_projs |

Rules of thumb: 256-entry int8 LUTs are slow on this delegate, 16-entry int4 LUTs are fast
(Apple's official qwen3-0.6B int4-km-g8 does 1000+); **per-block linear int8** (scale-multiply
dequant) is what unlocks int8 speed. The eager quantizer **silently skips tied weights** —
clone the embedding table first if you actually want the head quantized (and measure before
believing it helps).

## Numerics gating (how to judge a quantized bundle)

- Gate on **single-step top-1 vs an fp32 HF oracle**: a teacher-forced S=1 sweep over a fixed
  prompt (each position's argmax vs the oracle's per-position argmax — state noise accumulates
  through the bundle's own caches, which is exactly the path quantization corrupts) plus an
  oracle-cache-seeded decode step. 16/16 required.
- **Never gate on long greedy rollouts vs CPU** — GPU and CPU fp16 fork after ~17 tokens on
  natural prompts from accumulated fp16 noise. "≡ fp16-GPU sequence" is meaningful;
  "≠ CPU rollout" is not.
- Verify the **engine path** too, not just the python runtime: the on-GPU argmax sampler
  reproduced the python-probe sequences exactly in our runs, but it's one `llm-runner
  --warmup exact --warmup-length 1` invocation to confirm.
- Cross-device determinism held in our runs: iPhone GPU sequences were 24/24 token-identical
  to M4 Max GPU on both fixed prompts.

## Measured end state (Qwen3.5-0.8B int8lin, 2026-06-11)

| surface | prefill | decode |
|---|---:|---:|
| M4 Max, `llm-benchmark` | 198.8 | **204.1** |
| iPhone 17 Pro, one-shot runner | 51.2 | **50.3–51.5** |
| iPhone 17 Pro, chat app (CoreAIChat Qwen mode, 220-tok turn) | 50.6 | **47.9** |

vs the best previous iPhone config (fused int8 Metal-kernel static monolith): 42.5–45.4 —
**+12–20% with zero custom kernels**, and the same bundle runs on macOS at 204.

## What fits next / what doesn't

- **Fits with zero engine work**: pure transformers ≤~4B (KV-only states) — stock engine, no
  patch; the work is the `coreai_models`-style re-author + export.
- **Fits with the existing patch**: hybrids with ≤2 fixed-shape extra states — conv+attention
  (LFM2-class), Mamba2+attention (Granite-4.0-H / Falcon-H1-class: conv + ssm state = exactly
  2), GDN hybrids (qwen3.5 family incl. 2B — same scripts, `--hf-id`).
- **Needs an engine mod**: per-token host-gathered inputs (Gemma 4's PLE). Design sketch above.
- **Doesn't fit**: pure-RNN models (no growing KV — the engine's state model assumes one),
  models whose embed+head can't live in-graph, big MoE (different problem).

Apple's `coreai-models` repo is **issues-only** (no PRs), which is why the engine patch ships
as a patch file here instead of upstream. If Apple opens the engine, the two capabilities worth
upstreaming are: N extra fixed-shape states, and a per-token host-input hook.
