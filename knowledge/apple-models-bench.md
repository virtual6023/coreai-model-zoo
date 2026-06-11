# Apple coreai-models: measured benchmarks (the README Apple didn't write)

Apple's [coreai-models](https://github.com/apple/coreai-models) repo ships 21 export
recipes but publishes **zero performance numbers and zero sample apps**. This page is
the missing table: every model exported with **Apple's official recipe, unmodified**,
and measured with **Apple's official runners** (`llm-benchmark` / `llm-runner`) on
real hardware.

- **Hardware**: MacBook Pro M4 Max 128GB (macOS 27 beta) · iPhone 17 Pro (iOS 27 beta)
- **Method**: `llm-benchmark` defaults — 512 prompt tokens / 1024 generation tokens /
  5 trials, release build (`swift run -c release llm-benchmark --model <bundle>`).
  Load times from `llm-runner` "Model Load" line. Memory = peak physical footprint
  (`/usr/bin/time -l`). Cold load = first run after export (includes on-device
  specialization), warm = subsequent runs.
- All numbers are decode-side unless noted. `prompt tok/s` ≈ prefill throughput.

## LLMs — macOS (M4 Max)

| Model | Recipe (registry preset) | Artifact | Prompt tok/s | Gen tok/s | Model load (warm) | Peak mem | Notes |
|---|---|---|---|---|---|---|---|
| gpt-oss-20b (MoE) | `none` / bf16 / ctx 32768 (MXFP4 weights kept) | 13 GB | **1252** | **78.1** | 2.1 s (cold 13.2 s) | 33.9 GB RSS | first published Core AI big-MoE numbers; see chunk-threshold note |
| qwen3-0.6b | `4bit` / fp16 / ctx 8192 | 335 MB | 9396 | 484 (558 short-ctx) | 0.10 s (cold 0.85 s) | 0.77 GB RSS | decode is ctx-dependent: 558 @ 64p/128g |
| qwen3-4b | `4bit` / fp16 / ctx 40960 | 2.1 GB | 1635 | 145.4 (164 short-ctx) | 0.36 s (cold 1.95 s) | 4.6 GB RSS | |
| qwen3-8b | `4bit` / fp16 / ctx 40960 | 4.3 GB | 912 | 94.1 (102 short-ctx) | 0.64 s (cold 2.92 s) | 9.3 GB RSS | |
| gemma3-4b-it | `4bit` / bf16 / ctx 131072 | 2.1 GB | 1669 | 141.5 (157 short-ctx) | 0.32 s (cold 2.20 s) | 4.5 GB RSS | HF gated |
| gemma3-12b-it | `4bit` / bf16 / ctx 131072 | TBD | TBD | TBD | TBD | TBD | HF gated |
| mistral-7b-instruct-v0.3 | `4bit` / fp16 / ctx 8192 | 3.8 GB | 976 | 101.7 (109 short-ctx) | 0.56 s (cold 2.49 s) | 8.3 GB RSS | 27 GB download — see gotchas |

## LLMs — iPhone 17 Pro

Only Qwen has iOS presets (gemma3 / mistral / gpt-oss are macOS-only in the registry).

| Model | Recipe | Artifact | Prompt tok/s | Gen tok/s | Cold / warm load | Notes |
|---|---|---|---|---|---|---|
| qwen3-0.6b | mixed 4/8-bit palettized yaml / ctx 4096 | TBD | TBD | TBD | TBD | |
| qwen3-4b | mixed 4/8-bit palettized yaml / ctx 4096 | TBD | TBD | TBD | TBD | |

## Vision models — GPU vs ANE (Mac + iPhone)

Vision is the Neural Engine's home turf, and nobody publishes per-unit numbers for
these recipes. Method: load each official `.aimodel` export with
`SpecializationOptions.from_preferred_compute_unit_kind(<unit>)` (Python runtime),
synthetic inputs from the function descriptors, 3 warmup + 20 timed runs, median
single-inference latency. "Preferred" means the runtime may still place unsupported
ops elsewhere.

### M4 Max

| Model | Recipe | Artifact | GPU | ANE | CPU | Winner |
|---|---|---|---|---|---|---|
| clip-vit-base-patch32 | fp32 static (image+text joint) | 577 MB | 6.54 ms | **5.43 ms** | 18.76 ms | ANE |
| clip-vit-base-patch32 | **fp16** (official `--dtype float16`) | 289 MB | 6.31 ms | **3.68 ms** | — | **ANE, 1.7× over GPU** |
| yolos-base | fp32 static | 488 MB | **444.8 ms** | 456.7 ms | 733.7 ms | GPU (≈tie) |
| sam3 | fp32 static (promptable, bundled tokenizer) | 3.1 GB | **559.9 ms** | 565.7 ms | 2789.7 ms | GPU (≈tie) |
| depth-anything-3 (small) | fp32 static | 101 MB | 7.30 ms | **6.84 ms** | 34.58 ms | ANE |

Observation: **every official CV recipe DEFAULTS to float32**, and at fp32 the big
ViTs land in a GPU/ANE tie on M4 Max. But the scripts expose `--dtype float16`, and
fp16 is what the ANE runs natively: CLIP at fp16 drops to **3.68 ms on ANE (1.7×
faster than GPU, 1.5× faster than fp32-ANE) at half the artifact size**. If you're
deploying these recipes to ANE, pass `--dtype float16`. (First ANE load of a new
variant pays one-time specialization, ~5 s for CLIP; subsequent loads are sub-second.)

### iPhone 17 Pro

(pending — device CV harness after LLM device runs)

## Reproduction

```bash
git clone https://github.com/apple/coreai-models && cd coreai-models
uv run coreai.llm.export <preset>            # e.g. gpt-oss-20b, qwen3-0.6b
swift run -c release llm-benchmark --model exports/<bundle>
swift run -c release llm-runner --model exports/<bundle> --prompt "..."

# CV models (standalone recipes; add --dtype float16 for the ANE-optimal variant)
uv run models/clip/export.py --output-dir exports
python knowledge/scripts/bench_cv_aimodel.py exports/<model>.aimodel gpu neural_engine cpu
```

Want to chat with these bundles instead of benching them? See
[`apps/CoreAIChatMac`](../apps/CoreAIChatMac) — a minimal macOS chat app on the
official runtime with live load/TTFT/tok-s stats (works with every LLM row above).

## Deep dive: gpt-oss-20b (the first big-MoE numbers on Core AI)

Export is painless: ~8 min download (13.8 GB — only the MXFP4 shards; `original/` and
`metal/` weights are NOT fetched) + ~3 min convert on M4 Max. The MXFP4 weights pass
through unchanged (`compression: null` in metadata), giving a 13 GB artifact.

Official `llm-benchmark` (512 prompt / 1024 gen / 5 trials, release build):

| Metric | Value |
|---|---|
| Prefill (512 tok) | 1252 tok/s (σ < 0.5%) |
| Decode (1024 tok) | 78.1 tok/s (σ < 0.1%) |
| Cold load (first run ever, incl. GPU specialization) | 13.2 s |
| Warm load | 2.1 s |
| Peak RSS | 33.9 GB |

### The chunk-threshold dial (`COREAI_CHUNK_THRESHOLD` / `llm-runner --chunk-size`)

`llm-runner --help` hints "use 128 for MoE". On a 128 GB M4 Max the opposite is true —
but the hint is really a **memory dial**, and the memory numbers explain why it exists.
4096-token prefill, 3 trials:

| Chunk threshold | Prefill tok/s | Peak dirty footprint |
|---|---|---|
| 128 (the MoE hint) | 766 | **1.7 GB** |
| 1024 (default) | 1237 | (not measured) |
| 8192 (no chunking) | **1439** | **18.0 GB** |

Unchunked MoE prefill allocates huge expert activations (~18 GB dirty for 4096 tokens
on top of the mmap'd weights). On a 16–32 GB Mac that would swap or jetsam — chunk 128
caps it at 1.7 GB for a 1.9× prefill cost. On a big-RAM Mac, RAISE the threshold:
+16% prefill over the default for free. Decode is unaffected (~76–78 tok/s everywhere).

> Repro: `COREAI_CHUNK_THRESHOLD=8192 swift run -c release llm-benchmark --model exports/gpt_oss_20b_dynamic -p 4096 -g 128 -n 3`

Output sanity: greedy completion produces correct harmony-format output
(`<|channel|>analysis` → `final`), tool-use tokens intact.

### Core AI vs MLX — the full matrix (same M4 Max, same methodology)

Apple's `llm-benchmark` is explicitly modeled on `mlx-lm benchmark`, so the two are
directly comparable: identical synthetic-prompt protocol, 512 prompt / 1024 generation
/ 5 trials. MLX side: `mlx-lm 0.31.3`, `mlx-community` 4-bit conversions (gpt-oss uses
the same MXFP4 `openai/gpt-oss-20b` in both stacks).

Decode tok/s (and prefill in parentheses):

| Model | Core AI (official recipe) | MLX 0.31.3 | Decode verdict |
|---|---|---|---|
| gpt-oss-20b (MoE) | 78.1 (1252) | **100.2** (1528) | **MLX +28%** |
| qwen3-0.6b | **484** (9396) | 432 (9366) | **Core AI +12%** |
| qwen3-4b | 145.4 (**1635**) | 145.8 (1495) | tie |
| qwen3-8b | **94.1** (912) | 90.0 (825) | **Core AI +5%** |
| gemma3-4b-it | **141.5** (1669) | 136.3 (1631) | **Core AI +4%** |
| mistral-7b-v0.3 | **101.7** (976) | 97.5 (918) | **Core AI +4%** |

**The pattern: Core AI matches or beats MLX on every dense model (+4–12% decode,
+6–11% prefill on the bigger ones). MLX's one clear win is the MoE** — gpt-oss-20b
decode +28% — pointing at the expert-dispatch path, not the core engine, as Core AI's
current gap. (gpt-oss memory: MLX Metal peak 14.6 GB vs Core AI 33.9 GB RSS — not
directly comparable; RSS includes the mmap'd 13 GB weight file.)

> Quantization comparability: Core AI macOS presets = int4 weight-only, block 32;
> mlx-community = 4-bit affine, group 64. Same weight-byte class, slightly different
> schemes. gpt-oss is byte-identical MXFP4 in both.
>
> Repro (MLX): `pip install mlx-lm && python -m mlx_lm benchmark --model <repo> -p 512 -g 1024 -n 5`

## Gotchas (found while benching)

- `/usr/bin/time -l`'s "peak memory footprint" counts only dirty pages — the mmap'd
  weight file shows up in "maximum resident set size" instead. Report RSS for "how much
  RAM do I need", footprint for "how much does inference itself allocate".
- The first-ever run of a bundle includes on-device GPU specialization (gpt-oss-20b:
  13.2 s vs 2.1 s warm). Don't average it into load-time numbers — report both.
- **mistralai/Mistral-7B-Instruct-v0.3 downloads 27 GB, not 15** — the repo ships both
  transformers shards AND a redundant `consolidated.safetensors` (14 GB), and the export
  fetches everything. On a tight disk this ENOSPCs mid-export. Fix: delete the
  `consolidated.safetensors` blob from the HF cache, `hf download ... --include "tokenizer*" "*.json"`,
  then re-run the export with `HF_HUB_OFFLINE=1` (loads from the sharded files).
- Apple's exporters need scratch space ≈ one extra copy of the fp16 weights while
  serializing; budget disk accordingly for 7B+ models.
- `models/depth-anything/export.py` crashes with **OMP Error #15** (duplicate libomp —
  torch + DA3's deps both link OpenMP in the uv-resolved env). Workaround:
  `KMP_DUPLICATE_LIB_OK=TRUE uv run export.py`. Output verified fine.
- The CV recipes default to float32 but DO expose `--dtype float16|bfloat16|float32`.
  fp16 is the dtype ANE actually runs natively — see the fp16 row(s) in the vision table.
- **An `.aimodel` is a build artifact, not a pure function of the recipe**: the same
  `coreai.llm.export qwen3-0.6b` produced a 2.2× faster artifact on macOS 26 than on
  the 27 beta (native quantized-Linear lowering vs explicit dequant ops; same code,
  same wheels). Full forensics: apple-silicon-llm-bench
  `methodology/coreai-export-lowering.md`. Version-stamp and keep your artifacts.

## README 統合待ち (rows for the zoo README — community session owns that file)

Suggested additions when integrating:

1. Knowledge index: add `apple-models-bench.md — measured numbers for Apple's official
   recipes (M4 Max + iPhone 17 Pro), Core AI vs MLX, GPU vs ANE` to the knowledge list.
2. Apps table: add `CoreAIChatMac — macOS chat app for any exported bundle, official
   runtime, live tok/s stats (apps/CoreAIChatMac)`.
3. Headline numbers for the README top (if wanted): gpt-oss-20b 78 tok/s on M4 Max /
   qwen3-0.6b 484 tok/s / CLIP fp16 on ANE 3.7 ms.
