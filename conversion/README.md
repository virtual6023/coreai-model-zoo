# Conversion

PyTorch → Core AI `.aimodel`: re-authored models + convert / verify / compress scripts.

## How it relates to Apple's `coreai_models`

The re-authored decoders use `coreai_models` primitives (KVCache, RMSNorm, RoPE, SDPA, SSMState,
…) and the `coreai_models.export` pipeline. Apple's `coreai-models` does **not** take PRs and does
**not** register these newer models, so the model authoring + export wiring lives in **our fork /
overlay** of that package. Concretely, the additions are:

- `models/macos/qwen3_5.py`, `models/macos/qwen3_5_moe.py`, `models/macos/gemma4_text.py` — re-authored decoders (+ config shims).
- `models/registry.py` entries (`qwen3_5_text`, `gemma4_text`) + `model_registry.py` short-name
  presets + `export/{presets,metadata,macos,pipeline}.py` hooks (e.g. `export_core()` routing,
  macOS int8 palettization, multi-function front-end gather).

> ⚠️ These currently live as working-tree edits on a local `coreai-models` checkout. Packaging them
> as a clean installable overlay (a thin fork or a patch set applied on top of a pinned
> `coreai-models`) is a TODO so this repo is self-contained.

## Scripts (current locations, to be consolidated here)

- Gemma 4: `convert.py` / `convert_palettize.py` (int8 `all8`) / `convert_stateful*.py` (stateful +
  ring) / `convert_head.py` / `check_pipeline.py` / `verify_*` — the full convert+verify harness.
- Qwen3.5: parity ladder + fp16/int8 + head-split + stateful-palettize harnesses.
- On-device export (kept artifacts): `export_qwen3_5.py [0.8b|2b]`, `export_gemma4_frontend.py`.
- **Qwen3.5 pipelined fast path (in this dir): `export_qwen3_5_decode_pipelined.py`** —
  decode-only loop-free bundles for Apple's `coreai-pipelined` GPU engine. Ship config for
  BOTH sizes is `int8hu --head-sym` (per-block-32 **absmax** int8 head — clipping corrupts
  big-vocab heads; per-channel axis-0 is BROKEN on the beta GPU delegate, and the historical
  `*_perchan_sym` bundle names actually contain per-block-32 heads — see the qwen3.5 card):
  0.8B **210 tok/s M4 Max / 69.7–74.0 iPhone 17 Pro**
  (fp16-head int8lin: 204 / 50.3–51.5; custom-kernel CLI was 58.5); `--hf-id Qwen/Qwen3.5-2B`
  → 2B **161 / 28–30** (int8lin: 127 / 19–21). Needs the Swift engine patch
  `../apps/coreai-pipelined-extra-states.patch` and `COREAI_CHUNK_THRESHOLD=1` at run time.
- **LFM2.5 pipelined (in this dir): `export_lfm2_decode_pipelined.py [fp16|int8lin|int8hu]`** —
  the first non-Qwen rider: LiquidAI's conv+attention hybrid, decode-only S=1 (loop-free by
  construction — no scan anywhere), **253 tok/s int8lin / 276.5 with the int8 head
  (`int8hu --head-sym`) / 162 fp16 on M4 Max**, oracle gate 16/16 (all three). Model overlay: `models/macos/lfm2.py` on the `coreai-models` checkout — it bakes in
  two macOS-27-beta GPU-delegate workarounds (fused single conv-state write; fp32 attention
  projections). Same engine patch + `COREAI_CHUNK_THRESHOLD=1` run contract. See
  [`../zoo/lfm2.5.md`](../zoo/lfm2.5.md).
- **Gemma 4 E2B / E4B pipelined fast path (in this dir): `export_gemma4_decode_pipelined.py [int4lin]`** —
  decode-only S=1 bundle whose per-layer-embedding rows arrive as a per-token INPUT (the 9.4 GB
  PLE table stays a host mmap): in-graph embed + softcapped head, ONE unified padded KV pair,
  oracle 8/8, **70.9 tok/s decode on M4 Max** (+20-25% over the int4km-kernel CLI, zero custom
  kernels; int4-LINEAR per-block — eager-palettized k-means LUTs measure 2.25× slower at the
  same bytes). Needs the full patch stack incl.
  `../apps/coreai-pipelined-per-token-inputs.patch`, `COREAI_CHUNK_THRESHOLD=1`, and a
  `PerTokenInputProvider` that dequants the int8 PLE row dump per token. Add **`--tbl`** to
  export the variant whose PLE table is a STATIC graph input instead (in-graph gather; no
  provider, no per-token decode wait — **77.0 tok/s on M4 Max**, the best Mac gemma4 config;
  needs `../apps/coreai-pipelined-static-inputs.patch` + an app that binds the two dump files
  via `EngineOptions.staticInputBuffers` — buffer-mode traps in
  [`../knowledge/pipelined-engine.md`](../knowledge/pipelined-engine.md)).
  **`--hf-id` swaps the checkpoint**: Google's official QAT releases
  (`google/gemma-4-{E2B,E4B}-it-qat-q4_0-unquantized`) ride the same script — bundle names
  gain `_qat`, E2B-QAT measures 74.7/78.9 (provider/tbl) and **E4B (42L, 2 KV heads, dense
  — no MoE, zero model-code changes) 53.2/55.8**, all oracle 8/8; q4_0 IS per-block-32
  absmax int4, so these bundles carry Google's "≈ bf16" QAT quality claim. Regenerate the
  PLE dump (`--out`) and the oracle (`gen_gemma4_prompt.py --tag`) from the same
  checkpoint; `--lin-sym` exports the literal-q4_0-grid (absmax) variant (measured: same
  gate, same speed). See [`../zoo/gemma4-e4b.md`](../zoo/gemma4-e4b.md).
- **Granite 4.0-H pipelined (in this dir): `export_granite4h_decode_pipelined.py [fp16|int8lin|int8hu]`** —
  the first Mamba2/SSM-scan rider: at S=1 the selective scan is a single recurrence step
  (loop-free, no while_loop), states = KV (4 attn layers) + conv/SSM stacks (= the ≤2
  extra-states budget). 1b int8lin **136.5 tok/s** / fp16 103.6 on M4 Max, oracle gate
  16/16 (`int8hu --head-sym` also gates 16/16 but is Mac-flat at 134.2 — device re-test
  pending, the qwen "Mac no-win ≠ device no-win" pattern); `--hf-id ibm-granite/granite-4.0-h-350m` exports the 350m (ship fp16 there, 191
  tok/s — int8 fails the gate at that scale and is no faster). Model overlay:
  `models/macos/granite4h.py`. Same engine patch + `COREAI_CHUNK_THRESHOLD=1` run contract.
  See [`../zoo/granite-4.0-h.md`](../zoo/granite-4.0-h.md).
- **Qwen3-VL 2B pipelined — the first VLM (in this dir): `export_qwen3_vl_pipelined.py [fp16|int8lin|int8hu]`** —
  emits the text-decoder bundle (+ a `_s1` static-query twin that carries the python
  oracle gates) AND the fixed-grid vision encoder `.aimodel`. Multimodal state rides the
  static-inputs patch: image/deepstack embeds as rewritable owned buffers, image tokens
  as extension ids `V+slot`, interleaved M-RoPE derived in-graph from (ids, pos) + two
  `[1] i32` shift inputs. int8hu **187.6 tok/s decode** on M4 Max, multimodal oracle
  gates 4/4+16/16+HF-seeded vs fp32-HF; iPhone numerics 24/24 (text AND image prompts).
  Model overlay: `models/macos/qwen3_vl.py`. See [`../zoo/qwen3-vl.md`](../zoo/qwen3-vl.md).
- **Qwen3.6-35B-A3B pipelined — the first MoE (in this dir): `export_qwen3_6_decode_pipelined.py [int8lin|int8hu]`** —
  Qwen3.5's hybrid decoder + a 256-expert top-8 sparse-MoE FFN (+ shared expert), 40 layers,
  GVA GatedDeltaNet (32 value / 16 key heads). Experts ride Apple's `SwitchGLU`/`GatherMM`;
  the 4-D expert weights quantize with the documented SwitchLinear override
  (`block_size [1,1,1,32]`), router + shared-expert gate stay fp16, head is absmax int8.
  `int8hu --head-sym` = 35 GB bundle, **30.9 tok/s decode on M4 Max** (35B-A3B, ~3B active);
  fp16 full-scale eager ≡ bf16 HF oracle (margin rule), int8 eager teacher-forced gate.
  Mac-only (35 GB > iPhone jetsam). Model overlay: `models/macos/qwen3_5_moe.py` (MoE FFN +
  packed-expert loader) on top of `qwen3_5.py` (which gained the GVA head-repeat). **NOTE:
  raw `AIModel.load(gpu)` aborts on the MoE→ANE path (`ANE compilation writeToFile failed!`
  / 100 GB temp blowup); the real engine's `expectFrequentReshapes` avoids it — run via
  `llm-benchmark`/`llm-runner`, not raw load.** See [`../zoo/qwen3.6.md`](../zoo/qwen3.6.md).

## Reproduce (env)

Convert/verify needs the `coreai-core` + `coreai-torch` + `coreai-opt` Python env (macOS; the
`coreai-core` wheel is OS-coupled — re-verify after any OS bump). The HF reference oracles need a
transformers build with the target models. CLI: `coreai.llm.export <model> [--compression int8]`.

## License

The re-authored code derives from Apple's **BSD-3-clause** `coreai_models` and retains its
notices. This repo is licensed **BSD-3-Clause** (see the root [`LICENSE`](../LICENSE)).
