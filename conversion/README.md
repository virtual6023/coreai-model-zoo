# Conversion

PyTorch → Core AI `.aimodel`: re-authored models + convert / verify / compress scripts.

## How it relates to Apple's `coreai_models`

The re-authored decoders use `coreai_models` primitives (KVCache, RMSNorm, RoPE, SDPA, SSMState,
…) and the `coreai_models.export` pipeline. Apple's `coreai-models` does **not** take PRs and does
**not** register these newer models, so the model authoring + export wiring lives in **our fork /
overlay** of that package. Concretely, the additions are:

- `models/macos/qwen3_5.py`, `models/macos/gemma4_text.py` — re-authored decoders (+ config shims).
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
- **Qwen3.5 pipelined fast path (in this dir): `export_qwen3_5_decode_pipelined.py [int8lin]`** —
  decode-only loop-free bundle for Apple's `coreai-pipelined` GPU engine, 204 tok/s on M4 Max
  (3.5× the custom-kernel CLI). Needs the Swift engine patch
  `../apps/coreai-pipelined-extra-states.patch` and `COREAI_CHUNK_THRESHOLD=1` at run time.

## Reproduce (env)

Convert/verify needs the `coreai-core` + `coreai-torch` + `coreai-opt` Python env (macOS; the
`coreai-core` wheel is OS-coupled — re-verify after any OS bump). The HF reference oracles need a
transformers build with the target models. CLI: `coreai.llm.export <model> [--compression int8]`.

## License

The re-authored code derives from Apple's **BSD-3-clause** `coreai_models` and retains its
notices. This repo is licensed **BSD-3-Clause** (see the root [`LICENSE`](../LICENSE)).
