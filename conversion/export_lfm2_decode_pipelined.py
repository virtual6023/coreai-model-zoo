"""Export a DECODE-ONLY (S=1) LFM2 / LFM2.5 bundle for the Core AI pipelined engine.

LFM2 is a conv + full-attention hybrid (LiquidAI; 1.2B: 10 short-conv layers +
6 GQA attention layers). Unlike Qwen3.5 there is NO recurrent scan anywhere —
the conv mixer is a kernel-3 depthwise causal conv — so the decode graph is
loop-free by construction. State = growing KV (6 attn layers) + ONE extra
fixed-shape conv state [10, 1, hidden, kernel-1], which fits the
coreai-pipelined-extra-states.patch budget (<=2 extra states) with room to
spare.

input_ids is STATIC [1,1]; position_ids and the KV seq dim stay dynamic, so
`EngineFactory` classifies the bundle as dynamic -> pipelined engine. Prefill
runs as pipelined S=1 steps: set `COREAI_CHUNK_THRESHOLD=1` (prompt tok/s ~
decode tok/s).

Numerics gate: 16/16 teacher-forced single-step top-1 vs the fp32 HF oracle +
HF-cache-seeded decode step (run _smoke/test_lfm2_decode_oracle_gate.py).

Requires the lfm2 model overlay on `coreai-models` (see conversion/README.md)
plus the pipelined-engine extra-states patch on the Swift side to RUN the
bundle (the engine carries the conv state as a fixed-shape extra state).

Run:  python export_lfm2_decode_pipelined.py [fp16|int8lin|int4lin] \
          [--hf-id LiquidAI/LFM2.5-1.2B-Instruct] [--out-dir exports]

Modes: fp16 - baseline; int8lin - per-block-32 linear int8 (the qwen3.5 ship
recipe: scale-multiply dequant, no LUT — 256-entry LUT gathers are slow on the
GPU delegate); int4lin - per-block-32 linear int4 (the gemma4-verified
bytes-without-LUT recipe — decode is BW-bound on device, so halving the
quantized bytes is the remaining decode lever; gate 16/16 before believing
it). Embedding/conv1d/norms/lm_head/attention projections stay high precision
in all quantized modes.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch

from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
from coreai_models.export.macos import _EXTERNALIZE_SPECS, export_to_coreai
from coreai_models.models.macos.lfm2 import (
    DECODE_STATE_NAMES,
    build_decode_state,
    lfm2_from_hf,
)
from coreai_models.primitives.macos.cache import KVCache

DTYPE = torch.float16


def linear_quant_config(dtype: str = "int8", rescue_int8_regex: str | None = None,
                        block: int = 32) -> dict:
    """Weight-only linear per-block-32 — scale-multiply dequant, no LUT.
    Embedding/conv/norms excluded by type; lm_head excluded by name (tied
    table stays fp16 — the eager quantizer skips shared params anyway).
    The attention q/k/v/out projections are ALSO excluded: they are the
    GPU-delegate precision-critical path (kept fp32, see the lfm2 module
    header) and quantizing them flips near-tie argmaxes (14/16 vs 16/16).
    The MLP + conv-mixer linears (the bulk of the weights) still quantize.
    ``rescue_int8_regex`` keeps the matching modules at int8 per-block-32
    while the rest take ``dtype`` (the int4 rescue lever: same linear scheme,
    8-bit — NOT the per_tensor rescue that failed on qwen)."""
    def spec(d: str, b: int = 32) -> dict:
        return {
            "op_state_spec": {
                "weight": {
                    "dtype": d,
                    "qscheme": "symmetric_with_clipping",
                    "granularity": {"type": "per_block", "block_size": b, "axis": 1},
                }
            },
            "op_input_spec": None,
            "op_output_spec": None,
        }

    name_configs: dict = {
        r".*lm_head$": None,
        r".*self_attn\.(q_proj|k_proj|v_proj|out_proj)$": None,
    }
    if rescue_int8_regex:
        name_configs[rescue_int8_regex] = spec("int8")
    return {
        "execution_mode": "eager",
        "global_config": spec(dtype, block),
        "module_type_configs": {
            "coreai_models.primitives.macos.sdpa.SDPA": None,
            "coreai_models.primitives.macos.rms_norm.RMSNorm": None,
            "torch.nn.modules.sparse.Embedding": None,
            "torch.nn.modules.conv.Conv1d": None,
        },
        "module_name_configs": name_configs,
    }


def write_bundle_metadata(out_dir: Path, name: str, hf_id: str, cfg, max_ctx: int) -> None:
    meta = {
        "metadata_version": "0.2",
        "kind": "llm",
        "name": name,
        "assets": {"main": f"{name}.aimodel"},
        "language": {
            "tokenizer": hf_id,
            "vocab_size": cfg.vocab_size,
            "max_context_length": max_ctx,
            "embedded_tokenizer": True,
            "function_map": {"main": ["main"]},
        },
        "source": {"model_definition": "torch", "hf_model_id": hf_id},
        "compression": None,
        "compilation": {"date": datetime.now(timezone.utc).isoformat(), "targets": []},
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", nargs="?", default="int8lin",
                    choices=["fp16", "int8lin", "int4lin", "int4lin8"])
    ap.add_argument("--hf-id", default="LiquidAI/LFM2.5-1.2B-Instruct")
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--rescue-regex", default=None,
                    help="int4lin8 only: override the int8-rescue module regex "
                         "(default = the conv-mixer projections)")
    ap.add_argument("--int4-block", type=int, default=32,
                    help="per-block granularity for the int4 modes (16 halves "
                         "the per-weight range error at ~+6%% size)")
    ap.add_argument("--tag", default="",
                    help="suffix for the bundle name (rescue-bisect probes)")
    args = ap.parse_args()

    short = args.hf_id.rsplit("/", 1)[-1].lower().replace(".", "_").replace("-", "_")
    name = f"{short}_decode_{args.mode}{args.tag}"

    print(f"loading {args.hf_id} fp16 ...")
    model = lfm2_from_hf(args.hf_id, target_dtype=DTYPE, stateful=True)
    cfg = model.config
    print(f"{cfg.num_hidden_layers} layers ({cfg.num_full_layers} full / "
          f"{cfg.num_conv_layers} conv), ff_dim={cfg.ff_dim}, vocab={cfg.vocab_size}")

    # Decode trace: S=1 static query, dynamic full-length positions, dynamic KV seq.
    trace_past = 64
    input_ids = torch.randint(1, cfg.vocab_size, (1, 1), dtype=torch.int32)
    position_ids = torch.arange(trace_past + 1, dtype=torch.int32).unsqueeze(0)
    state = build_decode_state(cfg, max_seq_len=TRACE_KV_CACHE_SEQ_LEN, dtype=DTYPE)

    reference_inputs = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "k_cache": state["k_cache"],
        "v_cache": state["v_cache"],
        "conv_state": state["conv_state"],
    }
    seq_pos = torch.export.Dim("seq_pos", min=2, max=args.max_ctx - 1)
    k_seq = torch.export.Dim("k_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=args.max_ctx)
    v_seq = torch.export.Dim("v_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=args.max_ctx)
    dynamic_shapes = {
        "input_ids": None,  # static [1, 1]
        "position_ids": {1: seq_pos},
        "k_cache": {KVCache.seq_len_dim(): k_seq},
        "v_cache": {KVCache.seq_len_dim(): v_seq},
        "conv_state": None,  # fixed-shape extra state
    }

    if args.mode in ("int8lin", "int4lin", "int4lin8"):
        from coreai_models.export.compression import quantize_pytorch_model

        dtype = "int8" if args.mode == "int8lin" else "int4"
        # int4lin8 = int4 everywhere quantized EXCEPT the rescue set, which
        # stays int8 (default: the conv-mixer projections — the
        # mixer-projection class is the int4-sensitive path, the qwen3.5
        # lesson; --rescue-regex widens the set for bisects).
        rescue = None
        if args.mode == "int4lin8":
            rescue = args.rescue_regex or r".*conv\.(in_proj|out_proj)$"
        block = args.int4_block if dtype == "int4" else 32
        print(f"quantizing (linear {dtype} per-block-{block}"
              f"{', conv-mixer rescued to int8' if rescue else ''}) ...")
        model = quantize_pytorch_model(
            model, tuple(reference_inputs.values()), dynamic_shapes,
            linear_quant_config(dtype, rescue_int8_regex=rescue, block=block))

    # No GatedDeltaUpdate in LFM2 — drop its externalize spec (the exporter
    # would otherwise look for uncalled submodules). RMSNorm/SDPA stay fused
    # composites for the GPU delegate.
    specs = [s for s in _EXTERNALIZE_SPECS if s.composite_op_name != "gated_delta_update"]
    print("exporting decode-only graph to Core AI dialect ...")
    prog = export_to_coreai(
        model,
        reference_inputs,
        dynamic_shapes=dynamic_shapes,
        input_names=("input_ids", "position_ids"),
        output_names=("logits",),
        state_names=DECODE_STATE_NAMES,
        externalize_modules=specs,
    )
    print("optimizing ...")
    prog.optimize()

    out_dir = Path(args.out_dir) / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    import coreai.runtime as rt

    aimodel = out_dir / f"{name}.aimodel"
    print(f"saving {aimodel} ...")
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())

    write_bundle_metadata(out_dir, name, args.hf_id, cfg, args.max_ctx)
    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained(args.hf_id).save_pretrained(out_dir / "tokenizer")
    print(f"bundle ready: {out_dir}")
    print(f"run: COREAI_CHUNK_THRESHOLD=1 llm-benchmark --model {out_dir} -p 128 -g 256 -n 3")


if __name__ == "__main__":
    main()
