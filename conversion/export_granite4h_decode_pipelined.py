"""Export a DECODE-ONLY (S=1) Granite 4.0-H bundle for the Core AI pipelined engine.

Granite 4.0-H (IBM) is a Mamba2 + attention hybrid — the first SSM-scan
architecture on the pipelined path. The dense "H" decoders interleave a few
GQA NoPE attention layers into a Mamba2 stack (1b: 36 mamba + 4 attention;
350m: 28 + 4). At S=1 the Mamba2 selective scan collapses to a single
recurrence step (state = state*dA + dt*B*x), so the decode graph is loop-free
the same way qwen3.5's GDN step is — no while_loop, lowers on the MPSGraph
GPU delegate. State = growing KV (4 attn layers) + TWO fixed-shape extra
states (stacked conv columns + stacked SSM states), exactly the
coreai-pipelined-extra-states.patch budget (<=2).

input_ids is STATIC [1,1]; position_ids and the KV seq dim stay dynamic, so
`EngineFactory` classifies the bundle as dynamic -> pipelined engine. Prefill
runs as pipelined S=1 steps: set `COREAI_CHUNK_THRESHOLD=1` (prompt tok/s ~
decode tok/s).

Numerics gate: 16/16 teacher-forced single-step top-1 vs the fp32 HF oracle +
HF-cache-seeded decode step (run _smoke/test_granite_decode_oracle_gate.py),
on an oracle whose top-2 margin is >= 0.1 at every position.

Requires the granite4h model overlay on `coreai-models` (see
conversion/README.md) plus the pipelined-engine extra-states patch on the
Swift side to RUN the bundle.

Run:  python export_granite4h_decode_pipelined.py [fp16|int8lin] \
          [--hf-id ibm-granite/granite-4.0-h-1b] [--out-dir exports]

Modes: fp16 - baseline; int8lin - per-block-32 linear int8 (the qwen3.5 ship
recipe: scale-multiply dequant, no LUT). Embedding/conv1d/norms/lm_head stay
fp16. Ship guidance: 1b -> int8lin (gate 16/16, ~32% faster); 350m -> fp16
(int8 is numerically marginal there AND no faster — the model is
overhead-bound, not bandwidth-bound, at that size).
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
from coreai_models.models.macos.granite4h import (
    DECODE_STATE_NAMES,
    Granite4HForCausalLMStateful,
    build_decode_state,
)
from coreai_models.primitives.macos.cache import KVCache

DTYPE = torch.float16


def linear_quant_config(dtype: str = "int8") -> dict:
    """Weight-only linear int8 per-block-32 — scale-multiply dequant, no LUT.
    Embedding/conv/norms (incl. the gated Mamba output norm) excluded by type;
    the tied lm_head excluded by name. Unlike LFM2.5 the attention projections
    quantize cleanly here (NoPE, no q/k norm gains to amplify fp16 noise)."""
    return {
        "execution_mode": "eager",
        "global_config": {
            "op_state_spec": {
                "weight": {
                    "dtype": dtype,
                    "qscheme": "symmetric_with_clipping",
                    "granularity": {"type": "per_block", "block_size": 32, "axis": 1},
                }
            },
            "op_input_spec": None,
            "op_output_spec": None,
        },
        "module_type_configs": {
            "coreai_models.primitives.macos.sdpa.SDPA": None,
            "coreai_models.primitives.macos.rms_norm.RMSNorm": None,
            "coreai_models.models.macos.granite4h.GraniteGatedRMSNorm": None,
            "torch.nn.modules.sparse.Embedding": None,
            "torch.nn.modules.conv.Conv1d": None,
        },
        "module_name_configs": {r".*lm_head$": None},
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
    ap.add_argument("mode", nargs="?", default="int8lin", choices=["fp16", "int8lin"])
    ap.add_argument("--hf-id", default="ibm-granite/granite-4.0-h-1b")
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--max-ctx", type=int, default=4096)
    args = ap.parse_args()

    short = args.hf_id.rsplit("/", 1)[-1].lower().replace(".", "_").replace("-", "_")
    name = f"{short}_decode_{args.mode}"

    print(f"loading {args.hf_id} fp16 ...")
    model = Granite4HForCausalLMStateful.from_hf(args.hf_id, target_dtype=DTYPE)
    model.eval()
    cfg = model.config
    print(f"{cfg.num_hidden_layers} layers ({cfg.num_mamba_layers} mamba / "
          f"{cfg.num_attn_layers} attention), hidden={cfg.hidden_size}, "
          f"vocab={cfg.vocab_size}")

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
        "rec_state": state["rec_state"],
    }
    seq_pos = torch.export.Dim("seq_pos", min=2, max=args.max_ctx - 1)
    k_seq = torch.export.Dim("k_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=args.max_ctx)
    v_seq = torch.export.Dim("v_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=args.max_ctx)
    dynamic_shapes = {
        "input_ids": None,  # static [1, 1] — single recurrence step, no scan
        "position_ids": {1: seq_pos},
        "k_cache": {KVCache.seq_len_dim(): k_seq},
        "v_cache": {KVCache.seq_len_dim(): v_seq},
        "conv_state": None,  # fixed-shape extra states
        "rec_state": None,
    }

    if args.mode == "int8lin":
        from coreai_models.export.compression import quantize_pytorch_model

        print("quantizing (linear int8 per-block-32) ...")
        model = quantize_pytorch_model(
            model, tuple(reference_inputs.values()), dynamic_shapes,
            linear_quant_config())

    # No GatedDeltaUpdate modules in Granite 4.0-H — drop its externalize spec
    # (the exporter would otherwise look for uncalled submodules).
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
