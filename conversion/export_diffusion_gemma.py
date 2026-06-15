"""DiffusionGemma -> Core AI engine export: zoo's FIRST block-diffusion LLM.

Unlike an autoregressive LLM, DiffusionGemma denoises a whole CANVAS of tokens in
parallel over a short schedule (<=48 steps, early-stop). This export builds three
handoff graphs the host drives in a denoise loop (sampler/temperature stay on the
host; the engine runs the graphs).

IMPORTANT — the model is QAT (quantization-aware-trained) int4 for its MoE experts:
``google/diffusiongemma-26B-A4B-it`` ships bf16 MASTER weights that DEGENERATE in full
precision (the full-precision forward is incoherent; this reproduces identically in HF
Transformers and unquantized MLX, so it is a property of the weights, not a port/backend
bug). Coherent generation needs the QAT-adjusted int4 expert grid: point ``--model-dir``
at the dequantized PUBLISHED mlx-4bit weights (``_diffgemma_pub_bf16``), NOT google's
masters; the int8 quantization here preserves that grid finely.

Three graphs (the image-pipeline component-handoff pattern; KV is a value handoff,
NOT mutated state, so no ``state_names``); CANVAS = ``--canvas`` (default 256):

  encoder.aimodel  (input_ids[1,S], position_ids[1,S])      -> 60 KV (30 layers x k,v)
  decoder.aimodel  (canvas_ids[1,C], position_ids[1,C],     -> logits[1,C,vocab]
                    soft_embeds[1,C,hidden], <60 enc_kv>)      (run <=48x/block; KV fixed/block)
  soft_proj.aimodel(proc_logits[1,C,vocab])                 -> soft_embeds[1,C,hidden]

The encoder is CANVAS-independent (KV are [1,nkv,S,head_dim]); re-export only the decoder
+ soft_proj when changing the canvas and reuse the existing encoder.aimodel
(``--only decoder,soft_proj``). soft_proj factors out the 262144-wide ``softmax @ embed``
self-conditioning builder so (a) the decoder trace is branch-free (soft_embeds=0 ==
self_cond=None) and (b) it runs as its own graph. The host divides logits by the per-step
temperature between decoder and soft_proj (driver/sampler stay on the host, Swift sampler).

CANVAS choice drives ENGINE REUSE: at C=256 the heavy decoder chunk graph deadlocks the GPU
command queue when its loaded function handle is REUSED across steps; at C=64 the 4x-lighter
graph reuses the handle like the q=1 LLM-decode regime, so the fast no-reload loop runs the
real denoise schedule (~3 s/step). Quality is safe at C=64 (the short answer fits the canvas;
the port and MLX both converge to the same tokens at C=64).

Kernels / externalization: RMSNorm, RoPE, GatherMM (MoE) -> engine composites. SDPA is
NOT externalized (decomposes): the dual-mode Attention carries BOTH ``sdpa_causal``
(encoder) and ``sdpa_full`` (decoder), so externalizing SDPA by class would mark the
unused one and fail to find it in the graph (the Gemma4 "attribute not in program"
trap); the q=C prefill-shape attention decomposes fine on MPSGraph (and this dodges
the q=1 decode scratch-heap kernel concern). MoE = standard int8 SwitchGLU/GatherMM:
``MetalSwitchGLU``/``gather_qmm`` is DECODE-ONLY (q=1) so it CANNOT serve the q=C
bidirectional canvas (and at q=C the C×top-8 routing touches ~all experts, so the
dense read is not the over-read it is at q=1) -- the gather_qmm sym8-metalize note does
not apply here.

Run (ship bundle, int8, fast-reuse canvas=64, reusing the canvas-independent encoder):
      cd coreai-models  # with the diffusion_gemma model overlay (see this dir's README)
      .venv/bin/python ../coreai-models-community/conversion/export_diffusion_gemma.py \
          --mode int8 --static --trace-seq 26 --decoder-chunk 6 --canvas 64 \
          --model-dir ../_diffgemma_pub_bf16 --only decoder,soft_proj \
          --out ../_diffgemma_coreai_s26_qat_cl64
      # then symlink/copy the existing encoder.aimodel into the new bundle dir.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import torch
from torch import nn

from coreai_models.export.macos import _EXTERNALIZE_SPECS, export_to_coreai
from coreai_models.models.macos.diffusion_gemma import DiffusionGemmaForBlockDiffusion

# Externalize RMSNorm/RoPE/GatherMM AND SDPA -> the engine's GPU attention kernel. The
# decomposed q=256 SDPA was too slow / unstable on the GPU command queue (MTL4CommandQueue
# error at the decoder). The dual-mode Attention carries BOTH sdpa_causal (encoder) and
# sdpa_full (decoder); to dodge the "externalized class instance not in the traced graph"
# trap we NULL the unused one per graph (see _null_unused_sdpa) so only the traced SDPA is marked.
_SPECS = [s for s in _EXTERNALIZE_SPECS
          if s.composite_op_name in ("rms_norm", "rope", "gather_mm", "scaled_dot_product_attention")]


def _null_unused_sdpa(port, keep: str):
    """keep='causal' (encoder) or 'full' (decoder): null the OTHER SDPA on every layer so the
    externalize-by-class pass doesn't mark an SDPA instance absent from this graph's trace.
    Returns a restore list."""
    other = "sdpa_full" if keep == "causal" else "sdpa_causal"
    saved = []
    for layer in port.model.layers:
        attn = layer.self_attn
        saved.append((attn, other, getattr(attn, other)))
        setattr(attn, other, None)
    return saved


def _restore_sdpa(saved) -> None:
    for attn, name, mod in saved:
        setattr(attn, name, mod)


def _model_dir() -> str:
    return glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/*"))[0]


class EncoderExport(nn.Module):
    """prompt -> 60 KV tensors (k0,v0,...) + the final encoder hidden. References only
    embed_tokens / layers / norm (NOT self_conditioning), so every externalized RMSNorm
    instance it holds is actually traced in encoder mode.

    ``norm(x)`` is returned (the decoder ignores it) so the LAST layer's FFN and
    ``model.norm`` stay LIVE: with KV-only outputs, the final layer's FFN feeds nothing
    (KV come from attention; intermediate FFNs feed the next layer's traced KV), so
    torch.export's DCE prunes it and the externalize pass can't find its RMSNorm op."""

    def __init__(self, port: DiffusionGemmaForBlockDiffusion) -> None:
        super().__init__()
        self.embed_tokens = port.model.embed_tokens
        self.layers = port.model.layers
        self.norm = port.model.norm

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor):
        x = self.embed_tokens(input_ids)
        flat: list[torch.Tensor] = []
        for layer in self.layers:
            x, (k, v) = layer.forward_encoder(x, position_ids)
            flat.append(k)
            flat.append(v)
        return tuple(flat) + (self.norm(x),)  # pytree-flattens; order -> output_names


class DecoderExport(nn.Module):
    """(canvas, pos, soft_embeds, 60 enc_kv) -> softcapped logits. Feeds soft_embeds
    straight to decode_with_soft (the softmax@embed lives in soft_proj)."""

    def __init__(self, port: DiffusionGemmaForBlockDiffusion) -> None:
        super().__init__()
        self.m = port.model
        self.lm_head = port.lm_head
        self.softcap = port.config.final_logit_softcapping
        self.n = port.config.num_hidden_layers

    def forward(self, canvas_ids, position_ids, soft_embeds, enc_kv: list[torch.Tensor]):
        enc_kvs = [(enc_kv[2 * i], enc_kv[2 * i + 1]) for i in range(self.n)]
        hidden = self.m.decode_with_soft(canvas_ids, position_ids, enc_kvs, soft_embeds)
        logits = self.lm_head(hidden).float()
        if self.softcap is not None:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        return logits


class SoftProjExport(nn.Module):
    """proc_logits[1,256,V] -> soft_embeds[1,256,hidden] = (softmax_fp32 @ embed)*scale.
    The self-conditioning soft-embedding builder, factored out so the 262144-wide
    softmax@embed (the op torch-MPS-eager corrupts under memory pressure) runs as its
    own MPSGraph graph. Mirrors DiffusionGemma.decode exactly."""

    def __init__(self, port: DiffusionGemmaForBlockDiffusion) -> None:
        super().__init__()
        self.embed = port.model.embed_tokens  # ScaledEmbedding: .weight [V,hidden], .embed_scale

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        soft = torch.matmul(probs.to(self.embed.weight.dtype), self.embed.weight)
        return soft * self.embed.embed_scale


class FirstDecoderChunkExport(nn.Module):
    """First decoder chunk: (canvas, pos, soft_embeds, enc_kv[0:end]) -> hidden[1,256,H].

    Splits the 30-layer decoder into smaller graphs: the full graph overflows the GPU
    command queue at q=256 (MTL4CommandQueueError storm — MPSGraph can't bring up the
    whole 30-layer working set at once), but ~6-layer slices fit (validated). Holds ONLY
    embed_tokens + self_conditioning + its own layer slice so the externalize-by-class
    pass never sees an RMSNorm/SDPA/GatherMM instance that isn't in this graph's trace."""

    def __init__(self, port: DiffusionGemmaForBlockDiffusion, end: int) -> None:
        super().__init__()
        self.embed_tokens = port.model.embed_tokens
        self.self_conditioning = port.model.self_conditioning
        self.layers = nn.ModuleList(list(port.model.layers[:end]))

    def forward(self, canvas_ids, position_ids, soft_embeds, enc_kv: list[torch.Tensor]):
        x = self.self_conditioning(self.embed_tokens(canvas_ids), soft_embeds)
        for i, layer in enumerate(self.layers):
            x = layer.forward_decoder(x, position_ids, enc_kv[2 * i], enc_kv[2 * i + 1])
        return x


class BodyDecoderChunkExport(nn.Module):
    """A middle/last decoder chunk: (hidden[1,256,H], pos, enc_kv[start:end]) -> hidden, or
    softcapped logits on the LAST chunk (which also holds norm + lm_head). Holds ONLY its
    own layer slice (+ norm/lm_head when last), same externalize-safety as the first chunk."""

    def __init__(self, port: DiffusionGemmaForBlockDiffusion, start: int, end: int, is_last: bool) -> None:
        super().__init__()
        self.is_last = is_last
        self.layers = nn.ModuleList(list(port.model.layers[start:end]))
        if is_last:
            self.norm = port.model.norm
            self.lm_head = port.lm_head
            self.softcap = port.config.final_logit_softcapping

    def forward(self, hidden, position_ids, enc_kv: list[torch.Tensor]):
        x = hidden
        for i, layer in enumerate(self.layers):
            x = layer.forward_decoder(x, position_ids, enc_kv[2 * i], enc_kv[2 * i + 1])
        if not self.is_last:
            return x
        logits = self.lm_head(self.norm(x)).float()
        if self.softcap is not None:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        return logits


def linear_quant_config() -> dict:
    """int8 per-block-32 linear; 4D SwitchLinear override for MoE experts; router +
    head kept fp16/absmax (mirrors the qwen3.6 MoE recipe, adapted to diffusion paths)."""
    block_2d = {"dtype": "int8", "qscheme": "symmetric_with_clipping",
                "granularity": {"type": "per_block", "block_size": 32, "axis": 1}}
    block_4d = {"dtype": "int8", "qscheme": "symmetric_with_clipping",
                "granularity": {"type": "per_block", "block_size": [1, 1, 1, 32], "axis": None}}
    head = {"op_state_spec": {"weight": {"dtype": "int8", "qscheme": "symmetric",
                                         "granularity": {"type": "per_block", "block_size": 32, "axis": 1}}},
            "op_input_spec": None, "op_output_spec": None}
    return {
        "execution_mode": "eager",
        "global_config": {"op_state_spec": {"weight": block_2d},
                          "op_input_spec": None, "op_output_spec": None},
        "module_type_configs": {
            "coreai_models.primitives.macos.sdpa.SDPA": None,
            "coreai_models.primitives.macos.rope.RoPE": None,
            "coreai_models.primitives.macos.rms_norm.RMSNorm": None,
            "torch.nn.modules.sparse.Embedding": None,   # embed/soft_proj table stays fp16
            "coreai_models.primitives.macos.switch.SwitchLinear": {
                "module_state_spec": {"weight": block_4d},
                "op_input_spec": None, "op_output_spec": None},
        },
        # router proj stays fp16 (quantizing it flips discrete expert selection);
        # lm_head = absmax int8 (big-vocab-head rule).
        "module_name_configs": {r".*router\.proj$": None, r".*lm_head$": head},
    }


def _save(prog, out_dir: Path, name: str) -> None:
    import coreai.runtime as rt
    prog.optimize()
    f = out_dir / f"{name}.aimodel"
    print(f"  saving {f} ...", flush=True)
    prog.save_asset(f, rt.AIModelAssetMetadata())


def _kv_names(n: int) -> tuple[str, ...]:
    names: list[str] = []
    for i in range(n):
        names += [f"enc_k_{i}", f"enc_v_{i}"]
    return tuple(names)


def _kv_names_range(start: int, end: int) -> tuple[str, ...]:
    """enc_k/enc_v names for the absolute layer range [start, end) — a decoder chunk
    receives only the encoder KV for its own layers, named by ABSOLUTE layer index so the
    Swift gate slices the encoder's enc_k_0..enc_k_{N-1} outputs straight into each chunk."""
    names: list[str] = []
    for i in range(start, end):
        names += [f"enc_k_{i}", f"enc_v_{i}"]
    return tuple(names)


def _ref_kv_range(cfg, start: int, end: int, seq: int, dtype) -> list[torch.Tensor]:
    """Reference KV tensors for layers [start, end) with the correct per-layer shape
    (sliding = n_kv/head_dim differ from full layers)."""
    ref: list[torch.Tensor] = []
    for li in range(start, end):
        nkv, hd = cfg.n_kv_of(li), cfg.head_dim_of(li)
        ref += [torch.zeros(1, nkv, seq, hd, dtype=dtype),
                torch.zeros(1, nkv, seq, hd, dtype=dtype)]
    return ref


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="int8", choices=["fp16", "int8"])
    ap.add_argument("--model-dir", default=None,
                    help="checkpoint dir (default: google masters). Use _diffgemma_pub_bf16 "
                         "(dequantized published model = WORKING QAT-expert weights).")
    ap.add_argument("--num-layers", type=int, default=None, help="truncate backbone (structural smoke)")
    ap.add_argument("--out", default="_diffgemma_coreai")
    ap.add_argument("--trace-seq", type=int, default=32, help="encoder trace prompt length")
    ap.add_argument("--canvas", type=int, default=256,
                    help="bidirectional denoise canvas length (baked into the static decoder/soft_proj "
                         "traces). 256 is the q=256 default; shrink to 64/128 for a lighter decoder graph "
                         "(near the q=1-decode regime that reuses the fn handle without the GPU deadlock).")
    ap.add_argument("--max-prompt", type=int, default=4096, help="max dynamic prompt length")
    ap.add_argument("--static", action="store_true",
                    help="fixed prompt length = --trace-seq (no dynamic shapes; dodges the "
                         "MPSGraph shape-function-bytecode SIGSEGV on dynamic graphs)")
    ap.add_argument("--no-quant-mmap", action="store_true")
    ap.add_argument("--only", default="all",
                    help="comma-list of components to export (all|encoder|decoder|soft_proj). The encoder "
                         "is canvas-independent, so `--only decoder,soft_proj` re-exports just the "
                         "canvas-dependent graphs and the existing encoder.aimodel can be reused/symlinked.")
    ap.add_argument("--decoder-chunk", type=int, default=0,
                    help="split the decoder into chunks of <= N layers (0 = one monolithic "
                         "decoder graph). The full 30-layer q=256 decoder overflows the GPU "
                         "command queue (MTL4CommandQueueError); ~6-layer chunks fit. Chunks are "
                         "chained host-side (hidden handoff); writes decoder_chunks.json. Static only.")
    args = ap.parse_args()
    dtype = torch.float16
    TS, CANVAS = args.trace_seq, args.canvas

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading DiffusionGemma ({args.num_layers or 'full 30'} layers, fp16) ...", flush=True)
    port = DiffusionGemmaForBlockDiffusion.from_local(
        args.model_dir or _model_dir(), num_layers=args.num_layers, dtype=dtype).eval()
    cfg = port.config
    N, V, H = cfg.num_hidden_layers, cfg.vocab_size, cfg.hidden_size
    print(f"  E={cfg.num_experts}/top{cfg.top_k_experts}, hidden={H}, vocab={V}, layers={N}, "
          f"trace_seq={TS}, canvas={CANVAS}", flush=True)

    quant_mmap = None
    if args.mode == "int8":
        from coreai_models.export.compression import quantize_pytorch_model
        if not args.no_quant_mmap:
            quant_mmap = out_dir / "_quant_mmap"
            if quant_mmap.exists():
                shutil.rmtree(quant_mmap)
            quant_mmap.mkdir()
        # Weight-only int8: trace the full port forward (encode+decode+head, self_cond=None
        # branch) once with small inputs to FIND every Linear / SwitchLinear; PT2E quantizes
        # the shared modules in place, then the three wrappers export the quantized weights.
        q_inp = (torch.randint(1, V, (1, 4), dtype=torch.int32),
                 torch.randint(1, V, (1, 8), dtype=torch.int32))
        print("quantizing (int8 per-block-32; experts 4D; router/embed fp16; head absmax) ...", flush=True)
        port = quantize_pytorch_model(port, q_inp, None, linear_quant_config(),
                                      mmap_dir=str(quant_mmap) if quant_mmap else None)

    _only = ({"encoder", "decoder", "soft_proj"} if args.only == "all"
             else {s.strip() for s in args.only.split(",")})
    do = lambda which: which in _only  # noqa: E731

    # ---- encoder ----
    if do("encoder"):
        print(f"exporting encoder ({'static seq=' + str(TS) if args.static else 'dynamic'}) ...", flush=True)
        saved = _null_unused_sdpa(port, "causal")  # encoder uses sdpa_causal; null sdpa_full
        enc = EncoderExport(port).eval()
        enc_dyn = None if args.static else {
            "input_ids": {1: (seq := torch.export.Dim("seq", min=2, max=args.max_prompt))},
            "position_ids": {1: seq}}
        prog = export_to_coreai(
            enc,
            {"input_ids": torch.randint(1, V, (1, TS), dtype=torch.int32),
             "position_ids": torch.arange(TS, dtype=torch.int32).unsqueeze(0)},
            dynamic_shapes=enc_dyn,
            input_names=("input_ids", "position_ids"),
            output_names=_kv_names(N) + ("enc_hidden",),
            externalize_modules=_SPECS)
        _save(prog, out_dir, "encoder")
        _restore_sdpa(saved)
        del enc, prog

    # ---- decoder ----
    if do("decoder"):
        saved = _null_unused_sdpa(port, "full")  # decoder uses sdpa_full; null sdpa_causal
        dec_pos = torch.arange(TS, TS + CANVAS, dtype=torch.int32).unsqueeze(0)
        if args.decoder_chunk and args.decoder_chunk > 0:
            C = args.decoder_chunk
            ranges = [(s, min(s + C, N)) for s in range(0, N, C)]
            print(f"exporting decoder in {len(ranges)} chunks of <= {C} layers: {ranges}", flush=True)
            for j, (s, e) in enumerate(ranges):
                is_first, is_last = (j == 0), (j == len(ranges) - 1)
                print(f"  chunk {j}: layers [{s},{e}) first={is_first} last={is_last}", flush=True)
                ref_kv = _ref_kv_range(cfg, s, e, TS, dtype)
                if is_first:
                    mod = FirstDecoderChunkExport(port, e).eval()
                    ref = {"canvas_ids": torch.randint(1, V, (1, CANVAS), dtype=torch.int32),
                           "position_ids": dec_pos,
                           "soft_embeds": torch.zeros(1, CANVAS, H, dtype=dtype),
                           "enc_kv": ref_kv}
                    in_names = ("canvas_ids", "position_ids", "soft_embeds") + _kv_names_range(s, e)
                else:
                    mod = BodyDecoderChunkExport(port, s, e, is_last).eval()
                    ref = {"hidden": torch.zeros(1, CANVAS, H, dtype=dtype),
                           "position_ids": dec_pos, "enc_kv": ref_kv}
                    in_names = ("hidden", "position_ids") + _kv_names_range(s, e)
                prog = export_to_coreai(
                    mod, ref, dynamic_shapes=None, input_names=in_names,
                    output_names=("logits",) if is_last else ("hidden",),
                    externalize_modules=_SPECS)
                _save(prog, out_dir, f"decoder_chunk{j}")
                del mod, prog
            (out_dir / "decoder_chunks.json").write_text(json.dumps(
                {"chunk_size": C, "n_layers": N, "ranges": [list(r) for r in ranges]}))
            print(f"  wrote {out_dir / 'decoder_chunks.json'} ({len(ranges)} chunks)", flush=True)
        else:
            print("exporting decoder (monolithic) ...", flush=True)
            dec = DecoderExport(port).eval()
            ref_kv = _ref_kv_range(cfg, 0, N, TS, dtype)
            dec_dyn = None if args.static else {
                "canvas_ids": None, "position_ids": None, "soft_embeds": None,
                "enc_kv": [{2: torch.export.Dim("s_enc", min=2, max=args.max_prompt)} for _ in range(2 * N)]}
            prog = export_to_coreai(
                dec,
                {"canvas_ids": torch.randint(1, V, (1, CANVAS), dtype=torch.int32),
                 "position_ids": dec_pos,
                 "soft_embeds": torch.zeros(1, CANVAS, H, dtype=dtype),
                 "enc_kv": ref_kv},
                dynamic_shapes=dec_dyn,
                input_names=("canvas_ids", "position_ids", "soft_embeds") + _kv_names(N),
                output_names=("logits",),
                externalize_modules=_SPECS)
            _save(prog, out_dir, "decoder")
            del dec, prog
        _restore_sdpa(saved)

    # ---- soft_proj ----
    if do("soft_proj"):
        print("exporting soft_proj ...", flush=True)
        sp = SoftProjExport(port).eval()
        prog = export_to_coreai(
            sp,
            {"logits": torch.zeros(1, CANVAS, V, dtype=torch.float32)},
            dynamic_shapes=None,
            input_names=("logits",),
            output_names=("soft_embeds",),
            externalize_modules=_SPECS)  # no RMSNorm/RoPE/GatherMM instances -> no-op
        _save(prog, out_dir, "soft_proj")
        del sp, prog

    if quant_mmap is not None:
        shutil.rmtree(quant_mmap, ignore_errors=True)
    print(f"bundles ready in {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
