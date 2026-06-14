"""Export the MiniCPM-V-4.6 VISION encoder (.aimodel) + self-gate vs the fp32 oracle.

Fixed single-slice grid (32×32 patches @448px → 64 visual tokens). The grid is baked as a
python constant so the window-index / argsort / bucketized pos-ids fold to constants and lower
cleanly to the Core AI GPU delegate. fp16 (vision ship dtype). Reuses the parity-validated
`_smoke/minicpmv46_vision.py` math.

Run (GPU; _GPU_LOCK held):
    coreai-models/.venv/bin/python ../coreai-models-community/conversion/export_minicpmv46_vision.py
"""
from __future__ import annotations

import asyncio
import glob
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from torch import nn

sys.path.insert(0, "/Users/majimadaisuke/code/coreai/_smoke")
from minicpmv46_vision import MiniCPMV46Vision  # noqa: E402

import coreai.runtime as rt  # noqa: E402
from coreai_models.export.macos import export_to_coreai  # noqa: E402

DTYPE = torch.float16
GRID = 32
SNAP = glob.glob("/Users/majimadaisuke/.cache/huggingface/hub/"
                 "models--openbmb--MiniCPM-V-4.6/snapshots/*")[0]
REF = "/Users/majimadaisuke/code/coreai/_smoke/minicpmv46_ref.npz"
OUT = Path("/Users/majimadaisuke/code/coreai/coreai-models/exports/minicpmv46_vision")


class VisionExport(nn.Module):
    """pixel_values [1,3,448,448] -> image_features [64,1024], grid baked to 32."""

    def __init__(self, core: MiniCPMV46Vision):
        super().__init__()
        self.core = core

    def forward(self, pixel_values):
        x, gh2, gw2 = self.core.vision_tower(pixel_values, GRID, GRID)  # post-insert grid 16
        return self.core.merger(x, gh2, gw2)                             # [64,1024]


def load_vision(model: MiniCPMV46Vision) -> None:
    sd = {}
    with safe_open(glob.glob(SNAP + "/model.safetensors")[0], framework="pt", device="cpu") as f:
        for k in f.keys():  # noqa: SIM118
            if k.startswith("model.vision_tower.") or k.startswith("model.merger."):
                sd[k[len("model."):]] = f.get_tensor(k).to(DTYPE)
    model.load_state_dict(sd, strict=False, assign=True)


async def gate(oracle) -> bool:
    aimodel = OUT / f"{OUT.name}.aimodel"
    print(f"[gate] loading {aimodel.name} on GPU ...", flush=True)
    m = await rt.AIModel.load(
        str(aimodel),
        rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu()))
    fn = m.load_function("main")
    z = np.load(REF)
    pv = z["pixel_values"].astype(np.float16)
    res = await asyncio.wait_for(fn(inputs={"pixel_values": rt.NDArray(np.ascontiguousarray(pv))}),
                                 timeout=300)
    feats = torch.from_numpy(res["image_features"].numpy().astype(np.float32))
    o = torch.from_numpy(oracle)
    pertok = torch.nn.functional.cosine_similarity(feats, o, dim=-1)
    print(f"[gate] shape {tuple(feats.shape)} per-token cos mean {pertok.mean():.5f} "
          f"min {pertok.min():.5f} maxabs {(feats - o).abs().max():.4f}")
    return pertok.mean().item() > 0.99 and pertok.min().item() > 0.98


def main() -> None:
    core = MiniCPMV46Vision().to(DTYPE).eval()
    load_vision(core)
    core.bake_constants(GRID)  # constant pos-ids / window-index / inverse → no bucketize/argsort in graph
    model = VisionExport(core).eval()

    z = np.load(REF)
    pv = torch.from_numpy(z["pixel_values"]).to(DTYPE)

    # eager sanity (export-wrapper math == oracle)
    with torch.no_grad():
        eager = model(pv).float()
    o = z["image_features"]
    c = torch.nn.functional.cosine_similarity(eager, torch.from_numpy(o), dim=-1).mean().item()
    print(f"[eager] export-wrapper cos vs oracle {c:.5f} (fp16)")

    print("[export] vision -> Core AI dialect ...", flush=True)
    prog = export_to_coreai(
        model, {"pixel_values": pv}, dynamic_shapes=None,
        input_names=("pixel_values",), output_names=("image_features",),
        state_names=None, externalize_modules=[])
    prog.optimize()

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    aimodel = OUT / f"{OUT.name}.aimodel"
    print(f"[save] {aimodel}", flush=True)
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())
    meta = {
        "metadata_version": "0.2", "kind": "vision-encoder", "name": OUT.name,
        "assets": {"main": f"{OUT.name}.aimodel"},
        "vision": {"input": "pixel_values[1,3,448,448]", "output": "image_features[64,1024]",
                   "grid": GRID, "dtype": "fp16"},
        "source": {"model_definition": "torch", "hf_model_id": "openbmb/MiniCPM-V-4.6"},
        "compilation": {"date": datetime.now(timezone.utc).isoformat(), "targets": []},
    }
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=2))

    ok = asyncio.run(gate(o))
    print(f"\n{'✅ PASS' if ok else '❌ FAIL'} — vision .aimodel {'matches' if ok else 'DIVERGES from'} oracle")


if __name__ == "__main__":
    main()
