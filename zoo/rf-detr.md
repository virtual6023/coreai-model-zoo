# RF-DETR (nano / small / medium / large) — Core AI

**The zoo's first object detector**, and the answer to
[apple/coreai-models#14](https://github.com/apple/coreai-models/issues/14):
Roboflow's [RF-DETR](https://github.com/roboflow/rf-detr) — the real-time
detection transformer that broke 60 mAP on COCO — running as a single static
`.aimodel` on every Apple compute unit. **DETR family = no NMS**: host
post-processing is one sigmoid and a top-k.

Architecture (rfdetr 1.7.1): windowed-attention **DINOv2 ViT-S backbone** →
single-scale P4 projector → **two-stage proposal head** (top-300 of HW
anchors) → **deformable-attention decoder** (multi-scale deformable sampling,
2 points/head, iterative box refinement, `bbox_reparam`). All four sizes share
the architecture — only resolution, decoder depth and window count change:

| variant | input | params | dec layers | `.aimodel` |
|---|---|---|---|---|
| nano | 384² | 30.5M | 2 | 108 MB |
| small | 512² | 32.1M | 3 | 115 MB |
| medium | 576² | 33.7M | 4 | 121 MB |
| large | 704² | 33.9M | 4 | 122 MB |

## Graph contract

```
input  "image"  [1, 3, R, R]  float32, RGB in [0, 1]   (ImageNet mean/std folded in-graph)
output "dets"   [1, 300, 4]   cxcywh, normalized to [0, 1]
output "labels" [1, 300, 91]  raw logits; column index = ORIGINAL COCO id (0 unused, 1=person … 17=cat … 90)
```

Host decode: `score = sigmoid(labels)`, take top-k over the flattened
query×class plane, gather boxes — done. No anchors, no NMS, no letterboxing
(plain square resize, matching the upstream recipe).

## Measured (macOS 27 / iOS 27 beta, GPU, median)

| variant | M4 Max | M4 Max CPU | iPhone 17 Pro | iPhone end-to-end FPS |
|---|---|---|---|---|
| nano | **8.6 ms** (~116 FPS) | 27.1 ms | **~25 ms** | **33–39** |
| small | **12.0 ms** (~83 FPS) | 44.9 ms | — | — |
| medium | **14.8 ms** (~68 FPS) | 56.5 ms | **56–63 ms** | **15–17** |
| large | **19.1 ms** (~52 FPS) | 86.1 ms | — | — |

iPhone numbers are live-camera measurements from the DetectCamera example app
(Release, zero-copy capture pipeline: AVCaptureVideoPreviewLayer display +
hardware-scaled 32BGRA data buffers + vImage preprocessing overlapped with GPU
inference; 60 fps capture). Peak measured 39.6 FPS ≈ the nano model ceiling.
Sustained max-load throughput drops on a hot chassis (thermal); one-time
on-device specialization ~5 s on first load. The on-device gate reproduces the Mac fp32 oracle detections with
boxes within 1e-3 normalized and scores within 0.01 (medium) / 0.04 (nano) —
the documented GPU h16c noise class. `neural_engine` preference loads and
gates clean but measures ≈ GPU — the gather-heavy deformable decoder keeps the
graph on the GPU delegate.

**Ship dtype is fp32.** It gates bit-clean everywhere, and fp16 only bought 7%
latency (14.8 → 13.7 ms medium) while introducing near-tie detection noise
(score swings up to ~0.04 on duplicate predictions). Detection ≠ LLM: there is
no memory-bandwidth-bound decode loop to feed, so weight bytes barely matter.

Numerics gate: per real COCO image, every confident (>0.3) torch-fp32
detection must have a same-class IoU≥0.75 partner within 2e-3 score in the
`.aimodel` output (set-based matching — DETR emits near-duplicates whose ranks
swap under noise, so positional top-k compare overflags). **All four variants:
cpu AND gpu, 4 images each, worst-IoU ≥ 0.999, zero misses.**

<p align="center"><img src="https://huggingface.co/mlboydaisuke/RF-DETR-CoreAI/resolve/main/demo_coco_cats.jpg" width="420" alt="RF-DETR medium on Core AI — cats 0.94, remotes 0.93/0.86, couch 0.53"></p>

## ⬇️ Bundles

**[mlboydaisuke/RF-DETR-CoreAI](https://huggingface.co/mlboydaisuke/RF-DETR-CoreAI)** —
`rfdetr-{nano,small,medium,large}_float32.aimodel`. Apache-2.0 (upstream code
and COCO-pretrained weights are Apache-2.0).

Convert yourself: [`conversion/export_rf_detr.py`](../conversion/export_rf_detr.py)
(`pip install rfdetr==1.7.1`, torch ≤ 2.11 to match coreai-torch 0.4.0).

Live-camera reference app: **DetectCamera** in
[CoreAIKit](https://github.com/john-rocky/coreai-kit) (`Examples/DetectCamera` —
`CameraFeed` + `ObjectDetector`, box overlay, in-app Hub download; the whole ML
surface is two calls).

## The port in one lesson: deformable attention vs the Core AI stack

`grid_sample` has no Core AI lowering, but rfdetr 1.7.1 already ships a
gather-based bilinear fallback (their MPS path) — we force it on at export.
What actually fought back was four platform bugs, each pinned with a minimal
repro (details + repros in
[knowledge/conversion-guide.md](../knowledge/conversion-guide.md#detection-transformers)):

1. **`aten.arange` with float args aborts the converter** (`bad_optional_access`).
   `gen_sineembed_for_position(…, d_model / 2)` passes dim as a *float*. Repro:
   `torch.arange(8.0)` in any graph. Fix: precompute `dim_t` as a Python-list
   constant (also deletes the runtime arange/pow/floordiv chain).
2. **int64-comparison bool chains make the runtime clobber unrelated live
   buffers.** The bilinear's `((ix0 >= 0) & (ix0 < W)).to(float)` corrupted the
   decoder LayerNorm output *two ops upstream* — even when that tensor is a
   graph output; `clone()`/`contiguous()` guards don't help. Fix: compute
   in-bounds masks in pure float arithmetic
   (`1 - (x - x.clamp(lo, hi)).abs().clamp(max=1)` is an exact 0/1 mask on
   integer-valued floats).
3. **`aten.floor`/`trunc`/`ceil` lower to IDENTITY on the GPU delegate**
   (CPU is fine; `round` rounds ties away instead of to-even). And the two
   obvious workarounds also fail: `div(x, 1, floor)` folds to identity, and
   `float→long→float` roundtrips get cast-cancelled by the converter (so does
   on CPU!). The one floor that survives every unit:
   **`torch.div(x * 2.0, 2.0, rounding_mode="floor")`** (×2/2 is a power-of-two
   scale — exact in fp).
4. **`torch._assert` on a data-dependent equality** trips
   GuardOnDataDependentSymNode under torch-2.11 non-strict export — no-op it
   for static-shape export.

Everything else — two-stage top-k + gather, windowed DINOv2, 16 SDPA blocks,
the iterative refinement loop — converted and gated clean on the first try
once those four were out of the way.
