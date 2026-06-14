# MiniCPM-V-4.6 (1.3B, vision-language) — Core AI

**The strongest sub-2B open VLM, on-device.** A Core AI port of
[`openbmb/MiniCPM-V-4.6`](https://huggingface.co/openbmb/MiniCPM-V-4.6): image + text → text,
end-to-end on the GPU via the [pipelined-engine fast path](../knowledge/pipelined-engine.md) —
no engine changes beyond the published static-inputs patch. Pick a photo, ask, stream the answer.

Architecture (`model_type: minicpmv4_6`): a **SigLIP So400m vision tower** (980px / patch 14 /
27 layers, gelu-tanh) with a **window-attention insert-merger @ layer 6** (2×2) + a **downsample-MLP
merger** (2×2) → ÷16 = **64 visual tokens** per 448px slice, spliced (`masked_scatter`) into the
text embeddings at `<image>` positions; and a **Qwen3.5-hybrid text backbone** (`qwen3_5_text`:
0.8B, 24 L, GatedDeltaNet linear-attn ×3 : full-attn ×1, head_dim 256, vocab 248094, **tied head**,
plain 1D positions). The backbone reuses the zoo's **existing `qwen3_5.py` overlay verbatim** (the
qwen3.6 hybrid); only the SigLIP tower + mergers were authored fresh.

**⬇️ Converted `.aimodel` bundles:
[mlboydaisuke/MiniCPM-V-4.6-CoreAI](https://huggingface.co/mlboydaisuke/MiniCPM-V-4.6-CoreAI)** —
`gpu-pipelined/minicpmv46_vlm_decode_int8lin/` (VLM text decoder, int8, ship config) +
`gpu-pipelined/minicpmv46_vision/` (fixed-grid SigLIP vision encoder, fp16). Apache-2.0.

## How a VLM rides a text-only engine

The pipelined engine knows nothing about images. The whole multimodal state rides the
**static-input hook** (`apps/coreai-pipelined-static-inputs.patch`) + an id-space trick — the graph
stays `ids + positions → logits`:

- The host runs the vision encoder ONCE per image (resize 448, normalize `x/127.5−1`, CHW
  `[1,3,448,448]`) and writes `image_embeds [64,1024]` into one owned MTLBuffer the engine binds
  on every step (~0.13 MB).
- The prompt's `<|image_pad|>` ids (id 248056) are rewritten to **extension ids** `V + slot`
  (slot 0..63). In-graph: `embed = ids < V ? table[ids] : image_embeds[ids − V]`.
- **Positions are plain 1D** — no M-RoPE, no rope-shift. So this is the Qwen3-VL static-buffer
  recipe **minus deepstack and minus the M-RoPE machinery** (MiniCPM-V-4.6 dropped the perceiver
  resampler too; the connector is plain 2×2 merges + MLP).
- With zero embeds and no `V+slot` ids the decoder **is** a plain qwen3.5-hybrid text LLM — same
  bundle, no image required.

The backbone is the qwen3.5 **hybrid**, so the engine carries the SSM **conv + recurrent** states
alongside KV (the `expectFrequentReshapes` / extra-states path; cf. granite / lfm2). The vision
tower is a separate plain `.aimodel` with ALL positional work (bucketized pos-embed, window index
+ inverse) baked as constants for the fixed grid: `pixel_values [1,3,448,448] → image_features [64,1024]`.

## Measured (macOS 27 / iOS 27 beta, release, p=128 g=256, `COREAI_CHUNK_THRESHOLD=1`)

| config | bundle | platform | prefill | decode | numerics |
|---|---:|---|---:|---:|---|
| VLM int8 (image) | ~1.0 GB | iPhone 17 Pro | 52.3 | **51.5** | image→answer; engine reply == HF description (one int8 near-tie, reconverges) |
| text core int8 | ~1.0 GB | iPhone 17 Pro | 53.3 | **53.4** | **nat 24/24 + oracle 24/24** (engine ≡ python ≡ HF) |
| text core int8 | ~1.0 GB | M4 Max | 225.1 | **224.3** | `llm-benchmark` p128 g256 n3 (qwen3.5-0.8B class; VLM bundle decodes ~the same — llm-benchmark can't feed the image buffer so the text core is the Mac proxy) |
| vision encoder fp16 | ~1.0 GB | Mac | — | per-image (one-shot) | per-token cos **1.000000** vs fp32-HF |

- **Gated end-to-end**: fp32-torch ladder EXACT (vision image_features cos 1.000000; full overlay
  logits cos 1.00004, top-5 identical to HF) → fp16/int8 `.aimodel` (Mac GPU) → **engine ≡ python**
  ("The capital of France is" → " Paris", 5/5) → **full VLM on engine** (vision.aimodel + decoder →
  correct image caption) → **iPhone 17 Pro** (text 24/24; image → accurate grounded description).
- Real-photo example (iPhone, kakigōri): *"a bowl of shaved ice ... chunks of mango ... a dark blue
  saucer ... a menu or a book, hinting at a café ... a wooden table"* — accurate, fully on-device.
- ~1.5 GB resident (vision fp16 + int8 decoder) = iPhone increased-memory jetsam-safe; cold spec ~3–5 s.
- The image numerics "fork" vs the fp32 rollout is a single near-tie (`" This"` vs `"\nThe"`) that
  reconverges — the fp16/int8-noise class (cf. granite's nat rollout), not a path error.
- `head_dim 256` dodges the Gemma4-12B decode scratch-heap; tied head (no untied-head step). The
  qwen3.5-hybrid GatedDeltaNet runs via ANE-compile-fail → GPU-fallback (benign; cf. granite / qwen3.6).

## Convert / verify

```
# vision encoder (fixed 448 grid; bakes window-index/argsort/bucketize as constants)
python conversion/export_minicpmv46_vision.py
# VLM decoder (input_ids -> logits + image_embeds static buffer; qwen3_5 hybrid core)
python conversion/export_minicpmv46_vlm_pipelined.py int8lin
# standalone text decode core (input_ids -> logits, in-graph embed + tied head)
python conversion/export_minicpmv46_decode_pipelined.py int8lin
# head-split core variant (inputs_embeds -> hidden, embed/head on the front-end)
python conversion/export_minicpmv46_core_decode_pipelined.py int8lin
```

The backbone reuses `coreai-models/.../macos/qwen3_5.py` verbatim; the vision tower re-authoring is
faithful to transformers `minicpmv4_6` (oracle + ladders in `_smoke/`). The fp32 oracle uses
synthetic deterministic pixels so model parity is decoupled from preprocessing (real images use the
repo's resize-448 + mean/std 0.5 normalization, mirrored on the Swift host).

## Try it

`apps/CoreAIChat` has a **MiniCPM-V 4.6 mode with a photo picker** (the 8th model in the picker), and
there's a standalone `MiniCPMVLM` app — pick an image, ask about it, stream the answer. The vision
tower runs once per attached image; each turn re-prefills (S=1).
