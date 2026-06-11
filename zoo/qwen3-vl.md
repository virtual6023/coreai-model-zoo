# Qwen3-VL 2B (vision-language) — Core AI

**The zoo's first VLM on Core AI**: image + text → text, end-to-end on the
GPU via the [pipelined-engine fast path](../knowledge/pipelined-engine.md) —
no engine changes beyond the published static-inputs patch.

Architecture (`Qwen/Qwen3-VL-2B-Instruct`): a **pure-attention Qwen3 text
decoder** (28 layers, hidden 2048, 16 q / 8 kv heads, head_dim 128, vocab
151 936 tied head, SwiGLU 6144, rope θ 5e6) + a **24-layer ViT vision tower**
(hidden 1024, patch 16, temporal-2 duplicated frame, 2×2 spatial merge →
out 2048) with **DeepStack**: merger outputs at vision layers 5/11/17 are
added to the decoder hidden state at image positions after decoder layers
0/1/2. Positions are **interleaved M-RoPE** — 3D (t,h,w), section [24,20,20].

**⬇️ Converted `.aimodel` bundles:
[mlboydaisuke/Qwen3-VL-2B-CoreAI](https://huggingface.co/mlboydaisuke/Qwen3-VL-2B-CoreAI)** —
`gpu-pipelined/qwen3_vl_2b_instruct_decode_int8hu_s1/` (text decoder
LanguageBundle, ship config) + `gpu-pipelined/qwen3_vl_2b_instruct_vision/`
(fixed-grid vision encoder, fp16). Apache-2.0.

<p align="center"><img src="https://huggingface.co/mlboydaisuke/Qwen3-VL-2B-CoreAI/resolve/main/demo.gif" width="300" alt="CoreAIChat Qwen3-VL demo on iPhone 17 Pro"></p>

## How a VLM rides a text-only engine

The pipelined engine knows nothing about images. The whole multimodal state
rides the **static-input hook** (`apps/coreai-pipelined-static-inputs.patch`)
plus an id-space trick — the graph stays `ids + positions → logits`:

- The host runs the vision encoder ONCE per image (448×448 → 196 merged
  tokens) and writes `image_embeds [196,2048]` + `deepstack_embeds [588,2048]`
  into two owned MTLBuffers the engine binds on every encode (~3.2 MB).
- The prompt's `<|image_pad|>` ids are rewritten to **extension ids**
  `V + slot` (slot 0..195). In-graph: `embed = ids < V ? table[ids] :
  image_embeds[ids - V]`; the three DeepStack adds gather
  `deepstack_embeds[k·196 + slot]` at image positions.
- **M-RoPE is derived in-graph from (ids, position) alone.** Image tokens
  self-locate: `slot = ids - V`, image start `s0 = pos - slot`, then
  `t = s0, h = s0 + slot//14, w = s0 + slot%14`. Text tokens use `pos -
  shift` where `shift = amount·(pos ≥ start)` — two `[1] i32` static inputs
  the host sets per conversation (an image consumes only max(H,W)=14 rope
  positions, so post-image text shifts by 196−14=182). The interleaved
  layout is three constant 0/1 masks over head_dim mixing three standard
  RoPE rotations (frequency j: j%3==1 → h, j%3==2 → w, else t, j<60).
- With zero embeds and `start = 1<<30` the decoder **is** a plain Qwen3 text
  LLM — same bundle, no images required.

KV cache stays the engine's native pair (pure attention, no extra states).
The vision tower is a separate plain `.aimodel` with ALL positional work
(bilinear pos-embed interpolation, 2D rotary) baked as constants for the
fixed grid: `patches [784,1536] → (image_embeds, deepstack_embeds)`.

## Measured (macOS 27 beta / iOS 27 beta, release, p=128 g=256, `COREAI_CHUNK_THRESHOLD=1`)

| config | bundle | platform | prefill tok/s | decode tok/s | numerics |
|---|---:|---|---:|---:|---|
| **int8hu (ship): int8lin body + untied absmax int8 head** | 2.3 GB | M4 Max | **191.0** | **187.6** | A 211-tok multimodal sweep 4/4 + B decode 16/16 + D HF-seeded (cos 0.99995) vs fp32-HF; engine ≡ python 24/24 |
| int8lin (per-block-32 body, tied fp16 head) | 2.0 GB | M4 Max | 165.7 | 162.9 | same gate set PASS (D cos 0.99996) |
| **int8hu, iPhone 17 Pro** (settled, screen unlocked) | 2.3 GB | iPhone | **33.5–34.6** | **33.3 typical (32.9–33.8; one thermal-dip trial 27.5)** | **nat 24/24 + multimodal oracle 24/24 × 8 runs — token-identical to M4 Max. ~92% of the naive BW ceiling (~36 = 60 GB/s ÷ 1.66 GB/tok)** |
| vision encoder fp16 | 0.77 GB | both | — | 59–79 ms/image (Mac) | cos 0.9997 embeds / 0.99998 deepstack vs fp32-HF |

- The big-vocab head rule ([compression-reference](../knowledge/compression-reference.md))
  holds at vocab 151 936: absmax `symmetric` int8 head gates clean, +15% on Mac
  over int8lin.
- Cold GPU specialization on device 12.3 s (no AOT needed), warm load 0.6–5 s;
  needs the increased-memory entitlement (2.3 GB class). TTFT for a 211-token
  image prompt ≈ 6 s on iPhone (S=1 prefill), ~1.1 s on M4 Max.
- A LOCKED iPhone screen collapses GPU priority for devicectl-launched runs
  (decode 0.5–13 tok/s, oscillating) — bench with the screen unlocked and awake;
  numerics still pass 24/24 even throttled.
- End-to-end sanity (full `.aimodel` pipeline, Mac): the oracle scene caption
  forks from the fp32-HF rollout only at token 41/51 (fp16-greedy class); a
  fresh snowman scene is described correctly ("three stacked, white, circular
  shapes").
- The decoder also ships a **dynamic-query twin** (no `_s1` suffix) that would
  enable true chunked prefill; it exports and gates in torch but crashes the
  engine at generate on this beta (`NSArrayM nil insert` — first dynamic-ids
  graph on this path). S=1 prefill ≈ decode tok/s, so TTFT for a 211-token
  image prompt is ~1.1 s on M4 Max regardless.

## Convert / verify

```
# decoder (fp16 / int8lin / int8hu) + _s1 gate twin + vision encoder
python conversion/export_qwen3_vl_pipelined.py int8hu
# gates (vision + decoder A/B/D vs the fp32 oracle)
python _smoke/test_qwen3vl_aimodel_gate.py qwen3_vl_2b_instruct_decode_int8hu_s1 \
    --vision qwen3_vl_2b_instruct_vision
```

The torch re-authoring lives in
`coreai-models/python/src/coreai_models/models/macos/qwen3_vl.py` (model
overlay; see conversion/README.md). The position formula is verified EXACT
against HF `get_rope_index` before any conversion; the full gate ladder is
fp32 torch → fp16 `.aimodel` (Mac GPU) → engine → device.

## Try it

`apps/CoreAIChat` has a **Qwen3-VL mode with a photo picker**: pick an image,
ask about it. The vision tower runs once per attached image (~60-80 ms-class);
each turn re-prefills (S=1), so a 211-token image prompt streams its first
token in ~1 s on an M4 Max-class GPU.
