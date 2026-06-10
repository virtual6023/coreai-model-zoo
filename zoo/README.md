# Zoo — Core AI converted models

Model cards for models converted to Core AI `.aimodel`. **Ready-to-run bundles are on Hugging
Face** (one best verified configuration per platform × compute unit); each card also links the
source checkpoint and the `conversion/` script, plus parity numbers, sizes, and measured
throughput.

| Card | Family | Download | Status |
|---|---|---|---|
| [`qwen3.5.md`](qwen3.5.md) | Qwen3.5 (hybrid linear+full attn) | [🤗 qwen3.5-0.8B-CoreAI](https://huggingface.co/mlboydaisuke/qwen3.5-0.8B-CoreAI) | 0.8B + 2B, top-1 exact vs HF |
| [`gemma4-e2b.md`](gemma4-e2b.md) | Gemma 4 (multimodal; text decoder) | [🤗 gemma-4-E2B-CoreAI](https://huggingface.co/mlboydaisuke/gemma-4-E2B-CoreAI) | 8/8 exact vs HF |

## The matrix (every meaningful platform × compute-unit cell, greedy, top-1 vs HF)

<!-- Mac column RELEASE-VERIFIED 2026-06-10 (R2, ondevice/MACOS_RELEASE_README.md).
     qwen static iOS GPU 27.7 (ctx 2048, release config) = 2026-06-10 RELEASE-build device
     measurement (ctx-256 export measured 30.4).
     gemma4 iOS GPU 17.7 = 2026-06-10 int4km-kernel monolith (RELEASE_PLAN row #3); confirm
     build config + re-measure ANE in RELEASE before publish (debug-vs-release lesson:
     10.3 vs 30.4 / ~19 vs 58.5). -->

| | macOS GPU (M4 Max) | iOS GPU | iOS ANE |
|---|---|---|---|
| **Gemma 4 E2B** | ✅ 8/8 · 56.6–59.0 tok/s (int8 kernels) | ✅ 8/8 · **17.7 tok/s** (int4-k-means kernels) | ✅ 8/8 · ~6 tok/s (int8 chunks) |
| **Qwen3.5 0.8B** | ✅ 8/8 · 58.5 (int8 dynamic) | ✅ **27.7** (fp16 static, ctx 2048) / 12.5 (int8 dynamic) | ✅ 14.7 (int8 dynamic); static ✗ this beta (fp16 SSM recurrence) |

macOS ANE is intentionally out of scope (the runtime auto-prefers GPU on Mac for these
structures, and the Mac GPU dominates it anyway).

Parity is measured against the Hugging Face eager reference (cosine + top-1 argmax on a fixed
prompt): conversion on macOS, then re-verified end-to-end on-device (iPhone 17 Pro, iOS 27 beta).
Device numbers are int8, greedy, prompt "What is the capital of France?" / "The capital of France
is" → "Paris".
