# TensorOps: quantized matmul, cooperative tensors, and the neural accelerator (WWDC26 330)

> Foundation note for the **GPU-now** track — the hand-tuned-kernel layer UNDER
> [`custom-metal-kernels.md`](custom-metal-kernels.md). `TorchMetalKernel` gets your MSL into the
> `.aimodel`; **TensorOps is what that MSL should be written WITH** when it does matmul-shaped work.
> Source: WWDC26 session 330 "Optimize custom ML operations with Metal tensors" (verbatim transcript:
> `ondevice/_wwdc330_transcript.txt`, timestamps below), Apple session page, Metal Performance
> Primitives docs. Nothing here is device-measured by this project yet — marked ⚗️ where untested.

## Why this matters here

The stack, per the talk [00:28]: Core AI / MLX → MPS → **Metal Performance Primitives + TensorOps**
(an MSL API for tensor ops — matmul, convolution — inside your kernel). Two facts make it relevant
to this zoo:

1. **TensorOps auto-uses hardware acceleration across GPU generations** [01:29] — including the
   **neural accelerator, a new HW block in each shader core of the M5 family** [01:47], which Apple's
   session page extends to **A19** GPUs. It targets "dense compute-bound work **such as the prefill
   stage of an LLM**" [01:58]. A hand-rolled scalar/simdgroup MSL matmul gets NONE of this; the same
   kernel written on TensorOps does, with no per-generation code.
2. **TensorOps natively eats quantized tensors** — you pass int4/int8/fp8 data (+ scales) straight
   into `matmul2d` and "TensorOps will handle dequantization for you" [07:51], on the HW path.

This project's shipped custom kernels (the gemma4 attn-qo int8 kernels, the int8 dequant-LUT matvec)
are hand-rolled MSL — correct, measured, but blind to the neural accelerator. That's fine for
**decode** (memory-bandwidth-bound; int8 already reads at int8 memory). It is NOT fine for
**prefill** (compute-bound) on M5/A19-class hardware, where TensorOps is the only sanctioned way to
reach the new silicon. ⚗️ Expected shape of the win: prefill/TTFT, not decode tok/s.

## Quantized dtype support matrix (OS-gated — check before shipping)

| dtype | TensorOps support | OS floor |
|---|---|---|
| fp16 / fp32 | always | 26 |
| **int4, int8** | native (data type on the tensor) | **26 (point update)** [03:24] |
| **fp4, fp8, int2** | native | **27** [03:31] |
| **MX scaling formats / FP8 E8M0 block scale factors** | native scale plane | **27** [03:37, 04:21] |
| coop tensor **directly as matmul input** | `get_left_input_cooperative_tensor` | **27** [12:20] (26 = store/reload via threadgroup memory) |

⚠️ The sub-byte dtypes carry **extra alignment requirements** vs the larger types [09:20] — check the
Metal docs per dtype before assuming a layout. ⚠️ Zoo models that must run on OS 26 cannot use the
27-only column; gate kernels accordingly.

## Scale planes: one MTLTensor = data + scales (OS 27)

In OS 27 a single `MTLTensor` carries the quantized data plane **plus an auxiliary scale plane**
[04:23]: FP8 E8M0 block-wise scales, `blockFactors` defining the block (e.g. **32×1 → 32 data
elements share one scale** [06:17]). Host side: scale-plane descriptor (`dataType`, `blockFactors`)
→ auxiliary-plane map (kind = scales) → attach to the main `tensorDescriptor` →
`newTensorWithDescriptor` [04:44–05:12]. Kernel side: declare the scales-plane type + the full
tensor type (dtype + scales plane), bind to a buffer binding point — or construct a `tensor_inline`
on the shader stack from raw pointers + metadata when you don't want a host-side MTLTensor [06:43].

`slice()` on such a tensor slices **data and scale planes simultaneously**, respecting the block
size [07:26] — so threadgroup tiling code doesn't change.

## Quantized matmul2d — and the custom-format escape hatch

The quantized path is the SAME `matmul2d_descriptor` / op setup as fp16 [07:32]: descriptor (tile
sizes), op (simdgroups per threadgroup), `run`. Feed quantized tensors directly; dequant happens
inside, HW-accelerated [07:59].

If your format is NOT one TensorOps understands (e.g. a palettized LUT codebook — this zoo's
int8 LUT matvec): dequantize yourself, but **into a cooperative tensor** (storage distributed
across the thread-private registers of the participating threads [08:50]) and pass THAT as the
matmul input — instead of staging through threadgroup memory [08:30–09:05]. Saves the extra
load/store round trip the naive approach pays.

## FlashAttention on TensorOps (the recipe, [09:31–13:25])

The talk builds fused attention (QK^T → softmax → ×V, one kernel) from four primitives:

1. **`execution_simdgroup` operation scope** — each simdgroup runs an independent matmul and owns
   **complete rows** of the intermediate matrix, so softmax needs no cross-simdgroup exchange
   [10:12–10:39]; slice input tiles by simdgroup ID.
2. **Cooperative tensor** holds the intermediate matrix — never written to device memory [10:43].
3. **`reduce_rows`** computes the per-row max (reduction op = max, init = −INFINITY) into a second,
   smaller cooperative tensor [11:00–11:33]; **`map_iterator`** maps each 2D element's iterator to
   its row-reduction element [11:41–12:02]; dereference both to compute softmax in place.
4. Second matmul ×V takes the cooperative tensor **directly as left input** (OS 27,
   `get_left_input_cooperative_tensor`) — but layouts vary by dtype/etc., so call
   **`is_compatible_as_left_input` / `..._right_input` first**; `false` ⇒ you must store/reload
   via threadgroup memory after all [12:47–13:16]. `op.run` is identical either way.

**Core AI integration is the already-documented path** — the talk wires this exact kernel into a
Sam3 segmentation model via `TorchMetalKernel` (MSL body as a Python string, swap the HF attention
impl, export) [13:35–14:33]. See [`custom-metal-kernels.md`](custom-metal-kernels.md); nothing new
is needed to ship a TensorOps kernel inside a `.aimodel`.

## What this changes for this zoo (assessment, ⚗️ until measured)

- **Calibration first** — Apple's own [MLX-on-M5 LLM measurements](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
  (same accelerator class as A19): **prefill/TTFT 3.33–4.06× vs M4; decode 1.19–1.27×**, i.e. almost
  exactly the 120→153 GB/s bandwidth delta alone. The accelerator moves compute-bound prefill and
  contributes ~nothing to BW-bound decode. Set expectations accordingly for any work below.
- **Prefill lever on M5/A19**: a TensorOps rewrite of the big matmuls (or a TensorOps FlashAttention
  for the prefill graph) is the only route to the neural accelerator. Today's static q16 int8
  prefill (147 tok/s on iPhone 17 Pro) is strong; the question is what A19's accelerator adds on
  top — and whether the prefill is compute-bound at all at small chunk sizes (this project measured
  chunk-32 == chunk-16 per-token cost, pointing at unroll/SDPA/KV-fill, not matmul compute). Probe
  the bottleneck before writing kernels. Decode: don't bother — BW-bound, int8lin already near ½
  device bandwidth.
- **int4 re-test**: this project's int4km NO-GO was a numerics failure of OUR hand-rolled dequant
  path. Native int4 tensors + E8M0 block scales with TensorOps' own dequant is a DIFFERENT numerics
  path — one conversion + PSNR gate on the 27 beta would settle it.
- **fp8/fp4 as new design points**: between int8 (ships, clean) and int4 (failed), OS 27 adds fp8
  (int8 memory, more dynamic range) and fp4 (int4 memory, likely friendlier numerics than int4
  uniform). Pairs with the `coreai-opt` side: FP4_E2M1 / FP8_E4M3FN/E5M2 already exist in the
  compression API ([`compression-reference.md`](compression-reference.md)) — the Metal-side support
  closes the loop.
- **Keep preferring native SDPA for decode attention** ([`custom-metal-kernels.md`](custom-metal-kernels.md)
  rule). The FlashAttention recipe is for cases SDPA doesn't cover: quantized-KV attention, fused
  prefill attention, or shapes where the packed-state RMW tax killed the stateful-kernel monolith.
- **Start from the TensorOps sample code** [15:53] + MPP programming guide — captions omit the
  on-screen code, so exact signatures come from there, not from the transcript.
