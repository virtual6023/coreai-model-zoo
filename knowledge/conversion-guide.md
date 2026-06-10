# Conversion guide (PyTorch → `.aimodel`)

Re-author the model with `coreai_models` primitives (so it lowers cleanly), export via
`torch.export`, convert with `TorchConverter`, optimize, save. Verify numerically against the
Hugging Face reference (cosine / top-1 argmax) before trusting a bundle.

## Canonical API (gotchas burned in)

```python
import torch, shutil
from pathlib import Path
from coreai_torch import TorchConverter, get_decomp_table
import coreai.runtime as rt

ep = torch.export.export(m.eval(), args=(), kwargs=inputs).run_decompositions(get_decomp_table())
prog = (TorchConverter()
        .add_exported_program(exported_program=ep, input_names=[...], output_names=[...])
        .to_coreai())
prog.optimize()
shutil.rmtree(out, ignore_errors=True)                 # save_asset will NOT overwrite
prog.save_asset(Path(out), rt.AIModelAssetMetadata())  # Path (not str); minimum_os defaults to v27

model = await rt.AIModel.load(Path(out), rt.SpecializationOptions.cpu_only())  # async; cpu_only for parity
fn = model.load_function("main")                       # SYNC; model.function_names is a list attr
res = await fn({"x": rt.NDArray(np_or_tensor)})        # __call__ is ASYNC -> dict[str, NDArray]
```

## Gotchas that cost real time

- **`save_asset` won't overwrite** — `shutil.rmtree(out)` first.
- **`AIModel.load` is async; `load_function` is sync; calling the function is async.** Mixing these
  up is the most common first error.
- **Use `cpu_only()` for numeric parity checks** — h16c (fp16-compute) on GPU/ANE produces larger
  hidden-state diffs on high-magnitude activations (still argmax-exact, but noisier maxdiff).
- **Unsupported ATen ops surface at `add_exported_program` validate time**, not at runtime. e.g.
  `aten.remainder.Scalar` (tensor modulo) is unsupported — compute it with `floor`/`where` or a
  scalar symint instead. Either register a lowering (`register_torch_lowering`) or avoid the op.
- **In-place state writes** (KV cache via `slice_update`) need `remove_functionalization(ep)` after
  `run_decompositions`, before converting. Without it the mutation is dropped.
- **Composite externalization** (`ExternalizeSpec` for RMSNorm/RoPE/SDPA/...) marks ops by *class*.
  If your export unit holds submodules of that class which are NOT in the traced graph (e.g. a
  front-end norm kept as an attribute), externalizing fails ("custom op not found"). Opt out by
  exposing `coreai_externalize_specs = ()` on the module, or pass an empty externalize list.
- **fp16 conversion**: keep RoPE `inv_freq` a plain fp32 attribute (NOT a buffer) so `.half()`
  can't underflow the small frequencies to zero. Cast RoPE cos/sin to the query dtype.
- **Truncating layers for smoke tests**: keep architectural regions (KV-shared / double-wide MLP)
  aligned with the checkpoint's real layout, or weights won't load.

## Verify, don't trust

Compare to an HF eager reference: cosine ≈ 1.0 and **top-1 argmax match** on a fixed prompt is the
real pass criterion. A large hidden-state maxdiff alone is usually h16c compute noise, not a bug —
let argmax/top-5 decide. Re-run parity after any OS/toolchain bump (the runtime is OS-coupled).
