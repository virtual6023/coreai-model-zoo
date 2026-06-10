# Stateful decode & KV cache

A `.aimodel` can declare **states** — tensors the graph mutates in place — surfaced via a `state=`
kwarg (Python) / `InferenceFunction.MutableViews` (Swift). This is how KV / SSM caches persist
across decode steps without re-feeding the whole sequence.

## One dynamic graph for prefill + decode

Export a single graph where `offset = position_ids.len − query_len`:
- **prefill**: `query_len = S` tokens, full `position_ids [0..S)`, zero states.
- **decode**: `query_len = 1`, full `position_ids [0..past+1)`, persisted states.

Trace with `offset > 0` (position length > query length) so the query-length and position-length
dims stay independent. The cache tensors are static-allocated at context length (or grown 2×).
In-place writes use `slice_update` and require `remove_functionalization(ep)` before converting.

## Hybrid / multi-state caches

Standard models have 2 states (key/value). Newer architectures have more:
- **Qwen3.5** (hybrid linear+full attention): **4 states** — `keyCache`, `valueCache` (full-attn
  layers) + `convState`, `recState` (Mamba/SSM linear layers). The full KV grows with seq; the
  conv/rec states are fixed-shape.
- **Gemma 4** (dual head_dim): **4 states** — a sliding-window cache (head_dim 256) and a full
  cache (head_dim 512), each over only the non-shared layers (KV-sharing: producer layers write,
  shared layers read).

Apple's Swift `CoreAISequentialEngine` hard-codes 2 states; drive these with a generic N-state
runner (see [`swift-runtime.md`](swift-runtime.md)).

## Sliding-window ring buffer (long-context memory)

A sliding-window cache only needs the last `W` keys. Instead of a `ctx`-sized linear cache, use a
**width-`W` ring buffer**: write at `pos % W`, attend over the whole ring under a position-derived
mask. Since RoPE is baked into K at write time, slot order doesn't change the scores — the mask
just marks which slots hold an already-written, causal, in-window position (built in-graph from
`position_ids`, so the bundle signature is unchanged).

- Win: sliding KV memory becomes **constant** vs growing with context (Gemma 4 E2B: ~6 MB fp16
  regardless of ctx, vs 25 MB → 400 MB at 2K → 32K). On-disk weights unchanged.
- Exact for one-token decode and from-zero prefill. A chunked prefill block that *wraps* the ring
  drops the oldest in-window keys for the block's non-final tokens (final token stays exact) →
  feed from position 0, bound prefill chunks ≤ W.
- `aten.remainder` (tensor) doesn't lower → compute `(Pmax−r) % W` with a scalar symint + `where`.
