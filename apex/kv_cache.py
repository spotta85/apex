"""KVCache — per-layer accumulated K/V tensors for multi-token attention.

The decode loop writes one new (k, v) per layer per token; attention
then reads back the full history. Storage is one ndarray per layer
per side (keys, values), shape `(batch, num_kv_heads, seq_so_far, head_dim)`,
appended along axis -2.

Two limits worth knowing:

  - Append concatenates with `np.concatenate`, which is O(seq) per
    call (the existing buffer is reallocated each time). Fine for a
    smoke / decode demo where seq stays small; production loops
    should pre-allocate `max_seq` slots and write into them. Marked
    in `append`'s docstring so a future contributor doesn't get
    surprised by an O(n²) decode.
  - No eviction or rolling window. Sliding-window attention layers
    in Gemma 4 limit *attention scope* to the last 1024 tokens, but
    the cache itself still holds everything. A windowed cache would
    drop entries older than the window — out of scope for now.

`total_bytes` is exposed so step 17E can join it with the LRU expert
cache budget into a single `MemoryReport`-style snapshot. KV memory
isn't evictable mid-generation (the model would forget its context),
so it's tracked separately from `apex.memory.MemoryReport`.
"""

import numpy as np


class KVCache:
    """Per-layer key/value history for multi-token attention.

    Layer ids must be in `[0, num_layers)`. Each layer's `keys` and
    `values` start as `None` and become ndarrays after the first
    `append`; subsequent appends concatenate along axis -2 (the
    sequence axis).
    """

    def __init__(self, num_layers: int):
        if num_layers <= 0:
            raise ValueError(
                f"num_layers must be positive, got {num_layers}"
            )
        self.num_layers = num_layers
        self.keys: list = [None] * num_layers
        self.values: list = [None] * num_layers

    def append(self, layer_id: int, k, v):
        """Append `k`, `v` to layer `layer_id`'s history along the seq axis.

        Both `k` and `v` must share their last two dims with any
        previous append for this layer (num_kv_heads + head_dim);
        only the seq axis (-2) varies.

        Reallocates the backing buffer on every call (`np.concatenate`),
        so this is O(total_seq) per append — acceptable for the
        decode smoke, suboptimal for long-context production.
        """
        self._check_layer(layer_id)
        if k.shape[:-2] != v.shape[:-2] or k.shape[-1] != v.shape[-1]:
            raise ValueError(
                f"k {k.shape} and v {v.shape} must share batch/heads/head_dim"
            )
        if self.keys[layer_id] is None:
            self.keys[layer_id] = np.asarray(k)
            self.values[layer_id] = np.asarray(v)
        else:
            self.keys[layer_id] = np.concatenate(
                [self.keys[layer_id], k], axis=-2,
            )
            self.values[layer_id] = np.concatenate(
                [self.values[layer_id], v], axis=-2,
            )

    def get(self, layer_id: int):
        """Return `(keys, values)` for `layer_id`. May be `(None, None)`."""
        self._check_layer(layer_id)
        return self.keys[layer_id], self.values[layer_id]

    def length(self, layer_id: int) -> int:
        """Number of cached tokens for `layer_id` (0 if untouched)."""
        self._check_layer(layer_id)
        if self.keys[layer_id] is None:
            return 0
        return self.keys[layer_id].shape[-2]

    @property
    def total_bytes(self) -> int:
        """Sum of nbytes across every layer's keys + values.

        Used by 17E to track KV memory separately from the LRU expert
        cache budget. KV memory is non-evictable during generation,
        so it can't be folded into `apex.memory.MemoryReport`.
        """
        total = 0
        for k in self.keys:
            if k is not None:
                total += k.nbytes
        for v in self.values:
            if v is not None:
                total += v.nbytes
        return total

    def reset(self):
        """Drop all cached k/v, leaving the cache as if just constructed.

        Use between independent generation runs that share a KVCache
        instance (e.g. a server reusing the same allocator across
        requests). All slots return to `None`.
        """
        self.keys = [None] * self.num_layers
        self.values = [None] * self.num_layers

    def _check_layer(self, layer_id: int):
        if not 0 <= layer_id < self.num_layers:
            raise IndexError(
                f"layer_id {layer_id} out of range [0, {self.num_layers})"
            )
