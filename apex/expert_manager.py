"""Coordinates the cache and the store.

Everything in the runtime that needs an expert goes through
`ensure_loaded` — that's the choke point that decides whether to serve
from RAM (cache hit) or pay the disk cost (cache miss). Centralizing it
also means there's exactly one place that prints the hit/miss/eviction
trace, which is what makes the MVP demo legible.

Two callers come through here:
  - foreground: the runtime, mid-compute. We print "cache hit/miss" and
    bump foreground metrics.
  - prefetch: AsyncPrefetcher's worker thread. We stay quiet (the
    prefetcher prints its own line) and skip the foreground counters
    (the prefetcher tracks prefetch-specific metrics).

When a backend is supplied, `ensure_loaded` also calls
`backend.prepare_expert(...)` between `store.load` and `cache.put`.
That converts the raw on-disk tensor into the form the backend wants
to compute against (e.g. mx.array for MLXBackend), so the cache holds
ready-to-use objects and cache hits don't pay any conversion cost.
The "prepared expert for backend: <name>" trace line fires only when
prep actually transformed the array (i.e. the result is a different
object) — passthrough backends like NumPy stay quiet.
"""


class ExpertManager:
    def __init__(self, store, cache, metrics=None, backend=None):
        # The store knows how to fetch expert weights from disk.
        self.store = store
        # The cache decides which experts stay in memory.
        self.cache = cache
        # Optional Metrics object. Only foreground hits/misses bump
        # counters here; prefetch-specific stats live on the prefetcher.
        self.metrics = metrics
        # Optional Backend. When provided, ensure_loaded calls
        # backend.prepare_expert on the loaded array before caching it.
        self.backend = backend

    def ensure_loaded(self, layer_id: int, expert_id: int, source: str = "foreground"):
        # Experts are identified by (layer, expert) pairs.
        key = (layer_id, expert_id)
        is_foreground = source == "foreground"

        # Atomic check-and-promote so we don't race with a background
        # prefetcher evicting the key between has() and get().
        cached = self.cache.try_get(key)
        if cached is not None:
            # Fast path: already in RAM.
            if is_foreground:
                print(f"cache hit: layer {layer_id} expert {expert_id}")
                if self.metrics is not None:
                    self.metrics.on_foreground_hit(key)
            return cached

        # Slow path: not in RAM. Pay the disk read once, then cache it.
        # The cache's put() will print an "evicting ..." line if this
        # insert pushes us over the memory budget (verbose mode only).
        if is_foreground:
            print(f"cache miss: loading layer {layer_id} expert {expert_id}")
            if self.metrics is not None:
                self.metrics.on_foreground_miss(key)
        expert = self.store.load(layer_id, expert_id)

        # Backend-prep step: convert the loaded array into the backend's
        # preferred form so the cache holds compute-ready objects. We
        # only print when prep actually transformed the array — the
        # `is`-check avoids noise when prep is a passthrough (NumPy).
        if self.backend is not None:
            prepared = self.backend.prepare_expert(expert)
            if prepared is not expert:
                print(f"prepared expert for backend: {self.backend.name}")
            expert = prepared

        self.cache.put(key, expert)
        return expert
