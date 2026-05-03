"""Counters for the cache + prefetch pipeline.

Two distinctions matter:
  - **cache hit** vs **prefetch hit**: every foreground request that
    finds the expert in cache is a cache_hit. A prefetch_hit is the
    subset of those hits where the expert was put there by the
    prefetcher (i.e. the prediction *paid off*). Once a prefetched
    expert is consumed by the runtime, it stops counting as
    "outstanding" so we don't double-count later hits.
  - **foreground load** vs **prefetch completed**: the worker's loads
    don't count against the runtime's miss budget; they're tracked
    separately so we can later compare prefetch volume against actual
    usefulness (prefetch_hit_rate).

The metrics object is passed by reference into the cache / manager /
prefetcher; each component bumps its own counters. Optional everywhere
(None means tracking is off), so existing demos and tests can ignore it.
"""


class Metrics:
    def __init__(self):
        self.cache_hits = 0
        self.cache_misses = 0
        self.evictions = 0
        self.prefetch_scheduled = 0
        self.prefetch_completed = 0
        self.prefetch_hits = 0
        self.foreground_loads = 0
        # Set of (layer, expert) loaded by the prefetcher and not yet
        # consumed by the runtime. Used to detect prefetch_hits.
        self._outstanding_prefetches: set = set()

    def on_foreground_hit(self, key):
        # Runtime asked for an expert and the cache had it.
        self.cache_hits += 1
        # If the prefetcher put it there, credit the prediction.
        if key in self._outstanding_prefetches:
            self.prefetch_hits += 1
            self._outstanding_prefetches.discard(key)

    def on_foreground_miss(self, key):
        # Runtime asked for an expert and we had to load it inline.
        self.cache_misses += 1
        self.foreground_loads += 1

    def on_eviction(self, key):
        # Cache dropped an expert to stay under budget.
        self.evictions += 1
        # If the evicted key was prefetched but never used, the prefetch
        # was wasted — drop it from outstanding so a later re-load that
        # happens to be a foreground miss doesn't get mis-credited.
        self._outstanding_prefetches.discard(key)

    def on_prefetch_scheduled(self, key):
        # Prefetcher accepted a request (after dedupe + already-cached check).
        self.prefetch_scheduled += 1

    def on_prefetch_completed(self, key):
        # Worker finished loading a prefetched expert into the cache.
        self.prefetch_completed += 1
        self._outstanding_prefetches.add(key)

    @property
    def prefetch_hit_rate(self) -> float:
        # Returns 0.0 instead of dividing by zero when nothing was ever
        # scheduled — keeps the demo's print line safe when prefetch is off.
        if self.prefetch_scheduled == 0:
            return 0.0
        return self.prefetch_hits / self.prefetch_scheduled

    def print_summary(self):
        # Order chosen to match the expected demo output: prefetch
        # numbers first, then cache, then evictions.
        print("metrics:")
        print(f"prefetch scheduled: {self.prefetch_scheduled}")
        print(f"prefetch hits: {self.prefetch_hits}")
        print(f"prefetch hit rate: {self.prefetch_hit_rate * 100:.2f}%")
        print(f"cache hits: {self.cache_hits}")
        print(f"cache misses: {self.cache_misses}")
        print(f"evictions: {self.evictions}")
