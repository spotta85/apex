"""Thread-safe in-memory LRU cache for expert weights, bounded by a memory budget.

This is the load-bearing piece of APEX's "stay under max_memory_gb"
promise. The cache is told how many bytes it's allowed to use, tracks
the running total via each entry's `nbytes`, and evicts least-recently-
used experts whenever a new insert would push it over budget.

`verbose=False` silences the per-put memory + eviction print lines so
the predictor demo can show a focused trace without loader noise. The
metrics callback still fires for evictions in either mode.

Why thread-safety: AsyncPrefetcher writes to this cache from a
background worker thread while the runtime reads from the main thread.
Without locking, the OrderedDict / current_bytes / LRU ordering would
race. An RLock is used so a method holding the lock can safely call
another locked method without self-deadlock.

Why a memory budget instead of a count of experts: experts in real MoE
models vary wildly in size, and what users actually care about is "don't
let inference eat all my RAM." A count is a poor proxy.
"""

import threading
from collections import OrderedDict

from apex.tensor_utils import tensor_nbytes


class ExpertCache:
    def __init__(self, max_memory_mb: float, verbose: bool = True, metrics=None):
        # Convert the public MiB-based budget into the byte unit we
        # actually compare against. 1 MiB = 1024 * 1024 bytes.
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        # Running total of nbytes across everything in the cache.
        self.current_bytes = 0
        # High-water mark for current_bytes across the cache's lifetime.
        # Updated post-eviction in put() so it reflects what the cache
        # actually held at any moment, not transient mid-insert peaks.
        # Used by apex.memory + step 14 demos to assert "budget respected".
        self.peak_bytes = 0
        # OrderedDict gives O(1) insert/lookup and an in-order eviction
        # policy: oldest-touched at the front, newest at the back.
        self.cache = OrderedDict()
        # All mutating / order-sensitive operations grab this lock.
        self._lock = threading.RLock()
        # When False, suppress per-put trace lines (used by demos that
        # want a clean output focused on something other than caching).
        self.verbose = verbose
        # Optional Metrics object; cache notifies it on evictions.
        self.metrics = metrics

    def has(self, key):
        # `key` is a (layer_id, expert_id) tuple.
        with self._lock:
            return key in self.cache

    def get(self, key):
        # Touching an entry promotes it to most-recently-used.
        # Raises KeyError if missing — callers that race against eviction
        # should use try_get() instead, which is atomic.
        with self._lock:
            self.cache.move_to_end(key)
            return self.cache[key]

    def try_get(self, key):
        # Atomic "check-and-get-with-LRU-promote". Returns None on miss.
        # Avoids the TOCTOU race between has() and get() that an unlucky
        # thread interleaving would otherwise hit.
        with self._lock:
            if key not in self.cache:
                return None
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key, value):
        with self._lock:
            # Insert or refresh the entry and mark it most-recently-used.
            self.cache[key] = value
            self.cache.move_to_end(key)
            # tensor_nbytes handles ndarrays, mx.array, and ExpertWeights
            # bundles (post-step-4) — single source of truth for size.
            self.current_bytes += tensor_nbytes(value)

            # Evict from the LRU end until we're back under budget.
            # The `len > 1` guard prevents us from evicting the entry we
            # just inserted in the (degenerate) case where one expert
            # alone is bigger than the whole budget.
            while self.current_bytes > self.max_memory_bytes and len(self.cache) > 1:
                old_key, old_value = self.cache.popitem(last=False)
                self.current_bytes -= tensor_nbytes(old_value)
                if self.verbose:
                    print(f"evicting layer {old_key[0]} expert {old_key[1]}")
                if self.metrics is not None:
                    self.metrics.on_eviction(old_key)

            # Update peak after eviction so it reflects the steady-state
            # post-insert footprint, not the transient mid-insert peak
            # the eviction loop just stepped down from.
            if self.current_bytes > self.peak_bytes:
                self.peak_bytes = self.current_bytes

            # Verbose-only: report post-load memory usage so the user
            # can watch the cache stay near its budget. Metrics still
            # tracks exact byte count via current_bytes for queries.
            if self.verbose:
                print(f"current cache memory: {self.current_bytes} bytes")
