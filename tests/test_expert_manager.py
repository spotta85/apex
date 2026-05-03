"""Tests for ExpertManager: store is hit on miss, cache is hit on hit."""

import numpy as np

from apex.expert_cache import ExpertCache
from apex.expert_manager import ExpertManager


# Tiny in-memory stand-in for ExpertStore so the tests don't touch disk.
class FakeStore:
    def __init__(self):
        # Track how many times load() was called per key, so we can
        # assert the manager only hits disk when it should.
        self.load_calls = []

    def load(self, layer_id: int, expert_id: int):
        self.load_calls.append((layer_id, expert_id))
        # Identity-by-key array so we can check we got the right one back.
        return np.full(8, expert_id, dtype="float32")


def test_manager_loads_on_miss():
    # First request for a key should call the store exactly once.
    store = FakeStore()
    cache = ExpertCache(max_memory_mb=1)
    manager = ExpertManager(store, cache)

    manager.ensure_loaded(0, 1)
    assert store.load_calls == [(0, 1)]


def test_manager_skips_store_on_hit():
    # Second request for the same key should be served from the cache.
    store = FakeStore()
    cache = ExpertCache(max_memory_mb=1)
    manager = ExpertManager(store, cache)

    manager.ensure_loaded(0, 1)
    manager.ensure_loaded(0, 1)
    # Still just the one disk load — the second call hit the cache.
    assert store.load_calls == [(0, 1)]


def test_manager_reloads_after_eviction():
    # If the cache evicts a key, the next request for it should miss
    # again and re-fetch from the store.
    store = FakeStore()
    # 80-byte budget holds two 32-byte experts; the third forces eviction.
    cache = ExpertCache(max_memory_mb=80 / (1024 * 1024))
    manager = ExpertManager(store, cache)

    manager.ensure_loaded(0, 0)
    manager.ensure_loaded(0, 1)
    manager.ensure_loaded(0, 2)  # evicts (0, 0)
    manager.ensure_loaded(0, 0)  # should re-load from store

    assert store.load_calls == [(0, 0), (0, 1), (0, 2), (0, 0)]
