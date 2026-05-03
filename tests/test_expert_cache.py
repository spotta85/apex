"""Tests for ExpertCache: hit/miss accounting and memory-budgeted eviction.

Each fake "expert" is a small float32 array, so we know its nbytes
exactly and can reason about budget behavior in bytes (not megabytes).
"""

import numpy as np
import pytest

from apex.expert_cache import ExpertCache


# 8 float32s = 32 bytes per fake expert. Small numbers keep the
# arithmetic in the tests easy to follow.
def make_expert(value: float = 0.0):
    return np.full(8, value, dtype="float32")


# Helper: build a cache sized in bytes (tests want byte-precision; the
# public API only takes MiB, so we convert).
def cache_with_bytes(n: int) -> ExpertCache:
    return ExpertCache(max_memory_mb=n / (1024 * 1024))


def test_miss_then_hit():
    # A freshly-put key is reported as present, returns the same value.
    cache = cache_with_bytes(1024)
    expert = make_expert(1.0)

    assert not cache.has((0, 1))
    cache.put((0, 1), expert)
    assert cache.has((0, 1))
    assert cache.get((0, 1)) is expert


def test_eviction_when_over_budget():
    # Budget = 80 bytes holds two 32-byte experts (=64). Adding a third
    # pushes total to 96 bytes, so the LRU entry must be evicted.
    cache = cache_with_bytes(80)
    a, b, c = make_expert(1), make_expert(2), make_expert(3)

    cache.put((0, 0), a)
    cache.put((0, 1), b)
    cache.put((0, 2), c)

    # The oldest entry (key (0,0)) should be the one that got dropped.
    assert not cache.has((0, 0))
    assert cache.has((0, 1))
    assert cache.has((0, 2))
    # Memory accounting should reflect the surviving two experts.
    assert cache.current_bytes == 64


def test_lru_promotion_on_get():
    # Accessing an entry should make it most-recently-used, so a later
    # eviction targets a different (older) entry.
    cache = cache_with_bytes(80)
    a, b, c = make_expert(1), make_expert(2), make_expert(3)

    cache.put((0, 0), a)
    cache.put((0, 1), b)
    # Touch (0,0) -> it's now newer than (0,1).
    cache.get((0, 0))
    # Inserting (0,2) should evict the now-oldest entry, which is (0,1).
    cache.put((0, 2), c)

    assert cache.has((0, 0))
    assert not cache.has((0, 1))
    assert cache.has((0, 2))


def test_oversized_single_expert_kept():
    # An expert larger than the entire budget should still be retained
    # (otherwise the runtime would have nothing to run). The cache will
    # be over budget — that's fine and intentional for this edge case.
    cache = cache_with_bytes(8)  # smaller than one 32-byte expert
    big = make_expert(1)

    cache.put((0, 0), big)

    assert cache.has((0, 0))
    assert cache.current_bytes == big.nbytes


def test_memory_budget_conversion():
    # Public API takes MiB. Confirm the internal byte budget matches.
    cache = ExpertCache(max_memory_mb=1)
    assert cache.max_memory_bytes == 1 * 1024 * 1024
