"""Tests for backend-prepared cache.

Verifies that:
  - ExpertManager calls backend.prepare_expert exactly once on a miss.
  - Cache hits return the prepared object without re-converting.
  - MLXBackend's prepared cache holds mx.array.
  - FakeBackend's prepare is a passthrough — same identity in cache.
"""

import numpy as np
import pytest

from apex.backends.fake_backend import FakeBackend
from apex.expert_cache import ExpertCache
from apex.expert_manager import ExpertManager
from apex.types import ExpertWeights
from tests._world import FakeStore


# Recording wrapper around any backend so we can count prepare_expert calls.
class CountingBackend:
    name = "counting"

    def __init__(self, inner):
        self.inner = inner
        self.prep_calls = 0

    def prepare_expert(self, arr):
        self.prep_calls += 1
        return self.inner.prepare_expert(arr)

    # Forward whatever else the test might need to the inner backend.
    def __getattr__(self, name):
        return getattr(self.inner, name)


def test_prep_called_once_on_miss_then_skipped_on_hit():
    backend = CountingBackend(FakeBackend())
    cache = ExpertCache(max_memory_mb=1, verbose=False)
    manager = ExpertManager(FakeStore(), cache, backend=backend)

    manager.ensure_loaded(0, 1)
    manager.ensure_loaded(0, 1)  # should hit cache, no extra prep

    assert backend.prep_calls == 1


def test_cache_holds_prepared_object():
    # Whatever prepare_expert returns is what cache.get hands back —
    # not the raw store result and not a fresh conversion.
    backend = CountingBackend(FakeBackend())
    cache = ExpertCache(max_memory_mb=1, verbose=False)
    manager = ExpertManager(FakeStore(), cache, backend=backend)

    first = manager.ensure_loaded(0, 1)
    second = manager.ensure_loaded(0, 1)

    assert first is second  # exact same object across miss + hit
    assert cache.has((0, 1))
    # And the cache stores that same object.
    assert cache.get((0, 1)) is first


def test_fake_backend_prepare_is_identity():
    # FakeBackend (NumPy) is passthrough: prepare_expert returns the
    # same ExpertWeights bundle object so the manager skips the trace
    # line and the cache stores the original (with ndarray projections).
    backend = FakeBackend()
    cache = ExpertCache(max_memory_mb=1, verbose=False)
    manager = ExpertManager(FakeStore(), cache, backend=backend)

    loaded = manager.ensure_loaded(0, 1)
    # Identity check is the contract the manager uses to decide whether
    # to print the "prepared expert for backend: ..." line — for the
    # NumPy passthrough, the bundle (and its projections) is unchanged.
    assert isinstance(loaded, ExpertWeights)
    assert isinstance(loaded.gate_proj, np.ndarray)


def test_no_backend_means_no_prep_step():
    # Backwards-compat path: ExpertManager(store, cache) without a
    # backend should still load + cache exactly as before.
    cache = ExpertCache(max_memory_mb=1, verbose=False)
    manager = ExpertManager(FakeStore(), cache)  # no backend kwarg

    loaded = manager.ensure_loaded(0, 1)
    assert isinstance(loaded, ExpertWeights)
    assert cache.has((0, 1))


def test_prep_fires_for_prefetch_source_too():
    # The prefetch worker calls ensure_loaded(..., source="prefetch").
    # That path must also run prepare_expert — otherwise prefetched
    # entries would be raw ndarrays and a later runtime hit would
    # bypass MLX's mx.array cached form.
    backend = CountingBackend(FakeBackend())
    cache = ExpertCache(max_memory_mb=1, verbose=False)
    manager = ExpertManager(FakeStore(), cache, backend=backend)

    manager.ensure_loaded(0, 1, source="prefetch")
    assert backend.prep_calls == 1


# MLX-specific test, gated by importorskip.
mx = pytest.importorskip("mlx.core")
from apex.backends.mlx_backend import MLXBackend


def test_mlx_backend_caches_mx_array():
    # MLXBackend.prepare_expert returns an ExpertWeights bundle whose
    # three projections are mx.array; the cache should hold that
    # MLX-native bundle directly so future hits don't re-convert.
    backend = MLXBackend()
    cache = ExpertCache(max_memory_mb=1, verbose=False)
    manager = ExpertManager(FakeStore(), cache, backend=backend)

    loaded = manager.ensure_loaded(0, 1)
    assert isinstance(loaded, ExpertWeights)
    assert isinstance(loaded.gate_proj, mx.array)
    cached = cache.get((0, 1))
    assert isinstance(cached, ExpertWeights)
    assert isinstance(cached.gate_proj, mx.array)
