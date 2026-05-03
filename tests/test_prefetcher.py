"""Tests for AsyncPrefetcher.

These tests use the real ExpertManager + ExpertCache wired up with a
fake in-memory store, so they exercise the worker thread, the in_flight
dedupe, and the wait_for_layer synchronization.
"""

import threading
import time

import numpy as np
import pytest

from apex.expert_cache import ExpertCache
from apex.expert_manager import ExpertManager
from apex.prefetcher import AsyncPrefetcher


# Fake in-memory store that records every load call. We use a small
# sleep inside load() in some tests to widen race windows on purpose.
class FakeStore:
    def __init__(self, load_delay: float = 0.0):
        self.load_calls = []
        self._lock = threading.Lock()
        self.load_delay = load_delay

    def load(self, layer_id: int, expert_id: int):
        with self._lock:
            self.load_calls.append((layer_id, expert_id))
        if self.load_delay:
            time.sleep(self.load_delay)
        return np.full(8, expert_id, dtype="float32")


@pytest.fixture
def setup():
    # Build a fresh manager+cache+prefetcher per test, and tear down the
    # worker thread cleanly at the end so test threads don't leak.
    store = FakeStore()
    cache = ExpertCache(max_memory_mb=1)
    manager = ExpertManager(store, cache)
    prefetcher = AsyncPrefetcher(manager)
    yield store, cache, manager, prefetcher
    prefetcher.shutdown()


def test_prefetch_populates_cache(setup):
    # Schedule a prefetch and wait for it; the expert should land in
    # the cache before we return.
    store, cache, _, prefetcher = setup

    prefetcher.schedule(0, 1)
    prefetcher.wait_for_layer(0)

    assert cache.has((0, 1))
    assert store.load_calls == [(0, 1)]


def test_dedupe_inflight_requests(setup):
    # Use a deliberate load delay so the second schedule() arrives
    # while the first is still in-flight. Without dedupe, the store
    # would see two loads for the same key.
    cache = ExpertCache(max_memory_mb=1)
    store = FakeStore(load_delay=0.05)
    manager = ExpertManager(store, cache)
    prefetcher = AsyncPrefetcher(manager)
    try:
        prefetcher.schedule(0, 1)
        # Tiny sleep so the worker has dequeued before our second call;
        # the second schedule() should see (0,1) in _in_flight and skip.
        time.sleep(0.005)
        prefetcher.schedule(0, 1)

        prefetcher.wait_for_layer(0)
        assert store.load_calls == [(0, 1)]
    finally:
        prefetcher.shutdown()


def test_skip_already_cached(setup):
    # If the expert is already in the cache, scheduling a prefetch for
    # it should be a no-op (no extra store load).
    store, cache, manager, prefetcher = setup

    # Foreground load gets it into the cache.
    manager.ensure_loaded(0, 1)
    assert store.load_calls == [(0, 1)]

    # Prefetcher should see it's cached and skip.
    prefetcher.schedule(0, 1)
    prefetcher.wait_for_layer(0)
    assert store.load_calls == [(0, 1)]


def test_wait_for_layer_blocks_until_done(setup):
    # wait_for_layer should not return until *every* scheduled prefetch
    # for that layer is in the cache. We use a load delay so the worker
    # is genuinely still running when we call wait.
    cache = ExpertCache(max_memory_mb=1)
    store = FakeStore(load_delay=0.02)
    manager = ExpertManager(store, cache)
    prefetcher = AsyncPrefetcher(manager)
    try:
        prefetcher.prefetch_many(1, [0, 2])
        # Before wait, the cache may or may not have anything yet.
        prefetcher.wait_for_layer(1)
        # After wait, both must be present.
        assert cache.has((1, 0))
        assert cache.has((1, 2))
    finally:
        prefetcher.shutdown()


def test_wait_for_layer_no_op_when_nothing_scheduled(setup):
    # Waiting on a layer we never prefetched should return immediately,
    # not hang.
    _, _, _, prefetcher = setup
    # If this hangs forever, the test will time out.
    prefetcher.wait_for_layer(99)
