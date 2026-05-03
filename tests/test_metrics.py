"""Tests for Metrics: counter accounting + safe hit-rate math."""

from apex.metrics import Metrics


def test_foreground_hit_and_miss_increment():
    m = Metrics()

    m.on_foreground_miss((0, 1))
    m.on_foreground_miss((0, 3))
    m.on_foreground_hit((0, 1))

    assert m.cache_misses == 2
    assert m.cache_hits == 1
    assert m.foreground_loads == 2


def test_prefetch_scheduled_and_completed_increment():
    m = Metrics()

    m.on_prefetch_scheduled((1, 0))
    m.on_prefetch_scheduled((1, 2))
    m.on_prefetch_completed((1, 0))

    assert m.prefetch_scheduled == 2
    assert m.prefetch_completed == 1


def test_prefetch_hit_credited_only_when_was_prefetched():
    # A foreground hit on a key the prefetcher loaded counts as both a
    # cache_hit AND a prefetch_hit. A foreground hit on something that
    # was *not* prefetched only counts as a cache_hit.
    m = Metrics()

    # Prefetcher loaded (1, 0) into the cache.
    m.on_prefetch_completed((1, 0))
    # Foreground asks for (1, 0) -> cache hit, prefetch hit.
    m.on_foreground_hit((1, 0))
    # Foreground asks for (1, 0) again -> cache hit, but the prefetch
    # was already credited; don't double-count.
    m.on_foreground_hit((1, 0))
    # Foreground asks for (1, 5), which was never prefetched.
    m.on_foreground_hit((1, 5))

    assert m.cache_hits == 3
    assert m.prefetch_hits == 1


def test_eviction_clears_outstanding_prefetch():
    # If a prefetched expert is evicted before it's ever used, the
    # prefetch was wasted. A future foreground hit on the same key
    # (after a re-load) should NOT be credited to the original prefetch.
    m = Metrics()

    m.on_prefetch_completed((1, 0))
    m.on_eviction((1, 0))  # Evicted before consumed
    m.on_foreground_hit((1, 0))  # Re-cached somehow, but not from prefetch

    # The hit counts as a cache_hit but NOT a prefetch_hit.
    assert m.cache_hits == 1
    assert m.prefetch_hits == 0
    assert m.evictions == 1


def test_hit_rate_safe_when_zero_scheduled():
    # No prefetches scheduled -> hit_rate is 0.0, not a division error.
    m = Metrics()
    assert m.prefetch_hit_rate == 0.0


def test_hit_rate_calculated_correctly():
    # 4 scheduled, 3 hit -> 75%.
    m = Metrics()
    for key in [(1, 0), (1, 1), (1, 2), (1, 3)]:
        m.on_prefetch_scheduled(key)
        m.on_prefetch_completed(key)
    m.on_foreground_hit((1, 0))
    m.on_foreground_hit((1, 1))
    m.on_foreground_hit((1, 2))

    assert m.prefetch_scheduled == 4
    assert m.prefetch_hits == 3
    assert m.prefetch_hit_rate == 0.75
