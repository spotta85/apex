"""Memory budget reporting for the expert cache.

The runtime + cache + metrics each track one slice of "how much memory
APEX is using and whether it stayed under budget":

  - cache.max_memory_bytes  the budget itself
  - cache.current_bytes     bytes resident *right now*
  - cache.peak_bytes        high-water mark across the cache's lifetime
  - metrics.cache_hits      foreground requests served from cache
  - metrics.cache_misses    foreground requests that triggered a load
  - metrics.evictions       LRU drops because of budget pressure

`MemoryReport` collects all of those into one immutable snapshot so
demos / tests can pass the report around without dragging the live
cache + metrics objects with them. `snapshot(cache, metrics)` is the
canonical constructor; `print_summary` produces the human-readable
trace lines step 14's demo emits.

Why a separate module instead of folding into Metrics: peak_bytes lives
on the cache (cache is what observes its own bytes), while hits/misses
live on Metrics. A small reporter that joins them is the simplest
seam — Metrics keeps its single-responsibility, the cache keeps its
single-responsibility, and the reporter is the only place that has
to know about both.
"""

from dataclasses import dataclass


_MIB = 1024 * 1024


def format_mb(nbytes: float) -> str:
    """Pretty-print a byte count as MB, one decimal place."""
    return f"{nbytes / _MIB:.1f} MB"


@dataclass(frozen=True)
class MemoryReport:
    """Immutable snapshot of cache budget + cache state + metrics counters.

    `budget_bytes` is the configured `max_memory_mb` converted to bytes
    (matching `ExpertCache.max_memory_bytes`). `current_bytes` is the
    cache's footprint at snapshot time; `peak_bytes` is the highest it
    ever was since the cache was created.
    """

    budget_bytes: int
    current_bytes: int
    peak_bytes: int
    cache_hits: int
    cache_misses: int
    evictions: int

    @property
    def budget_respected(self) -> bool:
        """Whether the cache ever exceeded its configured budget.

        Compares `peak_bytes` against `budget_bytes` rather than
        `current_bytes`. If a single expert is bigger than the budget,
        the cache keeps it (otherwise the runtime would have nothing
        to compute on) — that's the one case where this returns False
        and that's the correct answer.
        """
        return self.peak_bytes <= self.budget_bytes

    def print_summary(self):
        """Emit the step 14 trace lines. Uses MB for human readability."""
        print(f"max memory budget: {format_mb(self.budget_bytes)}")
        print(f"current cache memory: {format_mb(self.current_bytes)}")
        print(f"peak cache memory: {format_mb(self.peak_bytes)}")
        print(f"cache hits: {self.cache_hits}")
        print(f"cache misses: {self.cache_misses}")
        print(f"evictions: {self.evictions}")
        print(f"budget respected: {str(self.budget_respected).lower()}")


def snapshot(cache, metrics) -> MemoryReport:
    """Build a MemoryReport from a live ExpertCache + Metrics pair."""
    return MemoryReport(
        budget_bytes=int(cache.max_memory_bytes),
        current_bytes=int(cache.current_bytes),
        peak_bytes=int(cache.peak_bytes),
        cache_hits=metrics.cache_hits,
        cache_misses=metrics.cache_misses,
        evictions=metrics.evictions,
    )
