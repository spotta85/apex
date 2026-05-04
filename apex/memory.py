"""Memory budget reporting for the expert cache + KV cache.

Three immutable snapshots:

  - MemoryReport         expert cache (LRU, evictable). Tracks budget,
                         current/peak bytes, hits/misses/evictions.
  - KVMemoryReport       KV cache (monotonic, non-evictable during
                         generation). Tracks per-layer + total bytes.
  - RuntimeMemoryReport  joined view: expert + kv + total. The shape
                         the step 17E spec calls out.

`snapshot(cache, metrics)` builds the expert one, `kv_snapshot(kv_cache)`
builds the KV one, and `runtime_snapshot(cache, metrics, kv_cache)`
builds the joined one (composing the other two so callers can still
drill into either side cleanly).

Why a separate module instead of folding into Metrics: peak_bytes lives
on the cache, hits/misses live on Metrics, KV lives on KVCache. A small
reporter that joins them is the simplest seam — each producer keeps its
single-responsibility and the reporter is the only place that has to
know about all three.
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


@dataclass(frozen=True)
class KVMemoryReport:
    """Per-layer KV cache memory snapshot.

    `per_layer_bytes[L]` is the sum of nbytes for layer L's keys +
    values (or 0 if the layer hasn't been touched). `total_bytes` is
    the sum across all layers. KV memory is non-evictable during
    generation, so there's no `budget_respected` notion here — operators
    set their generation budget by limiting sequence length, not by
    asking the cache to drop entries.
    """

    per_layer_bytes: tuple
    total_bytes: int

    def print_summary(self):
        """Print the total + each non-zero layer's contribution."""
        print(f"kv cache total: {format_mb(self.total_bytes)}")
        for layer_id, nbytes in enumerate(self.per_layer_bytes):
            # Skip empty layers — common during sub-model smokes that
            # only drive layer 0 of a larger num_layers KVCache.
            if nbytes:
                print(f"  layer {layer_id}: {format_mb(nbytes)}")


def kv_snapshot(kv_cache) -> KVMemoryReport:
    """Build a KVMemoryReport from a live KVCache.

    Walks every layer slot and asks for `keys`/`values` shape; uses
    `nbytes` when present, 0 otherwise. The total comes from
    `kv_cache.total_bytes` so the joined report stays consistent with
    whatever accounting KVCache itself uses.
    """
    per_layer = []
    for layer_id in range(kv_cache.num_layers):
        keys, values = kv_cache.get(layer_id)
        layer_bytes = 0
        if keys is not None:
            layer_bytes += int(keys.nbytes)
        if values is not None:
            layer_bytes += int(values.nbytes)
        per_layer.append(layer_bytes)
    return KVMemoryReport(
        per_layer_bytes=tuple(per_layer),
        total_bytes=int(kv_cache.total_bytes),
    )


@dataclass(frozen=True)
class RuntimeMemoryReport:
    """Joined view of expert (LRU, evictable) + KV (monotonic, not) memory.

    `expert_cache_bytes`, `kv_cache_bytes`, `total_runtime_bytes` are
    the three numbers the step 17 spec calls out. Drill into `.expert`
    or `.kv` for the full per-side detail (hits/misses/peak on the
    expert side, per-layer breakdown on the KV side).
    """

    expert: MemoryReport
    kv: KVMemoryReport

    @property
    def expert_cache_bytes(self) -> int:
        return self.expert.current_bytes

    @property
    def kv_cache_bytes(self) -> int:
        return self.kv.total_bytes

    @property
    def total_runtime_bytes(self) -> int:
        return self.expert_cache_bytes + self.kv_cache_bytes

    def print_summary(self):
        """One-screen view: expert (with peak/budget context), kv, total."""
        print(
            f"expert cache: {format_mb(self.expert_cache_bytes)} "
            f"(peak {format_mb(self.expert.peak_bytes)} / "
            f"budget {format_mb(self.expert.budget_bytes)})"
        )
        print(f"kv cache:     {format_mb(self.kv_cache_bytes)}")
        print(f"total:        {format_mb(self.total_runtime_bytes)}")


def runtime_snapshot(cache, metrics, kv_cache) -> RuntimeMemoryReport:
    """Joined snapshot from a live ExpertCache + Metrics + KVCache.

    The three are independent objects that each track one slice of
    runtime memory; this composes them so demos / tests can hand
    around a single immutable summary instead of three live handles.
    """
    return RuntimeMemoryReport(
        expert=snapshot(cache, metrics),
        kv=kv_snapshot(kv_cache),
    )
