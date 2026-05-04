"""Tests for step 14: cache budget enforcement under real expert pressure.

Three layers of coverage:

  - peak_bytes accounting: the cache's high-water mark is correct
    across put / evict cycles, including the oversized-single-expert
    edge case.
  - MemoryReport snapshot + budget_respected: the joiner module
    correctly reads cache + metrics and computes the pass/fail.
  - Real Gemma 4 NVFP4 (skip if missing): driving Runtime through two
    layers under a tight budget triggers eviction, stays in budget,
    and produces finite output.

Synthetic tests use small float32 arrays so the byte arithmetic stays
human-readable; the real-Gemma test exercises the same invariants on
NVFP4-decoded experts (~24 MB each).
"""

import numpy as np

from apex.backends.fake_backend import FakeBackend
from apex.dequant import DequantizingExpertStore
from apex.expert_cache import ExpertCache
from apex.expert_manager import ExpertManager
from apex.kv_cache import KVCache
from apex.memory import (
    KVMemoryReport,
    MemoryReport,
    RuntimeMemoryReport,
    kv_snapshot,
    runtime_snapshot,
    snapshot,
)
from apex.metrics import Metrics
from apex.runtime import Runtime
from tests._synthetic_nvfp4 import (
    real_checkpoint_path,
    real_layout_and_quant_store,
    skip_if_no_real_checkpoint,
)


# 64 bytes per fake expert: 16 float32s. Picked so two experts fit in
# a 128-byte cache exactly, and a third forces eviction.
_BYTES_PER_EXPERT = 64


def _expert(value: float) -> np.ndarray:
    return np.full(16, value, dtype=np.float32)


def _cache_with_bytes(n: int) -> ExpertCache:
    return ExpertCache(max_memory_mb=n / (1024 * 1024), verbose=False)


def test_peak_bytes_tracks_high_water_mark():
    """peak_bytes should rise as the cache fills and stay at the max."""
    cache = _cache_with_bytes(_BYTES_PER_EXPERT * 4)  # holds 4 experts
    assert cache.peak_bytes == 0

    cache.put((0, 0), _expert(1))
    assert cache.peak_bytes == _BYTES_PER_EXPERT
    cache.put((0, 1), _expert(2))
    assert cache.peak_bytes == 2 * _BYTES_PER_EXPERT
    cache.put((0, 2), _expert(3))
    assert cache.peak_bytes == 3 * _BYTES_PER_EXPERT


def test_peak_bytes_stays_at_high_water_after_eviction():
    """peak_bytes is a high-water mark; it must not drop after an eviction."""
    # Budget = 2 experts. Inserting a third forces one eviction.
    cache = _cache_with_bytes(_BYTES_PER_EXPERT * 2)

    cache.put((0, 0), _expert(1))
    cache.put((0, 1), _expert(2))
    assert cache.peak_bytes == 2 * _BYTES_PER_EXPERT
    cache.put((0, 2), _expert(3))

    # current_bytes is back at the budget (one expert evicted, one added).
    assert cache.current_bytes == 2 * _BYTES_PER_EXPERT
    # peak hasn't moved — even though we briefly held 3 experts mid-put,
    # the post-eviction steady state is what we measure.
    assert cache.peak_bytes == 2 * _BYTES_PER_EXPERT


def test_peak_bytes_oversized_single_expert():
    """Single-expert > budget: cache keeps it; peak reflects the actual size."""
    # Budget = 8 bytes; one 64-byte expert is 8x oversized.
    cache = _cache_with_bytes(8)
    cache.put((0, 0), _expert(1))

    assert cache.has((0, 0))
    assert cache.current_bytes == _BYTES_PER_EXPERT
    assert cache.peak_bytes == _BYTES_PER_EXPERT
    # And budget_respected reflects this honestly.
    report = snapshot(cache, Metrics())
    assert not report.budget_respected


def test_snapshot_aggregates_cache_and_metrics():
    """`snapshot()` should pull every field from the right source."""
    metrics = Metrics()
    cache = ExpertCache(max_memory_mb=1, verbose=False, metrics=metrics)
    cache.put((0, 0), _expert(1))
    metrics.on_foreground_hit((0, 0))
    metrics.on_foreground_miss((0, 1))
    metrics.on_eviction((0, 2))

    report = snapshot(cache, metrics)
    assert isinstance(report, MemoryReport)
    assert report.budget_bytes == cache.max_memory_bytes
    assert report.current_bytes == cache.current_bytes
    assert report.peak_bytes == cache.peak_bytes
    assert report.cache_hits == 1
    assert report.cache_misses == 1
    assert report.evictions == 1


def test_budget_respected_true_when_peak_under_budget():
    cache = _cache_with_bytes(_BYTES_PER_EXPERT * 4)
    cache.put((0, 0), _expert(1))
    report = snapshot(cache, Metrics())
    assert report.budget_respected


def test_runtime_under_tight_budget_evicts_and_stays_in_budget(tmp_path):
    """Synthetic Runtime + cache: small budget triggers eviction without bloat.

    Two-layer synthetic checkpoint with 4 experts/layer, top_k=4 (so each
    layer wants all 4 experts). Cache budget sized to hold 4 experts; the
    second layer's loads must evict layer 0's experts. Verify peak ≤ budget.
    """
    from apex.quantized_safetensors_store import QuantizedSafetensorsExpertStore
    from apex.weight_layout import FakeMoELayout
    from tests._synthetic_nvfp4 import build_multi_layer_checkpoint

    layout = FakeMoELayout()
    ckpt = tmp_path / "ck.safetensors"
    build_multi_layer_checkpoint(
        ckpt, layout, num_layers=2, num_experts=4,
        hidden=16, intermediate=32,
    )
    quant_store = QuantizedSafetensorsExpertStore(str(ckpt), layout=layout)
    store = DequantizingExpertStore(quant_store, verbose=False)
    metrics = Metrics()

    # Pre-load one expert through the dequant path so we can size the
    # budget against the actual fp32 footprint instead of guessing.
    sample = store.load(0, 0)
    sample_bytes = (
        sample.gate_proj.nbytes + sample.up_proj.nbytes + sample.down_proj.nbytes
    )
    # Budget = 4 experts so eviction kicks in once layer 1 starts loading.
    budget_bytes = sample_bytes * 4

    cache = ExpertCache(
        max_memory_mb=budget_bytes / (1024 * 1024),
        verbose=False,
        metrics=metrics,
    )
    manager = ExpertManager(store=store, cache=cache, metrics=metrics)

    class _Stub:
        name = "numpy"

        def route(self, layer_id, hidden):
            # Ask for all 4 experts at every layer, equal weights.
            ids = list(range(4))
            return ids, [0.25] * 4

    runtime = Runtime(
        router=_Stub(),
        expert_manager=manager,
        backend=FakeBackend(),
        num_layers=2,
        initial_hidden=np.full((1, 16), 0.5, dtype=np.float32),
    )
    with np.errstate(over="ignore", under="ignore",
                     invalid="ignore", divide="ignore"):
        out = runtime.run()

    assert out.shape == (1, 16)
    assert np.all(np.isfinite(out))
    report = snapshot(cache, metrics)
    assert report.budget_respected, (
        f"peak {report.peak_bytes} > budget {report.budget_bytes}"
    )
    # Layer 1 must have evicted at least one of layer 0's experts.
    assert report.evictions > 0


@skip_if_no_real_checkpoint()
def test_real_gemma_under_small_budget_evicts_and_stays_in_budget():
    """End-to-end on real Gemma: 256 MB budget forces eviction over 2 layers."""
    from apex.real_router import MultiLayerRealRouter

    path = real_checkpoint_path()
    layout, quant_store = real_layout_and_quant_store(path)
    router = MultiLayerRealRouter(path, layout, top_k=8)
    store = DequantizingExpertStore(quant_store, verbose=False)
    metrics = Metrics()
    # 256 MB ≈ ~10 of Gemma 4's ~24 MB experts; two layers' worth of
    # 16 experts (~384 MB total) won't fit, so eviction is forced.
    cache = ExpertCache(max_memory_mb=256, verbose=False, metrics=metrics)
    manager = ExpertManager(store=store, cache=cache, metrics=metrics)
    rng = np.random.default_rng(0)
    hidden_dim = router.shape_for_layer(0)[1]
    initial = rng.standard_normal((1, hidden_dim)).astype(np.float32)
    runtime = Runtime(
        router=router,
        expert_manager=manager,
        backend=FakeBackend(),
        num_layers=2,
        initial_hidden=initial,
    )

    with np.errstate(over="ignore", under="ignore",
                     invalid="ignore", divide="ignore"):
        out = runtime.run()

    assert out.shape == (1, hidden_dim)
    assert np.all(np.isfinite(out))
    report = snapshot(cache, metrics)
    assert report.budget_respected, (
        f"real Gemma run exceeded budget: peak {report.peak_bytes} > "
        f"budget {report.budget_bytes}"
    )
    # 16 selected experts total across the two layers, but the small
    # budget forces some evictions before layer 1 finishes loading.
    assert report.evictions > 0
    assert report.cache_misses == 16


# --- Step 17E: KV + joined runtime memory reports ----------------------

def _kv_array(seq_len: int, value: float = 0.0):
    """(1, num_kv_heads=2, seq_len, head_dim=4) ndarray = 32 * seq_len bytes."""
    return np.full((1, 2, seq_len, 4), value, dtype=np.float32)


def test_kv_snapshot_empty_cache_is_zero_bytes():
    """A fresh KVCache reports zero per-layer and zero total."""
    cache = KVCache(num_layers=3)
    report = kv_snapshot(cache)
    assert isinstance(report, KVMemoryReport)
    assert report.per_layer_bytes == (0, 0, 0)
    assert report.total_bytes == 0


def test_kv_snapshot_per_layer_bytes_match_appended_arrays():
    """Append to two of three layers; the unused layer reports 0."""
    cache = KVCache(num_layers=3)
    # Layer 0: 2 seq positions ⇒ 2 tensors × 2 × 2 × 4 floats × 4 bytes = 128.
    cache.append(0, _kv_array(2), _kv_array(2))
    # Layer 2: 1 position ⇒ 64 bytes.
    cache.append(2, _kv_array(1), _kv_array(1))
    report = kv_snapshot(cache)
    assert report.per_layer_bytes == (128, 0, 64)
    assert report.total_bytes == 192


def test_kv_snapshot_total_matches_kvcache_property():
    """The report's total must agree with KVCache.total_bytes."""
    cache = KVCache(num_layers=2)
    cache.append(0, _kv_array(3), _kv_array(3))
    cache.append(1, _kv_array(5), _kv_array(5))
    report = kv_snapshot(cache)
    assert report.total_bytes == cache.total_bytes


def test_runtime_snapshot_total_is_sum_of_expert_plus_kv():
    """RuntimeMemoryReport.total = expert.current_bytes + kv.total_bytes."""
    metrics = Metrics()
    cache = ExpertCache(max_memory_mb=1, verbose=False, metrics=metrics)
    cache.put((0, 0), _expert(1))   # _BYTES_PER_EXPERT = 64
    kv = KVCache(num_layers=1)
    kv.append(0, _kv_array(1), _kv_array(1))   # 64 bytes total

    report = runtime_snapshot(cache, metrics, kv)
    assert isinstance(report, RuntimeMemoryReport)
    assert report.expert_cache_bytes == _BYTES_PER_EXPERT
    assert report.kv_cache_bytes == 64
    assert report.total_runtime_bytes == _BYTES_PER_EXPERT + 64


def test_runtime_snapshot_drills_into_expert_and_kv_components():
    """`.expert` is the MemoryReport, `.kv` is the KVMemoryReport — round-trip."""
    metrics = Metrics()
    cache = ExpertCache(max_memory_mb=1, verbose=False, metrics=metrics)
    cache.put((0, 0), _expert(1))
    metrics.on_foreground_hit((0, 0))
    metrics.on_foreground_miss((0, 1))
    kv = KVCache(num_layers=2)
    kv.append(1, _kv_array(2), _kv_array(2))

    report = runtime_snapshot(cache, metrics, kv)
    # .expert mirrors snapshot(cache, metrics) field-for-field.
    expert_only = snapshot(cache, metrics)
    assert report.expert == expert_only
    assert report.expert.cache_hits == 1
    assert report.expert.cache_misses == 1
    # .kv mirrors kv_snapshot(kv) field-for-field.
    assert report.kv == kv_snapshot(kv)
    assert report.kv.per_layer_bytes == (0, 128)
