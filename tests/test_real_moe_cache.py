"""Tests for step 10: real MoE cache integration.

Two flavors of test live here:

  1. **Synthetic-NVFP4 tests** (always run) — build a tiny hand-coded
     NVFP4 expert checkpoint (4 experts, hidden=16, intermediate=32),
     wire the full DequantizingExpertStore + ExpertCache + ExpertManager
     stack against it, and run two passes through ``run_moe_layer``.
     These exercise the cache integration mechanics end-to-end without
     a multi-GB model download.
  2. **Real-checkpoint test** (skipped unless `REAL_MOE_PATH` env var
     points at a Gemma 4 NVFP4 directory) — same flow against actual
     weights. Confirms the synthetic tests' contract holds when the
     bundle dimensions, sidecar shapes, and routing are realistic.

The contract every test here enforces:

  - First pass through a set of experts ⇒ cache_misses += N, hits stay 0.
  - Second pass ⇒ cache_hits += N, no new misses, no evictions.
  - Outputs are finite both times AND bit-identical (deterministic
    cache means same input + same cached weights ⇒ same output).
  - The cache stores the *dequantized* (fp32 ndarray) bundle, not
    the QuantizedTensor bundle — so cache hits skip dequant entirely.
"""

import numpy as np

from apex.backends.fake_backend import FakeBackend
from apex.dequant import (
    DequantizingExpertStore,
    dequantize_bundle,
)
from apex.expert_cache import ExpertCache
from apex.expert_manager import ExpertManager
from apex.metrics import Metrics
from apex.moe_layer import run_moe_layer
from apex.quantized_safetensors_store import QuantizedSafetensorsExpertStore
from apex.types import ExpertWeights
from apex.weight_layout import FakeMoELayout, get_layout
from tests._synthetic_nvfp4 import (
    build_one_layer_checkpoint,
    real_layout_and_quant_store,
    real_checkpoint_path,
    skip_if_no_real_checkpoint,
)


def _make_synthetic_stack(tmp_path, num_experts: int = 4, hidden: int = 16,
                          intermediate: int = 32, max_memory_mb: float = 4):
    """Wire the standard demo stack against a synthetic NVFP4 checkpoint."""
    layout = FakeMoELayout()
    path = tmp_path / "synthetic_nvfp4.safetensors"
    build_one_layer_checkpoint(path, layout, num_experts, hidden, intermediate)

    quant_store = QuantizedSafetensorsExpertStore(str(path), layout)
    store = DequantizingExpertStore(quant_store, verbose=False)
    metrics = Metrics()
    cache = ExpertCache(max_memory_mb=max_memory_mb, verbose=False, metrics=metrics)
    manager = ExpertManager(store=store, cache=cache, metrics=metrics)
    backend = FakeBackend()
    return manager, cache, metrics, backend, hidden


# --- DequantizingExpertStore unit ------------------------------------------

def test_dequantizing_store_returns_fp32_bundle(tmp_path):
    # The whole point of the wrapper: cache hits should hit fp32, not
    # NVFP4-packed bundles. Without this, every cache hit would still
    # pay the ~60 ms NVFP4 decode cost.
    layout = FakeMoELayout()
    path = tmp_path / "single_expert.safetensors"
    build_one_layer_checkpoint(path, layout, num_experts=1, hidden=16, intermediate=32)

    quant_store = QuantizedSafetensorsExpertStore(str(path), layout)
    store = DequantizingExpertStore(quant_store)
    bundle = store.load(0, 0)

    assert isinstance(bundle, ExpertWeights)
    for proj in ("gate_proj", "up_proj", "down_proj"):
        arr = getattr(bundle, proj)
        assert isinstance(arr, np.ndarray), f"{proj} should be ndarray, got {type(arr)}"
        assert arr.dtype == np.float32
    # All-ones expert (by construction) — every projection should
    # dequantize to ones.
    np.testing.assert_array_equal(bundle.gate_proj, np.ones_like(bundle.gate_proj))


def test_dequantizing_store_verbose_prints_per_expert(tmp_path, capsys):
    layout = FakeMoELayout()
    path = tmp_path / "two_experts.safetensors"
    build_one_layer_checkpoint(path, layout, num_experts=2, hidden=16, intermediate=32)

    store = DequantizingExpertStore(
        QuantizedSafetensorsExpertStore(str(path), layout), verbose=True,
    )
    store.load(0, 1)
    out = capsys.readouterr().out
    # Trace line follows the demo spec exactly.
    assert "dequantized expert 1" in out


# --- run_moe_layer + ExpertManager integration -----------------------------

def test_first_run_misses_then_second_run_hits(tmp_path):
    """The headline assertion of step 10: same input, different cache state."""
    manager, cache, metrics, backend, hidden_dim = _make_synthetic_stack(
        tmp_path, num_experts=4,
    )

    selected = [0, 1, 2, 3]  # use all 4 experts for routing-independent test
    weights = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    hidden = np.ones((1, hidden_dim), dtype=np.float32)

    expert_loader = lambda eid: manager.ensure_loaded(0, eid)

    # --- first run: 4 misses, 0 hits.
    out_first = run_moe_layer(hidden, selected, weights, expert_loader, backend)
    assert metrics.cache_misses == 4
    assert metrics.cache_hits == 0
    assert metrics.evictions == 0
    assert out_first.shape == (1, hidden_dim)
    assert np.all(np.isfinite(out_first))

    # --- second run, same selection: 4 hits, no new misses.
    out_second = run_moe_layer(hidden, selected, weights, expert_loader, backend)
    assert metrics.cache_misses == 4, "should not re-load any experts"
    assert metrics.cache_hits == 4
    assert metrics.evictions == 0
    assert np.all(np.isfinite(out_second))

    # --- determinism: identical inputs through identical cache ⇒ bit-equal.
    np.testing.assert_array_equal(out_first, out_second)


def test_cache_holds_dequantized_bundles_after_first_run(tmp_path):
    """Cache contents must be fp32, otherwise the perf claim is a lie."""
    manager, cache, metrics, backend, hidden_dim = _make_synthetic_stack(
        tmp_path, num_experts=2,
    )
    expert_loader = lambda eid: manager.ensure_loaded(0, eid)
    run_moe_layer(
        np.ones((1, hidden_dim), dtype=np.float32),
        [0, 1], np.array([0.5, 0.5], dtype=np.float32),
        expert_loader, backend,
    )
    for eid in (0, 1):
        cached = cache.try_get((0, eid))
        assert isinstance(cached, ExpertWeights)
        assert isinstance(cached.gate_proj, np.ndarray)
        assert cached.gate_proj.dtype == np.float32


def test_cache_eviction_reverts_to_miss(tmp_path):
    """If the cache budget evicts an expert, a re-fetch should be a miss again.

    Confirms the metrics machinery is wired correctly to the cache,
    not just incremented optimistically by the manager.
    """
    # 4 experts, ~32 KiB each at fp32 (gate+up+down each (32,16)+(16,32) at f32).
    # Tight budget so two experts barely fit.
    manager, cache, metrics, backend, hidden_dim = _make_synthetic_stack(
        tmp_path, num_experts=4, max_memory_mb=0.012,  # ~12 KiB ⇒ holds 1 expert
    )
    expert_loader = lambda eid: manager.ensure_loaded(0, eid)
    hidden = np.ones((1, hidden_dim), dtype=np.float32)

    # First pass: load 4 experts; with a tight budget, earlier ones get
    # evicted as later ones come in.
    run_moe_layer(
        hidden, [0, 1, 2, 3], np.full(4, 0.25, dtype=np.float32),
        expert_loader, backend,
    )
    assert metrics.cache_misses == 4
    assert metrics.evictions >= 1, (
        "tight budget should have forced at least one eviction"
    )

    # Re-fetch expert 0 — should miss again because it was evicted.
    misses_before = metrics.cache_misses
    manager.ensure_loaded(0, 0)
    assert metrics.cache_misses == misses_before + 1


def test_partial_cache_overlap_mixes_hits_and_misses(tmp_path):
    """Realistic case: second routing decision overlaps the first by some experts."""
    manager, cache, metrics, backend, hidden_dim = _make_synthetic_stack(
        tmp_path, num_experts=4,
    )
    expert_loader = lambda eid: manager.ensure_loaded(0, eid)
    hidden = np.ones((1, hidden_dim), dtype=np.float32)
    weights = np.array([0.5, 0.5], dtype=np.float32)

    # First call: experts {0, 1}.
    run_moe_layer(hidden, [0, 1], weights, expert_loader, backend)
    assert metrics.cache_misses == 2
    assert metrics.cache_hits == 0

    # Second call: experts {1, 2}. One overlap (1) ⇒ one new hit + one
    # new miss. Tests that the cache is keyed on (layer, expert) and
    # not e.g. per-call.
    run_moe_layer(hidden, [1, 2], weights, expert_loader, backend)
    assert metrics.cache_misses == 3, "expert 2 should be a fresh miss"
    assert metrics.cache_hits == 1, "expert 1 should be a cache hit"


def test_cache_print_lines_match_demo_spec(tmp_path, capsys):
    """The demo trace's contract: cache miss → dequant → finalize.

    Locks in the output format. Future refactors that change the trace
    line wording will need to update this test (and the demo) together
    so they don't drift.
    """
    layout = FakeMoELayout()
    path = tmp_path / "trace.safetensors"
    build_one_layer_checkpoint(path, layout, num_experts=2, hidden=16, intermediate=32)

    metrics = Metrics()
    cache = ExpertCache(max_memory_mb=4, verbose=False, metrics=metrics)
    store = DequantizingExpertStore(
        QuantizedSafetensorsExpertStore(str(path), layout), verbose=True,
    )
    manager = ExpertManager(store=store, cache=cache, metrics=metrics)

    capsys.readouterr()  # drop any prior output
    manager.ensure_loaded(0, 0)
    out = capsys.readouterr().out
    # Order matters: "cache miss" prints first (the manager logs the
    # miss before delegating to the store), then the store's dequant
    # trace fires when load() runs.
    miss_idx = out.index("cache miss: loading layer 0 expert 0")
    deq_idx = out.index("dequantized expert 0")
    assert miss_idx < deq_idx, f"trace order wrong:\n{out}"

    capsys.readouterr()
    manager.ensure_loaded(0, 0)  # second call ⇒ hit, no dequant
    out = capsys.readouterr().out
    assert "cache hit: layer 0 expert 0" in out
    assert "dequantized" not in out


# --- real-checkpoint integration (skipped unless env var is set) -----------

@skip_if_no_real_checkpoint()
def test_real_gemma_layer_caches_top_8_then_reuses():
    """End-to-end on real weights: route → load+cache 8 → reuse 8."""
    from apex.real_router import MultiLayerRealRouter

    path = real_checkpoint_path()
    layout, quant_store = real_layout_and_quant_store(path)
    layer_id = 0
    top_k = 8

    router = MultiLayerRealRouter(path, layout, top_k=top_k)
    hidden_dim = router.shape_for_layer(layer_id)[1]
    rng = np.random.default_rng(0)
    hidden = rng.standard_normal((1, hidden_dim)).astype(np.float32)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        selected, routing_weights = router.route(layer_id, hidden)

    store = DequantizingExpertStore(quant_store, verbose=False)
    metrics = Metrics()
    cache = ExpertCache(max_memory_mb=256, verbose=False, metrics=metrics)
    manager = ExpertManager(store=store, cache=cache, metrics=metrics)
    backend = FakeBackend()

    expert_loader = lambda eid: manager.ensure_loaded(layer_id, eid)

    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        out_first = run_moe_layer(
            hidden, selected, routing_weights, expert_loader, backend,
        )
    assert metrics.cache_misses == top_k
    assert metrics.cache_hits == 0
    assert metrics.evictions == 0
    assert out_first.shape == (1, hidden_dim)
    assert np.all(np.isfinite(out_first))

    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        out_second = run_moe_layer(
            hidden, selected, routing_weights, expert_loader, backend,
        )
    assert metrics.cache_misses == top_k, "second run must not re-load"
    assert metrics.cache_hits == top_k
    assert metrics.evictions == 0
    assert np.all(np.isfinite(out_second))
    np.testing.assert_array_equal(out_first, out_second)
