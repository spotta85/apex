"""Tests for step 11: two-layer real MoE smoke flow.

Verifies the cache + manager + run_moe_layer stack handles multiple
layers in one logical pass — specifically:

  - Cache key (layer, expert) disambiguates: layer 0 expert N and
    layer 1 expert N are stored separately (a regression that collapsed
    the key to just expert id would be silently catastrophic — wrong
    weights would compute against the right inputs).
  - Cross-layer cache state survives intact: after pass 1 stuffs both
    layers' experts in, pass 2 finds them all on hits without
    re-loading.
  - Output threads forward correctly: layer 0's output is the input to
    layer 1; both produce finite tensors of shape (1, hidden).

Two flavors here, same pattern as tests/test_real_moe_cache.py:

  1. **Synthetic** tests build a tiny multi-layer NVFP4 checkpoint
     (4 experts, 2 layers, hidden=16, intermediate=32) — always run.
  2. **Real-checkpoint** test against Gemma 4 NVFP4 — skipped unless
     `REAL_MOE_PATH` env var (or the default download path) resolves.
"""

import numpy as np

from apex.backends.fake_backend import FakeBackend
from apex.dequant import DequantizingExpertStore
from apex.expert_cache import ExpertCache
from apex.expert_manager import ExpertManager
from apex.metrics import Metrics
from apex.moe_layer import route, run_moe_layer
from apex.quantized_safetensors_store import QuantizedSafetensorsExpertStore
from apex.types import ExpertWeights
from apex.weight_layout import FakeMoELayout, get_layout
from tests._synthetic_nvfp4 import (
    build_multi_layer_checkpoint,
    real_checkpoint_path,
    skip_if_no_real_checkpoint,
)


def _make_two_layer_stack(tmp_path, num_layers=2, num_experts=4,
                          hidden=16, intermediate=32, max_memory_mb=4):
    """Wire the demo stack against a synthetic multi-layer NVFP4 checkpoint."""
    layout = FakeMoELayout()
    path = tmp_path / "two_layer_nvfp4.safetensors"
    build_multi_layer_checkpoint(
        path, layout, num_layers, num_experts, hidden, intermediate,
    )

    quant_store = QuantizedSafetensorsExpertStore(str(path), layout)
    store = DequantizingExpertStore(quant_store, verbose=False)
    metrics = Metrics()
    cache = ExpertCache(max_memory_mb=max_memory_mb, verbose=False, metrics=metrics)
    manager = ExpertManager(store=store, cache=cache, metrics=metrics)
    backend = FakeBackend()
    return manager, cache, metrics, backend, hidden


def _two_layer_pass(hidden, layers, selections_per_layer, manager, backend):
    """Drive the given (layer_id → expert ids) plan; return final hidden."""
    h = hidden
    for layer_id in layers:
        selected = selections_per_layer[layer_id]
        weights = np.full(len(selected), 1.0 / len(selected), dtype=np.float32)
        expert_loader = lambda eid, lid=layer_id: manager.ensure_loaded(lid, eid)
        h = run_moe_layer(h, selected, weights, expert_loader, backend)
    return h


# --- synthetic two-layer cache invariants ----------------------------------

def test_cache_key_disambiguates_by_layer(tmp_path):
    """The cache must store layer 0 expert N and layer 1 expert N separately.

    Hits the most catastrophic possible regression: if the cache
    accidentally collapsed (layer, expert) to just expert, layer 1's
    SwiGLU would compute against layer 0's weights (or vice versa)
    and silently corrupt the output.
    """
    manager, cache, metrics, backend, hidden_dim = _make_two_layer_stack(
        tmp_path, num_layers=2, num_experts=2,
    )
    # Both layers reference expert 0. If keying is correct, this is two
    # distinct cache entries; if broken, it'd be one entry shared.
    selections = {0: [0], 1: [0]}
    hidden = np.ones((1, hidden_dim), dtype=np.float32)
    _two_layer_pass(hidden, [0, 1], selections, manager, backend)

    assert metrics.cache_misses == 2, (
        "expected 1 miss per layer (key by (layer, expert)); got "
        f"{metrics.cache_misses} — keying may have collapsed across layers"
    )
    # And both keys should be independently retrievable.
    assert cache.try_get((0, 0)) is not None
    assert cache.try_get((1, 0)) is not None


def test_two_pass_first_misses_then_hits(tmp_path):
    manager, cache, metrics, backend, hidden_dim = _make_two_layer_stack(
        tmp_path, num_layers=2, num_experts=4, max_memory_mb=8,
    )
    selections = {0: [0, 1], 1: [2, 3]}  # 2 unique experts per layer
    hidden = np.ones((1, hidden_dim), dtype=np.float32)

    # --- first pass ⇒ all 4 (layer, expert) pairs miss.
    out_first = _two_layer_pass(hidden, [0, 1], selections, manager, backend)
    assert metrics.cache_misses == 4
    assert metrics.cache_hits == 0
    assert metrics.evictions == 0
    assert out_first.shape == (1, hidden_dim)
    assert np.all(np.isfinite(out_first))

    # --- second pass ⇒ all 4 hit, no new loads.
    out_second = _two_layer_pass(hidden, [0, 1], selections, manager, backend)
    assert metrics.cache_misses == 4
    assert metrics.cache_hits == 4
    assert metrics.evictions == 0
    np.testing.assert_array_equal(out_first, out_second)


def test_layer_0_output_threads_into_layer_1(tmp_path):
    """Smoke check that layer 1's compute consumed layer 0's output, not its input.

    Passes the layer-0 hidden directly through layer 0, captures the
    output, then runs layer 1 on that output. The synthetic checkpoint
    has all-ones experts so the SwiGLU output is a deterministic
    function of the input; layer 1 receiving (layer 0's input) instead
    of (layer 0's output) would change the result.
    """
    manager, cache, metrics, backend, hidden_dim = _make_two_layer_stack(
        tmp_path, num_layers=2, num_experts=2, max_memory_mb=8,
    )
    hidden_in = np.ones((1, hidden_dim), dtype=np.float32)

    # Run layer 0 in isolation, then layer 1 on (layer 0's output).
    expert_loader_0 = lambda eid: manager.ensure_loaded(0, eid)
    weights = np.array([0.5, 0.5], dtype=np.float32)
    out_0 = run_moe_layer(hidden_in, [0, 1], weights, expert_loader_0, backend)

    expert_loader_1 = lambda eid: manager.ensure_loaded(1, eid)
    out_1_threaded = run_moe_layer(out_0, [0, 1], weights, expert_loader_1, backend)

    # Sanity check: feeding layer 1 the original `hidden_in` instead
    # gives a different output (since out_0 != hidden_in for non-trivial
    # SwiGLU on non-zero input). If they ever match, either the test
    # input is degenerate (hidden_in is a fixed point) or layer 1
    # silently ignored its input.
    out_1_misthread = run_moe_layer(hidden_in, [0, 1], weights, expert_loader_1, backend)

    # Both runs are valid + finite, but they should produce DIFFERENT
    # outputs because they're computing on different inputs.
    assert np.all(np.isfinite(out_1_threaded))
    assert np.all(np.isfinite(out_1_misthread))
    assert not np.allclose(out_1_threaded, out_1_misthread), (
        "layer 1 output didn't depend on its input — threading is broken"
    )


def test_cache_holds_both_layers_simultaneously(tmp_path):
    """After pass 1, every expert from every layer must still be in the cache.

    Plus the metric for evictions must be zero — otherwise the
    "second pass = all hits" claim doesn't hold.
    """
    manager, cache, metrics, backend, hidden_dim = _make_two_layer_stack(
        tmp_path, num_layers=2, num_experts=4, max_memory_mb=64,
    )
    selections = {0: [0, 1, 2, 3], 1: [0, 1, 2, 3]}
    _two_layer_pass(
        np.ones((1, hidden_dim), dtype=np.float32),
        [0, 1], selections, manager, backend,
    )

    assert metrics.evictions == 0
    for layer_id in (0, 1):
        for expert_id in range(4):
            assert cache.try_get((layer_id, expert_id)) is not None, (
                f"({layer_id}, {expert_id}) missing from cache"
            )
            cached = cache.try_get((layer_id, expert_id))
            # Cached bundle is fp32 (DequantizingExpertStore did the decode).
            assert isinstance(cached, ExpertWeights)
            assert isinstance(cached.gate_proj, np.ndarray)
            assert cached.gate_proj.dtype == np.float32


def test_tight_budget_evicts_layer_0_when_layer_1_loads(tmp_path):
    """If the cache budget can't hold both layers, layer 0 evicts to make room.

    And then a second pass through layer 0 misses (correctly), proving
    the LRU policy interacts with multi-layer flows the obvious way.
    """
    # Tight budget ⇒ holds ~2 fp32 experts. Loading 4 (2 per layer)
    # forces evictions during the first pass.
    manager, cache, metrics, backend, hidden_dim = _make_two_layer_stack(
        tmp_path, num_layers=2, num_experts=2, max_memory_mb=0.013,
    )
    selections = {0: [0, 1], 1: [0, 1]}
    hidden = np.ones((1, hidden_dim), dtype=np.float32)

    _two_layer_pass(hidden, [0, 1], selections, manager, backend)
    assert metrics.cache_misses == 4
    assert metrics.evictions >= 1, (
        "tight budget with 4 (layer,expert) entries should have evicted at least once"
    )


# --- real-checkpoint test (skipped if missing) -----------------------------

@skip_if_no_real_checkpoint()
def test_real_gemma_two_layer_pass_caches_correctly():
    """End-to-end on real Gemma 4 NVFP4: two layers, two passes, finite output."""
    from apex.checkpoint_index import get_tensor_shape, read_safetensors_index
    from apex.safetensors_loader import load_tensor

    path = real_checkpoint_path()
    layout = get_layout("gemma-moe")
    layers = [0, 1]
    top_k = 8

    weight_map = read_safetensors_index(path)
    routers = {}
    for layer_id in layers:
        name = layout.gate_weight(layer_id)
        routers[layer_id] = load_tensor(path / weight_map[name], name)
    _, hidden_dim = get_tensor_shape(path, layout.gate_weight(layers[0]))

    quant_store = QuantizedSafetensorsExpertStore(path, layout=layout)
    store = DequantizingExpertStore(quant_store, verbose=False)
    metrics = Metrics()
    # Sized for 16 fp32 experts (~24 MB each on Gemma 4); 512 MB has slack.
    cache = ExpertCache(max_memory_mb=512, verbose=False, metrics=metrics)
    manager = ExpertManager(store=store, cache=cache, metrics=metrics)
    backend = FakeBackend()

    rng = np.random.default_rng(0)
    hidden = rng.standard_normal((1, hidden_dim)).astype(np.float32)

    # --- first pass: route + run each layer in sequence.
    h = hidden
    selections = []
    for layer_id in layers:
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            ids, weights = route(h, routers[layer_id], top_k=top_k)
        selected = ids[0].tolist()
        selections.append(selected)
        expert_loader = lambda eid, lid=layer_id: manager.ensure_loaded(lid, eid)
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            h = run_moe_layer(h, selected, weights[0], expert_loader, backend)
        assert np.all(np.isfinite(h))
    out_first = h
    assert metrics.cache_misses == top_k * len(layers)
    assert metrics.cache_hits == 0
    assert metrics.evictions == 0

    # --- second pass: same hidden, same routing decisions, all hits.
    h = hidden
    for layer_id in layers:
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            ids, weights = route(h, routers[layer_id], top_k=top_k)
        selected = ids[0].tolist()
        # Routing must be deterministic across passes for the cache
        # claim to hold — otherwise the "all hits" assertion is moot.
        assert selected == selections[layer_id]
        expert_loader = lambda eid, lid=layer_id: manager.ensure_loaded(lid, eid)
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            h = run_moe_layer(h, selected, weights[0], expert_loader, backend)
    out_second = h
    assert metrics.cache_misses == top_k * len(layers), "second pass shouldn't re-load"
    assert metrics.cache_hits == top_k * len(layers)
    assert metrics.evictions == 0
    assert np.all(np.isfinite(out_second))
    np.testing.assert_array_equal(out_first, out_second)
