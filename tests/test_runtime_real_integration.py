"""Tests for Step 12: Runtime drives router-weighted real-checkpoint MoE."""

import json

import numpy as np

from apex.backends.fake_backend import FakeBackend
from apex.dequant import DequantizingExpertStore
from apex.expert_cache import ExpertCache
from apex.expert_manager import ExpertManager
from apex.metrics import Metrics
from apex.quantized_safetensors_store import QuantizedSafetensorsExpertStore
from apex.runtime import Runtime
from apex.types import ExpertWeights
from apex.weight_layout import FakeMoELayout
from tests._synthetic_nvfp4 import (
    build_one_layer_checkpoint,
    real_layout_and_quant_store,
    real_checkpoint_path,
    skip_if_no_real_checkpoint,
    write_safetensors,
)


class _StubRouter:
    name = "numpy"

    def __init__(self, plan):
        self.plan = plan

    def route(self, layer_id, hidden):
        return self.plan[layer_id]


class _TinyManager:
    def __init__(self, experts):
        self.experts = experts

    def ensure_loaded(self, layer_id, expert_id):
        return self.experts[(layer_id, expert_id)]


def _expert(down_scale: float):
    gate = np.ones((2, 2), dtype=np.float32)
    up = np.ones((2, 2), dtype=np.float32)
    down = np.full((2, 2), down_scale, dtype=np.float32)
    return ExpertWeights(gate_proj=gate, up_proj=up, down_proj=down)


def test_runtime_uses_router_weights_for_combine():
    """Router weights must drive combination when supplied."""
    backend = FakeBackend()
    hidden = np.ones((1, 2), dtype=np.float32)
    manager = _TinyManager({(0, 0): _expert(1.0), (0, 1): _expert(2.0)})
    router = _StubRouter({0: ([0, 1], [0.9, 0.1])})

    out = Runtime(
        router=router,
        expert_manager=manager,
        backend=backend,
        num_layers=1,
        initial_hidden=hidden,
    ).run()

    out0 = backend.run_expert(hidden, manager.ensure_loaded(0, 0))
    out1 = backend.run_expert(hidden, manager.ensure_loaded(0, 1))
    expected = 0.9 * out0 + 0.1 * out1
    np.testing.assert_allclose(out, expected, rtol=1e-6)


def test_runtime_uniform_when_router_returns_none():
    """weights=None preserves the old uniform-mean Runtime behavior."""
    backend = FakeBackend()
    hidden = np.ones((1, 2), dtype=np.float32)
    manager = _TinyManager({(0, 0): _expert(1.0), (0, 1): _expert(2.0)})
    router = _StubRouter({0: ([0, 1], None)})

    out = Runtime(
        router=router,
        expert_manager=manager,
        backend=backend,
        num_layers=1,
        initial_hidden=hidden,
    ).run()

    out0 = backend.run_expert(hidden, manager.ensure_loaded(0, 0))
    out1 = backend.run_expert(hidden, manager.ensure_loaded(0, 1))
    np.testing.assert_array_equal(out, (out0 + out1) / 2)


def test_runtime_accepts_arbitrary_hidden_dim(tmp_path):
    """Runtime no longer hardcodes hidden=8."""
    layout = FakeMoELayout()
    path = tmp_path / "ck.safetensors"
    build_one_layer_checkpoint(
        path, layout, num_experts=2, hidden=64, intermediate=128,
    )
    store = DequantizingExpertStore(
        QuantizedSafetensorsExpertStore(str(path), layout), verbose=False,
    )
    cache = ExpertCache(max_memory_mb=4, verbose=False)
    manager = ExpertManager(store=store, cache=cache)
    initial = np.zeros((1, 64), dtype=np.float32)
    router = _StubRouter({0: ([0, 1], None)})

    runtime = Runtime(
        router=router,
        expert_manager=manager,
        backend=FakeBackend(),
        num_layers=1,
        initial_hidden=initial,
    )
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        out = runtime.run()

    assert out.shape == (1, 64)
    assert np.all(np.isfinite(out))


def test_multi_layer_real_router_routes_from_sharded_index(tmp_path):
    from apex.real_router import MultiLayerRealRouter

    layout = FakeMoELayout()
    router_name = layout.gate_weight(0)
    router_weight = np.array(
        [
            [1, 0, 0, 0],
            [5, 0, 0, 0],
            [3, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    shard = tmp_path / "router.safetensors"
    write_safetensors(
        shard, [(router_name, "F32", list(router_weight.shape), router_weight.tobytes())],
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": {router_name: shard.name}})
    )

    router = MultiLayerRealRouter(tmp_path, layout, top_k=2)
    ids, weights = router.route(0, np.array([[1, 0, 0, 0]], dtype=np.float32))

    assert router.shape_for_layer(0) == (3, 4)
    assert ids == [1, 2]
    np.testing.assert_allclose(sum(weights), 1.0, rtol=1e-6)


def test_real_router_dequantizes_gemma_hf_affine_router(tmp_path):
    from apex.real_router import MultiLayerRealRouter
    from apex.weight_layout import GemmaHFMoELayout

    layout = GemmaHFMoELayout()
    names = layout.router_tensors(0)
    router_weight = np.array([[1], [5], [3]], dtype=np.uint32)
    scales = np.ones((3, 1), dtype=np.float32)
    biases = np.zeros((3, 1), dtype=np.float32)
    shard = tmp_path / "router.safetensors"
    items = [
        (names["weight"], "U32", list(router_weight.shape), router_weight.tobytes()),
        (names["scales"], "F32", list(scales.shape), scales.tobytes()),
        (names["biases"], "F32", list(biases.shape), biases.tobytes()),
    ]
    write_safetensors(shard, items)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": {
            name: shard.name for name, *_ in items
        }})
    )

    router = MultiLayerRealRouter(tmp_path, layout, top_k=2)
    ids, weights = router.route(0, np.array([[1, 0, 0, 0]], dtype=np.float32))

    assert router.shape_for_layer(0) == (3, 4)
    assert ids == [1, 2]
    np.testing.assert_allclose(sum(weights), 1.0, rtol=1e-6)


@skip_if_no_real_checkpoint()
def test_real_gemma_two_layer_runtime_run_then_run():
    """Real Gemma 4 NVFP4 goes through Runtime, not just smoke scripts."""
    from apex.real_router import MultiLayerRealRouter

    path = real_checkpoint_path()
    layout, quant_store = real_layout_and_quant_store(path)
    router = MultiLayerRealRouter(path, layout, top_k=8)
    store = DequantizingExpertStore(quant_store, verbose=False)
    metrics = Metrics()
    cache = ExpertCache(max_memory_mb=512, verbose=False, metrics=metrics)
    manager = ExpertManager(store=store, cache=cache, metrics=metrics)
    hidden_dim = router.shape_for_layer(0)[1]
    rng = np.random.default_rng(0)
    initial = rng.standard_normal((1, hidden_dim)).astype(np.float32)
    runtime = Runtime(
        router=router,
        expert_manager=manager,
        backend=FakeBackend(),
        num_layers=2,
        initial_hidden=initial,
    )

    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        out_first = runtime.run()
    assert metrics.cache_misses == 16
    assert metrics.cache_hits == 0
    assert metrics.evictions == 0
    assert out_first.shape == (1, hidden_dim)
    assert np.all(np.isfinite(out_first))

    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        out_second = runtime.run()
    assert metrics.cache_misses == 16
    assert metrics.cache_hits == 16
    assert metrics.evictions == 0
    np.testing.assert_array_equal(out_first, out_second)
