"""Tests for step 15: AsyncPrefetcher driving real-checkpoint Runtime.

Two layers of coverage:

  - Synthetic NVFP4 stack: Runtime + AsyncPrefetcher must produce
    foreground hits on layer 1 once the prefetcher has warmed it,
    and the metrics counters must agree (prefetch_hits == top_k *
    (num_layers - 1)).
  - Real Gemma 4 NVFP4 (skip if missing): the same invariants hold
    against actual ~24 MB experts loaded from the sharded checkpoint.

Both tests run with a comfortable cache budget so prefetch isn't
sabotaged by mid-run eviction. Verified prefetch+budget interaction
is the job of step 14's tests, not these.
"""

import numpy as np

from apex.backends.fake_backend import FakeBackend
from apex.dequant import DequantizingExpertStore
from apex.expert_cache import ExpertCache
from apex.expert_manager import ExpertManager
from apex.metrics import Metrics
from apex.prefetcher import AsyncPrefetcher
from apex.quantized_safetensors_store import QuantizedSafetensorsExpertStore
from apex.runtime import Runtime
from apex.weight_layout import FakeMoELayout
from tests._synthetic_nvfp4 import (
    build_multi_layer_checkpoint,
    real_checkpoint_path,
    real_layout_and_quant_store,
    skip_if_no_real_checkpoint,
)


class _StubRouter:
    """Deterministic per-layer plan. Returns (ids, weights) tuples.

    The Runtime calls route() twice per non-final layer (once to
    compute, once to peek for prefetch) — the same plan answers both
    because hidden state isn't inspected.
    """

    name = "numpy"

    def __init__(self, plan):
        self.plan = plan

    def route(self, layer_id, hidden):
        return self.plan[layer_id]


def test_prefetcher_warms_next_layer_in_synthetic_runtime(tmp_path):
    """Two-layer synthetic NVFP4 run: layer 1 served entirely from prefetch."""
    layout = FakeMoELayout()
    ckpt = tmp_path / "ck.safetensors"
    build_multi_layer_checkpoint(
        ckpt, layout, num_layers=2, num_experts=4,
        hidden=16, intermediate=32,
    )
    quant_store = QuantizedSafetensorsExpertStore(str(ckpt), layout=layout)
    store = DequantizingExpertStore(quant_store, verbose=False)
    metrics = Metrics()
    # Big enough to hold both layers' experts so prefetched layer 1
    # entries don't get evicted before layer 1 consumes them.
    cache = ExpertCache(max_memory_mb=4, verbose=False, metrics=metrics)
    manager = ExpertManager(store=store, cache=cache, metrics=metrics)
    prefetcher = AsyncPrefetcher(manager, verbose=False, metrics=metrics)

    # Layers route to disjoint expert sets so layer-1 cache hits prove
    # prefetch did its job (not layer-0 carryover).
    plan = {
        0: ([0, 1], [0.5, 0.5]),
        1: ([2, 3], [0.5, 0.5]),
    }
    runtime = Runtime(
        router=_StubRouter(plan),
        expert_manager=manager,
        backend=FakeBackend(),
        prefetcher=prefetcher,
        num_layers=2,
        initial_hidden=np.full((1, 16), 0.5, dtype=np.float32),
    )
    try:
        with np.errstate(over="ignore", under="ignore",
                         invalid="ignore", divide="ignore"):
            out = runtime.run()
    finally:
        prefetcher.shutdown()

    assert out.shape == (1, 16)
    assert np.all(np.isfinite(out))
    # Layer 0 -> 2 cold misses; layer 1 -> 2 prefetched hits.
    assert metrics.cache_misses == 2, metrics.cache_misses
    assert metrics.cache_hits == 2, metrics.cache_hits
    # Prefetcher saw layer 1's two experts and the foreground later
    # consumed both => two prefetch_hits credited.
    assert metrics.prefetch_scheduled == 2
    assert metrics.prefetch_hits == 2
    assert metrics.prefetch_hit_rate == 1.0


def test_prefetch_hit_rate_zero_when_predictor_misses(tmp_path):
    """Predictor that mispredicts contributes scheduled but no hits."""
    from apex.predictor import MarkovPredictor

    layout = FakeMoELayout()
    ckpt = tmp_path / "ck.safetensors"
    build_multi_layer_checkpoint(
        ckpt, layout, num_layers=2, num_experts=4,
        hidden=16, intermediate=32,
    )
    quant_store = QuantizedSafetensorsExpertStore(str(ckpt), layout=layout)
    store = DequantizingExpertStore(quant_store, verbose=False)
    metrics = Metrics()
    cache = ExpertCache(max_memory_mb=4, verbose=False, metrics=metrics)
    manager = ExpertManager(store=store, cache=cache, metrics=metrics)
    prefetcher = AsyncPrefetcher(manager, verbose=False, metrics=metrics)

    # Predictor pre-trained to claim layer 1 wants experts {0, 1}, but
    # the actual router asks for {2, 3} — wasted prefetches.
    trace_path = tmp_path / "wrong.jsonl"
    trace_path.write_text(
        '{"current_layer": 0, "current_experts": [0, 1], '
        '"next_layer": 1, "next_experts": [0, 1]}\n'
    )
    predictor = MarkovPredictor()
    predictor.fit(trace_path)

    plan = {
        0: ([0, 1], [0.5, 0.5]),
        1: ([2, 3], [0.5, 0.5]),
    }
    runtime = Runtime(
        router=_StubRouter(plan),
        expert_manager=manager,
        backend=FakeBackend(),
        prefetcher=prefetcher,
        predictor=predictor,
        num_layers=2,
        initial_hidden=np.full((1, 16), 0.5, dtype=np.float32),
    )
    try:
        with np.errstate(over="ignore", under="ignore",
                         invalid="ignore", divide="ignore"):
            runtime.run()
    finally:
        prefetcher.shutdown()

    # Predictor scheduled experts {0, 1} for layer 1; foreground asked
    # for {2, 3} so both prefetches were wasted (0 hits credited).
    assert metrics.prefetch_scheduled == 2
    assert metrics.prefetch_hits == 0
    assert metrics.prefetch_hit_rate == 0.0


@skip_if_no_real_checkpoint()
def test_real_gemma_prefetch_warms_layer1():
    """Two real Gemma 4 layers via Runtime + AsyncPrefetcher.

    Layer 0 cold-misses 8 experts; the prefetcher loads layer 1's 8
    experts in the background; layer 1 hits the cache on every lookup.
    """
    from apex.real_router import MultiLayerRealRouter

    path = real_checkpoint_path()
    layout, quant_store = real_layout_and_quant_store(path)
    router = MultiLayerRealRouter(path, layout, top_k=8)
    store = DequantizingExpertStore(quant_store, verbose=False)
    metrics = Metrics()
    # 768 MB easily holds 16 ~24 MB experts; no eviction interferes
    # with the prefetch hit-count we're measuring.
    cache = ExpertCache(max_memory_mb=768, verbose=False, metrics=metrics)
    manager = ExpertManager(store=store, cache=cache, metrics=metrics)
    prefetcher = AsyncPrefetcher(manager, verbose=False, metrics=metrics)
    rng = np.random.default_rng(0)
    hidden_dim = router.shape_for_layer(0)[1]
    initial = rng.standard_normal((1, hidden_dim)).astype(np.float32)

    runtime = Runtime(
        router=router,
        expert_manager=manager,
        backend=FakeBackend(),
        prefetcher=prefetcher,
        num_layers=2,
        initial_hidden=initial,
    )
    try:
        with np.errstate(over="ignore", under="ignore",
                         invalid="ignore", divide="ignore"):
            out = runtime.run()
    finally:
        prefetcher.shutdown()

    assert out.shape == (1, hidden_dim)
    assert np.all(np.isfinite(out))
    assert metrics.cache_misses == 8
    assert metrics.cache_hits == 8
    assert metrics.prefetch_scheduled == 8
    assert metrics.prefetch_hits == 8
    assert metrics.evictions == 0
