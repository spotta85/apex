"""Tests for Runtime + GateRouter integration.

Sets up a tiny on-disk MoE world (gate + expert files) per test so we
can exercise the runtime end-to-end without depending on the real
experts/ directory or scripts/make_fake_moe_weights.py output.
"""

import json

import numpy as np
import pytest

from apex.expert_cache import ExpertCache
from apex.expert_manager import ExpertManager
from apex.expert_store import ExpertStore
from apex.gate_router import GateRouter
from apex.predictor import MarkovPredictor
from apex.prefetcher import AsyncPrefetcher
from apex.backends.fake_backend import FakeBackend
from apex.runtime import Runtime
from apex.trace_recorder import TraceRecorder
from tests._world import HIDDEN, write_npy_world


# Recording wrapper around GateRouter so tests can inspect what
# (layer, hidden_state) the runtime fed it.
class RecordingGateRouter:
    name = "numpy"  # runtime prints "router backend: <name>"

    def __init__(self, inner):
        self.inner = inner
        self.calls = []

    def route(self, layer_id, hidden_state):
        self.calls.append((layer_id, hidden_state.copy()))
        return self.inner.route(layer_id, hidden_state)


def test_runtime_passes_hidden_state_to_router(tmp_path):
    # Runtime should call router.route(layer_id, hidden_state) for each
    # layer (twice for layer 0 because it also peeks for prefetch
    # planning), with shape (1, HIDDEN).
    root = write_npy_world(tmp_path)
    router = RecordingGateRouter(GateRouter(root, top_k=2))
    cache = ExpertCache(max_memory_mb=1, verbose=False)
    manager = ExpertManager(ExpertStore(root), cache)

    runtime = Runtime(router=router, expert_manager=manager, backend=FakeBackend())
    runtime.run()

    # 2 layers x [1 normal call] + 1 next-layer peek for prefetch = 3 calls.
    assert len(router.calls) == 3
    layers_seen = [c[0] for c in router.calls]
    assert layers_seen == [0, 1, 1]
    for _layer, hidden in router.calls:
        assert hidden.shape == (1, HIDDEN)
        assert hidden.dtype == np.float32


def test_selected_experts_are_loaded(tmp_path):
    # Whatever the gate picks for each layer should end up in the cache.
    root = write_npy_world(tmp_path)
    router = GateRouter(root, top_k=2)
    cache = ExpertCache(max_memory_mb=1, verbose=False)
    manager = ExpertManager(ExpertStore(root), cache)

    # Pre-compute what the router will pick for layer 0 (input is ones).
    hidden0 = np.ones((1, HIDDEN), dtype="float32")
    layer0_picks = router.route(0, hidden0)

    runtime = Runtime(router=router, expert_manager=manager, backend=FakeBackend())
    runtime.run()

    for expert_id in layer0_picks:
        assert cache.has((0, expert_id))


def test_trace_recorder_captures_gate_selections(tmp_path):
    # The recorded transition's current_experts must equal what the gate
    # actually picked for layer 0, not some hardcoded value.
    root = write_npy_world(tmp_path)
    router = GateRouter(root, top_k=2)
    cache = ExpertCache(max_memory_mb=1, verbose=False)
    manager = ExpertManager(ExpertStore(root), cache)
    trace_path = tmp_path / "trace.jsonl"
    recorder = TraceRecorder(trace_path)

    # What the gate will pick for layer 0 with ones-hidden:
    expected_layer0 = router.route(0, np.ones((1, HIDDEN), dtype="float32"))

    runtime = Runtime(
        router=router,
        expert_manager=manager,
        backend=FakeBackend(),
        trace_recorder=recorder,
    )
    runtime.run()

    with open(trace_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    # Single transition: layer 0 -> layer 1.
    assert len(rows) == 1
    assert rows[0]["current_layer"] == 0
    assert rows[0]["current_experts"] == expected_layer0
    assert rows[0]["next_layer"] == 1
    # next_experts should be a list of two ints — the actual selection
    # for layer 1; we don't assert specific IDs since they depend on
    # the post-layer-0 hidden state.
    assert len(rows[0]["next_experts"]) == 2


def test_predictor_prefetches_next_layer_from_gate_traces(tmp_path):
    # End-to-end: train predictor on a recorded run, then run again with
    # prefetch active and confirm the predicted experts land in cache.
    root = write_npy_world(tmp_path)

    # Phase 1: warmup run with a fresh cache, just to capture the trace.
    warmup_cache = ExpertCache(max_memory_mb=1, verbose=False)
    warmup_manager = ExpertManager(ExpertStore(root), warmup_cache)
    trace_path = tmp_path / "warmup.jsonl"
    warmup_recorder = TraceRecorder(trace_path)
    warmup_runtime = Runtime(
        router=GateRouter(root, top_k=2),
        expert_manager=warmup_manager,
        backend=FakeBackend(),
        trace_recorder=warmup_recorder,
    )
    warmup_runtime.run()

    # Phase 2: train + run with predictor + real prefetcher.
    predictor = MarkovPredictor()
    predictor.fit(trace_path)

    # Read what was actually recorded so we know which layer-1 experts
    # should land in the cache.
    with open(trace_path) as f:
        record = json.loads(f.readline())
    expected_layer1 = record["next_experts"]

    cache = ExpertCache(max_memory_mb=1, verbose=False)
    manager = ExpertManager(ExpertStore(root), cache)
    prefetcher = AsyncPrefetcher(manager, verbose=False)
    try:
        runtime = Runtime(
            router=GateRouter(root, top_k=2),
            expert_manager=manager,
            backend=FakeBackend(),
            prefetcher=prefetcher,
            predictor=predictor,
        )
        runtime.run()

        for expert_id in expected_layer1:
            assert cache.has((1, expert_id))
    finally:
        prefetcher.shutdown()
