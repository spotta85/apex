"""Tests for Runtime + MarkovPredictor integration.

Covers three behaviors:
  - When the predictor returns a prediction, the prefetcher receives it.
  - When the predictor returns [], the runtime falls back to the router.
  - End-to-end: a trained predictor + real prefetcher leaves the
    next-layer experts in cache before the runtime needs them.
"""

import json

import numpy as np
import pytest

from apex.expert_cache import ExpertCache
from apex.expert_manager import ExpertManager
from apex.prefetcher import AsyncPrefetcher
from apex.predictor import MarkovPredictor
from apex.router import FakeRouter
from apex.backends.fake_backend import FakeBackend
from apex.runtime import Runtime
from tests._world import FakeStore


class RecordingPrefetcher:
    # Captures prefetch_many calls without spinning a real thread.
    # Lets us assert exactly what the runtime asked to prefetch.
    def __init__(self):
        self.scheduled = []

    def prefetch_many(self, layer_id, expert_ids):
        self.scheduled.append((layer_id, list(expert_ids)))

    def wait_for_layer(self, layer_id):
        # No background work to wait on.
        pass


# --- helpers --------------------------------------------------------------

def write_traces(tmp_path, lines):
    path = tmp_path / "traces.jsonl"
    with open(path, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    return path


def make_runtime(predictor=None, prefetcher=None):
    cache = ExpertCache(max_memory_mb=1)
    manager = ExpertManager(FakeStore(), cache)
    return Runtime(
        router=FakeRouter(),
        expert_manager=manager,
        backend=FakeBackend(),
        prefetcher=prefetcher,
        predictor=predictor,
    )


# --- tests ----------------------------------------------------------------

def test_predictor_output_is_used_for_prefetch(tmp_path):
    # Predictor predicts [9, 7] for (0, [1, 3], 1) — a deliberately
    # wrong-but-distinct answer so we can confirm the prefetcher saw
    # the *predictor's* choice, not the router's [0, 2].
    path = write_traces(tmp_path, [
        {"current_layer": 0, "current_experts": [1, 3], "next_layer": 1, "next_experts": [9, 7]},
        {"current_layer": 0, "current_experts": [1, 3], "next_layer": 1, "next_experts": [9, 7]},
    ])
    predictor = MarkovPredictor()
    predictor.fit(path)
    prefetcher = RecordingPrefetcher()

    runtime = make_runtime(predictor=predictor, prefetcher=prefetcher)
    runtime.run()

    # FakeRouter routes layer 0 -> [1, 3]; predictor maps that to [9, 7].
    assert prefetcher.scheduled == [(1, [9, 7])]


def test_falls_back_to_router_when_predictor_empty(tmp_path):
    # Predictor was trained on a transition that never matches what the
    # router actually does in this run, so predict() returns []. The
    # runtime should prefetch what the router will route to next.
    path = write_traces(tmp_path, [
        # Different current_experts than the router's [1, 3] -> miss.
        {"current_layer": 0, "current_experts": [5, 7], "next_layer": 1, "next_experts": [9, 9]},
    ])
    predictor = MarkovPredictor()
    predictor.fit(path)
    prefetcher = RecordingPrefetcher()

    runtime = make_runtime(predictor=predictor, prefetcher=prefetcher)
    runtime.run()

    # FakeRouter: layer 1 -> [0, 2]. That's the fallback target.
    assert prefetcher.scheduled == [(1, [0, 2])]


def test_prefetched_experts_become_hits_with_real_prefetcher(tmp_path):
    # End-to-end: train predictor, plug in the real AsyncPrefetcher,
    # run, and confirm the next-layer experts ended up in cache.
    # wait_for_layer (called by the runtime) ensures determinism.
    path = write_traces(tmp_path, [
        {"current_layer": 0, "current_experts": [1, 3], "next_layer": 1, "next_experts": [0, 2]},
    ])
    predictor = MarkovPredictor()
    predictor.fit(path)

    cache = ExpertCache(max_memory_mb=1)
    manager = ExpertManager(FakeStore(), cache)
    prefetcher = AsyncPrefetcher(manager, verbose=False)
    try:
        runtime = Runtime(
            router=FakeRouter(),
            expert_manager=manager,
            backend=FakeBackend(),
            prefetcher=prefetcher,
            predictor=predictor,
        )
        runtime.run()

        # After the run, both predicted layer-1 experts should be cached.
        assert cache.has((1, 0))
        assert cache.has((1, 2))
    finally:
        prefetcher.shutdown()
