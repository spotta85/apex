"""Tests for apex/moe_layer.py: routing math + per-expert combine.

All deterministic — no real checkpoint required. The fake bundles are
built with `make_expert_bundle` so the SwiGLU FFN does real (small)
matmuls under HF orientation.
"""

import numpy as np
import pytest

from apex.backends.fake_backend import FakeBackend
from apex.moe_layer import route, run_moe_layer
from apex.types import ExpertWeights
from tests._world import HIDDEN, INTERMEDIATE, make_expert_bundle


# --- route -----------------------------------------------------------------

def test_route_picks_top_k_by_logit():
    # Hand-build a router whose row sums are 1, 5, 3, 2 — so top-2 of
    # logits computed against ones-hidden are experts 1 (sum 5) and 2 (sum 3).
    router = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],  # row sum 1
            [1, 1, 1, 1, 1, 0, 0, 0],  # row sum 5
            [1, 1, 1, 0, 0, 0, 0, 0],  # row sum 3
            [1, 1, 0, 0, 0, 0, 0, 0],  # row sum 2
        ],
        dtype="float32",
    )
    hidden = np.ones((1, 8), dtype="float32")

    expert_ids, weights = route(hidden, router, top_k=2)
    assert expert_ids.shape == (1, 2)
    assert weights.shape == (1, 2)
    # IDs come back ordered by logit DESC: [1 (sum 5), 2 (sum 3)].
    assert expert_ids[0].tolist() == [1, 2]


def test_route_weights_are_softmax_of_top_k_logits():
    # Same router as above; verify the weight values numerically.
    router = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
        ],
        dtype="float32",
    )
    hidden = np.ones((1, 8), dtype="float32")
    _, weights = route(hidden, router, top_k=2)

    # Top-2 logits are [5, 3]. Softmax([5, 3]) = exp([2, 0]) / (exp(2)+1).
    exp_diff = np.exp(2.0)
    expected = np.array([[exp_diff / (exp_diff + 1.0), 1.0 / (exp_diff + 1.0)]], dtype="float32")
    np.testing.assert_allclose(weights, expected, rtol=1e-5)


def test_route_weights_sum_to_one_per_token():
    # Random router + random hidden; weights along the top_k axis must
    # sum to 1 (softmax invariant), so combination produces a true
    # weighted average rather than an arbitrary scaling.
    rng = np.random.default_rng(0)
    router = rng.standard_normal((32, 16)).astype("float32")
    hidden = rng.standard_normal((4, 16)).astype("float32")  # batch=4

    _, weights = route(hidden, router, top_k=8)
    np.testing.assert_allclose(weights.sum(axis=-1), 1.0, rtol=1e-5)


def test_route_supports_batch_dim():
    rng = np.random.default_rng(0)
    router = rng.standard_normal((10, 6)).astype("float32")
    hidden = rng.standard_normal((3, 6)).astype("float32")

    expert_ids, weights = route(hidden, router, top_k=4)
    assert expert_ids.shape == (3, 4)
    assert weights.shape == (3, 4)


def test_route_top_k_too_large_raises():
    router = np.zeros((4, 8), dtype="float32")
    hidden = np.zeros((1, 8), dtype="float32")
    with pytest.raises(ValueError):
        route(hidden, router, top_k=8)


def test_route_rejects_non_2d_hidden():
    router = np.zeros((4, 8), dtype="float32")
    with pytest.raises(ValueError):
        route(np.zeros(8, dtype="float32"), router, top_k=2)


# --- run_moe_layer ---------------------------------------------------------

def test_run_moe_layer_shape_matches_single_expert_output():
    # Output of run_moe_layer should have the same shape as one
    # expert's output (per-token hidden-dim vector).
    hidden = np.ones((1, HIDDEN), dtype="float32")
    bundle = make_expert_bundle(fill=0.05)
    backend = FakeBackend()

    out = run_moe_layer(
        hidden,
        expert_ids=[0, 1],
        weights=[0.6, 0.4],
        expert_loader=lambda eid: bundle,
        backend=backend,
    )
    assert out.shape == (1, HIDDEN)
    assert out.dtype == np.float32


def test_run_moe_layer_uniform_weights_match_single_expert_when_loader_returns_same():
    # If every selected expert returns the same per-expert output
    # (because the loader hands back the same bundle), the weighted
    # sum equals one expert's output as long as the weights sum to 1.
    hidden = np.ones((1, HIDDEN), dtype="float32")
    bundle = make_expert_bundle(fill=0.05)
    backend = FakeBackend()
    single_output = backend.run_expert(hidden, bundle)

    out = run_moe_layer(
        hidden,
        expert_ids=[7, 7, 7, 7],
        weights=[0.25, 0.25, 0.25, 0.25],
        expert_loader=lambda eid: bundle,
        backend=backend,
    )
    np.testing.assert_allclose(out, single_output, rtol=1e-5, atol=1e-5)


def test_run_moe_layer_weights_scale_contribution():
    # Loader returns bundles whose forward pass we can compute by
    # hand: a fixed bundle that produces the same per-expert output.
    # Weights [1.0, 0.0] should equal expert 0's output exactly;
    # weights [0.5, 0.5] should equal half the output (since both
    # experts produce the same thing).
    hidden = np.ones((1, HIDDEN), dtype="float32")
    bundle = make_expert_bundle(fill=0.05)
    backend = FakeBackend()
    single = backend.run_expert(hidden, bundle)

    # Weight [1.0] on the first, [0.0] on the second — output is the
    # first expert's contribution unchanged.
    only_first = run_moe_layer(
        hidden, [0, 1], [1.0, 0.0], lambda eid: bundle, backend,
    )
    np.testing.assert_allclose(only_first, single, rtol=1e-5)

    half_each = run_moe_layer(
        hidden, [0, 1], [0.5, 0.5], lambda eid: bundle, backend,
    )
    np.testing.assert_allclose(half_each, single, rtol=1e-5)


def test_run_moe_layer_loader_called_per_expert_id():
    # The loader is called exactly once per expert_id, in the order
    # they appear — which is the order route() returned them.
    bundle = make_expert_bundle(fill=0.05)
    backend = FakeBackend()
    seen = []

    def recording_loader(expert_id):
        seen.append(expert_id)
        return bundle

    run_moe_layer(
        np.ones((1, HIDDEN), dtype="float32"),
        expert_ids=[3, 1, 4, 1, 5],
        weights=[0.2, 0.2, 0.2, 0.2, 0.2],
        expert_loader=recording_loader,
        backend=backend,
    )
    assert seen == [3, 1, 4, 1, 5]


def test_run_moe_layer_mismatched_lengths_raises():
    bundle = make_expert_bundle()
    with pytest.raises(ValueError):
        run_moe_layer(
            np.ones((1, HIDDEN), dtype="float32"),
            expert_ids=[0, 1, 2],
            weights=[0.5, 0.5],
            expert_loader=lambda eid: bundle,
            backend=FakeBackend(),
        )


# --- end-to-end (route + run_moe_layer) -----------------------------------

def test_route_plus_run_moe_layer_end_to_end():
    # Simulate a full layer: build a fake-MoE world, route, run all
    # selected experts, combine, confirm shape/finiteness.
    rng = np.random.default_rng(0)
    NUM_EXPERTS = 8
    TOP_K = 3

    router = rng.standard_normal((NUM_EXPERTS, HIDDEN)).astype("float32")
    bundles = [
        ExpertWeights(
            gate_proj=rng.standard_normal((INTERMEDIATE, HIDDEN)).astype("float32"),
            up_proj=rng.standard_normal((INTERMEDIATE, HIDDEN)).astype("float32"),
            down_proj=rng.standard_normal((HIDDEN, INTERMEDIATE)).astype("float32"),
        )
        for _ in range(NUM_EXPERTS)
    ]
    hidden = rng.standard_normal((1, HIDDEN)).astype("float32")
    backend = FakeBackend()

    expert_ids, weights = route(hidden, router, top_k=TOP_K)
    assert set(expert_ids[0].tolist()).issubset(range(NUM_EXPERTS))

    out = run_moe_layer(
        hidden,
        expert_ids[0].tolist(),
        weights[0],
        lambda eid: bundles[eid],
        backend,
    )
    assert out.shape == (1, HIDDEN)
    assert np.all(np.isfinite(out))
