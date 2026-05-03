"""Tests for GateRouter: gate score → top-k expert selection."""

import numpy as np
import pytest

from apex.gate_router import GateRouter


# Helper: write a per-layer gate matrix to a tmp experts/-style root.
def make_gates(tmp_path, gates_by_layer):
    for layer_id, gate in gates_by_layer.items():
        layer_dir = tmp_path / f"layer_{layer_id}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        np.save(layer_dir / "gate.npy", gate.astype("float32"))
    return tmp_path


def test_returns_exactly_top_k_experts(tmp_path):
    # 4-expert gate; top_k=2 → result has length 2.
    gate = np.array(
        [
            [1.0, 5.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    root = make_gates(tmp_path, {0: gate})

    router = GateRouter(root, top_k=2)
    hidden = np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32")
    selected, weights = router.route(0, hidden)

    # Highest-scoring two columns are 1 (score 5) and 2 (score 2),
    # in descending order.
    assert selected == [1, 2]
    assert len(selected) == 2
    assert weights is None


def test_deterministic_for_same_input(tmp_path):
    # Same gate + same hidden_state → same output, every call.
    rng = np.random.default_rng(0)
    gate = rng.standard_normal((8, 4)).astype("float32")
    root = make_gates(tmp_path, {0: gate})

    router = GateRouter(root, top_k=2)
    hidden = np.ones((1, 8), dtype="float32")
    a, _ = router.route(0, hidden)
    b, _ = router.route(0, hidden)
    assert a == b


def test_different_hidden_chooses_different_experts(tmp_path):
    # Construct a gate where each input dimension steers a different expert.
    # Hidden = identity rows -> picks the matching expert as top-1.
    gate = np.eye(4, 4, dtype="float32")  # 4 hidden dims, 4 experts
    root = make_gates(tmp_path, {0: gate})

    router = GateRouter(root, top_k=1)

    hidden_a = np.array([[1, 0, 0, 0]], dtype="float32")
    hidden_b = np.array([[0, 0, 1, 0]], dtype="float32")

    selected_a, _ = router.route(0, hidden_a)
    selected_b, _ = router.route(0, hidden_b)
    assert selected_a == [0]
    assert selected_b == [2]


def test_returns_python_ints_not_numpy_ints(tmp_path):
    # Cache keys / JSON serialization expect plain Python ints.
    gate = np.eye(4, 4, dtype="float32")
    root = make_gates(tmp_path, {0: gate})

    router = GateRouter(root, top_k=2)
    hidden = np.ones((1, 4), dtype="float32")
    selected, _ = router.route(0, hidden)
    for expert_id in selected:
        assert type(expert_id) is int


def test_lazy_load_caches_gate(tmp_path):
    # First call hits disk, subsequent calls reuse the in-memory cache.
    # We swap the file out underneath to confirm a second route() doesn't
    # re-read from disk.
    gate1 = np.array([[1.0, 0.0]], dtype="float32")
    root = make_gates(tmp_path, {0: gate1})

    router = GateRouter(root, top_k=1)
    hidden = np.array([[1.0]], dtype="float32")
    first, _ = router.route(0, hidden)

    # Overwrite the file with a gate that would prefer expert 1 if reread.
    np.save(tmp_path / "layer_0" / "gate.npy", np.array([[0.0, 1.0]], dtype="float32"))
    second, _ = router.route(0, hidden)

    assert first == second  # cached gate is still the original


def test_invalid_layer_raises_file_not_found(tmp_path):
    # Asking for a layer with no gate.npy on disk surfaces a clear error.
    root = tmp_path  # empty
    router = GateRouter(root, top_k=2)
    hidden = np.ones((1, 8), dtype="float32")

    with pytest.raises(FileNotFoundError):
        router.route(99, hidden)
