"""Tests for MLXBackend.

Skipped on machines without MLX installed (e.g. non-Apple-Silicon CI),
so the rest of the suite stays portable. Locally on this Mac, all
should pass.
"""

import numpy as np
import pytest

# Skip the entire module if mlx isn't importable.
mx = pytest.importorskip("mlx.core")

from apex.backends.fake_backend import FakeBackend
from apex.backends.mlx_backend import MLXBackend
from apex.types import ExpertWeights
from tests._world import (
    HIDDEN,
    INTERMEDIATE,
    make_expert_bundle,
    write_gguf_world,
    write_npy_world,
    write_safetensors_world,
)


# Identity-projection bundle in HF orientation: gate_proj/up_proj are
# (intermediate, hidden) with the first `hidden` rows = I, down_proj is
# (hidden, intermediate) with the first `hidden` cols = I. Backends do
# h @ proj.T, so this projects h to itself in the active hidden dim
# without caring about specific numbers — useful for "the math
# actually computed" sanity checks.
def _identity_bundle(hidden=HIDDEN, intermediate=INTERMEDIATE):
    gate = np.zeros((intermediate, hidden), dtype="float32")
    gate[:hidden, :hidden] = np.eye(hidden, dtype="float32")
    up = gate.copy()
    down = np.zeros((hidden, intermediate), dtype="float32")
    down[:hidden, :hidden] = np.eye(hidden, dtype="float32")
    return ExpertWeights(gate_proj=gate, up_proj=up, down_proj=down)


def test_output_shape_matches_numpy_backend():
    # SwiGLU through both backends should produce the same shape.
    # Inputs are pre-prepared into each backend's preferred form because
    # run_expert no longer auto-converts (step 1: prep is the manager's
    # job, not run_expert's).
    hidden = np.ones((1, HIDDEN), dtype="float32")
    bundle = make_expert_bundle(fill=1.0)

    fake = FakeBackend()
    mlx_b = MLXBackend()
    numpy_out = fake.run_expert(fake.prepare_hidden(hidden), fake.prepare_expert(bundle))
    mlx_out = mlx_b.run_expert(mlx_b.prepare_hidden(hidden), mlx_b.prepare_expert(bundle))

    assert mlx_out.shape == numpy_out.shape == (1, HIDDEN)


def test_output_values_close_to_numpy_backend():
    # MLX's SwiGLU should agree with NumPy's to within float32 precision.
    # This is the key correctness check for step 4 — same algebra, two
    # different backends, same answer.
    rng = np.random.default_rng(0)
    hidden = rng.standard_normal((1, HIDDEN)).astype("float32")
    bundle = ExpertWeights(
        # HF orientation: gate/up are (intermediate, hidden); down is reversed.
        gate_proj=rng.standard_normal((INTERMEDIATE, HIDDEN)).astype("float32"),
        up_proj=rng.standard_normal((INTERMEDIATE, HIDDEN)).astype("float32"),
        down_proj=rng.standard_normal((HIDDEN, INTERMEDIATE)).astype("float32"),
    )

    fake = FakeBackend()
    mlx_b = MLXBackend()
    numpy_out = fake.run_expert(fake.prepare_hidden(hidden), fake.prepare_expert(bundle))
    mlx_out = mlx_b.run_expert(mlx_b.prepare_hidden(hidden), mlx_b.prepare_expert(bundle))

    np.testing.assert_allclose(
        mlx_b.to_numpy(mlx_out), fake.to_numpy(numpy_out), rtol=1e-5, atol=1e-5
    )


def test_run_expert_returns_mx_array():
    # Step 2 contract: run_expert no longer eagerly converts to numpy.
    backend = MLXBackend()
    hidden = backend.prepare_hidden(np.ones((1, HIDDEN), dtype="float32"))
    bundle = backend.prepare_expert(make_expert_bundle())

    out = backend.run_expert(hidden, bundle)
    assert isinstance(out, mx.array)


def test_prepare_expert_returns_mx_bundle():
    # Step 4 contract: prepare_expert converts each of the 3 projections
    # to mx.array; the cache holds an MLX-native ExpertWeights.
    bundle = make_expert_bundle()
    prepared = MLXBackend().prepare_expert(bundle)

    assert isinstance(prepared, ExpertWeights)
    assert isinstance(prepared.gate_proj, mx.array)
    assert isinstance(prepared.up_proj, mx.array)
    assert isinstance(prepared.down_proj, mx.array)


def test_runtime_run_returns_numpy_at_boundary(tmp_path):
    # End-to-end: even though intermediate state is mx.array under
    # MLXBackend, runtime.run() must hand the caller a numpy ndarray
    # (so demos / tests have a stable contract).
    from apex.expert_store import ExpertStore
    from apex.expert_cache import ExpertCache
    from apex.expert_manager import ExpertManager
    from apex.router import FakeRouter
    from apex.runtime import Runtime

    write_npy_world(tmp_path, with_gates=False)

    backend = MLXBackend()
    cache = ExpertCache(max_memory_mb=1, verbose=False)
    manager = ExpertManager(ExpertStore(tmp_path), cache, backend=backend)
    runtime = Runtime(router=FakeRouter(), expert_manager=manager, backend=backend)

    final = runtime.run()
    assert isinstance(final, np.ndarray)
    assert final.shape == (1, HIDDEN)


def test_backend_has_name_field():
    assert MLXBackend.name == "mlx"


def test_runtime_works_with_mlx_backend(tmp_path):
    # End-to-end: build a runtime with MLXBackend + the npy stack
    # (FakeRouter, ExpertStore) and confirm it runs without errors.
    from apex.expert_store import ExpertStore
    from apex.expert_cache import ExpertCache
    from apex.expert_manager import ExpertManager
    from apex.router import FakeRouter
    from apex.runtime import Runtime

    write_npy_world(tmp_path, with_gates=False)

    backend = MLXBackend()
    cache = ExpertCache(max_memory_mb=1, verbose=False)
    manager = ExpertManager(ExpertStore(tmp_path), cache, backend=backend)
    runtime = Runtime(router=FakeRouter(), expert_manager=manager, backend=backend)
    final = runtime.run()
    assert final.shape == (1, HIDDEN)
    assert final.dtype == np.float32


def test_safetensors_store_works_with_mlx_backend(tmp_path):
    # Confirm the on-disk weight format is independent of the backend:
    # safetensors-loaded experts feed cleanly into MLXBackend's matmul.
    from apex.expert_cache import ExpertCache
    from apex.expert_manager import ExpertManager
    from apex.router import FakeRouter
    from apex.runtime import Runtime
    from apex.safetensors_expert_store import SafetensorsExpertStore
    from apex.weight_layout import FakeMoELayout

    weights = write_safetensors_world(tmp_path / "x.safetensors")
    layout = FakeMoELayout()

    backend = MLXBackend()
    cache = ExpertCache(max_memory_mb=1, verbose=False)
    manager = ExpertManager(
        SafetensorsExpertStore(str(weights), layout), cache, backend=backend
    )
    runtime = Runtime(router=FakeRouter(), expert_manager=manager, backend=backend)
    final = runtime.run()
    assert final.shape == (1, HIDDEN)


def test_gguf_store_works_with_mlx_backend(tmp_path):
    # Same coverage, but with GGUF weights. MLXBackend should treat
    # the np.memmap views from GGUF the same as any other ndarray.
    from apex.expert_cache import ExpertCache
    from apex.expert_manager import ExpertManager
    from apex.gguf_expert_store import GGUFExpertStore
    from apex.router import FakeRouter
    from apex.runtime import Runtime
    from apex.weight_layout import FakeMoELayout

    weights = write_gguf_world(tmp_path / "x.gguf")
    layout = FakeMoELayout()

    backend = MLXBackend()
    cache = ExpertCache(max_memory_mb=1, verbose=False)
    manager = ExpertManager(
        GGUFExpertStore(str(weights), layout), cache, backend=backend
    )
    runtime = Runtime(router=FakeRouter(), expert_manager=manager, backend=backend)
    final = runtime.run()
    assert final.shape == (1, HIDDEN)
