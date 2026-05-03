"""Tests for MMapExpertStore: same interface as ExpertStore, mmap-backed projections."""

import numpy as np
import pytest

from apex.expert_store import ExpertStore
from apex.mmap_expert_store import MMapExpertStore
from apex.types import ExpertWeights
from tests._world import write_npy_world


def test_load_returns_bundle_of_memmaps(tmp_path):
    # Most important: each projection in the bundle is an np.memmap.
    write_npy_world(tmp_path, with_gates=False)

    bundle = MMapExpertStore(tmp_path).load(0, 1)

    assert isinstance(bundle, ExpertWeights)
    assert isinstance(bundle.gate_proj, np.memmap)
    assert isinstance(bundle.up_proj, np.memmap)
    assert isinstance(bundle.down_proj, np.memmap)


def test_shapes_match_eager_load(tmp_path):
    # The mmap variant should agree with the eager ExpertStore on
    # every observable attribute — same shape, same dtype, same nbytes.
    write_npy_world(tmp_path, with_gates=False)

    eager = ExpertStore(tmp_path).load(0, 2)
    mapped = MMapExpertStore(tmp_path).load(0, 2)

    for proj in ("gate_proj", "up_proj", "down_proj"):
        assert getattr(mapped, proj).shape == getattr(eager, proj).shape
        assert getattr(mapped, proj).dtype == getattr(eager, proj).dtype
        assert getattr(mapped, proj).nbytes == getattr(eager, proj).nbytes


def test_data_matches_eager_load(tmp_path):
    # Reading from the memmap should yield the same values as eager load.
    write_npy_world(tmp_path, with_gates=False)

    eager = ExpertStore(tmp_path).load(0, 3)
    mapped = MMapExpertStore(tmp_path).load(0, 3)

    np.testing.assert_array_equal(np.asarray(mapped.gate_proj), eager.gate_proj)
    np.testing.assert_array_equal(np.asarray(mapped.up_proj), eager.up_proj)
    np.testing.assert_array_equal(np.asarray(mapped.down_proj), eager.down_proj)


def test_works_with_swiglu_matmul(tmp_path):
    # The fake backend does SwiGLU FFN. Confirm the memmap participates
    # in matmul without fuss (this is the path that triggers actual
    # page-in from disk).
    from apex.backends.fake_backend import FakeBackend

    write_npy_world(tmp_path, with_gates=False)
    bundle = MMapExpertStore(tmp_path).load(0, 0)

    hidden = np.ones((1, 8), dtype="float32")
    out = FakeBackend().run_expert(hidden, bundle)
    assert out.shape == (1, 8)


def test_invalid_path_raises_file_not_found(tmp_path):
    store = MMapExpertStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.load(99, 99)


def test_works_with_cache_and_manager(tmp_path):
    from apex.expert_cache import ExpertCache
    from apex.expert_manager import ExpertManager

    write_npy_world(tmp_path, with_gates=False)
    cache = ExpertCache(max_memory_mb=1, verbose=False)
    manager = ExpertManager(MMapExpertStore(tmp_path), cache)

    first = manager.ensure_loaded(0, 1)
    second = manager.ensure_loaded(0, 1)

    assert isinstance(first, ExpertWeights)
    assert first is second  # same cached object on the hit
    expected = (
        first.gate_proj.nbytes + first.up_proj.nbytes + first.down_proj.nbytes
    )
    assert cache.current_bytes == expected
