"""Tests for tensor_nbytes — the cache's single source of truth for size."""

import numpy as np
import pytest

from apex.tensor_utils import tensor_nbytes
from apex.types import ExpertWeights


def test_ndarray_uses_native_nbytes():
    arr = np.zeros((4, 5), dtype="float32")
    assert tensor_nbytes(arr) == arr.nbytes == 4 * 5 * 4


def test_memmap_uses_native_nbytes(tmp_path):
    arr = np.zeros((3, 3), dtype="float32")
    np.save(tmp_path / "x.npy", arr)
    mapped = np.load(tmp_path / "x.npy", mmap_mode="r")
    assert tensor_nbytes(mapped) == 3 * 3 * 4


def test_expert_weights_sums_three_projections():
    # The cache treats an ExpertWeights bundle as one entry; size is
    # the sum of its three projection matrices.
    bundle = ExpertWeights(
        gate_proj=np.zeros((4, 16), dtype="float32"),
        up_proj=np.zeros((4, 16), dtype="float32"),
        down_proj=np.zeros((16, 4), dtype="float32"),
    )
    expected = (4 * 16 + 4 * 16 + 16 * 4) * 4  # 3 matrices, all float32
    assert tensor_nbytes(bundle) == expected


def test_unsupported_type_raises():
    # Anything that exposes neither nbytes nor (size, itemsize) should
    # surface a clear TypeError, not a silent wrong-size.
    with pytest.raises(TypeError):
        tensor_nbytes("not a tensor")


# MLX-specific path, gated.
mx = pytest.importorskip("mlx.core")


def test_mx_array_falls_back_to_size_times_itemsize():
    # mx.array doesn't expose `.nbytes` (as of mlx 0.31), but does
    # expose `.size` and `.itemsize` — tensor_nbytes uses those.
    arr = mx.zeros((4, 5), dtype=mx.float32)
    assert tensor_nbytes(arr) == 4 * 5 * 4
