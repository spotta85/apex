"""Tests for apex/safetensors_loader.py: load any tensor as numpy.

Builds tiny safetensors files by hand (since safetensors-numpy can't
write BF16 / F8_E4M3) and checks the loader returns the right ndarray
for each dtype, including the "exotic" ones.
"""

import numpy as np
import pytest
from safetensors.numpy import save_file

from apex.safetensors_loader import load_tensor
from tests._synthetic_nvfp4 import write_safetensors


def test_load_fp32_tensor(tmp_path):
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    path = tmp_path / "fp32.safetensors"
    save_file({"x": arr}, str(path))

    loaded = load_tensor(path, "x")
    assert loaded.dtype == np.float32
    assert loaded.shape == (3, 4)
    np.testing.assert_array_equal(loaded, arr)


def test_load_bf16_tensor(tmp_path):
    # Hand-pack BF16 bits for 1.0, -2.0, 3.5, 0.0.
    # fp32(1.0) = 0x3F800000  → upper 16 = 0x3F80
    # fp32(-2.0) = 0xC0000000 → upper 16 = 0xC000
    # fp32(3.5) = 0x40600000  → upper 16 = 0x4060
    # fp32(0.0) = 0x00000000  → upper 16 = 0x0000
    bits = np.array([0x3F80, 0xC000, 0x4060, 0x0000], dtype=np.uint16)
    path = tmp_path / "bf16.safetensors"
    write_safetensors(
        path, [("router.weight", "BF16", [4], bits.tobytes())],
    )

    loaded = load_tensor(path, "router.weight")
    assert loaded.dtype == np.float32  # BF16 is converted to fp32 on load.
    np.testing.assert_array_equal(
        loaded, np.array([1.0, -2.0, 3.5, 0.0], dtype=np.float32),
    )


def test_load_f8_e4m3_returns_uint8_view(tmp_path):
    # F8_E4M3 is left as uint8 for the caller (NVFP4 dequant decodes it).
    bytes_in = np.array([0x38, 0x40, 0x00, 0xB8], dtype=np.uint8)  # 1, 2, 0, -1
    path = tmp_path / "f8.safetensors"
    write_safetensors(
        path, [("scale.weight", "F8_E4M3", [4], bytes_in.tobytes())],
    )

    loaded = load_tensor(path, "scale.weight")
    assert loaded.dtype == np.uint8
    assert loaded.shape == (4,)
    np.testing.assert_array_equal(loaded, bytes_in)


def test_load_uint8_tensor(tmp_path):
    arr = np.arange(8, dtype=np.uint8)
    path = tmp_path / "u8.safetensors"
    save_file({"packed": arr}, str(path))

    loaded = load_tensor(path, "packed")
    assert loaded.dtype == np.uint8
    np.testing.assert_array_equal(loaded, arr)


def test_load_missing_tensor_raises_key_error(tmp_path):
    arr = np.zeros((2,), dtype=np.float32)
    path = tmp_path / "x.safetensors"
    save_file({"present": arr}, str(path))

    with pytest.raises(KeyError) as excinfo:
        load_tensor(path, "absent")
    assert "absent" in str(excinfo.value)


def test_unsupported_dtype_raises_value_error(tmp_path):
    # Construct a header with a totally fake dtype string. Safetensors
    # itself wouldn't write this, but we want a clear failure mode if
    # a future format extends the dtype set unexpectedly.
    path = tmp_path / "weird.safetensors"
    write_safetensors(
        path, [("x", "F128_FANCY", [1], b"\x00" * 4)],
    )

    with pytest.raises(ValueError) as excinfo:
        load_tensor(path, "x")
    assert "F128_FANCY" in str(excinfo.value)
