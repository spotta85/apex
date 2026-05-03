"""Tests for apex/dequant.py: passthrough, RawTensor BF16, NVFP4 round-trip.

The NVFP4 tests use hand-built ground-truth tensors:

  - Pick FP4 codes whose lookup values are known exactly
    (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 and their negatives).
  - Pick F8_E4M3 byte values whose decoded fp32 value is known exactly
    (e.g. 0x38 = 1.0, 0x40 = 2.0).
  - Multiply by a known scalar weight_scale_2.

The expected output is then exact to the fp32 representable values,
so the round-trip can use array_equal rather than allclose.
"""

import numpy as np
import pytest

from apex.dequant import (
    NVFP4_GROUP_SIZE,
    NVFP4_LOOKUP,
    UnsupportedQuantizationError,
    _decode_e4m3,
    dequantize,
    dequantize_bundle,
)
from apex.tensor_types import QuantizedTensor, RawTensor
from apex.types import ExpertWeights


# --- passthrough -----------------------------------------------------------

def test_dequantize_ndarray_is_passthrough():
    arr = np.full((4, 8), 7.0, dtype=np.float32)
    out = dequantize(arr)
    # Identity preserved (no copy on the passthrough path).
    assert out is arr


def test_dequantize_unhandled_type_raises_type_error():
    with pytest.raises(TypeError):
        dequantize(object())


# --- RawTensor BF16 → fp32 conversion --------------------------------------

def test_raw_tensor_bf16_decodes_to_fp32():
    # BF16 of 1.0 is the upper 16 bits of fp32(1.0). fp32(1.0) bits =
    # 0x3F800000; the upper 16 bits = 0x3F80. Loaded as uint16, padded
    # to 0x3F800000, viewed as fp32 => 1.0.
    bits = np.array([0x3F80, 0xC000, 0x4040], dtype=np.uint16)  # 1.0, -2.0, 3.0
    rt = RawTensor(data=bits, dtype="BF16", shape=(3,))
    out = dequantize(rt)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, np.array([1.0, -2.0, 3.0], dtype=np.float32))


def test_raw_tensor_fp32_passthrough():
    arr = np.full((2, 2), 3.14, dtype=np.float32)
    rt = RawTensor(data=arr, dtype="F32", shape=(2, 2))
    out = dequantize(rt)
    np.testing.assert_array_equal(out, arr)


def test_raw_tensor_bf16_with_wrong_underlying_dtype_raises():
    # BF16-typed but data isn't uint16 → unrecoverable, surface clearly.
    rt = RawTensor(data=np.zeros(4, dtype=np.float32), dtype="BF16", shape=(4,))
    with pytest.raises(UnsupportedQuantizationError) as excinfo:
        dequantize(rt)
    assert "BF16" in str(excinfo.value)


# --- NVFP4 dequant ground-truth -------------------------------------------

def _e4m3_one():
    """Return the F8_E4M3 byte that decodes to exactly 1.0."""
    # exp=7 (=bias), mant=0, sign=0  =>  S 0111 000 = 0x38
    return 0x38


def _nibble_pack(low_codes, high_codes) -> np.ndarray:
    """Pack alternating low/high 4-bit codes into uint8 bytes (vectorised)."""
    return ((np.asarray(high_codes, dtype=np.uint8) << 4) |
            np.asarray(low_codes, dtype=np.uint8)).astype(np.uint8)


def _u32_pack_nibbles(codes) -> np.ndarray:
    """Pack eight 4-bit codes into little-endian uint32 words."""
    codes = np.asarray(codes, dtype=np.uint32)
    assert codes.size % 8 == 0
    return sum(codes[i::8] << (4 * i) for i in range(8)).astype(np.uint32)


def _u32_pack_bytes(values) -> np.ndarray:
    """Pack four uint8 values into little-endian uint32 words."""
    values = np.asarray(values, dtype=np.uint32)
    assert values.size % 4 == 0
    return sum(values[i::4] << (8 * i) for i in range(4)).astype(np.uint32)


def _build_nvfp4_tensor(rows, codes_per_row, block_scale_byte, weight_scale_2):
    """Build a small NVFP4 QuantizedTensor with controlled inputs.

    `codes_per_row` is a list of 4-bit codes (length must be a
    multiple of 2, since pack two per byte, and a multiple of
    NVFP4_GROUP_SIZE so block scales line up).
    """
    cols = len(codes_per_row)
    assert cols % NVFP4_GROUP_SIZE == 0, "test data must align to group size"
    low = codes_per_row[0::2]
    high = codes_per_row[1::2]
    one_row_bytes = _nibble_pack(low, high)  # shape (cols // 2,)
    packed = np.tile(one_row_bytes, (rows, 1))
    block_scales = np.full(
        (rows, cols // NVFP4_GROUP_SIZE), block_scale_byte, dtype=np.uint8,
    )
    return QuantizedTensor(
        data=packed,
        dtype="NVFP4",
        shape=packed.shape,
        weight_scale=block_scales,
        weight_scale_2=np.float32(weight_scale_2),
    )


def test_nvfp4_all_ones_round_trip():
    # Codes: 0x2 = 1.0 in NVFP4. Block scale 1.0, weight_scale_2 = 1.0.
    # Expected: an all-ones matrix.
    qt = _build_nvfp4_tensor(
        rows=2,
        codes_per_row=[0x2] * NVFP4_GROUP_SIZE,
        block_scale_byte=_e4m3_one(),
        weight_scale_2=1.0,
    )
    out = dequantize(qt)
    assert out.shape == (2, NVFP4_GROUP_SIZE)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, np.ones((2, NVFP4_GROUP_SIZE), dtype=np.float32))


def test_nvfp4_alternating_values_validates_nibble_order():
    # Low nibble = 0x2 (=1.0), high nibble = 0x4 (=2.0). Result should
    # be [1.0, 2.0, 1.0, 2.0, ...] — confirms the unpacker treats the
    # low nibble as element-2k (first), upper as element-2k+1 (second).
    qt = _build_nvfp4_tensor(
        rows=1,
        codes_per_row=([0x2, 0x4] * (NVFP4_GROUP_SIZE // 2)),
        block_scale_byte=_e4m3_one(),
        weight_scale_2=1.0,
    )
    out = dequantize(qt)
    expected = np.tile([1.0, 2.0], NVFP4_GROUP_SIZE // 2).astype(np.float32)[None, :]
    np.testing.assert_array_equal(out, expected)


def test_nvfp4_weight_scale_2_scales_output():
    qt = _build_nvfp4_tensor(
        rows=1,
        codes_per_row=[0x2] * NVFP4_GROUP_SIZE,  # all 1.0
        block_scale_byte=_e4m3_one(),             # block scale 1.0
        weight_scale_2=0.5,                       # tensor-wide 0.5
    )
    out = dequantize(qt)
    np.testing.assert_array_equal(out, np.full((1, NVFP4_GROUP_SIZE), 0.5, dtype=np.float32))


def test_nvfp4_block_scale_applies_per_group():
    # Two groups in one row. Group 0 scale = 1.0 (0x38), group 1 scale = 2.0 (0x40).
    # All FP4 values = 1.0. Expected: first 16 = 1.0, second 16 = 2.0.
    cols = NVFP4_GROUP_SIZE * 2
    codes = [0x2] * cols
    one_row_bytes = _nibble_pack(codes[0::2], codes[1::2])
    packed = one_row_bytes.reshape(1, -1)
    weight_scale = np.array([[0x38, 0x40]], dtype=np.uint8)  # E4M3: 1.0, 2.0
    qt = QuantizedTensor(
        data=packed,
        dtype="NVFP4",
        shape=packed.shape,
        weight_scale=weight_scale,
        weight_scale_2=np.float32(1.0),
    )
    out = dequantize(qt)
    expected = np.concatenate(
        [
            np.ones((1, NVFP4_GROUP_SIZE), dtype=np.float32),
            np.full((1, NVFP4_GROUP_SIZE), 2.0, dtype=np.float32),
        ],
        axis=1,
    )
    np.testing.assert_array_equal(out, expected)


def test_nvfp4_negative_codes_decode_to_negative_values():
    # Code 0xA = -1.0 (sign=1, exp=1, mant=0).
    qt = _build_nvfp4_tensor(
        rows=1,
        codes_per_row=[0xA] * NVFP4_GROUP_SIZE,
        block_scale_byte=_e4m3_one(),
        weight_scale_2=1.0,
    )
    out = dequantize(qt)
    np.testing.assert_array_equal(
        out, np.full((1, NVFP4_GROUP_SIZE), -1.0, dtype=np.float32),
    )


def test_nvfp4_u32_unpacks_eight_codes_per_word():
    codes = [0x2, 0x4] * (NVFP4_GROUP_SIZE // 2)
    qt = QuantizedTensor(
        data=_u32_pack_nibbles(codes).reshape(1, -1),
        dtype="NVFP4_U32",
        shape=(1, 2),
        weight_scale=np.array([[0x38]], dtype=np.uint8),
    )

    out = dequantize(qt)

    expected = np.tile([1.0, 2.0], NVFP4_GROUP_SIZE // 2).astype(np.float32)
    np.testing.assert_array_equal(out, expected.reshape(1, -1))


def test_affine_uint8_u32_unpacks_bytes_and_applies_groups():
    qt = QuantizedTensor(
        data=_u32_pack_bytes([0, 1, 2, 3, 4, 5, 6, 7]).reshape(1, -1),
        dtype="AFFINE_UINT8_U32",
        shape=(1, 2),
        weight_scale=np.array([[0.5, -1.0]], dtype=np.float32),
        weight_scale_2=np.array([[10.0, 20.0]], dtype=np.float32),
    )

    out = dequantize(qt)

    expected = np.array([[10.0, 10.5, 11.0, 11.5, 16.0, 15.0, 14.0, 13.0]])
    np.testing.assert_array_equal(out, expected.astype(np.float32))


# --- NVFP4 failure paths --------------------------------------------------

def test_nvfp4_missing_weight_scale_raises_clear_error():
    qt = QuantizedTensor(
        data=np.zeros((1, 8), dtype=np.uint8),
        dtype="NVFP4",
        shape=(1, 8),
        weight_scale=None,
        weight_scale_2=np.float32(1.0),
    )
    with pytest.raises(UnsupportedQuantizationError) as excinfo:
        dequantize(qt, name="experts.0.gate_proj.weight")
    msg = str(excinfo.value)
    assert "experts.0.gate_proj.weight" in msg
    assert "NVFP4" in msg
    assert "weight_scale" in msg


def test_nvfp4_missing_weight_scale_2_raises_clear_error():
    qt = QuantizedTensor(
        data=np.zeros((1, 8), dtype=np.uint8),
        dtype="NVFP4",
        shape=(1, 8),
        weight_scale=np.zeros((1, 1), dtype=np.uint8),
        weight_scale_2=None,
    )
    with pytest.raises(UnsupportedQuantizationError):
        dequantize(qt)


def test_unknown_quant_scheme_raises_with_dtype_in_message():
    qt = QuantizedTensor(
        data=np.zeros((1, 8), dtype=np.uint8),
        dtype="MXFP4",
        shape=(1, 8),
    )
    with pytest.raises(UnsupportedQuantizationError) as excinfo:
        dequantize(qt, name="some.tensor")
    msg = str(excinfo.value)
    assert "MXFP4" in msg
    assert "some.tensor" in msg


def test_nvfp4_wrong_block_scale_shape_raises():
    # Packed (1, 8) → unpacks to (1, 16); expected scales shape is
    # (1, 1) for one block of 16. A wrong shape (1, 2) must raise.
    qt = QuantizedTensor(
        data=np.full((1, 8), 0x22, dtype=np.uint8),
        dtype="NVFP4",
        shape=(1, 8),
        weight_scale=np.full((1, 2), 0x38, dtype=np.uint8),
        weight_scale_2=np.float32(1.0),
    )
    with pytest.raises(UnsupportedQuantizationError) as excinfo:
        dequantize(qt)
    assert "weight_scale shape" in str(excinfo.value)


# --- E4M3 decoder spot checks ---------------------------------------------

def test_decode_e4m3_known_values():
    # Verified against the formula by hand:
    # 0x38 = +1.0     0x40 = +2.0     0xB8 = -1.0     0x00 = +0.0
    # 0x18 = +0.25 (exp=3, normal: (1+0/8)*2^(3-7) = 1*0.0625 = 0.0625? wait)
    # Let me recheck: 0x18 = 0001_1000? No, 0x18 = 0001_1000 = sign=0, exp=0011=3, mant=000.
    #   Normal: (1 + 0/8) * 2^(3-7) = 1 * 2^-4 = 0.0625
    bytes_in = np.array([0x38, 0x40, 0xB8, 0x00, 0x18], dtype=np.uint8)
    expected = np.array([1.0, 2.0, -1.0, 0.0, 0.0625], dtype=np.float32)
    np.testing.assert_array_equal(_decode_e4m3(bytes_in), expected)


# --- dequantize_bundle convenience ---------------------------------------

def test_dequantize_bundle_runs_through_each_projection():
    # Mix of an ndarray (passthrough) and a NVFP4 QuantizedTensor.
    raw = np.full((1, 4), 9.0, dtype=np.float32)
    qt = _build_nvfp4_tensor(
        rows=1,
        codes_per_row=[0x2] * NVFP4_GROUP_SIZE,
        block_scale_byte=_e4m3_one(),
        weight_scale_2=1.0,
    )
    bundle = ExpertWeights(gate_proj=qt, up_proj=raw, down_proj=raw)
    out = dequantize_bundle(bundle)
    assert isinstance(out, ExpertWeights)
    np.testing.assert_array_equal(
        out.gate_proj, np.ones((1, NVFP4_GROUP_SIZE), dtype=np.float32),
    )
    assert out.up_proj is raw  # passthrough preserved
    assert out.down_proj is raw
