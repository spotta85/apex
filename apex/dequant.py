"""Dequantize tensor wrappers into ndarrays ready for SwiGLU compute.

Single entry point: ``dequantize(t)``. Accepts:

  - ``np.ndarray``      → returned as-is (passthrough; lets backends
                          call dequantize() unconditionally without
                          branching at every call site).
  - ``RawTensor``       → returns its underlying ndarray. BF16-typed
                          RawTensors get an explicit fp32 conversion
                          (uint16 holding the BF16 bits → pad with 16
                          zero bits → view as float32).
  - ``QuantizedTensor`` → routes to a per-scheme dequantizer. NVFP4 is
                          implemented; other schemes raise
                          ``UnsupportedQuantizationError`` with the
                          offending dtype + shape so the caller can
                          act on the gap rather than silently fall
                          through to garbage output.

Optional ``dequantize_bundle(bundle)`` runs ``dequantize`` on each of
an ExpertWeights bundle's three projections and returns a new bundle
of plain ndarrays — the canonical "make this expert SwiGLU-runnable"
boundary.

NVFP4 details (see also: hf_quant_config.json on real Gemma 4):

  - Each element is 4 bits in E2M1 layout (sign | 2-bit exp | 1-bit
    mantissa, bias=1). Two elements pack into one uint8 byte: the
    first element in the lower nibble, the second in the upper nibble.
  - Two-level scaling: every block of GROUP_SIZE=16 consecutive
    elements along the last axis shares one F8_E4M3 scale
    (``weight_scale``), and the whole tensor has one F32 scale
    (``weight_scale_2``). Reconstructed value =
        fp4_value × decode_e4m3(block_scale) × weight_scale_2.

Correctness-first implementation: pure numpy, no kernels, vectorized
where it's free but not micro-optimized. Fast enough for the smoke
test; future iterations can tune.
"""

from typing import Any

import numpy as np

from apex.tensor_types import QuantizedTensor, RawTensor
from apex.types import ExpertWeights


# NVFP4 (E2M1) lookup table indexed by the 4-bit code.
# Code layout: bit3 = sign, bits2-1 = exponent (bias=1), bit0 = mantissa.
# Codes 0–7: positive values [0, 0.5, 1, 1.5, 2, 3, 4, 6].
# Codes 8–15: same magnitudes negated. (-0.0 is harmless arithmetically.)
NVFP4_LOOKUP = np.array(
    [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ],
    dtype=np.float32,
)

# Group size used by NVFP4: every 16 consecutive elements share one
# F8_E4M3 weight_scale. Confirmed via the model's hf_quant_config.json.
NVFP4_GROUP_SIZE = 16


class UnsupportedQuantizationError(Exception):
    """Raised when ``dequantize`` can't handle a tensor's quant scheme.

    Carries the offending dtype + shape on attributes so callers can
    surface them in higher-level error messages without parsing the
    string representation.
    """

    def __init__(self, *, tensor: str, dtype: str, shape, message: str):
        self.tensor = tensor
        self.dtype = dtype
        self.shape = tuple(shape)
        self.message = message
        super().__init__(
            f"{message}\n  tensor: {tensor}\n  dtype: {dtype}\n  shape: {self.shape}"
        )


def dequantize(t: Any, *, name: str = "<unnamed>") -> np.ndarray:
    """Convert any tensor wrapper (or raw array) into a SwiGLU-ready ndarray.

    `name` is purely informational — included in error messages so the
    caller knows which tensor the failure refers to.
    """
    if isinstance(t, np.ndarray):
        return t
    if isinstance(t, RawTensor):
        return _dequantize_raw(t, name=name)
    if isinstance(t, QuantizedTensor):
        if t.dtype == "NVFP4":
            return _dequantize_nvfp4(t, name=name)
        if t.dtype == "NVFP4_U32":
            return _dequantize_nvfp4_u32(t, name=name)
        if t.dtype == "AFFINE_UINT8_U32":
            return _dequantize_affine_uint8_u32(t, name=name)
        raise UnsupportedQuantizationError(
            tensor=name,
            dtype=t.dtype,
            shape=t.shape,
            message=f"dequant not implemented for quantization scheme {t.dtype!r}",
        )
    # Fall through for anything ndarray-compatible (np.memmap, mlx.array,
    # etc.) — let np.asarray handle the conversion.
    if hasattr(t, "shape"):
        return np.asarray(t)
    raise TypeError(f"dequantize cannot handle object of type {type(t).__name__}")


def dequantize_bundle(bundle: ExpertWeights) -> ExpertWeights:
    """Dequantize all three projections of an expert bundle.

    Returns a new ExpertWeights whose fields are plain ndarrays.
    Convenience for "load + dequant" callers; equivalent to calling
    ``dequantize`` on each field by hand.
    """
    return ExpertWeights(
        gate_proj=dequantize(bundle.gate_proj, name="gate_proj"),
        up_proj=dequantize(bundle.up_proj, name="up_proj"),
        down_proj=dequantize(bundle.down_proj, name="down_proj"),
    )


class DequantizingExpertStore:
    """Wrap a store-of-quantized-bundles so it returns dequantized bundles.

    Composes with any object exposing ``load(layer_id, expert_id)``:

        base  = QuantizedSafetensorsExpertStore(path, layout)
        store = DequantizingExpertStore(base)
        manager = ExpertManager(store, cache, ...)

    The point of this adapter is **where the dequant happens**. Without
    it, the manager would cache the QuantizedTensor bundle and every
    cache hit would still pay the ~60 ms NVFP4 decode (gate/up/down).
    By dequantizing inside ``load``, the cache holds fp32 ndarrays and
    cache hits skip dequant entirely — which is the whole point of
    "real MoE cache integration."

    `verbose=True` prints ``dequantized expert <eid>`` after each load so
    the demo trace can show that dequant is happening on misses (and
    NOT on hits — the line never fires from the manager's hit path).
    """

    def __init__(self, base, *, verbose: bool = False):
        self.base = base
        self.verbose = verbose

    def load(self, layer_id: int, expert_id: int) -> ExpertWeights:
        bundle = self.base.load(layer_id, expert_id)
        out = dequantize_bundle(bundle)
        if self.verbose:
            print(f"dequantized expert {expert_id}")
        return out


# --- Raw tensor handling (mostly BF16) -------------------------------------

def _dequantize_raw(t: RawTensor, *, name: str) -> np.ndarray:
    """Convert a RawTensor into ndarray. BF16 needs explicit byte expansion."""
    if t.dtype == "BF16":
        # The data was loaded as uint16 holding the BF16 bits. Pad each
        # value with 16 zero bits to make a 32-bit IEEE float — BF16 IS
        # the upper 16 bits of an fp32, so this is a true (lossless-
        # within-bf16's precision) conversion.
        if t.data.dtype != np.uint16:
            raise UnsupportedQuantizationError(
                tensor=name,
                dtype=t.dtype,
                shape=t.shape,
                message=f"BF16 RawTensor must hold uint16 bits, got dtype {t.data.dtype}",
            )
        u32 = t.data.astype(np.uint32) << 16
        return u32.view(np.float32).copy()
    return np.asarray(t.data)


# --- NVFP4 dequantization --------------------------------------------------

def _dequantize_nvfp4(t: QuantizedTensor, *, name: str) -> np.ndarray:
    """Dequantize NVFP4 packed-uint8 → fp32.

    Layout:
      data (uint8):           shape (rows, cols // 2). Two FP4 values
                              per byte, lower nibble first.
      weight_scale (uint8):   shape (rows, cols // GROUP_SIZE).
                              F8_E4M3 encoded — one block scale per
                              group of GROUP_SIZE=16 consecutive
                              elements along the last axis.
      weight_scale_2 (f32):   scalar (or 0-d array). Tensor-wide scale.

    Returns:
      fp32 ndarray of shape (rows, cols), where cols = data.shape[1] * 2.
    """
    if t.weight_scale is None or t.weight_scale_2 is None:
        raise UnsupportedQuantizationError(
            tensor=name,
            dtype=t.dtype,
            shape=t.shape,
            message="NVFP4 dequant requires both weight_scale and weight_scale_2 sidecars",
        )

    packed = np.asarray(t.data, dtype=np.uint8)
    if packed.ndim != 2:
        raise UnsupportedQuantizationError(
            tensor=name,
            dtype=t.dtype,
            shape=t.shape,
            message=f"NVFP4 dequant expects 2-D packed data, got {packed.ndim}-D",
        )

    rows, packed_cols = packed.shape
    cols = packed_cols * 2

    # Unpack two FP4 values per byte. Element 2k = low nibble of byte k,
    # element 2k+1 = high nibble. (CUDA-conventional packing order.)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    nibbles = np.empty((rows, cols), dtype=np.uint8)
    nibbles[:, 0::2] = low
    nibbles[:, 1::2] = high

    # FP4 code → fp32 magnitude via lookup table.
    fp4_values = NVFP4_LOOKUP[nibbles]

    # Decode the per-block F8_E4M3 scales and broadcast each to its
    # GROUP_SIZE elements along the last axis.
    block_scales_u8 = np.asarray(t.weight_scale, dtype=np.uint8)
    if block_scales_u8.shape != (rows, cols // NVFP4_GROUP_SIZE):
        raise UnsupportedQuantizationError(
            tensor=name,
            dtype=t.dtype,
            shape=t.shape,
            message=(
                f"NVFP4 weight_scale shape {block_scales_u8.shape} does not match "
                f"expected (rows={rows}, cols/{NVFP4_GROUP_SIZE}={cols // NVFP4_GROUP_SIZE})"
            ),
        )
    block_scales_f32 = _decode_e4m3(block_scales_u8)
    block_scales_expanded = np.repeat(block_scales_f32, NVFP4_GROUP_SIZE, axis=1)

    # Apply two-level scaling.
    weight_scale_2 = float(np.asarray(t.weight_scale_2).item())
    return fp4_values * block_scales_expanded * weight_scale_2


def _dequantize_nvfp4_u32(t: QuantizedTensor, *, name: str) -> np.ndarray:
    """Dequantize mlx-community U32-packed NVFP4 weights.

    The local lowercase Gemma checkpoint stores eight 4-bit FP values in
    each uint32 and keeps one F8_E4M3 scale per 16 unpacked values. It
    has no tensor-wide `weight_scale_2` sidecar, unlike the NVIDIA-style
    path above.
    """
    if t.weight_scale is None:
        raise UnsupportedQuantizationError(
            tensor=name,
            dtype=t.dtype,
            shape=t.shape,
            message="NVFP4_U32 dequant requires a .scales sidecar",
        )

    packed = np.asarray(t.data, dtype=np.uint32)
    if packed.ndim < 1:
        raise UnsupportedQuantizationError(
            tensor=name,
            dtype=t.dtype,
            shape=t.shape,
            message="NVFP4_U32 dequant expects at least 1-D packed data",
        )

    nibbles = _unpack_u32_lanes(packed, lane_bits=4)
    fp4_values = NVFP4_LOOKUP[nibbles]

    block_scales_u8 = np.asarray(t.weight_scale, dtype=np.uint8)
    expected = packed.shape[:-1] + (nibbles.shape[-1] // NVFP4_GROUP_SIZE,)
    if block_scales_u8.shape != expected:
        raise UnsupportedQuantizationError(
            tensor=name,
            dtype=t.dtype,
            shape=t.shape,
            message=(
                f"NVFP4_U32 scales shape {block_scales_u8.shape} does not match "
                f"expected {expected}"
            ),
        )
    block_scales = _decode_e4m3(block_scales_u8)
    return fp4_values * np.repeat(block_scales, NVFP4_GROUP_SIZE, axis=-1)


def _dequantize_affine_uint8_u32(t: QuantizedTensor, *, name: str) -> np.ndarray:
    """Dequantize U32-packed affine uint8 weights.

    Used by the lowercase Gemma router projection. Each uint32 stores
    four quantized bytes. Per-group BF16 sidecars are represented on the
    wrapper as `weight_scale` (.scales) and `weight_scale_2` (.biases),
    with reconstruction `q * scale + bias`.
    """
    if t.weight_scale is None or t.weight_scale_2 is None:
        raise UnsupportedQuantizationError(
            tensor=name,
            dtype=t.dtype,
            shape=t.shape,
            message="AFFINE_UINT8_U32 dequant requires .scales and .biases sidecars",
        )

    packed = np.asarray(t.data, dtype=np.uint32)
    q = _unpack_u32_lanes(packed, lane_bits=8).astype(np.float32)
    scales = np.asarray(t.weight_scale, dtype=np.float32)
    biases = np.asarray(t.weight_scale_2, dtype=np.float32)
    if scales.shape != biases.shape:
        raise UnsupportedQuantizationError(
            tensor=name,
            dtype=t.dtype,
            shape=t.shape,
            message=f"scales shape {scales.shape} != biases shape {biases.shape}",
        )
    if q.shape[-1] % scales.shape[-1] != 0:
        raise UnsupportedQuantizationError(
            tensor=name,
            dtype=t.dtype,
            shape=t.shape,
            message=(
                f"unpacked width {q.shape[-1]} is not divisible by "
                f"scale groups {scales.shape[-1]}"
            ),
        )
    group_size = q.shape[-1] // scales.shape[-1]
    return q * np.repeat(scales, group_size, axis=-1) + np.repeat(
        biases, group_size, axis=-1,
    )


def _unpack_u32_lanes(packed: np.ndarray, *, lane_bits: int) -> np.ndarray:
    """Unpack little-endian uint32 words into 32/lane_bits lanes per word.

    Lane 0 is bits [0, lane_bits); lane 1 is [lane_bits, 2*lane_bits);
    etc. Lanes are placed at strided positions along the last axis so
    `out[..., i]` corresponds to the i-th lane in word `i // lanes`,
    matching the on-disk MLX/HF packing order.
    """
    lanes = 32 // lane_bits
    mask = (1 << lane_bits) - 1
    out = np.empty(packed.shape[:-1] + (packed.shape[-1] * lanes,), dtype=np.uint8)
    for i in range(lanes):
        out[..., i::lanes] = ((packed >> (lane_bits * i)) & mask).astype(np.uint8)
    return out


def _decode_e4m3(u8: np.ndarray) -> np.ndarray:
    """Vectorized F8_E4M3 (uint8 view) → fp32.

    Layout of one byte (MSB → LSB): S EEEE MMM (sign | 4-bit exp | 3-bit mant).
    Bias = 7. Subnormal when exp == 0.

    NaN encoding (S.1111.111) is NOT special-cased here — it would
    decode to ±480.0 instead of NaN. The upstream NVFP4 quantizer
    never produces NaN scales (block scales are computed from real
    weight magnitudes), so this is safe in practice.
    """
    sign = np.where(u8 & 0x80, -1.0, 1.0).astype(np.float32)
    exp = ((u8 >> 3) & 0xF).astype(np.int32)
    mant = (u8 & 0x7).astype(np.float32)
    # Subnormal: value = sign × (mant/8) × 2^(1 - bias) = sign × (mant/8) × 2^-6
    sub = sign * (mant / 8.0) * (2.0 ** -6)
    # Normal: value = sign × (1 + mant/8) × 2^(exp - bias)
    norm = sign * (1.0 + mant / 8.0) * np.power(2.0, (exp - 7).astype(np.float32))
    return np.where(exp == 0, sub, norm)
