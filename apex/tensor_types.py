"""Tensor wrappers that carry dtype + (optional) quantization sidecars.

Two thin dataclasses sit between "raw bytes loaded from disk" and
"ndarray ready for SwiGLU." They let stores hand back tensors that
remember what they are without forcing the caller to re-discover
that information.

  - `RawTensor`:       an unquantized tensor + its dtype string. Used
                       for fp32 / bf16 / fp16 / int8 etc. The data
                       field is whatever the loader produced (numpy
                       array; for bf16 it's already real fp32 after
                       manual byte conversion).
  - `QuantizedTensor`: a quantized tensor + every sidecar needed to
                       reconstruct fp32 values. For NVFP4: weight_scale
                       (per-block F8_E4M3 scales) plus weight_scale_2
                       (one F32 tensor-wide scale) plus optional
                       input_scale (for activation quantization at
                       compute time, kept here for completeness even
                       though we don't yet quantize activations).

Both wrappers are dtype-agnostic on `data` (typed `Any`) so the same
class works under numpy, mlx, or anything else with a shape attribute.
The `dtype` string is the source-of-truth for what's actually inside.

Don't put behavior on these wrappers — they're pure data. Conversion
lives in `apex/dequant.py` so the wrapper itself stays trivial and
the dispatch logic is in one place.
"""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class RawTensor:
    """An unquantized tensor with explicit dtype + shape metadata.

    `dtype` follows the safetensors string convention ("F32", "BF16",
    "F16", "U8", "I32", etc.) so it round-trips cleanly through the
    safetensors loader without translation.
    """

    data: Any
    dtype: str
    shape: tuple


@dataclass
class QuantizedTensor:
    """A quantized tensor + sidecars needed to reconstruct fp32.

    `data` is the packed tensor (typically uint8). `shape` is its
    on-disk shape (matches `data.shape`); the unpacked logical shape
    depends on the quantization scheme and is the dequantizer's job
    to compute. For NVFP4: logical shape is (rows, cols * 2) because
    two FP4 values live in each uint8 byte.

    `dtype` is the quantization scheme name, not a numpy dtype:
    "NVFP4" or similar future schemes. This is what the dequantizer
    dispatches on.

    Sidecars are optional at the type level so partial states are
    representable, but a dequantizer for any specific scheme will
    require its specific sidecars and raise if they're missing.
    """

    data: Any
    dtype: str
    shape: tuple
    input_scale: Optional[Any] = None
    weight_scale: Optional[Any] = None
    weight_scale_2: Optional[Any] = None
