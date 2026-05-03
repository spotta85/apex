"""Load any tensor from a safetensors file as a numpy array.

Bypasses ``safetensors.numpy``'s dtype restrictions (BF16, F8_E4M3,
and other "exotic" types raise in current safetensors-numpy because
numpy doesn't have native dtypes for them). We parse the safetensors
header by hand and reinterpret the bytes ourselves:

  - F32 / F16 / I8 / U8 / I32 / I64 / BOOL → frombuffer with the
    matching numpy dtype.
  - BF16 → loaded as uint16, then padded with 16 zero bits and
    viewed as fp32. BF16 IS the upper 16 bits of an IEEE fp32, so
    this is a true conversion (no precision lost beyond what BF16
    already lost when it was stored).
  - F8_E4M3 → returned as uint8 view; the caller (typically the
    NVFP4 dequantizer) handles the f8→f32 decode because they
    usually want to combine it with other arithmetic anyway.

This module exists to keep the safetensors low-level handling in one
place. ``apex/checkpoint_index.py`` only does metadata reads (no bulk
data); this module owns the data read path. Stores that need raw
tensor bytes (e.g. QuantizedSafetensorsExpertStore) call here.

Why no torch fallback: the manual conversion is exact for BF16 and
correct for F8_E4M3 reinterpretation, and avoids pulling in torch
just for two dtypes. If a future quant scheme needs a dtype only
torch can decode, that's the time to add an optional torch path.
"""

import json
from pathlib import Path

import numpy as np


# Standard numpy-representable safetensors dtypes. The "exotic" ones
# (BF16, F8_*) get explicit handling below.
_NUMPY_DTYPES = {
    "F32": np.float32,
    "F16": np.float16,
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U8": np.uint8,
    "U16": np.uint16,
    "U32": np.uint32,
    "U64": np.uint64,
    "BOOL": np.bool_,
}


def load_tensor(path, name: str) -> np.ndarray:
    """Load one tensor from a single .safetensors file as a numpy array.

    Always returns a copy (not a view into the file's bytes), so the
    file handle is closed before this function returns.

    Raises ``KeyError`` if the tensor isn't in the file, ``ValueError``
    for unsupported dtypes that we genuinely can't decode.
    """
    path = Path(path)
    with open(path, "rb") as f:
        # Safetensors layout: 8-byte little-endian header size, then
        # JSON header, then bulk tensor data. Tensor metadata in the
        # header includes data_offsets relative to the start of bulk.
        header_size = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_size))
        if name not in header:
            raise KeyError(f"tensor {name!r} not found in {path}")
        meta = header[name]
        offset_start, offset_end = meta["data_offsets"]
        f.seek(8 + header_size + offset_start)
        raw = f.read(offset_end - offset_start)
    return _interpret_bytes(raw, meta["dtype"], tuple(meta["shape"]))


def _interpret_bytes(raw: bytes, dtype: str, shape: tuple) -> np.ndarray:
    """Turn raw safetensors bytes + dtype/shape metadata into a numpy array."""
    if dtype in _NUMPY_DTYPES:
        np_dtype = _NUMPY_DTYPES[dtype]
        # frombuffer + reshape + copy: copy detaches us from the file
        # buffer so we can close the file and still hold the data.
        return np.frombuffer(raw, dtype=np_dtype).reshape(shape).copy()
    if dtype == "BF16":
        # BF16 is the upper 16 bits of fp32. Read as uint16, shift left
        # 16, view as fp32 — exact reconstruction within BF16 precision.
        u16 = np.frombuffer(raw, dtype=np.uint16)
        u32 = u16.astype(np.uint32) << 16
        return u32.view(np.float32).reshape(shape).copy()
    if dtype.startswith("F8"):
        # F8_E4M3 / F8_E5M2 etc — return as uint8 view. Callers that
        # need fp32 (e.g. apex.dequant._decode_e4m3) handle the decode.
        return np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy()
    raise ValueError(
        f"safetensors dtype {dtype!r} not handled by apex.safetensors_loader"
    )
