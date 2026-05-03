"""Tests for the RawTensor / QuantizedTensor wrapper dataclasses.

Pure data classes — the tests are mostly there to pin the field
contract since downstream code (dequant, the quantized store) reaches
into these by name.
"""

import numpy as np

from apex.tensor_types import QuantizedTensor, RawTensor


def test_raw_tensor_holds_data_dtype_shape():
    arr = np.zeros((4, 8), dtype=np.float32)
    rt = RawTensor(data=arr, dtype="F32", shape=(4, 8))
    assert rt.data is arr
    assert rt.dtype == "F32"
    assert rt.shape == (4, 8)


def test_quantized_tensor_required_vs_optional_fields():
    arr = np.zeros((4, 8), dtype=np.uint8)
    qt = QuantizedTensor(data=arr, dtype="NVFP4", shape=(4, 8))
    # Required fields populated.
    assert qt.data is arr
    assert qt.dtype == "NVFP4"
    assert qt.shape == (4, 8)
    # Sidecars default to None — important for the dequantizer's
    # "missing scales" failure path.
    assert qt.input_scale is None
    assert qt.weight_scale is None
    assert qt.weight_scale_2 is None


def test_quantized_tensor_with_all_sidecars():
    qt = QuantizedTensor(
        data=np.zeros((1, 1), dtype=np.uint8),
        dtype="NVFP4",
        shape=(1, 1),
        input_scale=np.float32(0.5),
        weight_scale=np.zeros((1, 1), dtype=np.uint8),
        weight_scale_2=np.float32(0.25),
    )
    assert qt.input_scale == np.float32(0.5)
    assert qt.weight_scale.shape == (1, 1)
    assert qt.weight_scale_2 == np.float32(0.25)
