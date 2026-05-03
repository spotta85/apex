"""Tests for GGUFTensorIndex: name-keyed tensor lookup over a .gguf file."""

import numpy as np
import pytest
from gguf import GGUFWriter

from apex.gguf_tensor_index import GGUFTensorIndex


def write_fake_gguf(path, tensors):
    # Helper: write a {name: ndarray} dict to a .gguf file with the
    # canonical writer phases. Always uses arch='probe' for tests.
    writer = GGUFWriter(str(path), "probe")
    for name, arr in tensors.items():
        writer.add_tensor(name, arr)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_has_and_names_inventory(tmp_path):
    # Index should report exactly the tensors present in the file.
    path = tmp_path / "x.gguf"
    write_fake_gguf(
        path,
        {
            "a.weight": np.zeros((2, 2), dtype="float32"),
            "b.weight": np.zeros((4, 1), dtype="float32"),
        },
    )

    idx = GGUFTensorIndex(path)
    assert idx.has("a.weight")
    assert idx.has("b.weight")
    assert not idx.has("missing")
    assert sorted(idx.names()) == ["a.weight", "b.weight"]


def test_tensor_returns_correct_data(tmp_path):
    # Round-trip: writing an ndarray and reading it through the index
    # gives an array with matching shape/dtype/values.
    arr = np.arange(8 * 5, dtype="float32").reshape(8, 5)
    path = tmp_path / "x.gguf"
    write_fake_gguf(path, {"layers.0.experts.1.weight": arr})

    idx = GGUFTensorIndex(path)
    loaded = idx.tensor("layers.0.experts.1.weight")
    assert loaded.shape == (8, 5)
    assert loaded.dtype == np.float32
    np.testing.assert_array_equal(np.asarray(loaded), arr)


def test_metadata_reports_numpy_shape_not_ggml_shape(tmp_path):
    # Important quirk: ReaderTensor.shape is reversed (ggml convention).
    # Our metadata() must report the natural numpy shape so callers
    # don't have to know about ggml's ne[] ordering.
    path = tmp_path / "x.gguf"
    write_fake_gguf(path, {"t": np.zeros((3, 7), dtype="float32")})

    idx = GGUFTensorIndex(path)
    md = idx.metadata("t")
    assert md["shape"] == (3, 7)
    assert md["dtype"] == "float32"
    assert md["element_count"] == 21
    assert md["name"] == "t"


def test_missing_tensor_raises_key_error(tmp_path):
    # Asking for a tensor not in the file -> KeyError mentioning the
    # name and the file path.
    path = tmp_path / "x.gguf"
    write_fake_gguf(path, {"present.weight": np.zeros((1, 1), dtype="float32")})

    idx = GGUFTensorIndex(path)
    with pytest.raises(KeyError) as excinfo:
        idx.tensor("absent.weight")
    msg = str(excinfo.value)
    assert "absent.weight" in msg
    assert str(path) in msg


def test_tensor_data_is_mmap_backed(tmp_path):
    # GGUFReader uses mmap; the returned data should be an np.memmap so
    # callers get the same lazy-page-in behavior as MMapExpertStore.
    path = tmp_path / "x.gguf"
    write_fake_gguf(path, {"t": np.zeros((4, 4), dtype="float32")})

    idx = GGUFTensorIndex(path)
    loaded = idx.tensor("t")
    assert isinstance(loaded, np.memmap)
