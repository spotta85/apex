"""Tests for apex/checkpoint_index.py: list_tensor_names + get_tensor_shape.

Covers all three input shapes (single .safetensors, sharded directory,
single .gguf) for both queries, plus the error paths that the CLI
scripts depend on for clean failure messages.
"""

import json

import numpy as np
import pytest
from safetensors.numpy import save_file

from apex.checkpoint_index import get_tensor_shape, list_tensor_names
from tests._world import write_gguf_world, write_safetensors_world


# --- list_tensor_names: happy paths ---------------------------------------

def test_list_safetensors_keys(tmp_path):
    path = tmp_path / "fake.safetensors"
    write_safetensors_world(path)

    names = list_tensor_names(path)
    assert names == sorted(names)  # sorted contract
    assert "layers.0.gate.weight" in names
    assert "layers.0.experts.0.gate_proj.weight" in names
    assert "layers.1.experts.3.down_proj.weight" in names


def test_list_gguf_keys(tmp_path):
    path = tmp_path / "fake.gguf"
    write_gguf_world(path)

    names = list_tensor_names(path)
    assert names == sorted(names)
    assert "layers.0.gate.weight" in names
    assert "layers.0.experts.0.gate_proj.weight" in names


def test_list_sharded_safetensors_keys(tmp_path):
    # Two shards plus an index that splits tensors across them — the
    # "is the union sorted?" property needs both shards' keys.
    save_file(
        {"layers.0.gate.weight": np.zeros((4, 4), dtype="float32")},
        str(tmp_path / "model-00001-of-00002.safetensors"),
    )
    save_file(
        {"layers.1.gate.weight": np.zeros((4, 4), dtype="float32")},
        str(tmp_path / "model-00002-of-00002.safetensors"),
    )
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 0},
        "weight_map": {
            "layers.0.gate.weight": "model-00001-of-00002.safetensors",
            "layers.1.gate.weight": "model-00002-of-00002.safetensors",
        },
    }))

    names = list_tensor_names(tmp_path)
    assert names == ["layers.0.gate.weight", "layers.1.gate.weight"]


# --- list_tensor_names: error paths ----------------------------------------

def test_list_unsupported_extension_raises_value_error(tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"")
    with pytest.raises(ValueError) as excinfo:
        list_tensor_names(path)
    # The message has to be actionable — name the supported shapes.
    assert ".safetensors" in str(excinfo.value)
    assert ".gguf" in str(excinfo.value)


def test_list_directory_without_index_raises_file_not_found(tmp_path):
    # Empty directory, no *.safetensors.index.json.
    with pytest.raises(FileNotFoundError):
        list_tensor_names(tmp_path)


def test_list_directory_with_corrupt_index_raises_value_error(tmp_path):
    # Index file present but missing the required weight_map field.
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 0},
    }))
    with pytest.raises(ValueError) as excinfo:
        list_tensor_names(tmp_path)
    assert "weight_map" in str(excinfo.value)


# --- get_tensor_shape: happy paths -----------------------------------------

def test_get_tensor_shape_safetensors(tmp_path):
    path = tmp_path / "fake.safetensors"
    write_safetensors_world(path)

    # write_safetensors_world produces HF-oriented projections:
    # gate/up are (intermediate=16, hidden=8); down is (8, 16).
    assert get_tensor_shape(path, "layers.0.experts.0.gate_proj.weight") == (16, 8)
    assert get_tensor_shape(path, "layers.0.experts.0.down_proj.weight") == (8, 16)
    assert get_tensor_shape(path, "layers.0.gate.weight") == (8, 4)


def test_get_tensor_shape_gguf(tmp_path):
    path = tmp_path / "fake.gguf"
    write_gguf_world(path)

    assert get_tensor_shape(path, "layers.0.experts.0.gate_proj.weight") == (16, 8)
    assert get_tensor_shape(path, "layers.0.experts.0.down_proj.weight") == (8, 16)


def test_get_tensor_shape_sharded(tmp_path):
    # Build a sharded layout where one shard owns the gate and another
    # owns the expert; shape lookup must follow the index to the right
    # file rather than scanning the whole directory.
    save_file(
        {"layers.0.gate.weight": np.zeros((8, 4), dtype="float32")},
        str(tmp_path / "shard-a.safetensors"),
    )
    save_file(
        {"layers.0.experts.0.gate_proj.weight": np.zeros((8, 16), dtype="float32")},
        str(tmp_path / "shard-b.safetensors"),
    )
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 0},
        "weight_map": {
            "layers.0.gate.weight": "shard-a.safetensors",
            "layers.0.experts.0.gate_proj.weight": "shard-b.safetensors",
        },
    }))

    assert get_tensor_shape(tmp_path, "layers.0.gate.weight") == (8, 4)
    assert get_tensor_shape(tmp_path, "layers.0.experts.0.gate_proj.weight") == (8, 16)


# --- get_tensor_shape: error paths ----------------------------------------

def test_get_tensor_shape_missing_raises_key_error(tmp_path):
    # Sharded variant: tensor isn't in the index map.
    save_file(
        {"layers.0.gate.weight": np.zeros((8, 4), dtype="float32")},
        str(tmp_path / "shard.safetensors"),
    )
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 0},
        "weight_map": {"layers.0.gate.weight": "shard.safetensors"},
    }))

    with pytest.raises(KeyError) as excinfo:
        get_tensor_shape(tmp_path, "definitely.not.in.checkpoint")
    assert "definitely.not.in.checkpoint" in str(excinfo.value)


def test_get_tensor_shape_unsupported_extension_raises(tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"")
    with pytest.raises(ValueError):
        get_tensor_shape(path, "anything")
