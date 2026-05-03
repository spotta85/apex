"""Tests for scripts/inspect_weight_names.py.

Covers the three input shapes the script supports:
  - single .safetensors file
  - directory of sharded safetensors with model.safetensors.index.json
  - single .gguf file

Each test invokes the script via subprocess so it exercises the full
CLI path (sys.argv parsing, exit codes, stdout format), not just the
internal helper functions.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from tests._world import write_gguf_world, write_safetensors_world


SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "inspect_weight_names.py"


def run_inspect(path):
    """Run the script with the given path; return (returncode, stdout, stderr)."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(path)],
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout, result.stderr


# --- single .safetensors ---------------------------------------------------

def test_lists_keys_from_single_safetensors_file(tmp_path):
    path = tmp_path / "fake.safetensors"
    write_safetensors_world(path)

    rc, stdout, _ = run_inspect(path)
    assert rc == 0
    # write_safetensors_world produces these tensor names via
    # FakeMoELayout. Spot-check that some appear.
    names = stdout.strip().splitlines()
    assert "layers.0.gate.weight" in names
    assert "layers.0.experts.0.gate_proj.weight" in names
    assert "layers.1.experts.3.down_proj.weight" in names


def test_safetensors_output_is_sorted(tmp_path):
    path = tmp_path / "fake.safetensors"
    write_safetensors_world(path)

    rc, stdout, _ = run_inspect(path)
    assert rc == 0
    names = stdout.strip().splitlines()
    assert names == sorted(names)


# --- sharded safetensors directory ----------------------------------------

def test_lists_keys_across_sharded_safetensors(tmp_path):
    # Build two real shards plus a synthetic index that splits tensors
    # across them. The script must read the index and union both
    # shards' key sets — there's no "everything's in one file" shortcut.
    shard1 = tmp_path / "model-00001-of-00002.safetensors"
    shard2 = tmp_path / "model-00002-of-00002.safetensors"

    save_file(
        {"layers.0.gate.weight": np.zeros((4, 4), dtype="float32")},
        str(shard1),
    )
    save_file(
        {"layers.1.gate.weight": np.zeros((4, 4), dtype="float32")},
        str(shard2),
    )

    index = {
        "metadata": {"total_size": 64},
        "weight_map": {
            "layers.0.gate.weight": shard1.name,
            "layers.1.gate.weight": shard2.name,
        },
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

    rc, stdout, _ = run_inspect(tmp_path)
    assert rc == 0
    names = stdout.strip().splitlines()
    assert names == ["layers.0.gate.weight", "layers.1.gate.weight"]


def test_directory_without_index_returns_helpful_error(tmp_path):
    # No index file at all — script should fail loudly, not silently
    # treat the empty dir as "zero tensors".
    rc, _, stderr = run_inspect(tmp_path)
    assert rc == 1
    assert "safetensors.index.json" in stderr


# --- single .gguf ----------------------------------------------------------

def test_lists_keys_from_single_gguf_file(tmp_path):
    path = tmp_path / "fake.gguf"
    write_gguf_world(path)

    rc, stdout, _ = run_inspect(path)
    assert rc == 0
    names = stdout.strip().splitlines()
    assert "layers.0.gate.weight" in names
    assert "layers.0.experts.0.gate_proj.weight" in names


# --- error paths -----------------------------------------------------------

def test_nonexistent_path_returns_error(tmp_path):
    rc, _, stderr = run_inspect(tmp_path / "nope.safetensors")
    assert rc == 1
    assert "does not exist" in stderr


def test_unsupported_extension_returns_error(tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"")
    rc, _, stderr = run_inspect(path)
    assert rc == 1
    assert "expected a .safetensors file" in stderr


def test_no_args_returns_usage_error():
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "usage:" in result.stderr
