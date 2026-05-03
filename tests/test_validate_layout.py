"""Tests for scripts/validate_layout.py.

Each test invokes the script via subprocess so the full CLI path
(argparse, exit codes, stderr formatting) gets exercised — not just
internal helpers. The contract validate_layout enforces is critical
for the "did the layout actually match the checkpoint?" workflow,
so the failure messages are tested with as much care as the pass
cases.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from apex.weight_layout import FakeMoELayout
from tests._world import write_gguf_world, write_safetensors_world


SCRIPT = (
    Path(__file__).resolve().parent.parent / "scripts" / "validate_layout.py"
)


def run_validate(*args):
    """Run the script with the given args; return (returncode, stdout, stderr)."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout, result.stderr


# --- pass paths ------------------------------------------------------------

def test_passes_on_complete_fake_safetensors(tmp_path):
    path = tmp_path / "fake.safetensors"
    write_safetensors_world(path)

    rc, stdout, _ = run_validate(path, "--layout", "fake")
    assert rc == 0
    assert "layout validation passed" in stdout
    assert "[ok] gate (layer 0)" in stdout
    assert "detected: 2 layers x 4 experts" in stdout


def test_passes_on_complete_fake_gguf(tmp_path):
    path = tmp_path / "fake.gguf"
    write_gguf_world(path)

    rc, stdout, _ = run_validate(path, "--layout", "fake")
    assert rc == 0
    assert "layout validation passed" in stdout


def test_passes_on_sharded_directory(tmp_path):
    # Distribute fake-MoE tensors across two shards; the script should
    # still find every name via the index.
    layout = FakeMoELayout()
    rng = np.random.default_rng(0)

    shard_a_tensors = {layout.gate_weight(0): rng.standard_normal((8, 4)).astype("float32")}
    shard_b_tensors = {}
    for L in range(2):
        for E in range(4):
            for proj_key, name in layout.expert_tensors(L, E).items():
                # All experts go to shard B; gate stays in shard A.
                # HF orientation: gate/up are (intermediate, hidden); down is reversed.
                shape = (16, 8) if proj_key in ("gate_proj", "up_proj") else (8, 16)
                shard_b_tensors[name] = rng.standard_normal(shape).astype("float32")
        if L == 1:
            shard_a_tensors[layout.gate_weight(L)] = rng.standard_normal((8, 4)).astype("float32")

    save_file(shard_a_tensors, str(tmp_path / "shard-a.safetensors"))
    save_file(shard_b_tensors, str(tmp_path / "shard-b.safetensors"))
    weight_map = {n: "shard-a.safetensors" for n in shard_a_tensors}
    weight_map.update({n: "shard-b.safetensors" for n in shard_b_tensors})
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 0},
        "weight_map": weight_map,
    }))

    rc, stdout, _ = run_validate(tmp_path, "--layout", "fake")
    assert rc == 0
    assert "layout validation passed" in stdout


# --- --check-shapes --------------------------------------------------------

def test_check_shapes_reports_shapes(tmp_path):
    path = tmp_path / "fake.safetensors"
    write_safetensors_world(path)

    rc, stdout, _ = run_validate(path, "--layout", "fake", "--check-shapes")
    assert rc == 0
    # Spot-check that the shapes section landed and contains the
    # canonical fake-world dimensions in HF orientation.
    assert "shapes (expert 0,0):" in stdout
    assert "(16, 8)" in stdout  # gate_proj / up_proj: (intermediate, hidden)
    assert "(8, 16)" in stdout  # down_proj: (hidden, intermediate)
    assert "(8, 4)" in stdout   # gate router (hidden=8, num_experts=4)


def test_check_shapes_fails_on_inconsistent_projections(tmp_path):
    # Hand-craft a checkpoint where up_proj's shape doesn't match
    # gate_proj's. The SwiGLU math can't run, so --check-shapes must
    # surface it before any forward pass is attempted downstream.
    layout = FakeMoELayout()
    rng = np.random.default_rng(0)
    tensors = {layout.gate_weight(0): rng.standard_normal((8, 4)).astype("float32")}
    for E in range(4):
        names = layout.expert_tensors(0, E)
        # HF orientation: gate/up are (intermediate=16, hidden=8); down is (8, 16).
        tensors[names["gate_proj"]] = rng.standard_normal((16, 8)).astype("float32")
        # WRONG: up_proj has the wrong intermediate dim.
        tensors[names["up_proj"]] = rng.standard_normal((12, 8)).astype("float32")
        tensors[names["down_proj"]] = rng.standard_normal((8, 16)).astype("float32")

    path = tmp_path / "broken.safetensors"
    save_file(tensors, str(path))

    rc, stdout, stderr = run_validate(path, "--layout", "fake", "--check-shapes")
    assert rc == 1
    # Names check should have passed first (they're all there) — the
    # failure is purely about shape consistency.
    assert "[ok]" in stdout
    assert "MISMATCH" in stderr
    assert "(12, 8)" in stderr
    assert "(16, 8)" in stderr


# --- failure paths ---------------------------------------------------------

def test_fails_on_wrong_layout(tmp_path):
    # Fake checkpoint, mixtral layout — the gate format itself doesn't
    # match anything in the file, so we should bail out early with a
    # "wrong layout?" hint.
    path = tmp_path / "fake.safetensors"
    write_safetensors_world(path)

    rc, _, stderr = run_validate(path, "--layout", "mixtral")
    assert rc == 1
    assert "no gate weight matched layout 'mixtral'" in stderr
    assert "model.layers.0.block_sparse_moe.gate.weight" in stderr
    assert "inspect_weight_names.py" in stderr  # actionable hint


def test_fails_when_one_expert_projection_is_missing(tmp_path):
    # Build a complete fake-MoE checkpoint, then drop expert (1, 3)'s
    # down_proj before saving. The exact missing tensor name has to
    # show up in the failure output so the user can fix the layout
    # (or accept a broken checkpoint).
    layout = FakeMoELayout()
    rng = np.random.default_rng(0)
    tensors = {}
    for L in range(2):
        tensors[layout.gate_weight(L)] = rng.standard_normal((8, 4)).astype("float32")
        for E in range(4):
            names = layout.expert_tensors(L, E)
            # HF orientation: gate/up are (intermediate, hidden); down is reversed.
            tensors[names["gate_proj"]] = rng.standard_normal((16, 8)).astype("float32")
            tensors[names["up_proj"]] = rng.standard_normal((16, 8)).astype("float32")
            tensors[names["down_proj"]] = rng.standard_normal((8, 16)).astype("float32")

    missing_name = layout.expert_down_proj(1, 3)
    del tensors[missing_name]

    path = tmp_path / "incomplete.safetensors"
    save_file(tensors, str(path))

    rc, stdout, stderr = run_validate(path, "--layout", "fake")
    assert rc == 1
    # The missing-marker line goes to stdout as part of the per-name
    # checklist; the summary failure goes to stderr.
    assert "[MISSING]" in stdout
    assert missing_name in stdout
    assert "FAILED" in stderr


def test_fails_when_layer_zero_gate_missing(tmp_path):
    # Drop only the layer-0 gate (keep layer 1's). The probe finds no
    # gate at layer 0, scans forward, finds layer 1's gate, treats
    # layer 1 as the sole layer. Then the per-name check on
    # gate (layer 0) fails.
    layout = FakeMoELayout()
    rng = np.random.default_rng(0)
    tensors = {layout.gate_weight(1): rng.standard_normal((8, 4)).astype("float32")}
    for E in range(4):
        names = layout.expert_tensors(1, E)
        # HF orientation: gate/up are (intermediate, hidden); down is reversed.
        tensors[names["gate_proj"]] = rng.standard_normal((16, 8)).astype("float32")
        tensors[names["up_proj"]] = rng.standard_normal((16, 8)).astype("float32")
        tensors[names["down_proj"]] = rng.standard_normal((8, 16)).astype("float32")

    path = tmp_path / "no_layer_zero.safetensors"
    save_file(tensors, str(path))

    rc, stdout, stderr = run_validate(path, "--layout", "fake")
    assert rc == 1
    assert "[MISSING] gate (layer 0)" in stdout
    assert "FAILED" in stderr


# --- argparse plumbing -----------------------------------------------------

def test_unknown_layout_is_rejected_by_argparse(tmp_path):
    path = tmp_path / "fake.safetensors"
    write_safetensors_world(path)

    rc, _, stderr = run_validate(path, "--layout", "not-a-layout")
    # argparse exits 2 for invalid choice values.
    assert rc == 2
    assert "invalid choice" in stderr or "choose from" in stderr


def test_no_layout_argument_is_rejected_by_argparse(tmp_path):
    path = tmp_path / "fake.safetensors"
    write_safetensors_world(path)

    rc, _, stderr = run_validate(path)
    assert rc == 2
    assert "--layout" in stderr


def test_nonexistent_path_returns_error(tmp_path):
    rc, _, stderr = run_validate(tmp_path / "nope.safetensors", "--layout", "fake")
    assert rc == 1
    assert "does not exist" in stderr
