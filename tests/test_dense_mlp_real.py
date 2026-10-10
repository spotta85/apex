"""Step 24B: dense-MLP shared path tests on real Gemma 4.

Confirms that:
  - the layout exposes dense MLP tensor names for Gemma 4
  - load_dense_mlp_weights returns correctly-shaped fp32 arrays
  - MoEModel.from_pretrained populates per_layer_dense_mlp
  - the dense path actually contributes (output differs vs disabled)

The "actually contributes" test is the structural sanity that step 24B
is doing real work — without it a regression that silently disabled
the dense branch would only surface as quality drift (hard to spot).

We do NOT test that quality improves — that's the operator's
judgement after running scripts/benchmark_quality.py (step 24F).
"""

import json

import numpy as np
import pytest

import apex
from apex.checkpoint_loaders import load_dense_mlp_weights
from apex.weight_layout import get_layout
from tests._synthetic_nvfp4 import (
    E4M3_ONE,
    real_checkpoint_path,
    skip_if_no_real_checkpoint,
    write_safetensors,
)


@skip_if_no_real_checkpoint()
def test_layout_exposes_dense_mlp_for_gemma_hf():
    """All three projections must be exposed (shape consistency)."""
    layout = get_layout("gemma-hf-moe")
    gate = layout.dense_mlp_gate_proj(0)
    up = layout.dense_mlp_up_proj(0)
    down = layout.dense_mlp_down_proj(0)
    for names in (gate, up, down):
        assert names is not None
        # The layout names all three; `biases` exists on disk only in
        # 8-bit-affine uploads (see the encoding tests below).
        assert {"weight", "scales", "biases"} <= set(names.keys())


@skip_if_no_real_checkpoint()
def test_load_dense_mlp_weights_returns_expected_shapes():
    """Dequant produces (intermediate=2112, hidden=2816) for gate/up,
    (hidden=2816, intermediate=2112) for down — HF orientation."""
    layout = get_layout("gemma-hf-moe")
    weights = load_dense_mlp_weights(real_checkpoint_path(), layout, 0)
    assert weights is not None
    assert weights["gate_weight"].shape == (2112, 2816)
    assert weights["up_weight"].shape == (2112, 2816)
    assert weights["down_weight"].shape == (2816, 2112)
    for w in weights.values():
        assert w.dtype == np.float32


def test_load_dense_mlp_weights_returns_none_for_layout_without_one():
    """Mixtral / Qwen / fake layouts return None — back-compat preserved."""
    for name in ("fake", "mixtral", "qwen-moe"):
        layout = get_layout(name)
        # layout.dense_mlp_gate_proj returns None ⇒ loader short-circuits.
        # We pass a path that doesn't matter because the loader bails
        # before any disk I/O.
        result = load_dense_mlp_weights("/nonexistent", layout, 0)
        assert result is None, f"layout {name!r} unexpectedly returned dense MLP"


# --- Synthetic encoding tests (always on) ----------------------------------
# The dense MLP's on-disk encoding differs between uploads of the same
# mlx-community repo: 8-bit affine (weight + scales + biases) before the
# 2026-07 re-upload, NVFP4 (weight + scales only) after. The loader has
# to read whichever one the checkpoint carries.

_SYN_DIM = 16  # hidden == intermediate keeps all three projections (16, 16)


def _write_dense_mlp_checkpoint(tmp_path, items):
    """Write `items` as a one-shard sharded-safetensors directory."""
    shard = tmp_path / "model-00001-of-00001.safetensors"
    write_safetensors(shard, items)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 0},
        "weight_map": {name: shard.name for name, *_ in items},
    }))


def test_load_dense_mlp_weights_nvfp4_encoding(tmp_path):
    """NVFP4 dense MLP: U32-packed 4-bit weight + U8 E4M3 scales, no biases."""
    layout = get_layout("gemma-hf-moe")
    # FP4 code 0x2 is 1.0, eight codes per uint32; one E4M3 scale byte
    # per 16 values. Distinct scales per projection (1.0 / 2.0 / 0.5)
    # prove each projection is read from its own tensors.
    weight = np.full((_SYN_DIM, _SYN_DIM // 8), 0x22222222, dtype=np.uint32)
    items = []
    for names, scale_byte in (
        (layout.dense_mlp_gate_proj(0), E4M3_ONE),   # 1.0
        (layout.dense_mlp_up_proj(0), 0x40),         # 2.0
        (layout.dense_mlp_down_proj(0), 0x30),       # 0.5
    ):
        scales = np.full((_SYN_DIM, 1), scale_byte, dtype=np.uint8)
        items.append((names["weight"], "U32", list(weight.shape), weight.tobytes()))
        items.append((names["scales"], "U8", list(scales.shape), scales.tobytes()))
    _write_dense_mlp_checkpoint(tmp_path, items)

    weights = load_dense_mlp_weights(tmp_path, layout, 0)

    for key, expected in (
        ("gate_weight", 1.0), ("up_weight", 2.0), ("down_weight", 0.5),
    ):
        assert weights[key].shape == (_SYN_DIM, _SYN_DIM)
        assert weights[key].dtype == np.float32
        np.testing.assert_array_equal(weights[key], expected)


def test_load_dense_mlp_weights_affine_encoding(tmp_path):
    """8-bit affine dense MLP: U32-packed bytes + per-group scales + biases."""
    layout = get_layout("gemma-hf-moe")
    # Every byte is q=3; reconstruction is q * scale + bias = 3*2 + 1.
    weight = np.full((_SYN_DIM, _SYN_DIM // 4), 0x03030303, dtype=np.uint32)
    scales = np.full((_SYN_DIM, 1), 2.0, dtype=np.float32)
    biases = np.full((_SYN_DIM, 1), 1.0, dtype=np.float32)
    items = []
    for names in (
        layout.dense_mlp_gate_proj(0),
        layout.dense_mlp_up_proj(0),
        layout.dense_mlp_down_proj(0),
    ):
        items.append((names["weight"], "U32", list(weight.shape), weight.tobytes()))
        items.append((names["scales"], "F32", list(scales.shape), scales.tobytes()))
        items.append((names["biases"], "F32", list(biases.shape), biases.tobytes()))
    _write_dense_mlp_checkpoint(tmp_path, items)

    weights = load_dense_mlp_weights(tmp_path, layout, 0)

    for key in ("gate_weight", "up_weight", "down_weight"):
        assert weights[key].shape == (_SYN_DIM, _SYN_DIM)
        np.testing.assert_array_equal(weights[key], 7.0)


# --- MoEModel integration tests --------------------------------------------

@pytest.fixture(scope="module")
def model_2l_with_dense():
    """2-layer model — small enough to fit a fixture, both layers sliding."""
    return apex.MoEModel.from_pretrained(
        real_checkpoint_path(),
        max_memory_gb=4.0,
        num_layers=2,
        verbose=False,
    )


@skip_if_no_real_checkpoint()
def test_moemodel_loads_dense_mlp_per_layer(model_2l_with_dense):
    """Each loaded layer must carry a dense MLP weights dict."""
    assert len(model_2l_with_dense.per_layer_dense_mlp) == 2
    for L in range(2):
        dense = model_2l_with_dense.per_layer_dense_mlp[L]
        assert dense is not None
        assert "gate_weight" in dense
        assert "up_weight" in dense
        assert "down_weight" in dense


@skip_if_no_real_checkpoint()
def test_dense_mlp_changes_logits_vs_disabled(model_2l_with_dense):
    """forward(prompt) with dense_mlp on must differ from dense_mlp off.

    Patches per_layer_dense_mlp to all-None to disable the parallel
    branch, runs forward, restores, and compares to the production
    forward. If the two are byte-identical the dense branch is a no-op
    (regression).
    """
    prompt = "Hello"
    logits_on = model_2l_with_dense.forward(prompt)
    saved = model_2l_with_dense.per_layer_dense_mlp
    model_2l_with_dense.per_layer_dense_mlp = [None] * model_2l_with_dense.num_layers
    try:
        logits_off = model_2l_with_dense.forward(prompt)
    finally:
        model_2l_with_dense.per_layer_dense_mlp = saved
    assert not np.allclose(logits_on, logits_off, atol=1e-3), (
        "dense MLP path produces identical logits to no-dense — branch is "
        "either a no-op or contributing zero"
    )


@skip_if_no_real_checkpoint()
def test_dense_mlp_keeps_logits_finite(model_2l_with_dense):
    """End-to-end finite check — dense branch must not introduce NaN/inf."""
    logits = model_2l_with_dense.forward("Hello")
    assert np.all(np.isfinite(logits))
