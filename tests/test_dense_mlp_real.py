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

import numpy as np
import pytest

import apex
from apex.checkpoint_loaders import load_dense_mlp_weights
from apex.weight_layout import get_layout
from tests._synthetic_nvfp4 import (
    real_checkpoint_path,
    skip_if_no_real_checkpoint,
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
        # AFFINE_UINT8_U32 quantization: weight + scales + biases trio.
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
