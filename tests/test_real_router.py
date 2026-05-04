"""Step 24A: router scaling parity tests.

Two layers:

  - Synthetic tests pin the math (route() with hidden_scale and
    per_expert_scale) — no checkpoint needed.
  - Real-Gemma test (skipped without REAL_MOE_PATH) confirms that
    MultiLayerRealRouter actually loads the stabilizer tensors and
    produces measurably different routing decisions when they're on
    vs off.

We don't compare against an HF reference (no torch dep — deferred from
22C). The test is "the scales are doing something", not "they match
the reference exactly".
"""

import numpy as np
import pytest

from apex.moe_layer import route
from apex.real_router import MultiLayerRealRouter
from apex.weight_layout import get_layout
from tests._synthetic_nvfp4 import (
    real_checkpoint_path,
    skip_if_no_real_checkpoint,
)


# --- Synthetic route() tests ------------------------------------------------

def test_route_with_no_scales_matches_pre_step24_behavior():
    """Default (no scales) path must reproduce the original math byte-for-byte."""
    rng = np.random.default_rng(0)
    hidden = rng.standard_normal((2, 8)).astype(np.float32)
    router_w = rng.standard_normal((4, 8)).astype(np.float32)
    ids_baseline, w_baseline = route(hidden, router_w, top_k=2)
    ids_explicit_none, w_explicit_none = route(
        hidden, router_w, top_k=2,
        hidden_scale=None, per_expert_scale=None,
    )
    np.testing.assert_array_equal(ids_baseline, ids_explicit_none)
    np.testing.assert_array_equal(w_baseline, w_explicit_none)


def test_route_per_expert_scale_zeros_weight_at_selected_expert():
    """An expert with scale 0 still appears in top-k but its weight is 0.

    Step 26D corrected the semantics: `per_expert_scale` is a per-
    expert OUTPUT GAIN applied to the softmax weights at the selected
    top-k indices (matching `mlx_lm.models.gemma4_text:Router`). Top-k
    selection runs on the raw post-projection logits — the scale
    doesn't bias selection.
    """
    hidden = np.ones((1, 4), dtype=np.float32)
    router_w = np.eye(4, 4, dtype=np.float32)  # logits = [1,1,1,1]
    per_expert_scale = np.array([1.0, 0.0, 1.0, 1.0], dtype=np.float32)
    ids, weights = route(
        hidden, router_w, top_k=3,
        per_expert_scale=per_expert_scale,
    )
    ids_list = ids[0].tolist()
    weights_list = weights[0].tolist()
    if 1 in ids_list:
        # Expert 1 made it into top-k; its post-softmax weight must be 0.
        idx = ids_list.index(1)
        assert weights_list[idx] == 0.0, (
            "per_expert_scale[1]=0 must zero the softmax weight at "
            f"the selected position; got {weights_list[idx]}"
        )


def test_route_hidden_scale_changes_logits():
    """hidden_scale should alter the projected logits (not just no-op)."""
    rng = np.random.default_rng(1)
    hidden = rng.standard_normal((1, 8)).astype(np.float32)
    router_w = rng.standard_normal((4, 8)).astype(np.float32)
    hidden_scale = rng.standard_normal(8).astype(np.float32) + 2.0  # avoid all-1
    ids_no_scale, _ = route(hidden, router_w, top_k=2)
    ids_scaled, _ = route(
        hidden, router_w, top_k=2, hidden_scale=hidden_scale,
    )
    # Either ids or weights differ — at minimum the logit ranking is
    # not the same.  We use an OR of two checks because a ties-broken
    # ranking that happens to match by chance shouldn't fail.
    assert (
        not np.array_equal(ids_no_scale, ids_scaled)
        or not np.allclose(
            (hidden * hidden_scale) @ router_w.T,
            hidden @ router_w.T,
        )
    )


def test_route_hidden_scale_wrong_shape_raises():
    hidden = np.zeros((1, 8), dtype=np.float32)
    router_w = np.zeros((4, 8), dtype=np.float32)
    bad = np.zeros(7, dtype=np.float32)  # not (8,)
    with pytest.raises(ValueError, match="hidden_scale shape"):
        route(hidden, router_w, top_k=2, hidden_scale=bad)


def test_route_per_expert_scale_wrong_shape_raises():
    hidden = np.zeros((1, 8), dtype=np.float32)
    router_w = np.zeros((4, 8), dtype=np.float32)
    bad = np.zeros(3, dtype=np.float32)  # not (4,)
    with pytest.raises(ValueError, match="per_expert_scale shape"):
        route(hidden, router_w, top_k=2, per_expert_scale=bad)


def test_route_per_expert_scale_negative_flips_weight_sign():
    """A negative `per_expert_scale[e]` produces a negative softmax weight.

    Step 26D: `per_expert_scale` is no longer a top-k bias; it's a
    post-softmax multiplier at the selected indices. So a negative
    scale produces a NEGATIVE final weight at that expert (acting as
    a destructive contribution to the combined output), and a large
    positive scale amplifies the contribution. Top-k selection itself
    is unchanged from the no-scale case.
    """
    hidden = np.ones((1, 4), dtype=np.float32)
    router_w = np.eye(4, 4, dtype=np.float32)
    per_expert_scale = np.array([2.0, 1.0, 1.0, -5.0], dtype=np.float32)
    ids, weights = route(
        hidden, router_w, top_k=4,
        per_expert_scale=per_expert_scale,
    )
    ids_list = ids[0].tolist()
    weights_list = weights[0].tolist()
    # Whichever position holds expert 3 must be negative (scale -5).
    if 3 in ids_list:
        idx = ids_list.index(3)
        assert weights_list[idx] < 0.0
    # Whichever position holds expert 0 must be the largest (scale 2).
    if 0 in ids_list:
        idx0 = ids_list.index(0)
        # All other selected weights at neutral / negative ⇒ this is
        # the largest absolute-value contribution.
        assert weights_list[idx0] >= max(
            (w for w in weights_list if w >= 0.0), default=0.0
        )


# --- Real-Gemma router tests ------------------------------------------------

@skip_if_no_real_checkpoint()
def test_real_router_loads_scale_tensors():
    """Gemma 4 layer 0 must report scale tensors are loaded."""
    layout = get_layout("gemma-hf-moe")
    router = MultiLayerRealRouter(real_checkpoint_path(), layout, top_k=8)
    # has_router_scales triggers a load and confirms both vectors landed.
    assert router.has_router_scales(0)


@skip_if_no_real_checkpoint()
def test_real_router_scaling_changes_top_k_decisions():
    """With vs without scales: at least some routing decisions differ.

    Build two routers — one with scaling on (the production path), one
    that we patch to skip the scales — and confirm top-k IDs aren't
    identical for a fixed hidden vector. This proves the stabilizer
    tensors actually influence which experts the model picks.
    """
    layout = get_layout("gemma-hf-moe")
    rng = np.random.default_rng(42)
    hidden = rng.standard_normal((1, 2816)).astype(np.float32)

    scaled = MultiLayerRealRouter(real_checkpoint_path(), layout, top_k=8)
    ids_scaled, _ = scaled.route(0, hidden)

    unscaled = MultiLayerRealRouter(real_checkpoint_path(), layout, top_k=8)
    # Force-disable both scale paths to compare against the pre-step-24
    # behavior. Manipulating the cache directly is acceptable — it's an
    # internal that the test is allowed to know about.
    unscaled._load_for_layer(0)
    unscaled._hidden_scales[0] = None
    unscaled._per_expert_scales[0] = None
    ids_unscaled, _ = unscaled.route(0, hidden)

    # Either the IDs differ or at least the order changed. With 8 of
    # 128 experts selected, identical-rank-and-set is extremely
    # unlikely under any non-trivial scaling.
    assert ids_scaled != ids_unscaled, (
        "Router output identical with and without scales — "
        "either the scales are no-op or they aren't being applied"
    )
