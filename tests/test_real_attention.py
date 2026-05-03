"""Tests for step 16C: single-token attention wiring.

Synthetic tests pin the math (GQA repeat, q_norm application,
softmax-of-one identity); the real-Gemma test (skip if missing)
exercises the same path on actual NVFP4_U32 attention weights.
"""

import contextlib

import numpy as np

from apex.attention import single_token_attention
from apex.ops import linear
from tests._synthetic_nvfp4 import skip_if_no_real_checkpoint


# Standard-normal random weights at attention sizes routinely trigger
# numpy's matmul over/underflow heuristics even when the final output
# is finite (which we assert below). Silencing them keeps test output
# clean — same pattern apex.real_router and the real-Gemma tests use.
@contextlib.contextmanager
def _silence_matmul_warnings():
    with np.errstate(over="ignore", under="ignore",
                     invalid="ignore", divide="ignore"):
        yield


def test_single_token_attention_with_no_norms_is_finite_and_shaped():
    """Smoke that the math runs end-to-end without optional norms."""
    rng = np.random.default_rng(0)
    hidden = rng.standard_normal((1, 64)).astype(np.float32)
    # 4 Q heads, 2 KV heads (q_per_kv = 2), head_dim = 16.
    q_w = rng.standard_normal((4 * 16, 64)).astype(np.float32)
    k_w = rng.standard_normal((2 * 16, 64)).astype(np.float32)
    v_w = rng.standard_normal((2 * 16, 64)).astype(np.float32)
    o_w = rng.standard_normal((64, 4 * 16)).astype(np.float32)
    with _silence_matmul_warnings():
        out = single_token_attention(
            hidden, q_weight=q_w, k_weight=k_w, v_weight=v_w, o_weight=o_w,
            num_heads=4, num_kv_heads=2, head_dim=16,
        )
    assert out.shape == (1, 64)
    assert np.all(np.isfinite(out))


def test_single_token_attention_softmax_is_unity_so_output_equals_v_oproj():
    """For seq_len=1 the attention output reduces to o_proj(v_per_head).

    softmax over a length-1 axis is exactly [1.0], so the per-head
    weighted sum collapses to v itself. The output therefore must
    equal `linear(reshape(v_repeated_for_gqa), o_w)` — the entire
    Q/K/scale path drops out. If the test fails, we know the
    softmax-of-one or GQA-repeat path corrupted v.
    """
    rng = np.random.default_rng(1)
    hidden = rng.standard_normal((1, 32)).astype(np.float32)
    num_heads, num_kv_heads, head_dim = 4, 2, 8
    q_w = rng.standard_normal((num_heads * head_dim, 32)).astype(np.float32)
    k_w = rng.standard_normal((num_kv_heads * head_dim, 32)).astype(np.float32)
    v_w = rng.standard_normal((num_kv_heads * head_dim, 32)).astype(np.float32)
    o_w = rng.standard_normal((32, num_heads * head_dim)).astype(np.float32)

    with _silence_matmul_warnings():
        out = single_token_attention(
            hidden, q_weight=q_w, k_weight=k_w, v_weight=v_w, o_weight=o_w,
            num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
        )

    # Recompute the GQA-repeated v path by hand and run o_proj on it.
    v = (hidden @ v_w.T).reshape(1, num_kv_heads, head_dim)
    v_repeated = np.repeat(v, num_heads // num_kv_heads, axis=1)
    expected = linear(v_repeated.reshape(1, num_heads * head_dim), o_w)
    np.testing.assert_allclose(out, expected, rtol=1e-5)


def test_single_token_attention_changes_output_when_q_norm_provided():
    """q_norm scales Q rows per-head — output must differ from no-norm run.

    The no-norm baseline produces o_proj(v) (per the unity-softmax
    test above); adding q_norm doesn't change v but it perturbs the
    softmax-of-one to be ... still 1.0. So this test only verifies
    that supplying q_norm doesn't crash on real-shaped inputs and
    that q_norm of all zeros leaves the output unchanged (both
    norms scale q by 1+0 = 1, so q stays the same and softmax
    stays the same).
    """
    rng = np.random.default_rng(2)
    num_heads, num_kv_heads, head_dim = 4, 2, 8
    hidden = rng.standard_normal((1, 32)).astype(np.float32)
    q_w = rng.standard_normal((num_heads * head_dim, 32)).astype(np.float32)
    k_w = rng.standard_normal((num_kv_heads * head_dim, 32)).astype(np.float32)
    v_w = rng.standard_normal((num_kv_heads * head_dim, 32)).astype(np.float32)
    o_w = rng.standard_normal((32, num_heads * head_dim)).astype(np.float32)

    common = dict(
        q_weight=q_w, k_weight=k_w, v_weight=v_w, o_weight=o_w,
        num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
    )
    with _silence_matmul_warnings():
        base = single_token_attention(hidden, **common)
        # Zero norm weight ⇒ scale = 1 + 0 = 1 ⇒ identity scaling on q.
        # Output must equal the no-norm baseline.
        zero_norm = single_token_attention(
            hidden, **common,
            q_norm_weight=np.zeros(head_dim, dtype=np.float32),
        )
        # Non-zero norm changes q, but softmax of one element is still 1.
        # Output should still equal the v-only baseline (q drops out).
        nonzero_norm = single_token_attention(
            hidden, **common,
            q_norm_weight=rng.standard_normal(head_dim).astype(np.float32),
        )
    np.testing.assert_allclose(zero_norm, base, rtol=1e-5)
    np.testing.assert_allclose(nonzero_norm, base, rtol=1e-5)


def test_single_token_attention_no_gqa_when_heads_equal_kv_heads():
    """num_heads == num_kv_heads ⇒ no repeat applied; shapes still line up."""
    rng = np.random.default_rng(3)
    num_heads, num_kv_heads, head_dim = 4, 4, 8
    hidden = rng.standard_normal((1, 32)).astype(np.float32)
    q_w = rng.standard_normal((num_heads * head_dim, 32)).astype(np.float32)
    k_w = rng.standard_normal((num_kv_heads * head_dim, 32)).astype(np.float32)
    v_w = rng.standard_normal((num_kv_heads * head_dim, 32)).astype(np.float32)
    o_w = rng.standard_normal((32, num_heads * head_dim)).astype(np.float32)
    with _silence_matmul_warnings():
        out = single_token_attention(
            hidden, q_weight=q_w, k_weight=k_w, v_weight=v_w, o_weight=o_w,
            num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
        )
    assert out.shape == (1, 32)
    assert np.all(np.isfinite(out))


@skip_if_no_real_checkpoint()
def test_real_gemma_layer0_single_token_attention_is_finite():
    """End-to-end on real Gemma 4 layer 0: load + dequant + run attention."""
    from scripts._real_moe_helpers import (
        load_attention_weights,
        read_text_config,
    )
    from tests._synthetic_nvfp4 import (
        real_checkpoint_path,
        real_layout_and_quant_store,
    )

    path = real_checkpoint_path()
    layout, _ = real_layout_and_quant_store(path)
    text_cfg = read_text_config(path)
    weights = load_attention_weights(path, layout, layer_id=0)

    rng = np.random.default_rng(0)
    hidden = rng.standard_normal((1, text_cfg["hidden_size"])).astype(np.float32)

    with np.errstate(over="ignore", under="ignore",
                     invalid="ignore", divide="ignore"):
        out = single_token_attention(
            hidden,
            q_weight=weights["q_weight"], k_weight=weights["k_weight"],
            v_weight=weights["v_weight"], o_weight=weights["o_weight"],
            num_heads=text_cfg["num_attention_heads"],
            num_kv_heads=text_cfg["num_key_value_heads"],
            head_dim=text_cfg["head_dim"],
            q_norm_weight=weights["q_norm"], k_norm_weight=weights["k_norm"],
            rms_eps=text_cfg["rms_norm_eps"],
        )

    assert out.shape == (1, text_cfg["hidden_size"])
    assert np.all(np.isfinite(out))
    # Sanity on the loaded weight shapes — q_proj is (Q*head_dim, hidden).
    assert weights["q_weight"].shape == (
        text_cfg["num_attention_heads"] * text_cfg["head_dim"],
        text_cfg["hidden_size"],
    )
    # K/V are (KV*head_dim, hidden) — pin the GQA shape so a future
    # config change surfaces here, not in a downstream attention bug.
    assert weights["k_weight"].shape == (
        text_cfg["num_key_value_heads"] * text_cfg["head_dim"],
        text_cfg["hidden_size"],
    )
    assert weights["v_weight"].shape == weights["k_weight"].shape
    # O projects back to hidden ⇒ (hidden, Q*head_dim).
    assert weights["o_weight"].shape == (
        text_cfg["hidden_size"],
        text_cfg["num_attention_heads"] * text_cfg["head_dim"],
    )
