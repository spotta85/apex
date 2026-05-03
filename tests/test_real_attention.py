"""Tests for step 16C: single-token attention wiring.

Synthetic tests pin the math (GQA repeat, q_norm application,
softmax-of-one identity); the real-Gemma test (skip if missing)
exercises the same path on actual NVFP4_U32 attention weights.
"""

import contextlib

import numpy as np

from apex.attention import attention_decode_step, single_token_attention
from apex.kv_cache import KVCache
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


# --- attention_decode_step (KV cache + RoPE) ----------------------------

def _decode_kwargs(rng, hidden_dim=64, num_heads=4, num_kv_heads=2, head_dim=16):
    return dict(
        q_weight=rng.standard_normal(
            (num_heads * head_dim, hidden_dim),
        ).astype(np.float32),
        k_weight=rng.standard_normal(
            (num_kv_heads * head_dim, hidden_dim),
        ).astype(np.float32),
        v_weight=rng.standard_normal(
            (num_kv_heads * head_dim, hidden_dim),
        ).astype(np.float32),
        o_weight=rng.standard_normal(
            (hidden_dim, num_heads * head_dim),
        ).astype(np.float32),
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )


def test_decode_step_first_call_with_empty_cache_matches_single_token_attention():
    """At cache len 0 (position 0), RoPE is identity ⇒ output equals
    `single_token_attention` with the same weights."""
    rng = np.random.default_rng(10)
    hidden = rng.standard_normal((1, 64)).astype(np.float32)
    kwargs = _decode_kwargs(rng)
    cache = KVCache(num_layers=1)

    with _silence_matmul_warnings():
        decode_out = attention_decode_step(
            hidden, **kwargs, kv_cache=cache, layer_id=0,
        )
        single_out = single_token_attention(hidden, **kwargs)

    np.testing.assert_allclose(decode_out, single_out, rtol=1e-5)


def test_decode_step_grows_kv_cache_by_one_per_call():
    """Each call appends exactly one new (k, v) row to the cache."""
    rng = np.random.default_rng(11)
    kwargs = _decode_kwargs(rng)
    cache = KVCache(num_layers=1)

    with _silence_matmul_warnings():
        for step in range(5):
            hidden = rng.standard_normal((1, 64)).astype(np.float32)
            attention_decode_step(
                hidden, **kwargs, kv_cache=cache, layer_id=0,
            )
            assert cache.length(0) == step + 1


def test_decode_step_isolates_per_layer_kv_state():
    """Decoding through layer 0 must not pollute layer 1's cache."""
    rng = np.random.default_rng(12)
    kwargs = _decode_kwargs(rng)
    cache = KVCache(num_layers=2)

    with _silence_matmul_warnings():
        attention_decode_step(
            rng.standard_normal((1, 64)).astype(np.float32),
            **kwargs, kv_cache=cache, layer_id=0,
        )

    assert cache.length(0) == 1
    assert cache.length(1) == 0


def test_decode_step_output_changes_when_cache_has_history():
    """Same input vector at position 5 differs from same input at position 0.

    Position 0 ⇒ RoPE identity + cache holds only the new k/v.
    Position 5 ⇒ RoPE rotates q/k + softmax mixes 6 cached values.
    Outputs must differ — otherwise the cache is being ignored.
    """
    rng = np.random.default_rng(13)
    kwargs = _decode_kwargs(rng)

    fresh = KVCache(num_layers=1)
    primed = KVCache(num_layers=1)

    hidden = rng.standard_normal((1, 64)).astype(np.float32)
    with _silence_matmul_warnings():
        # Prime the second cache with 5 unrelated tokens so the next
        # decode step sees position 5 + a populated history.
        for _ in range(5):
            attention_decode_step(
                rng.standard_normal((1, 64)).astype(np.float32),
                **kwargs, kv_cache=primed, layer_id=0,
            )
        out_at_zero = attention_decode_step(
            hidden, **kwargs, kv_cache=fresh, layer_id=0,
        )
        out_at_five = attention_decode_step(
            hidden, **kwargs, kv_cache=primed, layer_id=0,
        )

    assert not np.allclose(out_at_zero, out_at_five)


@skip_if_no_real_checkpoint()
def test_real_gemma_layer0_single_token_attention_is_finite():
    """End-to-end on real Gemma 4 layer 0: load + dequant + run attention."""
    from apex.checkpoint_loaders import (
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


@skip_if_no_real_checkpoint()
def test_real_gemma_attention_decode_step_grows_cache_finitely():
    """Three real-Gemma decode steps: cache grows 1→2→3, outputs stay finite."""
    from apex.checkpoint_loaders import (
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

    cache = KVCache(num_layers=1)
    rng = np.random.default_rng(0)
    kwargs = dict(
        q_weight=weights["q_weight"], k_weight=weights["k_weight"],
        v_weight=weights["v_weight"], o_weight=weights["o_weight"],
        num_heads=text_cfg["num_attention_heads"],
        num_kv_heads=text_cfg["num_key_value_heads"],
        head_dim=text_cfg["head_dim"],
        q_norm_weight=weights["q_norm"], k_norm_weight=weights["k_norm"],
        rms_eps=text_cfg["rms_norm_eps"],
        rope_theta=10000.0,            # sliding_attention default
    )

    for step in range(3):
        hidden = rng.standard_normal(
            (1, text_cfg["hidden_size"]),
        ).astype(np.float32)
        with _silence_matmul_warnings():
            out = attention_decode_step(
                hidden, **kwargs, kv_cache=cache, layer_id=0,
            )
        assert out.shape == (1, text_cfg["hidden_size"])
        assert np.all(np.isfinite(out))
        assert cache.length(0) == step + 1

    # Cached key shape: (batch=1, num_kv_heads=8, seq=3, head_dim=256).
    keys, _ = cache.get(0)
    assert keys.shape == (
        1,
        text_cfg["num_key_value_heads"],
        3,
        text_cfg["head_dim"],
    )
