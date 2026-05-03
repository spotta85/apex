"""Tests for step 16D: full decoder layer composition.

Synthetic tests pin the residual + sandwich-norm structure (zero
norms ⇒ identity; zero MoE call ⇒ output equals attention residual).
The real-Gemma test (skipif) drives layer 0 end-to-end and asserts
finite + correctly shaped output.
"""

import contextlib

import numpy as np

from apex.attention import single_token_attention
from apex.decoder import decoder_decode_step, single_token_decoder_layer
from apex.kv_cache import KVCache
from apex.ops import rms_norm
from tests._synthetic_nvfp4 import skip_if_no_real_checkpoint


@contextlib.contextmanager
def _silence_matmul_warnings():
    with np.errstate(over="ignore", under="ignore",
                     invalid="ignore", divide="ignore"):
        yield


def _attn_kwargs(rng, hidden_dim=64, num_heads=4, num_kv_heads=2, head_dim=16):
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


def _zero_norms(hidden_dim: int):
    # gemma_unit_offset=True with weight=0 ⇒ scale = (1 + 0) = 1, so
    # the norm becomes "RMSNorm with weight 1" = pure normalization.
    z = np.zeros(hidden_dim, dtype=np.float32)
    return {"input": z, "post_attn": z, "pre_ffn": z, "post_ffn": z}


def test_decoder_with_identity_moe_call_runs_end_to_end():
    """Identity MoE callable ⇒ decoder still produces finite, shaped output."""
    rng = np.random.default_rng(0)
    hidden = rng.standard_normal((1, 64)).astype(np.float32)
    norms = _zero_norms(64)
    attn_kwargs = _attn_kwargs(rng)

    with _silence_matmul_warnings():
        out = single_token_decoder_layer(
            hidden,
            norms=norms,
            attn_kwargs=attn_kwargs,
            moe_call=lambda h: h,    # identity passes the FFN sub-block
        )

    assert out.shape == (1, 64)
    assert np.all(np.isfinite(out))


def test_decoder_residual_carries_attention_output_through_to_final():
    """Residual structure: with zero MoE output, final = (resid_after_attn) + 0.

    Specifically: if moe_call returns zeros, the FFN sub-block
    contributes only the post_ffn-normalized zero, which is 0. So the
    decoder's output equals the attention sub-block's residual sum.
    """
    rng = np.random.default_rng(1)
    hidden = rng.standard_normal((1, 64)).astype(np.float32)
    norms = _zero_norms(64)
    attn_kwargs = _attn_kwargs(rng)

    with _silence_matmul_warnings():
        full = single_token_decoder_layer(
            hidden,
            norms=norms,
            attn_kwargs=attn_kwargs,
            moe_call=lambda h: np.zeros_like(h),
        )
        # Recompute the attention sub-block by hand.
        h_normed = rms_norm(
            hidden, norms["input"], eps=1e-6, gemma_unit_offset=True,
        )
        attn_out = single_token_attention(h_normed, **attn_kwargs)
        attn_normed = rms_norm(
            attn_out, norms["post_attn"], eps=1e-6, gemma_unit_offset=True,
        )
        attention_residual = hidden + attn_normed

    np.testing.assert_allclose(full, attention_residual, rtol=1e-5, atol=1e-6)


def test_decoder_calls_moe_callable_with_normalized_hidden():
    """moe_call receives the pre-FFN-normalized hidden, not raw hidden."""
    rng = np.random.default_rng(2)
    hidden = rng.standard_normal((1, 64)).astype(np.float32)
    norms = _zero_norms(64)
    attn_kwargs = _attn_kwargs(rng)
    captured = {}

    def spying_moe(h):
        captured["arg"] = h.copy()
        return np.zeros_like(h)

    with _silence_matmul_warnings():
        single_token_decoder_layer(
            hidden, norms=norms, attn_kwargs=attn_kwargs, moe_call=spying_moe,
        )

    # The captured arg must NOT equal the original hidden — there's a
    # pre_ffn rms_norm applied first. With gemma_unit_offset=True and
    # weight=0 the norm is pure RMS-normalization, which changes magnitudes.
    assert "arg" in captured
    assert captured["arg"].shape == hidden.shape
    assert not np.allclose(captured["arg"], hidden)


# --- decoder_decode_step (with KVCache) ---------------------------------

def test_decoder_decode_step_first_call_matches_single_token_layer():
    """First call (empty cache, RoPE = identity) ⇒ identical output."""
    rng = np.random.default_rng(10)
    hidden = rng.standard_normal((1, 64)).astype(np.float32)
    norms = _zero_norms(64)
    attn_kwargs = _attn_kwargs(rng)

    cache = KVCache(num_layers=1)
    moe = lambda h: np.zeros_like(h)

    with _silence_matmul_warnings():
        decode_out = decoder_decode_step(
            hidden,
            norms=norms,
            attn_kwargs=attn_kwargs,
            moe_call=moe,
            kv_cache=cache,
            layer_id=0,
        )
        single_out = single_token_decoder_layer(
            hidden,
            norms=norms,
            attn_kwargs=attn_kwargs,
            moe_call=moe,
        )
    np.testing.assert_allclose(decode_out, single_out, rtol=1e-5)
    # And the cache must have grown by exactly one entry.
    assert cache.length(0) == 1


def test_decoder_decode_step_grows_kv_cache_per_call():
    rng = np.random.default_rng(11)
    norms = _zero_norms(64)
    attn_kwargs = _attn_kwargs(rng)
    cache = KVCache(num_layers=1)
    moe = lambda h: np.zeros_like(h)

    with _silence_matmul_warnings():
        for step in range(4):
            hidden = rng.standard_normal((1, 64)).astype(np.float32)
            out = decoder_decode_step(
                hidden,
                norms=norms,
                attn_kwargs=attn_kwargs,
                moe_call=moe,
                kv_cache=cache,
                layer_id=0,
            )
            assert out.shape == (1, 64)
            assert np.all(np.isfinite(out))
            assert cache.length(0) == step + 1


@skip_if_no_real_checkpoint()
def test_real_gemma_layer0_decoder_decode_step_grows_cache():
    """3 decode steps on real Gemma 4 layer 0 with KV cache."""
    from apex.backends.fake_backend import FakeBackend
    from apex.dequant import DequantizingExpertStore
    from apex.expert_cache import ExpertCache
    from apex.expert_manager import ExpertManager
    from apex.metrics import Metrics
    from apex.moe_layer import run_moe_layer
    from apex.real_router import MultiLayerRealRouter
    from apex.checkpoint_loaders import (
        load_attention_weights,
        load_layer_norms,
        read_text_config,
    )
    from scripts._real_moe_helpers import quantized_store_for_layout
    from tests._synthetic_nvfp4 import (
        real_checkpoint_path,
        real_layout_and_quant_store,
    )

    path = real_checkpoint_path()
    layout, _ = real_layout_and_quant_store(path)
    text_cfg = read_text_config(path)

    metrics = Metrics()
    cache = ExpertCache(max_memory_mb=512, verbose=False, metrics=metrics)
    quant_store = quantized_store_for_layout(path, layout)
    store = DequantizingExpertStore(quant_store, verbose=False)
    manager = ExpertManager(store=store, cache=cache, metrics=metrics)
    backend = FakeBackend()
    router = MultiLayerRealRouter(path, layout, top_k=8)

    attn_weights = load_attention_weights(path, layout, layer_id=0)
    norms = load_layer_norms(path, layout, layer_id=0)
    attn_kwargs = dict(
        q_weight=attn_weights["q_weight"], k_weight=attn_weights["k_weight"],
        v_weight=attn_weights["v_weight"], o_weight=attn_weights["o_weight"],
        num_heads=text_cfg["num_attention_heads"],
        num_kv_heads=text_cfg["num_key_value_heads"],
        head_dim=text_cfg["head_dim"],
        q_norm_weight=attn_weights["q_norm"],
        k_norm_weight=attn_weights["k_norm"],
        rms_eps=text_cfg["rms_norm_eps"],
        rope_theta=10000.0,
    )

    def moe_call(h):
        with _silence_matmul_warnings():
            ids, weights = router.route(0, h)
            return run_moe_layer(
                h, ids, weights,
                lambda eid: manager.ensure_loaded(0, eid),
                backend,
            )

    kv = KVCache(num_layers=1)
    rng = np.random.default_rng(0)
    for step in range(3):
        hidden = rng.standard_normal(
            (1, text_cfg["hidden_size"]),
        ).astype(np.float32)
        with _silence_matmul_warnings():
            out = decoder_decode_step(
                hidden,
                norms=norms,
                attn_kwargs=attn_kwargs,
                moe_call=moe_call,
                kv_cache=kv,
                layer_id=0,
                rms_eps=text_cfg["rms_norm_eps"],
            )
        assert out.shape == (1, text_cfg["hidden_size"])
        assert np.all(np.isfinite(out))
        assert kv.length(0) == step + 1


@skip_if_no_real_checkpoint()
def test_real_gemma_layer0_decoder_produces_finite_output():
    """End-to-end on real Gemma 4 layer 0: attn + MoE through the decoder."""
    from apex.backends.fake_backend import FakeBackend
    from apex.dequant import DequantizingExpertStore
    from apex.expert_cache import ExpertCache
    from apex.expert_manager import ExpertManager
    from apex.metrics import Metrics
    from apex.moe_layer import run_moe_layer
    from apex.real_router import MultiLayerRealRouter
    from apex.checkpoint_loaders import (
        load_attention_weights,
        load_layer_norms,
        read_text_config,
    )
    from scripts._real_moe_helpers import quantized_store_for_layout
    from tests._synthetic_nvfp4 import (
        real_checkpoint_path,
        real_layout_and_quant_store,
    )

    path = real_checkpoint_path()
    layout, _ = real_layout_and_quant_store(path)
    text_cfg = read_text_config(path)

    metrics = Metrics()
    cache = ExpertCache(max_memory_mb=512, verbose=False, metrics=metrics)
    quant_store = quantized_store_for_layout(path, layout)
    store = DequantizingExpertStore(quant_store, verbose=False)
    manager = ExpertManager(store=store, cache=cache, metrics=metrics)
    backend = FakeBackend()
    router = MultiLayerRealRouter(path, layout, top_k=8)

    attn_weights = load_attention_weights(path, layout, layer_id=0)
    norms = load_layer_norms(path, layout, layer_id=0)

    rng = np.random.default_rng(0)
    hidden = rng.standard_normal((1, text_cfg["hidden_size"])).astype(np.float32)

    def moe_call(h):
        with _silence_matmul_warnings():
            ids, weights = router.route(0, h)
            return run_moe_layer(
                h, ids, weights,
                lambda eid: manager.ensure_loaded(0, eid),
                backend,
            )

    attn_kwargs = dict(
        q_weight=attn_weights["q_weight"], k_weight=attn_weights["k_weight"],
        v_weight=attn_weights["v_weight"], o_weight=attn_weights["o_weight"],
        num_heads=text_cfg["num_attention_heads"],
        num_kv_heads=text_cfg["num_key_value_heads"],
        head_dim=text_cfg["head_dim"],
        q_norm_weight=attn_weights["q_norm"],
        k_norm_weight=attn_weights["k_norm"],
        rms_eps=text_cfg["rms_norm_eps"],
    )

    with _silence_matmul_warnings():
        out = single_token_decoder_layer(
            hidden,
            norms=norms,
            attn_kwargs=attn_kwargs,
            moe_call=moe_call,
            rms_eps=text_cfg["rms_norm_eps"],
        )

    assert out.shape == (1, text_cfg["hidden_size"])
    assert np.all(np.isfinite(out))
    # Sanity that the cache actually saw work (8 cold misses for the
    # MoE sub-block); if zero, our moe_call wiring is broken.
    assert metrics.cache_misses == 8, metrics.cache_misses
    # Norm shapes must match hidden_size.
    for key in ("input", "post_attn", "pre_ffn", "post_ffn"):
        assert norms[key].shape == (text_cfg["hidden_size"],)
