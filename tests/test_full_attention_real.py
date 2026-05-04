"""Step 23: real-Gemma full_attention generation tests.

Locks in the step 23 unlock — `num_layers > num_sliding_layers` (5 on
Gemma 4) now works. Layers 5/11/17/23/29 use full_attention with
K=V tied, head_dim=512, num_kv_heads=2, partial RoPE (factor=0.25,
theta=1e6).

Module-scope fixtures amortize the expensive embedding dequant
(~3 GB) across all tests; each fixture builds at one specific
num_layers.
"""

import pytest

import apex
from tests._synthetic_nvfp4 import (
    real_checkpoint_path,
    skip_if_no_real_checkpoint,
)


@pytest.fixture(scope="module")
def model_6l():
    """Smallest model that includes a full_attention layer (layer 5)."""
    return apex.MoEModel.from_pretrained(
        real_checkpoint_path(),
        max_memory_gb=4.0,
        num_layers=6,
        verbose=False,
    )


@pytest.fixture(scope="module")
def model_8l():
    """8 layers: 5 sliding + 1 full + 2 sliding (layer 5 is the only full)."""
    return apex.MoEModel.from_pretrained(
        real_checkpoint_path(),
        max_memory_gb=4.0,
        num_layers=8,
        verbose=False,
    )


@skip_if_no_real_checkpoint()
def test_full_attention_layer_actually_loaded(model_6l):
    """Layer 5's loaded attn dict must have v_weight=None (K=V tied)."""
    assert model_6l.config.is_full_attention(5)
    assert model_6l.per_layer_attn[5]["v_weight"] is None
    # Layer 0 (sliding) still has v_weight.
    assert model_6l.per_layer_attn[0]["v_weight"] is not None


@skip_if_no_real_checkpoint()
def test_full_attention_layer_uses_correct_head_dim(model_6l):
    """Layer 5's q_weight shape must reflect head_dim=512, not 256."""
    cfg = model_6l.config
    # q_proj output dim = num_heads * head_dim_full = 16 * 512 = 8192.
    expected_q_out = cfg.num_attention_heads * cfg.head_dim_full
    q_w = model_6l.per_layer_attn[5]["q_weight"]
    assert q_w.shape[0] == expected_q_out, (
        f"layer 5 q_weight out-dim {q_w.shape[0]} != expected {expected_q_out}"
    )
    # k_proj output dim = num_kv_heads_full * head_dim_full = 2 * 512 = 1024.
    expected_k_out = cfg.num_kv_heads_full * cfg.head_dim_full
    k_w = model_6l.per_layer_attn[5]["k_weight"]
    assert k_w.shape[0] == expected_k_out, (
        f"layer 5 k_weight out-dim {k_w.shape[0]} != expected {expected_k_out}"
    )


@skip_if_no_real_checkpoint()
def test_generate_with_full_attention_returns_string(model_6l):
    """6-layer generation (layers 0-4 sliding + 5 full) returns plain str."""
    text = model_6l.generate("Hello", max_tokens=2, verbose=False)
    assert isinstance(text, str)


@skip_if_no_real_checkpoint()
def test_generate_with_full_attention_respects_budget(model_6l):
    model_6l.generate("Hello", max_tokens=2, verbose=False)
    report = model_6l.memory_report()
    assert report.expert.budget_respected


@skip_if_no_real_checkpoint()
def test_full_attention_layer_kv_is_smaller_than_sliding(model_6l):
    """Full layer KV per token: 2 * 512 * 2 * 4 = 8192 bytes (vs sliding 16384).

    On Gemma 4: sliding has 8 KV heads × 256 head_dim = 2048 dims per K;
    full has 2 KV heads × 512 head_dim = 1024 dims per K. So a full
    layer's KV is half the size of a sliding layer's per token.
    """
    model_6l.generate("Hello", max_tokens=1, verbose=False)
    per_layer = model_6l.memory_report().kv.per_layer_bytes
    # Layers 0-4 (sliding) should all share the same nonzero size.
    sliding_sizes = set(per_layer[L] for L in range(5))
    assert len(sliding_sizes) == 1
    sliding_per_layer = sliding_sizes.pop()
    # Layer 5 (full) is exactly half the sliding size.
    assert per_layer[5] * 2 == sliding_per_layer, (
        f"full layer KV {per_layer[5]} != sliding {sliding_per_layer} / 2"
    )


@skip_if_no_real_checkpoint()
def test_generate_with_8_layers_includes_one_full(model_8l):
    """8-layer model has full layer 5 in the middle; gen still finite."""
    # Sanity that the layer mix is what we expect.
    full_layers = [
        L for L in range(8) if model_8l.config.is_full_attention(L)
    ]
    assert full_layers == [5]
    text = model_8l.generate("Hello", max_tokens=2, verbose=False)
    assert isinstance(text, str)


@skip_if_no_real_checkpoint()
def test_generate_8_layers_logits_are_finite(model_8l):
    """Two prompts; each forward must produce finite logits."""
    import numpy as np
    for prompt in ("Hello", "The capital of France is"):
        logits = model_8l.forward(prompt)
        assert np.all(np.isfinite(logits))


@skip_if_no_real_checkpoint()
def test_eight_layers_bumps_kv_total_above_six(model_6l, model_8l):
    """Adding more layers should add more KV (no shape regression)."""
    model_6l.generate("Hello", max_tokens=1, verbose=False)
    model_8l.generate("Hello", max_tokens=1, verbose=False)
    assert (
        model_8l.memory_report().kv_cache_bytes
        > model_6l.memory_report().kv_cache_bytes
    )
