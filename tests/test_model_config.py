"""Tests for apex/config.py — ModelConfig.from_path on real Gemma 4."""

import pytest

from apex.config import ModelConfig
from tests._synthetic_nvfp4 import (
    real_checkpoint_path,
    skip_if_no_real_checkpoint,
)


def test_from_path_missing_dir_raises_clear_error(tmp_path):
    bogus = tmp_path / "no-checkpoint-here"
    with pytest.raises(FileNotFoundError, match="checkpoint directory"):
        ModelConfig.from_path(bogus)


@skip_if_no_real_checkpoint()
def test_from_path_real_gemma_populates_known_fields():
    cfg = ModelConfig.from_path(real_checkpoint_path())
    assert cfg.hidden_size == 2816
    assert cfg.num_attention_heads == 16
    assert cfg.num_kv_heads == 8
    assert cfg.head_dim == 256
    assert cfg.num_experts == 128
    assert cfg.top_k_experts == 8
    assert cfg.bos_token_id == 2
    assert cfg.final_logit_softcapping == 30.0
    assert cfg.rope_theta_sliding == 10000.0
    # eos_token_id may be scalar (1) or list — must be a tuple either way.
    assert isinstance(cfg.eos_token_ids, tuple)
    assert len(cfg.eos_token_ids) >= 1


@skip_if_no_real_checkpoint()
def test_num_sliding_layers_counts_leading_sliding_layers():
    cfg = ModelConfig.from_path(real_checkpoint_path())
    # Gemma 4: layers 0-4 sliding, layer 5 full → 5 leading sliding.
    assert cfg.num_sliding_layers == 5
    assert cfg.layer_types[5] != "sliding_attention"


@skip_if_no_real_checkpoint()
def test_from_path_is_frozen():
    cfg = ModelConfig.from_path(real_checkpoint_path())
    with pytest.raises(Exception):
        cfg.hidden_size = 1  # frozen dataclass disallows assignment


# --- Step 23A: full_attention extras --------------------------------------

@skip_if_no_real_checkpoint()
def test_full_attention_extras_match_gemma4_known_values():
    cfg = ModelConfig.from_path(real_checkpoint_path())
    assert cfg.head_dim_full == 512
    assert cfg.num_kv_heads_full == 2
    assert cfg.attention_k_eq_v is True
    assert cfg.partial_rotary_factor_full == 0.25
    assert cfg.rope_theta_full == 1_000_000.0


@skip_if_no_real_checkpoint()
def test_per_layer_accessors_distinguish_sliding_vs_full():
    cfg = ModelConfig.from_path(real_checkpoint_path())
    # Layer 0 = sliding
    assert not cfg.is_full_attention(0)
    assert cfg.head_dim_for_layer(0) == cfg.head_dim
    assert cfg.num_kv_heads_for_layer(0) == cfg.num_kv_heads
    assert cfg.rope_theta_for_layer(0) == cfg.rope_theta_sliding
    assert cfg.rotary_dim_for_layer(0) == cfg.head_dim  # full rotation
    assert not cfg.attention_k_eq_v_for_layer(0)
    # Layer 5 = full
    assert cfg.is_full_attention(5)
    assert cfg.head_dim_for_layer(5) == 512
    assert cfg.num_kv_heads_for_layer(5) == 2
    assert cfg.rope_theta_for_layer(5) == 1_000_000.0
    # 512 * 0.25 = 128 rotated dims
    assert cfg.rotary_dim_for_layer(5) == 128
    assert cfg.attention_k_eq_v_for_layer(5)


@skip_if_no_real_checkpoint()
def test_out_of_range_layer_treated_as_sliding():
    """Defensive: OOB layer_id falls back to sliding semantics, no crash."""
    cfg = ModelConfig.from_path(real_checkpoint_path())
    assert not cfg.is_full_attention(9999)
    assert cfg.head_dim_for_layer(9999) == cfg.head_dim


# --- Step 24C: sliding-window scope -----------------------------------------

@skip_if_no_real_checkpoint()
def test_sliding_window_parsed_from_text_config():
    """Gemma 4 sets `sliding_window=1024`; ModelConfig must surface it."""
    cfg = ModelConfig.from_path(real_checkpoint_path())
    assert cfg.sliding_window == 1024


@skip_if_no_real_checkpoint()
def test_sliding_window_for_layer_distinguishes_sliding_vs_full():
    """Sliding layers cap at the configured window; full layers attend globally."""
    cfg = ModelConfig.from_path(real_checkpoint_path())
    # Layer 0 is sliding → window cap.
    assert cfg.sliding_window_for_layer(0) == 1024
    # Layer 5 is full_attention → no cap (None ⇒ attend globally).
    assert cfg.sliding_window_for_layer(5) is None
