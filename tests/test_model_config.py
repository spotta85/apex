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
