"""Tests for the user-facing apex.MoEModel API.

Three groups:

  - pure-import smokes (apex.MoEModel actually exists at the top
    level, no real checkpoint needed)
  - clear-error smokes (synthetic tmp_path checkpoints to trigger
    the operator-facing failure modes)
  - real-Gemma round trips (skipped when the local checkpoint is
    absent, gated by `skip_if_no_real_checkpoint`)

Real-Gemma tests share a single `MoEModel` via a session-scoped fixture
because `from_pretrained` dequantizes the 3 GB embedding table — way
too slow to repeat across tests.
"""

import pytest

import apex
from tests._synthetic_nvfp4 import (
    real_checkpoint_path,
    skip_if_no_real_checkpoint,
)


def test_moemodel_importable_from_apex_top_level():
    """`from apex import MoEModel` must work — that's the headline API."""
    assert hasattr(apex, "MoEModel")
    assert apex.MoEModel is not None


def test_apex_top_level_exports_match_all():
    """Every name in __all__ resolves; no typos in the public surface."""
    for name in apex.__all__:
        assert hasattr(apex, name), f"apex.__all__ promises {name} but it's missing"


def test_from_pretrained_missing_path_raises_clear_error(tmp_path):
    """Pointing at a directory that doesn't exist should fail with a clear msg."""
    bogus = tmp_path / "does-not-exist"
    with pytest.raises(FileNotFoundError, match="checkpoint directory not found"):
        apex.MoEModel.from_pretrained(bogus)


def test_from_pretrained_unsupported_backend_raises():
    """Asking for a backend we don't ship is a fast ValueError."""
    with pytest.raises(ValueError, match="backend"):
        # Path won't be reached — backend validation runs first.
        apex.MoEModel.from_pretrained("/nonexistent", backend="cuda")


def test_from_pretrained_nonpositive_memory_raises():
    """Zero or negative memory budget is a config bug, not a runtime decision."""
    with pytest.raises(ValueError, match="max_memory_gb must be positive"):
        apex.MoEModel.from_pretrained("/nonexistent", max_memory_gb=0)


# --- Real-Gemma round-trip tests --------------------------------------------

@pytest.fixture(scope="module")
def real_model():
    """Session-shared MoEModel; built once because embedding dequant is slow."""
    return apex.MoEModel.from_pretrained(
        real_checkpoint_path(),
        max_memory_gb=2.0,
        num_layers=1,
        verbose=False,
    )


@skip_if_no_real_checkpoint()
def test_real_from_pretrained_loads_config(real_model):
    """The ModelConfig surfaces Gemma 4's known hyperparameters."""
    cfg = real_model.config
    assert cfg.hidden_size == 2816
    assert cfg.num_attention_heads == 16
    assert cfg.num_kv_heads == 8
    assert cfg.head_dim == 256
    assert cfg.num_experts == 128
    assert cfg.top_k_experts == 8
    assert cfg.bos_token_id == 2
    # Gemma 4: layers 0-4 sliding, layer 5 full.
    assert cfg.num_sliding_layers == 5


@skip_if_no_real_checkpoint()
def test_real_from_pretrained_caps_num_layers_at_sliding(real_model):
    """num_layers > sliding cap is rejected with a clear message."""
    with pytest.raises(ValueError, match="exceeds the .* sliding-attention"):
        apex.MoEModel.from_pretrained(
            real_checkpoint_path(),
            max_memory_gb=2.0,
            num_layers=99,
            verbose=False,
        )


@skip_if_no_real_checkpoint()
def test_real_generate_returns_string(real_model):
    """End-to-end: text in, text out, type str."""
    text = real_model.generate("Hello", max_tokens=2, verbose=False)
    assert isinstance(text, str)


@skip_if_no_real_checkpoint()
def test_real_generate_respects_max_tokens(real_model):
    """At most `max_tokens` token ids are appended (may be fewer if EOS hit).

    Counts via `_last_generated_ids` directly — re-encoding the decoded
    text via the tokenizer doesn't round-trip when the model (running
    only one layer) emits non-printable byte sequences that BPE
    re-segments into more tokens than were originally generated.
    """
    real_model.generate("Hello", max_tokens=3, verbose=False)
    assert len(real_model._last_generated_ids) <= 3


@skip_if_no_real_checkpoint()
def test_real_memory_report_after_generate_includes_kv(real_model):
    """After a generate() call, the joined report has nonzero KV bytes."""
    real_model.generate("Hello", max_tokens=2, verbose=False)
    report = real_model.memory_report()
    # Expert cache may be 0 if all experts evicted; KV must be populated.
    assert report.kv_cache_bytes > 0
    assert report.total_runtime_bytes >= report.kv_cache_bytes


@skip_if_no_real_checkpoint()
def test_real_generate_zero_max_tokens_raises(real_model):
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        real_model.generate("Hello", max_tokens=0, verbose=False)
