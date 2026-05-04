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
def test_real_from_pretrained_caps_num_layers_at_total(real_model):
    """num_layers > total decoder layer count is rejected with a clear message.

    Step 23 lifted the sliding-only cap; both attention kinds are
    supported now, but asking for more layers than the checkpoint has
    is still a fast ValueError.
    """
    with pytest.raises(ValueError, match="exceeds the .* decoder layers"):
        apex.MoEModel.from_pretrained(
            real_checkpoint_path(),
            max_memory_gb=2.0,
            num_layers=999,
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


# --- generate_chat (step 22B) ----------------------------------------------

@skip_if_no_real_checkpoint()
def test_real_generate_chat_returns_string(real_model):
    """End-to-end: messages in, string out, type str. Same shape as generate."""
    text = real_model.generate_chat(
        [{"role": "user", "content": "Hello"}],
        max_tokens=2,
        verbose=False,
    )
    assert isinstance(text, str)


@skip_if_no_real_checkpoint()
def test_generate_chat_stops_on_eot_token(real_model, monkeypatch):
    """When the model predicts `<turn|>` (id 106), generate_chat must stop.

    Patch `_make_step_fn` to return a step that always emits id 106
    (the Gemma 4 EOT). With max_tokens=8, the generated list should
    have length 1 (just the EOT) — proving the chat-mode stop set
    includes EOT and triggers the early break.
    """
    # Sanity: the real tokenizer carries the EOT id we expect.
    assert real_model.tokenizer.eot_token_id == 106

    eot_id = real_model.tokenizer.eot_token_id

    def fake_step_fn(self, kv, *, temperature):
        # Returns a step closure that always predicts EOT, ignoring input.
        def step(_token_id, _eot=eot_id):
            return _eot
        return step

    monkeypatch.setattr(
        type(real_model), "_make_step_fn", fake_step_fn, raising=True,
    )
    real_model.generate_chat(
        [{"role": "user", "content": "Hello"}],
        max_tokens=8, verbose=False,
    )
    # Just the first EOT — the loop must NOT keep generating.
    assert real_model._last_generated_ids == [eot_id]


@skip_if_no_real_checkpoint()
def test_generate_does_not_stop_on_eot_token(real_model, monkeypatch):
    """Plain `generate` must NOT treat EOT as a stop token.

    Confirms the chat stop-set is `generate_chat`-only — `generate`
    keeps stopping just on EOS, since EOT has no special meaning
    outside chat mode.
    """
    eot_id = real_model.tokenizer.eot_token_id

    def fake_step_fn(self, kv, *, temperature):
        def step(_token_id, _eot=eot_id):
            return _eot
        return step

    monkeypatch.setattr(
        type(real_model), "_make_step_fn", fake_step_fn, raising=True,
    )
    real_model.generate("Hello", max_tokens=4, verbose=False)
    # Plain generate should run for the full max_tokens — every step
    # returns EOT but EOT isn't in the (EOS-only) stop set.
    assert len(real_model._last_generated_ids) == 4
    assert all(i == eot_id for i in real_model._last_generated_ids)


@skip_if_no_real_checkpoint()
def test_real_generate_chat_uses_more_prompt_tokens_than_plain(real_model):
    """Templated render expands the prompt — KV must reflect more tokens.

    `apply_chat_template` wraps the user content in `<|turn>user…<turn|>`
    markers + opens the model turn, so the encoded id list is strictly
    longer than `encode("Hello")`. This catches the failure mode where
    `generate_chat` accidentally falls back to plain encoding.
    """
    real_model.generate("Hello", max_tokens=1, verbose=False)
    plain_kv = real_model.memory_report().kv_cache_bytes
    real_model.generate_chat(
        [{"role": "user", "content": "Hello"}],
        max_tokens=1, verbose=False,
    )
    chat_kv = real_model.memory_report().kv_cache_bytes
    assert chat_kv > plain_kv, (
        f"chat KV {chat_kv} should exceed plain KV {plain_kv} "
        "(chat-template adds turn markers)"
    )
