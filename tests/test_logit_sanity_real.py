"""Step 22C / 24E: structural + plausibility logit sanity on real Gemma 4.

Without an HF / MLX reference baseline we can't compare logits
exactly. These tests instead lock in *structural* properties — the
runtime is producing plausibly-shaped logits and varies its output
across prompts. Catches regressions like:

  - all logits NaN/inf
  - argmax always same id regardless of input (degenerate)
  - logits exceed softcap (softcap accidentally bypassed)
  - shape drift (vocab_size mismatch with config)

The last section holds the quality bar: the full 30-layer model,
prompted through the chat template, must give the right answer. It
replaced step 24E's `xfail` raw-completion probes (see the note there).

Exact numerical agreement with mlx_lm — every layer, every position —
is checked by `scripts/parity_mlx.py`, which runs each model in its own
process because the two don't fit in memory together.
"""

import numpy as np
import pytest

import apex
from tests._synthetic_nvfp4 import (
    real_checkpoint_path,
    skip_if_no_real_checkpoint,
)


# Module-scope so the ~3 GB embedding dequant happens once for all
# logit-sanity tests. Use 4 layers so we exercise multi-layer KV +
# multiple expert lookups per token.
@pytest.fixture(scope="module")
def model_4l():
    return apex.MoEModel.from_pretrained(
        real_checkpoint_path(),
        max_memory_gb=4.0,
        num_layers=4,
        verbose=False,
    )


@skip_if_no_real_checkpoint()
def test_forward_returns_correctly_shaped_logits(model_4l):
    logits = model_4l.forward("Hello")
    assert logits.shape == (1, model_4l.config.vocab_size)


@skip_if_no_real_checkpoint()
def test_forward_logits_are_finite(model_4l):
    logits = model_4l.forward("Hello")
    assert np.all(np.isfinite(logits))


@skip_if_no_real_checkpoint()
def test_forward_respects_softcap(model_4l):
    """Gemma's `final_logit_softcapping=30.0` ⇒ |logits| ≤ 30."""
    logits = model_4l.forward("Hello")
    softcap = model_4l.config.final_logit_softcapping
    assert softcap is not None
    # tanh saturates at exactly ±1.0 in fp32 once |x| > ~9, so equality
    # with softcap is allowed (a strict `<` would flap).
    assert float(np.max(np.abs(logits))) <= softcap + 1e-3


@skip_if_no_real_checkpoint()
def test_different_prompts_yield_different_argmax(model_4l):
    """Smoke that the model isn't degenerate to one id regardless of prompt.

    Strongest "the runtime is computing something" signal we have
    without a reference baseline. Four prompts span code, narrative,
    factual, and greeting — all four argmax ids must not collapse to
    the same value.
    """
    prompts = [
        "Hello",
        "The capital of France is",
        "import numpy as np",
        "Once upon a time",
    ]
    argmaxes = []
    for p in prompts:
        logits = model_4l.forward(p)
        argmaxes.append(int(np.argmax(logits[0])))
    assert len(set(argmaxes)) > 1, (
        f"all prompts produced the same argmax id {argmaxes[0]} "
        "— the model has degenerated to a constant"
    )


@skip_if_no_real_checkpoint()
def test_forward_kv_length_matches_prompt_tokens(model_4l):
    """forward() should leave KV at exactly len(encoded prompt) per layer."""
    prompt_ids = model_4l.tokenizer.encode("Hello world")
    model_4l.forward("Hello world")
    for L in range(model_4l.num_layers):
        assert model_4l._last_kv.length(L) == len(prompt_ids), (
            f"layer {L} KV length {model_4l._last_kv.length(L)} != "
            f"prompt token count {len(prompt_ids)}"
        )


@skip_if_no_real_checkpoint()
def test_forward_top_10_softcap_saturation_is_not_an_error(model_4l):
    """At few-layer depth, top-K logits all saturate the softcap.

    Lock in the *expected* behavior: when running only a fraction of
    the model's layers, the residual stream has wild magnitudes and
    `softcap * tanh(x/softcap)` saturates at ±softcap for the highest
    logits. This test is an explicit acknowledgment, not a problem —
    it would fail if some future code path bypassed the softcap.
    """
    logits = model_4l.forward("Hello")[0]
    softcap = model_4l.config.final_logit_softcapping
    top_10_idx = np.argpartition(-logits, 10)[:10]
    top_10_vals = logits[top_10_idx]
    # At least one of the top-10 logits must be near the softcap
    # (the few-layer regime saturates them); none should exceed it.
    assert float(top_10_vals.max()) <= softcap + 1e-3
    assert float(top_10_vals.max()) > softcap * 0.9


@skip_if_no_real_checkpoint()
def test_forward_empty_prompt_raises(model_4l):
    """Empty prompt fails fast — no logits to return from a zero-token forward."""
    # `tokenizer.encode("")` returns [bos] not [], so we test the
    # encoded-empty case directly via the private helper.
    with pytest.raises(ValueError, match="zero tokens"):
        model_4l._forward_ids([])


# --- Full-depth chat answers (strict) ---------------------------------------
# These replaced three `xfail` probes that asked for a sensible top-20
# after raw-completion prompts ("The capital of France is", "2 + 2 =",
# "Once upon a time"). Those could never pass: this instruction-tuned
# 4-bit checkpoint degenerates on raw completion in the mlx_lm
# reference too ("作用作用作用…"), so they measured the prompt format
# rather than the runtime. Through the chat template the reference
# answers correctly, and so must APEX.

# Loaded once per module — embedding dequant + 30 layers is the
# expensive part. 6 GB of expert cache is deliberate: the non-evictable
# fp32 weights already occupy most of a 24 GB machine, and a larger
# cache tips it into swap (slower, not faster).
@pytest.fixture(scope="module")
def model_30l():
    return apex.MoEModel.from_pretrained(
        real_checkpoint_path(),
        max_memory_gb=6.0,
        num_layers=30,
        verbose=False,
    )


def _chat_reply(model, question: str, max_tokens: int = 24) -> str:
    """Greedy single-turn reply to `question` via the chat template."""
    return model.generate_chat(
        [{"role": "user", "content": question}],
        max_tokens=max_tokens, temperature=0.0,
    )


@skip_if_no_real_checkpoint()
def test_chat_answers_capital_of_france(model_30l):
    reply = _chat_reply(model_30l, "What is the capital of France?")
    assert "Paris" in reply, f"reply = {reply!r}"


@skip_if_no_real_checkpoint()
def test_chat_answers_two_plus_two(model_30l):
    reply = _chat_reply(model_30l, "What is 2+2?")
    assert "4" in reply, f"reply = {reply!r}"
