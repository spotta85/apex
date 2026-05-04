"""Step 22C / 24E: structural + plausibility logit sanity on real Gemma 4.

Without an HF / MLX reference baseline we can't compare logits
exactly. These tests instead lock in *structural* properties — the
runtime is producing plausibly-shaped logits and varies its output
across prompts. Catches regressions like:

  - all logits NaN/inf
  - argmax always same id regardless of input (degenerate)
  - logits exceed softcap (softcap accidentally bypassed)
  - shape drift (vocab_size mismatch with config)

Step 24E adds *plausibility* checks: for canonical prompts like
`"The capital of France is"` and `"2 + 2 ="` the top-K should
contain a recognizable continuation token. These are marked
`xfail(strict=False)` because the pre-coherence runtime may still
produce gibberish-but-stable output — the marker documents the goal
without blocking CI. When coherence lands they can be promoted to
strict.

Reference-logits comparison against an HF/MLX baseline is deferred
to its own sprint (it requires loading the same checkpoint via
`transformers` or `mlx-lm` — a substantial dependency we don't want
to take just for one test).
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


# --- Step 24E: plausible-top-K (xfail-tolerant) -----------------------------

# Full-depth fixture for the plausibility tests. Loaded once per module
# because embedding dequant + 30-layer load is the expensive piece. We
# keep a generous 10 GB cap so the LRU expert cache + ~2.1 GB of
# non-evictable dense MLP weights both fit without thrashing.
@pytest.fixture(scope="module")
def model_30l():
    return apex.MoEModel.from_pretrained(
        real_checkpoint_path(),
        max_memory_gb=10.0,
        num_layers=30,
        verbose=False,
    )


def _top_k_decoded(model, prompt: str, k: int = 20) -> list[str]:
    """Return the top-`k` token strings (decoded) for `prompt`'s next token.

    Used by the plausibility tests below. Decoding ids individually
    via the tokenizer can produce odd whitespace / leading-space
    fragments — that's expected and the assertions are case-insensitive
    substring searches, not exact matches.
    """
    logits = model.forward(prompt)[0]
    top_idx = np.argpartition(-logits, k - 1)[:k]
    return [model.tokenizer.decode([int(i)]) for i in top_idx]


@skip_if_no_real_checkpoint()
@pytest.mark.xfail(
    strict=False,
    reason="step 24A+B brought routing + dense MLP online; coherence likely "
           "still needs more (parallel norms, sliding window, reference parity).",
)
def test_plausible_capital_of_france(model_30l):
    """Top-20 should mention 'paris' or 'france' for the capital prompt."""
    decoded = _top_k_decoded(model_30l, "The capital of France is")
    joined = " ".join(decoded).lower()
    assert "paris" in joined or "france" in joined, (
        f"top-20 = {decoded!r}"
    )


@skip_if_no_real_checkpoint()
@pytest.mark.xfail(
    strict=False,
    reason="step 24A+B brought routing + dense MLP online; coherence likely "
           "still needs more (parallel norms, sliding window, reference parity).",
)
def test_plausible_two_plus_two(model_30l):
    """Top-20 should contain '4' for '2 + 2 ='."""
    decoded = _top_k_decoded(model_30l, "2 + 2 =")
    joined = " ".join(decoded)
    assert "4" in joined, f"top-20 = {decoded!r}"


@skip_if_no_real_checkpoint()
@pytest.mark.xfail(
    strict=False,
    reason="step 24A+B brought routing + dense MLP online; coherence likely "
           "still needs more (parallel norms, sliding window, reference parity).",
)
def test_plausible_once_upon_a_time(model_30l):
    """Top-20 should contain a common narrative continuation token.

    Permissive list: any of these would be a plausible next token for
    `"Once upon a time"`.
    """
    decoded = _top_k_decoded(model_30l, "Once upon a time")
    joined = " ".join(decoded).lower()
    assert any(
        marker in joined
        for marker in (",", " in", " there", " a ", " the ", " ago", " when")
    ), f"top-20 = {decoded!r}"
