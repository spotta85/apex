"""Tests for apex/prefill.py — prompt-token loop with mock step_fn.

Synthetic-only. The real-Gemma end-to-end behavior (prefill grows
KV cache, generated tokens depend on prompt tokens) is covered by
the demo_prefill.py script + the existing demo_generate_with_kv.py
which now uses `prefill` itself.
"""

import pytest

from apex.prefill import prefill


def test_prefill_returns_last_step_fn_result():
    """Last call to step_fn determines what prefill returns."""
    # step_fn returns its input + 100; the last input is the last
    # prompt token, so prefill returns prompt[-1] + 100.
    last = prefill([1, 2, 3, 4], step_fn=lambda t: t + 100)
    assert last == 104


def test_prefill_calls_step_fn_once_per_token_in_order():
    """Order matters — prefill must consume prompt_ids left-to-right."""
    seen = []

    def spy(token_id):
        seen.append(token_id)
        return -token_id

    prefill([7, 11, 13, 17], step_fn=spy)
    assert seen == [7, 11, 13, 17]


def test_prefill_empty_prompt_raises_value_error():
    """Empty prompt is a programming error, not a silent no-op."""
    # Zero calls to step_fn ⇒ no prediction to return ⇒ asking for
    # one is meaningless. Surface it loudly.
    with pytest.raises(ValueError, match="must not be empty"):
        prefill([], step_fn=lambda t: t)


def test_prefill_threads_state_through_step_fn_closure():
    """The KV-cache wiring works because step_fn's closure is mutable."""
    # Mock a "cache" that grows by one per call; assert the final
    # length matches len(prompt). This is the prefill-fills-cache
    # contract on real Gemma, just synthesized.
    cache = []

    def step(token_id):
        cache.append(token_id)
        return token_id * 10

    last = prefill([2, 4, 6], step_fn=step)
    assert cache == [2, 4, 6]    # cache grew by one per token, in order
    assert last == 60            # last_id is step(prompt[-1])


def test_prefill_single_token_runs_step_fn_exactly_once():
    """Boundary: a single-token "prompt" still works (and is the bridge
    between pre-step-18 demos that fed one token and the new prefill API)."""
    calls = 0

    def step(token_id):
        nonlocal calls
        calls += 1
        return 42

    assert prefill([99], step_fn=step) == 42
    assert calls == 1
