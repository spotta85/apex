"""Tests for apex/sampling.py — sample_argmax + sample with temperature."""

import numpy as np
import pytest

from apex.sampling import sample, sample_argmax


def test_sample_argmax_picks_highest_logit():
    logits = np.array([[0.1, 5.0, -2.0, 1.0]], dtype=np.float32)
    assert sample_argmax(logits) == 1


def test_sample_argmax_handles_ties_deterministically():
    # numpy argmax returns the first max; assert that contract holds.
    logits = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
    assert sample_argmax(logits) == 0


def test_sample_temperature_zero_collapses_to_argmax():
    logits = np.array([[0.1, 5.0, -2.0, 1.0]], dtype=np.float32)
    assert sample(logits, temperature=0.0) == sample_argmax(logits)


def test_sample_negative_temperature_raises():
    logits = np.array([[1.0, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="temperature must be >= 0"):
        sample(logits, temperature=-0.1)


def test_sample_with_temperature_returns_valid_id():
    """Token id must be a valid index into the vocab; sample many times."""
    logits = np.array([[0.5, 1.0, -0.5, 2.0, 0.0]], dtype=np.float32)
    rng = np.random.default_rng(42)
    for _ in range(50):
        tok = sample(logits, temperature=1.0, rng=rng)
        assert isinstance(tok, int)
        assert 0 <= tok < logits.shape[-1]


def test_sample_with_high_temperature_explores_more_tokens():
    """Low temperature should concentrate on the argmax; high should spread."""
    # Build logits where argmax is index 3.
    logits = np.array([[0.0, 0.5, 1.0, 5.0, 0.5]], dtype=np.float32)
    rng_low = np.random.default_rng(0)
    rng_high = np.random.default_rng(0)
    low_picks = {sample(logits, temperature=0.1, rng=rng_low) for _ in range(50)}
    high_picks = {sample(logits, temperature=10.0, rng=rng_high) for _ in range(50)}
    # At T=0.1 the argmax dominates; at T=10 several ids should appear.
    assert len(high_picks) > len(low_picks)
