"""Tests for apex/ops.py — RMSNorm / linear / softmax primitives.

Synthetic only (no checkpoint dependency). Every test pins behavior
that downstream callers in 16C/D/E will rely on; correctness now is
cheaper than debugging a wrong attention output later.
"""

import numpy as np
import pytest

from apex.ops import linear, rms_norm, softmax


# --- rms_norm -------------------------------------------------------------

def test_rms_norm_preserves_shape_and_dtype():
    # 3-D input is the realistic shape (batch, tokens, hidden) for
    # downstream attention; normalization is over the last axis.
    x = np.ones((2, 3, 8), dtype=np.float32)
    weight = np.ones(8, dtype=np.float32)
    out = rms_norm(x, weight, eps=1e-6)
    assert out.shape == (2, 3, 8)
    assert out.dtype == np.float32


def test_rms_norm_matches_hand_computed_value_on_fixed_input():
    # x = [1, 2, 3, 4]; mean(x^2) = (1+4+9+16)/4 = 7.5; rms = sqrt(7.5).
    # weight = ones ⇒ output = x / sqrt(7.5).
    x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    weight = np.ones(4, dtype=np.float32)
    out = rms_norm(x, weight, eps=0.0)
    expected = x / np.sqrt(7.5)
    np.testing.assert_allclose(out, expected, rtol=1e-6)


def test_rms_norm_applies_weight_per_axis():
    # Distinct weight per dim: output[i] = (x_normed[i]) * weight[i].
    x = np.array([[2.0, 2.0, 2.0, 2.0]], dtype=np.float32)
    weight = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    out = rms_norm(x, weight, eps=0.0)
    # x is constant ⇒ rms = 2 ⇒ x_normed is all ones ⇒ out == weight.
    np.testing.assert_allclose(out[0], weight, rtol=1e-6)


def test_rms_norm_gemma_unit_offset_adds_one_to_weight():
    # Same x, same weight; only the unit_offset flag differs. The
    # standard form scales by `weight`; Gemma form scales by
    # `1 + weight`. So gemma_out / standard_out == (1 + weight) / weight
    # element-wise (assuming weight != 0).
    x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    weight = np.array([0.5, 1.0, 2.0, 4.0], dtype=np.float32)
    standard = rms_norm(x, weight, eps=0.0, gemma_unit_offset=False)
    gemma = rms_norm(x, weight, eps=0.0, gemma_unit_offset=True)
    expected_ratio = (1.0 + weight) / weight
    np.testing.assert_allclose(gemma[0] / standard[0], expected_ratio, rtol=1e-6)


def test_rms_norm_eps_zero_does_not_divide_by_zero_on_nonzero_input():
    # Real Gemma uses eps=1e-6; eps=0 still has to be defined for
    # arbitrary-magnitude inputs since variance is nonzero.
    x = np.full((1, 4), 3.0, dtype=np.float32)
    weight = np.ones(4, dtype=np.float32)
    out = rms_norm(x, weight, eps=0.0)
    assert np.all(np.isfinite(out))
    # rms = 3 ⇒ x_normed = ones ⇒ out = weight.
    np.testing.assert_allclose(out[0], weight, rtol=1e-6)


# --- linear ---------------------------------------------------------------

def test_linear_hf_orientation_shape():
    # x: (batch=2, in=8); weight: (out=4, in=8); ⇒ y: (2, 4).
    x = np.ones((2, 8), dtype=np.float32)
    weight = np.ones((4, 8), dtype=np.float32)
    out = linear(x, weight)
    assert out.shape == (2, 4)


def test_linear_matches_explicit_matmul_when_no_bias():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 5)).astype(np.float32)
    weight = rng.standard_normal((7, 5)).astype(np.float32)
    np.testing.assert_array_equal(linear(x, weight), x @ weight.T)


def test_linear_broadcasts_bias_across_batch():
    x = np.ones((4, 3), dtype=np.float32)
    weight = np.ones((2, 3), dtype=np.float32)
    bias = np.array([10.0, 20.0], dtype=np.float32)
    out = linear(x, weight, bias=bias)
    # x @ weight.T = 3 (each row sums 1+1+1) ⇒ before bias = (4, 2) of 3s.
    # Plus broadcast bias [10, 20] ⇒ [13, 23] in every row.
    expected = np.tile(np.array([13.0, 23.0]), (4, 1))
    np.testing.assert_allclose(out, expected)


# --- softmax --------------------------------------------------------------

def test_softmax_rows_sum_to_one():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, 8)).astype(np.float32)
    out = softmax(x, axis=-1)
    np.testing.assert_allclose(out.sum(axis=-1), np.ones(5), rtol=1e-6)


def test_softmax_invariant_under_per_row_constant_shift():
    # Numerical stability: softmax(x + c) == softmax(x) for any
    # row-broadcast scalar c. This is the property the max-subtract
    # trick is supposed to preserve.
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 6)).astype(np.float32)
    shifted = x + 1e3   # Big enough to overflow naive exp if not stabilized.
    # Tolerance reflects fp32 rounding only; the algorithm itself is
    # exactly invariant. A naive (non-stabilized) softmax would have
    # produced inf/NaN here, not just a few ULPs of drift.
    np.testing.assert_allclose(
        softmax(x, axis=-1), softmax(shifted, axis=-1), rtol=1e-4,
    )


def test_softmax_axis_zero_normalizes_columns():
    # Pin the axis behavior so callers can't get the axis silently
    # wrong when reducing over a non-default dim.
    x = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    out = softmax(x, axis=0)
    np.testing.assert_allclose(out.sum(axis=0), np.ones(2), rtol=1e-6)
    # Each column has identical values ⇒ uniform softmax (0.5 each).
    np.testing.assert_allclose(out, np.full_like(x, 0.5), rtol=1e-6)


def test_softmax_single_element_axis_returns_unity():
    # For seq_len=1 attention (the 16C smoke case), softmax along the
    # length-1 axis MUST be exactly 1.0 — anything else zeros out the
    # value matrix in attention.
    x = np.array([[[5.0]]], dtype=np.float32)  # shape (1, 1, 1)
    out = softmax(x, axis=-1)
    np.testing.assert_array_equal(out, np.array([[[1.0]]]))
