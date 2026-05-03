"""Tests for step 16E: generation primitives.

Synthetic only — no checkpoint dependency. The real-Gemma end-to-end
test (skipif) lives separately in tests/test_real_generation.py so
the always-on synthetic suite stays fast.
"""

import numpy as np
import pytest

from apex.generation import embed_token, final_logits, sample_argmax


def test_embed_token_returns_correct_row_with_batch_dim():
    embedding = np.arange(20, dtype=np.float32).reshape(4, 5)  # vocab=4, hidden=5
    out = embed_token(embedding, token_id=2)
    assert out.shape == (1, 5)
    np.testing.assert_array_equal(out[0], embedding[2])


def test_embed_token_dtype_is_float32():
    # BF16-decoded embeddings come in as fp32 already; this just
    # pins that the contract holds even if a caller passes in fp16.
    embedding = np.zeros((3, 4), dtype=np.float16)
    out = embed_token(embedding, 0)
    assert out.dtype == np.float32


def test_final_logits_shape_and_no_softcap_matches_naive_matmul():
    rng = np.random.default_rng(0)
    hidden = rng.standard_normal((1, 8)).astype(np.float32)
    embedding = rng.standard_normal((6, 8)).astype(np.float32)  # vocab=6
    logits = final_logits(hidden, embedding, softcap=None)
    assert logits.shape == (1, 6)
    # final_logits is just `hidden @ embedding.T` when softcap is None.
    np.testing.assert_allclose(logits, hidden @ embedding.T, rtol=1e-6)


def test_final_logits_softcap_bounds_logits_in_range():
    """softcap = C ⇒ |output| ≤ C elementwise.

    Mathematically the bound is open (tanh ∈ (-1, 1)) but in fp32
    `tanh(x)` saturates to exactly ±1.0 once `|x|` exceeds ~9, so
    the floating-point bound is closed. Asserting `<=` lets us
    use realistically large logits without hand-tuning the input
    range to stay below saturation.
    """
    rng = np.random.default_rng(1)
    hidden = 50.0 * rng.standard_normal((1, 8)).astype(np.float32)
    embedding = 50.0 * rng.standard_normal((10, 8)).astype(np.float32)
    softcap = 30.0
    logits = final_logits(hidden, embedding, softcap=softcap)
    assert np.all(np.abs(logits) <= softcap)


def test_final_logits_softcap_matches_explicit_formula():
    rng = np.random.default_rng(2)
    hidden = rng.standard_normal((1, 4)).astype(np.float32)
    embedding = rng.standard_normal((5, 4)).astype(np.float32)
    softcap = 30.0
    raw = hidden @ embedding.T
    expected = softcap * np.tanh(raw / softcap)
    np.testing.assert_allclose(
        final_logits(hidden, embedding, softcap=softcap),
        expected, rtol=1e-6,
    )


def test_sample_argmax_returns_max_index():
    # Hand-crafted logits so the argmax is unambiguous.
    logits = np.array([[0.1, 0.2, 0.9, 0.3]], dtype=np.float32)
    assert sample_argmax(logits) == 2


def test_sample_argmax_returns_python_int_not_numpy_scalar():
    # Token ids land in dicts / json / tokenizer APIs — keeping the
    # type a plain int avoids numpy-scalar surprises downstream.
    logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    assert isinstance(sample_argmax(logits), int)
