"""Tiny numerical primitives for transformer building blocks.

Four pure functions, all NumPy:

  - rms_norm(x, weight, eps, *, gemma_unit_offset=False)
  - linear(x, weight, bias=None)
  - softmax(x, axis=-1)
  - apply_rope(x, *, position_offset, theta)

Why "tiny": these are the lowest-cost layer of math that decoder layers
need (attention pre-norm, projection matmuls, attention softmax). They
exist as standalone callables so:

  - the smoke tests in 16C / 16D / 16E can compose them without
    reaching into a backend object,
  - per-op unit tests stay focused (no FFN/attention scaffolding to
    exercise the primitive),
  - a future MLX variant can mirror the NumPy signatures one for one.

Pure NumPy, no kernels. Vectorized where it's free but not micro-optimized.
Same correctness-first stance as `apex.dequant`.
"""

import numpy as np


def rms_norm(x, weight, eps: float = 1e-6, *, gemma_unit_offset: bool = False):
    """RMSNorm normalized along the last axis.

    Standard form: ``output = x / sqrt(mean(x^2) + eps) * weight``.

    `gemma_unit_offset=True` switches to Gemma's RMSNorm convention
    where the stored weight is a delta-from-1.0
    (``output = x_normed * (1 + weight)``). This matters because
    Gemma's checkpoint weights are centered around zero, not one;
    forgetting the +1 silently zeros out activations and the smoke
    test would still "pass" (finite + correctly shaped, just wrong).

    Verified against ``transformers/.../modeling_gemma.py:GemmaRMSNorm``.

    Computation always runs in fp32 to keep variance accumulation
    stable, then casts back to the input dtype so callers can stack
    rms_norm into a bf16 / fp16 graph without surprises.
    """
    x_f32 = x.astype(np.float32, copy=False)
    var = np.mean(x_f32 * x_f32, axis=-1, keepdims=True)
    x_normed = x_f32 / np.sqrt(var + eps)
    scale = (1.0 + weight) if gemma_unit_offset else weight
    return (x_normed * scale).astype(x.dtype, copy=False)


def linear(x, weight, bias=None):
    """y = x @ weight.T (+ bias). HF orientation: weight is (out, in).

    Same convention apex's MoE projections use end to end (see
    apex/backends/fake_backend.py and apex/moe_layer.py): weights are
    stored as (out_features, in_features) and matmul applies the
    transpose, so callers don't pre-transpose anywhere in the pipeline.
    """
    out = x @ weight.T
    if bias is not None:
        out = out + bias
    return out


def softmax(x, axis: int = -1):
    """Numerically-stable softmax along `axis`.

    Subtracts the per-axis maximum before exponentiating so large
    inputs don't overflow to inf — the standard trick. Output sums
    to 1.0 along `axis` (within fp32 rounding).
    """
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def apply_rope(x, *, position_offset: int = 0, theta: float = 10000.0):
    """Apply Rotary Position Embedding along the last axis (head_dim).

    `x` must have shape `(..., seq, head_dim)` with `head_dim` even.
    `position_offset` is the starting position; the rotation matrix
    for row `i` along the seq axis uses position `position_offset + i`.

    Pass the KV cache's current length as `position_offset` when
    rotating a new decode-step token. Pass `0` to rotate a fresh
    prompt prefix from position 0.

    Convention: HF / Gemma "split half" pairing. The first `head_dim/2`
    features pair with the second `head_dim/2` features (NOT the
    interleaved `2i, 2i+1` pairing used by the original RoPE paper);
    HF Gemma stores weights in the split-half convention so this
    matches the checkpoint.

    `theta` is the base of the inverse-frequency series. Gemma 4
    sliding-attention layers use 10000.0 (the default); full-attention
    layers use 1000000.0 with a partial_rotary_factor — those need
    a separate code path that this primitive doesn't yet implement.

    Position 0 is the identity rotation (cos(0)=1, sin(0)=0), so
    `apply_rope(x, position_offset=0)` on a single-token tensor
    returns x unchanged — matches the "16C is correct at position 0"
    invariant from the smoke tests.
    """
    seq = x.shape[-2]
    head_dim = x.shape[-1]
    if head_dim % 2:
        raise ValueError(
            f"head_dim must be even for RoPE; got {head_dim}"
        )

    # Per-pair inverse frequencies in fp32. The doubled emb below
    # broadcasts the same per-pair angle to both halves, which is
    # what the split-half rotate_half expects.
    half = head_dim // 2
    inv_freq = 1.0 / (
        theta ** (np.arange(half, dtype=np.float32) * 2.0 / head_dim)
    )
    positions = (
        np.arange(seq, dtype=np.float32) + float(position_offset)
    )
    freqs = np.outer(positions, inv_freq)             # (seq, half)
    emb = np.concatenate([freqs, freqs], axis=-1)     # (seq, head_dim)
    cos = np.cos(emb)
    sin = np.sin(emb)

    # rotate_half(x) = cat([-x[half:], x[:half]]) — split-half pairing.
    rotated_half = np.concatenate(
        [-x[..., half:], x[..., :half]], axis=-1,
    )
    return (x * cos + rotated_half * sin).astype(x.dtype, copy=False)
