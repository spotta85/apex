"""Tiny numerical primitives for transformer building blocks.

Three pure functions, all NumPy:

  - rms_norm(x, weight, eps, *, gemma_unit_offset=False)
  - linear(x, weight, bias=None)
  - softmax(x, axis=-1)

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
