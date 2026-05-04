"""Tiny numerical primitives for transformer building blocks.

Six pure functions, all NumPy:

  - rms_norm(x, weight, eps, *, gemma_unit_offset=False)
  - linear(x, weight, bias=None)
  - softmax(x, axis=-1)
  - apply_rope(x, *, position_offset, theta)
  - silu(x)                   — x * sigmoid(x), the SwiGLU activation
  - gelu_tanh(x)              — Gemma 4's `gelu_pytorch_tanh` activation

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


def apply_rope(
    x,
    *,
    position_offset: int = 0,
    theta: float = 10000.0,
    rotary_dim: int | None = None,
):
    """Apply Rotary Position Embedding along the last axis (head_dim).

    `x` must have shape `(..., seq, head_dim)` with `head_dim` even.
    `position_offset` is the starting position; the rotation matrix
    for row `i` along the seq axis uses position `position_offset + i`.

    Pass the KV cache's current length as `position_offset` when
    rotating a new decode-step token. Pass `0` to rotate a fresh
    prompt prefix from position 0.

    Convention: HF / Gemma "split half" pairing on the rotated slice.
    The first `rotary_dim/2` features pair with the second `rotary_dim/2`
    features (NOT the interleaved `2i, 2i+1` pairing of the original
    RoPE paper); HF Gemma stores weights in the split-half convention.

    `theta` is the base of the inverse-frequency series. Gemma 4
    sliding-attention layers use 10000.0 (the default); full-attention
    layers use 1000000.0.

    `rotary_dim` controls partial RoPE (Gemma 4 full-attention layers
    rotate only the first 25% of dims):

      - `None` (default) ⇒ rotate the full head_dim — original behavior.
      - integer < head_dim ⇒ rotate only the first `rotary_dim` dims;
        the trailing `head_dim - rotary_dim` dims pass through unchanged.

    Why partial RoPE: leaving a tail of dims unrotated lets the model
    learn position-invariant features alongside positional ones — Gemma
    4's full-attention layers use this with `partial_rotary_factor=0.25`.

    Position 0 is the identity rotation (cos(0)=1, sin(0)=0), so
    `apply_rope(x, position_offset=0)` on a single-token tensor
    returns x unchanged regardless of `rotary_dim` — matches the
    "16C is correct at position 0" invariant from the smoke tests.
    """
    seq = x.shape[-2]
    head_dim = x.shape[-1]

    # Default to full rotation. Reject odd head_dim early so the
    # split-half pairing has even halves.
    if rotary_dim is None:
        rotary_dim = head_dim
    if rotary_dim > head_dim:
        raise ValueError(
            f"rotary_dim {rotary_dim} cannot exceed head_dim {head_dim}"
        )
    if rotary_dim % 2:
        raise ValueError(
            f"rotary_dim must be even for RoPE; got {rotary_dim}"
        )
    if rotary_dim == 0:
        # Degenerate "no RoPE" case — return unchanged. Easier to
        # support than to make every caller branch.
        return x

    # Per-pair inverse frequencies in fp32 over the rotated slice.
    # The doubled emb below broadcasts the same per-pair angle to both
    # halves, which is what the split-half rotate_half expects.
    half = rotary_dim // 2
    inv_freq = 1.0 / (
        theta ** (np.arange(half, dtype=np.float32) * 2.0 / rotary_dim)
    )
    positions = (
        np.arange(seq, dtype=np.float32) + float(position_offset)
    )
    freqs = np.outer(positions, inv_freq)             # (seq, half)
    emb = np.concatenate([freqs, freqs], axis=-1)     # (seq, rotary_dim)
    cos = np.cos(emb)
    sin = np.sin(emb)

    if rotary_dim == head_dim:
        # Full-rotation fast path — no slice / concat. Same math as
        # the original implementation; preserved so the no-partial
        # smoke benchmarks don't pay the slice cost.
        rotated_half = np.concatenate(
            [-x[..., half:], x[..., :half]], axis=-1,
        )
        return (x * cos + rotated_half * sin).astype(x.dtype, copy=False)

    # Partial RoPE: rotate the first `rotary_dim`, pass through the
    # tail. Slicing along axis -1 keeps the seq axis intact for the
    # broadcast against (seq, rotary_dim) cos/sin tables.
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    rotated_half = np.concatenate(
        [-x_rot[..., half:], x_rot[..., :half]], axis=-1,
    )
    rotated = x_rot * cos + rotated_half * sin
    return np.concatenate([rotated, x_pass], axis=-1).astype(
        x.dtype, copy=False,
    )


def silu(x):
    """SiLU (Swish) activation: x * sigmoid(x).

    The "Swi" in SwiGLU. Mixtral / Qwen MoE / older Gemma all use
    this in their FFN gate path.

    Computed as ``x / (1 + exp(-x))`` to avoid an explicit sigmoid
    call — same value, fewer FLOPs in fp32. For very negative x the
    `exp(-x)` overflows; we accept that since the fp32 dynamic range
    is wide enough that the resulting `inf` divides to 0 cleanly.
    """
    return x / (1.0 + np.exp(-x))


# Tanh-approximation GELU magic numbers (PyTorch's
# `nn.functional.gelu(x, approximate='tanh')` and Gemma's
# `gelu_pytorch_tanh`). Constants pre-computed in fp32 so the inner
# loop doesn't re-allocate for every call.
_GELU_TANH_CUBIC = np.float32(0.044715)
_GELU_TANH_SQRT_2_OVER_PI = np.float32(0.7978845608028654)  # sqrt(2/pi)


def gelu_tanh(x):
    """GELU with tanh approximation (Gemma 4's `gelu_pytorch_tanh`).

    Formula:
        0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))

    Bit-equivalent to PyTorch's
    ``F.gelu(x, approximate='tanh')`` and HF transformers'
    ``ACT2FN['gelu_pytorch_tanh']``. Used in Gemma 4's MLP gate path
    and dense feedforward — substituting SiLU here against a
    GeLU-trained model is the single biggest source of math drift
    we found at L=8 (it was emitting unused-token gibberish).

    SiLU and GeLU have similar shapes — both are smooth approximations
    of ReLU near zero — but they diverge meaningfully outside the
    [-3, 3] range, and that error compounds across 30 decoder layers.
    """
    inner = _GELU_TANH_SQRT_2_OVER_PI * (
        x + _GELU_TANH_CUBIC * x * x * x
    )
    return 0.5 * x * (1.0 + np.tanh(inner))


# Lookup table so backends + tests can pick the activation by the
# string name the config provides ("silu" / "gelu_pytorch_tanh"),
# instead of branching at the call site. Adding a new activation
# means appending one entry.
ACTIVATIONS = {
    "silu": silu,
    "swish": silu,                  # alias HF emits sometimes
    "gelu_pytorch_tanh": gelu_tanh,
    "gelu_new": gelu_tanh,          # HF's `gelu_new` is the same formula
}


def get_activation(name: str):
    """Resolve `name` to an activation callable.

    Raises ValueError naming the unknown activation rather than
    silently falling back — quietly defaulting to silu against a
    GeLU-trained model is exactly the bug we're fixing.
    """
    fn = ACTIVATIONS.get(name)
    if fn is None:
        raise ValueError(
            f"unknown activation {name!r}; "
            f"available: {sorted(ACTIVATIONS)}"
        )
    return fn
