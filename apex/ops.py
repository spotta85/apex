"""Tiny numerical primitives for transformer building blocks.

Six pure functions plus an activation lookup, all NumPy:

  - rms_norm(x, weight, eps, *, gemma_unit_offset=False)
  - linear(x, weight, bias=None)
  - softmax(x, axis=-1)
  - apply_rope(x, *, position_offset, theta, rotary_dim, rope_type,
               rope_factor)
  - silu(x)                   — x * sigmoid(x), the SwiGLU activation
  - gelu_tanh(x)              — Gemma 4's `gelu_pytorch_tanh` activation
  - get_activation(name)      — resolves a config activation name
                                ("silu", "gelu_pytorch_tanh", ...) via
                                the `ACTIVATIONS` table

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

    `weight=None` skips the scale entirely — equivalent to
    `mlx_lm.models.gemma4_text.RMSNormNoScale`. Used for V projection
    in Gemma 4 attention (V gets a plain RMS normalization with no
    learnable scale, while Q and K get the standard scaled RMSNorm).

    `gemma_unit_offset=True` switches to the Gemma 1/2 RMSNorm
    convention where the stored weight is a delta-from-1.0
    (``output = x_normed * (1 + weight)``). Gemma 4 does NOT use it:
    this checkpoint's norm weights are already full-scale (e.g. the
    q_norm γ is ≈ 1), so every production call site passes False (the
    default) and gets the plain ``x_normed * weight`` form, matching
    mlx's `nn.RMSNorm`. Applying the +1 to full-scale weights inflates
    every norm and the error compounds layer after layer while the
    output stays finite and correctly shaped. The flag is
    kept as an option; it is simply ignored when `weight=None` (there
    is no weight to offset).

    Computation always runs in fp32 to keep variance accumulation
    stable, then casts back to the input dtype so callers can stack
    rms_norm into a bf16 / fp16 graph without surprises.
    """
    x_f32 = x.astype(np.float32, copy=False)
    var = np.mean(x_f32 * x_f32, axis=-1, keepdims=True)
    x_normed = x_f32 / np.sqrt(var + eps)
    if weight is None:
        # RMSNormNoScale — no learnable scale, no +1 offset.
        return x_normed.astype(x.dtype, copy=False)
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
    rope_type: str = "default",
    rope_factor: float = 1.0,
):
    """Apply Rotary Position Embedding along the last axis (head_dim).

    `x` must have shape `(..., seq, head_dim)` with `head_dim` even.
    `position_offset` is the starting position; the rotation matrix
    for row `i` along the seq axis uses position `position_offset + i`.

    Pass the KV cache's current length as `position_offset` when
    rotating a new decode-step token. Pass `0` to rotate a fresh
    prompt prefix from position 0.

    Convention: HF / Gemma "split half" pairing across the FULL
    `head_dim`. Element index `i` (for `i < head_dim/2`) pairs with
    `i + head_dim/2`. This is invariant under partial RoPE: the
    pairing is determined by `head_dim`, not by `rotary_dim`. The
    pre-step-26F implementation paired within the rotated slice
    (`(i, i + rotary_dim/2)` for `i < rotary_dim/2`), which silently
    produces a different rotation than the model was trained with
    when `rotary_dim < head_dim`.

    `theta` is the base of the inverse-frequency series. Gemma 4
    sliding-attention layers use 10000.0; full-attention layers use
    1000000.0.

    `rotary_dim` controls partial RoPE:

      - `None` (default) ⇒ rotate the full head_dim.
      - integer < head_dim ⇒ rotate only the first `rotary_dim/2`
        pair indices (those that fall within the first half of the
        head_dim split). The remaining `(head_dim - rotary_dim) / 2`
        pair indices use freq=inf ⇒ angle=0 ⇒ no rotation. The
        ELEMENTS those pairs touch (`rotary_dim/2 ... head_dim/2`
        and `rotary_dim/2 + head_dim/2 ... head_dim`) are passed
        through unchanged.

    `rope_type='default'` uses the standard exponent
    `2i / rotary_dim` (matches `mlx.nn.RoPE`).

    `rope_type='proportional'` uses the Gemma 4 full_attention
    exponent `2i / head_dim` (matches `mlx_lm.models.rope_utils.
    ProportionalRoPE`). The exponent denominator is the FULL
    head_dim, so for `rotary_dim < head_dim` the per-pair angles
    are smaller than under default — meaningfully different
    rotation rates. `rope_factor` is an additional multiplier on
    the freq base (default 1.0).

    Position 0 is the identity rotation (cos(0)=1, sin(0)=0), so
    `apply_rope(x, position_offset=0)` on a single-token tensor
    returns x unchanged regardless of `rotary_dim` / `rope_type`.
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
    if head_dim % 2:
        raise ValueError(
            f"head_dim must be even for split-half RoPE; got {head_dim}"
        )
    if rotary_dim == 0:
        # Degenerate "no RoPE" case — return unchanged. Easier to
        # support than to make every caller branch.
        return x

    # Per-pair inverse frequencies in fp32. Length = head_dim/2 pairs.
    # Pairs at index i < rotary_dim/2 get rotated; pairs at index
    # i >= rotary_dim/2 get inv_freq=0 (no rotation, freq=inf).
    half_full = head_dim // 2
    half_rot = rotary_dim // 2

    if rope_type == "proportional":
        # ProportionalRoPE: exponents divide by head_dim (NOT rotary_dim).
        # Matches mlx_lm.models.rope_utils.ProportionalRoPE.
        exponents = (
            np.arange(0, rotary_dim, 2, dtype=np.float32) / head_dim
        )
        freq_base = np.float32(rope_factor) * (theta ** exponents)
    elif rope_type == "default":
        # Standard RoPE: exponents divide by rotary_dim.
        exponents = np.arange(half_rot, dtype=np.float32) * 2.0 / rotary_dim
        freq_base = theta ** exponents
    else:
        raise ValueError(
            f"unknown rope_type {rope_type!r}; "
            f"available: 'default', 'proportional'"
        )

    # Build full-head_dim/2 inv_freq array: rotated pairs use
    # 1/freq_base; trailing tail-pairs use 0 (no rotation).
    inv_freq = np.zeros(half_full, dtype=np.float32)
    inv_freq[:half_rot] = 1.0 / freq_base

    positions = (
        np.arange(seq, dtype=np.float32) + float(position_offset)
    )
    freqs = np.outer(positions, inv_freq)             # (seq, half_full)
    emb = np.concatenate([freqs, freqs], axis=-1)     # (seq, head_dim)
    cos = np.cos(emb)
    sin = np.sin(emb)

    # Split-half rotation across the FULL head_dim. The unrotated
    # tail-pairs (indices >= half_rot) have angle 0 ⇒ cos=1, sin=0
    # ⇒ rotation is identity for those elements. So we don't need
    # a separate slice path — the math is uniform.
    rotated_half = np.concatenate(
        [-x[..., half_full:], x[..., :half_full]], axis=-1,
    )
    return (x * cos + rotated_half * sin).astype(x.dtype, copy=False)


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
