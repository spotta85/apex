"""Single-token self-attention (no KV cache, no RoPE).

For seq_len=1 at position 0, the math reduces to:

  q · k.T / sqrt(d) → softmax (= [1.0]) → @ v → o_proj

The single-position softmax is trivially 1.0 but we run the full
scaled-dot-product path anyway so the compute graph mirrors what 16D
will need once a KV cache is populated. The smoke test asserts
``finite + correct shape`` only — exact numerical agreement vs HF
transformers Gemma is a 16D/16E concern.

Supports GQA via num_kv_heads < num_heads (Gemma 4 sliding: 16 Q
heads share 8 K/V heads, q_per_kv = 2). Optional q_norm / k_norm
follow Gemma's per-head RMSNorm on the head_dim axis using the
``gemma_unit_offset=True`` convention from apex.ops.rms_norm.

What is intentionally NOT here:
  - RoPE: position 0 has cos(0)=1, sin(0)=0 ⇒ identity. Add when
    multi-token / KV cache lands (16D).
  - KV cache: future tokens need it; for one token there's nothing
    to cache.
  - Sliding-window logic: with no past context the window is moot.
"""

import numpy as np

from apex.ops import linear, rms_norm, softmax


def single_token_attention(
    hidden,
    *,
    q_weight,
    k_weight,
    v_weight,
    o_weight,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_norm_weight=None,
    k_norm_weight=None,
    rms_eps: float = 1e-6,
):
    """Run one token of self-attention; return shape (1, hidden_size).

    `hidden` is (1, hidden_size). All weights are HF-orientation
    `(out_features, hidden_size)` for projections (q/k/v have
    out=num_heads_x*head_dim, o has out=hidden_size). Norm weights
    are 1-D `(head_dim,)` and apply per-head along the last axis.
    """
    # Project — each result is (1, num_heads_x * head_dim).
    q = linear(hidden, q_weight)
    k = linear(hidden, k_weight)
    v = linear(hidden, v_weight)

    # Per-head split. (1, num_heads_x, head_dim).
    q = q.reshape(1, num_heads, head_dim)
    k = k.reshape(1, num_kv_heads, head_dim)
    v = v.reshape(1, num_kv_heads, head_dim)

    # Per-head Q/K RMSNorm along the head_dim axis. Gemma 4 stores
    # these and they control attention-score magnitude — without them
    # logits explode but stay finite (the smoke contract still passes,
    # so a missing q_norm/k_norm would slip past unless we run the
    # full numerical check, which is 16D's job).
    if q_norm_weight is not None:
        q = rms_norm(q, q_norm_weight, eps=rms_eps, gemma_unit_offset=True)
    if k_norm_weight is not None:
        k = rms_norm(k, k_norm_weight, eps=rms_eps, gemma_unit_offset=True)

    # GQA: repeat each KV head q_per_kv times so Q and K/V align on
    # the head axis. For full equality (q_per_kv == 1) this is a
    # no-op; for Gemma 4 sliding (q_per_kv == 2) every KV head is
    # duplicated.
    q_per_kv = num_heads // num_kv_heads
    if q_per_kv > 1:
        k = np.repeat(k, q_per_kv, axis=1)
        v = np.repeat(v, q_per_kv, axis=1)

    # Scaled dot-product. seq_q=1, seq_k=1 ⇒ scores shape (1, num_heads, 1, 1).
    # softmax along the last axis ⇒ exactly [1.0] per head.
    scores = (q[:, :, None, :] @ k[:, :, :, None]) / np.sqrt(head_dim)
    weights = softmax(scores, axis=-1)
    attn = weights @ v[:, :, None, :]            # (1, num_heads, 1, head_dim)
    attn = attn.reshape(1, num_heads * head_dim)

    return linear(attn, o_weight)
