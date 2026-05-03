"""Self-attention — single-token (no cache) and decode-step (KV cache + RoPE).

Two callables for two stages of the generation pipeline:

  - `single_token_attention`: pre-step-17 smoke surface. One token
    at position 0, no cache, no RoPE (cos(0)=1 / sin(0)=0 makes RoPE
    identity at position 0). Kept because it's the simplest possible
    "attention works on real weights" test and remains useful for
    isolating regressions.

  - `attention_decode_step`: one new token with a KVCache. Reads
    `kv_cache.length(layer_id)` for the RoPE position, applies RoPE
    to q/k, appends k/v, then attends over the full cached history.
    No causal mask needed — seq_q is 1, so the new token's softmax
    over `[past..., current]` is unambiguous. Multi-token prefill
    (seq_q > 1) needs a causal mask + a different code path; not
    implemented yet.

Both support GQA via num_kv_heads < num_heads (Gemma 4 sliding:
16 Q heads share 8 KV heads, q_per_kv=2) and optional per-head
q_norm / k_norm with the Gemma `+1` RMSNorm convention.

What is intentionally NOT here:
  - Multi-token prefill with causal mask. Decode-only is the
    "simple loop first, optimize prefill later" stance from the
    step 17 spec. The decode function can drive a prefill loop
    one token at a time at O(prompt_len * total_seq) cost.
  - Sliding-window attention scope limit (Gemma 4 sliding layers
    cap attention to the last 1024 tokens). For decode demos with
    seq < 1024 this is moot.
  - Full-attention layers (Gemma 4 layers 5/11/17/23/29 tie K=V,
    head_dim=512, partial_rotary_factor=0.25, theta=1e6).
"""

import numpy as np

from apex.ops import apply_rope, linear, rms_norm, softmax


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


def attention_decode_step(
    hidden,
    *,
    q_weight,
    k_weight,
    v_weight,
    o_weight,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    kv_cache,
    layer_id: int,
    q_norm_weight=None,
    k_norm_weight=None,
    rms_eps: float = 1e-6,
    rope_theta: float = 10000.0,
):
    """One decode step of self-attention with a KVCache.

    `hidden` is `(1, hidden_size)` — a single new token. Reads
    `kv_cache.length(layer_id)` for the RoPE position, projects q/k/v,
    applies q_norm/k_norm + RoPE, appends k/v to the cache for this
    layer, and attends q over the full cached history (past + current).

    Returns `(1, hidden_size)`. No causal mask: seq_q=1 attends to
    `[past..., current]`, which is exactly what causal would allow
    anyway. Multi-token prefill (seq_q > 1) needs a separate function
    with explicit masking — not implemented yet.

    `rope_theta` defaults to 10000.0 to match Gemma 4 sliding-attention
    layers. Full-attention layers use a different theta + partial
    rotation; passing `theta` here is the surface for that path even
    though the full layer kind isn't yet supported end-to-end.
    """
    # Project — each result is (1, num_heads_x * head_dim).
    q = linear(hidden, q_weight)
    k = linear(hidden, k_weight)
    v = linear(hidden, v_weight)

    # Reshape to (1, num_heads_x, 1, head_dim). The seq=1 axis is
    # explicit so RoPE can broadcast over it and KVCache.append can
    # concatenate along it on subsequent decode steps.
    q = q.reshape(1, num_heads, 1, head_dim)
    k = k.reshape(1, num_kv_heads, 1, head_dim)
    v = v.reshape(1, num_kv_heads, 1, head_dim)

    if q_norm_weight is not None:
        q = rms_norm(q, q_norm_weight, eps=rms_eps, gemma_unit_offset=True)
    if k_norm_weight is not None:
        k = rms_norm(k, k_norm_weight, eps=rms_eps, gemma_unit_offset=True)

    # RoPE at this token's position in the sequence — the cache's
    # current length IS the next position (0-indexed). At position 0
    # RoPE is identity, which keeps step 16's smoke math correct
    # when wired through this code path with an empty cache.
    position_offset = kv_cache.length(layer_id)
    q = apply_rope(q, position_offset=position_offset, theta=rope_theta)
    k = apply_rope(k, position_offset=position_offset, theta=rope_theta)

    # Commit this step's k/v to the cache before reading; the new
    # token attends to itself too (the canonical "attend to all
    # past INCLUDING self" decode behavior).
    kv_cache.append(layer_id, k, v)
    k_all, v_all = kv_cache.get(layer_id)
    # k_all / v_all: (1, num_kv_heads, seq_total, head_dim).

    # GQA: replicate each KV head q_per_kv times so the per-head
    # einsum lines up with q.
    q_per_kv = num_heads // num_kv_heads
    if q_per_kv > 1:
        k_all = np.repeat(k_all, q_per_kv, axis=1)
        v_all = np.repeat(v_all, q_per_kv, axis=1)

    # Scaled dot product. q: (1, H, 1, D); k_all.T: (1, H, D, S).
    # scores: (1, H, 1, S). softmax over the S axis.
    scores = (q @ k_all.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
    weights = softmax(scores, axis=-1)
    attn = weights @ v_all                          # (1, H, 1, D)
    attn = attn.transpose(0, 2, 1, 3).reshape(1, num_heads * head_dim)
    return linear(attn, o_weight)
