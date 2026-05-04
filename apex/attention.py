"""Self-attention — single canonical implementation, two convenience wrappers.

`attention_decode_step` is the canonical implementation: one new
token, with a `KVCache` for past K/V and RoPE applied at the cache's
current length. It supports GQA via `num_kv_heads < num_heads` and
optional per-head q_norm / k_norm with the Gemma `+1` RMSNorm
convention.

`single_token_attention` is a thin wrapper that constructs a fresh
1-layer KVCache, calls `attention_decode_step` at layer 0, and
discards the cache. RoPE at position 0 is the identity rotation
(cos(0)=1, sin(0)=0) so this is mathematically equivalent to "no
RoPE, no cache" — the simplest possible smoke surface, kept for
the position-0 tests in step 16C / 16D / 16E.

Why one implementation instead of two: the previous code path
duplicated ~30 lines (project + reshape + per-head norm + GQA repeat
+ scaled-dot-product) between the two surfaces. Centralising avoids
the two paths drifting on a future shape / norm tweak.

What is intentionally NOT here:
  - Multi-token prefill with causal mask. Decode-only is the
    "simple loop first, optimize prefill later" stance from the
    step 17 spec. The decode function can drive a prefill loop
    one token at a time at O(prompt_len * total_seq) cost.

Step 23 added support for Gemma 4 full_attention layers via three
keyword args without forking the function:
  - `v_weight=None` ⇒ K/V tied (full layers don't ship a v_proj on
    disk; we project once via k_weight and use it for both).
  - `rotary_dim` plumbs through to `apply_rope` for partial RoPE
    (full layers rotate only `head_dim * partial_rotary_factor` dims).
  - `head_dim`/`num_kv_heads`/`rope_theta` are already per-call.

Step 24C added `window_size` for sliding-window attention scope:
when set, the read of K/V from the cache trims to the last
`window_size` entries before the dot product. The cache itself is
not modified (eviction inside KVCache is out of scope). For prompt
lengths ≤ window_size this is a no-op.
"""

import numpy as np

from apex.kv_cache import KVCache
from apex.ops import apply_rope, linear, rms_norm, softmax


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
    rotary_dim: int | None = None,
    window_size: int | None = None,
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
    layers. Full-attention layers pass 1e6 with `rotary_dim < head_dim`
    (partial RoPE). Both options thread cleanly through `apply_rope`.

    `v_weight=None` enables the K=V-tied path used by Gemma 4
    full_attention layers (which don't ship a v_proj). When None,
    `k_weight` is used for both K and V — one matmul instead of two,
    consistent with how the model was trained.

    `rotary_dim=None` rotates the full head_dim (sliding behavior).
    Pass an integer < head_dim for partial RoPE (full layers).
    """
    # Project Q always; project K once. V either gets its own
    # projection or is aliased to K below (after k_norm, before RoPE)
    # for the `attention_k_eq_v` path.
    q = linear(hidden, q_weight)
    k = linear(hidden, k_weight)
    v = None if v_weight is None else linear(hidden, v_weight)

    # Reshape to (1, num_heads_x, 1, head_dim). The seq=1 axis is
    # explicit so RoPE can broadcast over it and KVCache.append can
    # concatenate along it on subsequent decode steps.
    q = q.reshape(1, num_heads, 1, head_dim)
    k = k.reshape(1, num_kv_heads, 1, head_dim)
    if v is not None:
        v = v.reshape(1, num_kv_heads, 1, head_dim)

    # Per-head Q/K RMSNorm along the head_dim axis. Gemma 4 stores
    # these and they control attention-score magnitude — without them
    # logits explode but stay finite (the smoke contract still passes,
    # so a missing q_norm/k_norm would slip past unless we run the
    # full numerical check, which is a downstream concern).
    if q_norm_weight is not None:
        q = rms_norm(q, q_norm_weight, eps=rms_eps, gemma_unit_offset=True)
    if k_norm_weight is not None:
        k = rms_norm(k, k_norm_weight, eps=rms_eps, gemma_unit_offset=True)

    # K=V-tied path: V picks up the post-norm K but NOT RoPE. Alias
    # before the RoPE rebind below so V keeps the un-rotated values.
    if v is None:
        v = k

    # RoPE at this token's position in the sequence — the cache's
    # current length IS the next position (0-indexed). At position 0
    # RoPE is identity, which keeps step 16's smoke math correct
    # when wired through this code path with an empty cache.
    # `rotary_dim` is None for sliding (full rotation) or
    # `head_dim * partial_rotary_factor` for full (partial rotation).
    #
    # Q and K both get RoPE so the q·k dot product carries positional
    # information. V deliberately does NOT — that's the standard
    # transformer convention. In the K=V-tied path, `v` was aliased
    # before this block to the un-rotated projection, so the rebind
    # below leaves V with its pre-RoPE values (correct semantic;
    # only K rotates).
    position_offset = kv_cache.length(layer_id)
    q = apply_rope(
        q, position_offset=position_offset,
        theta=rope_theta, rotary_dim=rotary_dim,
    )
    k = apply_rope(
        k, position_offset=position_offset,
        theta=rope_theta, rotary_dim=rotary_dim,
    )

    # Commit this step's k/v to the cache before reading; the new
    # token attends to itself too (the canonical "attend to all
    # past INCLUDING self" decode behavior).
    kv_cache.append(layer_id, k, v)
    k_all, v_all = kv_cache.get(layer_id)
    # k_all / v_all: (1, num_kv_heads, seq_total, head_dim).

    # Step 24C: sliding-window scope. When set, attend to only the last
    # `window_size` cached tokens (the cache itself isn't trimmed —
    # this is purely an attention-scope slice). Mathematically a no-op
    # for cached lengths ≤ window_size, so short prompts are unaffected.
    # Full_attention layers pass window_size=None and attend globally.
    if window_size is not None and k_all.shape[-2] > window_size:
        k_all = k_all[..., -window_size:, :]
        v_all = v_all[..., -window_size:, :]

    # GQA: replicate each KV head q_per_kv times so the per-head
    # einsum lines up with q. q_per_kv == 1 ⇒ no-op.
    q_per_kv = num_heads // num_kv_heads
    if q_per_kv > 1:
        k_all = np.repeat(k_all, q_per_kv, axis=1)
        v_all = np.repeat(v_all, q_per_kv, axis=1)

    # Scaled dot product. q: (1, H, 1, D); k_all.T: (1, H, D, S).
    # scores: (1, H, 1, S). softmax over the S axis.
    # `np.sqrt(int)` returns numpy.float64; dividing an fp32 score
    # array by an fp64 scalar promotes the result to fp64 — and that
    # promotion then propagates through softmax, the v matmul, and
    # the o_weight projection, so the *layer's output* becomes fp64.
    # Layer L+1's hidden then carries fp64 into its KV cache, which
    # silently doubles the per-layer KV bytes (visible in
    # `kv_snapshot.per_layer_bytes`). Cast the divisor to fp32 to
    # keep the whole attention path in single precision.
    scale = np.float32(head_dim) ** -0.5
    scores = (q @ k_all.transpose(0, 1, 3, 2)) * scale
    weights = softmax(scores, axis=-1)
    attn = weights @ v_all                          # (1, H, 1, D)
    attn = attn.transpose(0, 2, 1, 3).reshape(1, num_heads * head_dim)
    return linear(attn, o_weight)


def single_token_attention(hidden, **kwargs):
    """Position-0, no-cache wrapper over `attention_decode_step`.

    Equivalent to constructing a fresh `KVCache(num_layers=1)`,
    calling `attention_decode_step` at layer 0, and dropping the
    cache. RoPE position 0 is the identity rotation (cos(0)=1,
    sin(0)=0), so the result is the same as "no RoPE, no cache" —
    matching the step 16C smoke contract.

    Useful for tests + demos that want to exercise the attention
    math without managing a cache. Don't pass `kv_cache`, `layer_id`,
    or `rope_theta` here — those belong on `attention_decode_step`
    directly when callers need them.
    """
    return attention_decode_step(
        hidden,
        **kwargs,
        kv_cache=KVCache(num_layers=1),
        layer_id=0,
    )
