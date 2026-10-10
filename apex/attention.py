"""Self-attention — single canonical implementation, two convenience wrappers.

`attention_decode_step` is the canonical implementation: one new
token, with a `KVCache` for past K/V and RoPE applied at the cache's
current length. It supports GQA via `num_kv_heads < num_heads` and
optional per-head q_norm / k_norm in the plain RMSNorm convention
(`x_normed * weight`, no `+1`; Gemma 4's norm weights are full-scale).

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
    disk; `k_weight` serves as both the K and the V projection — K
    then gets k_norm + RoPE, V gets a plain RMS norm and no RoPE).
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
    rope_type: str = "default",
    window_size: int | None = None,
    attn_scale: float | None = None,
):
    """One decode step of self-attention with a KVCache.

    `hidden` is `(1, hidden_size)` — a single new token. Reads
    `kv_cache.length(layer_id)` for the RoPE position, projects q/k/v,
    applies q_norm/k_norm + RoPE, appends k/v to the cache for this
    layer, and attends q over the full cached history (past + current),
    or only the last `window_size` entries when `window_size` is set
    (sliding-attention layers).

    Returns `(1, hidden_size)`. No causal mask: seq_q=1 attends to
    `[past..., current]`, which is exactly what causal would allow
    anyway. Multi-token prefill (seq_q > 1) needs a separate function
    with explicit masking — not implemented yet.

    `rope_theta` defaults to 10000.0 to match Gemma 4 sliding-attention
    layers. Full-attention layers pass 1e6 with `rotary_dim < head_dim`
    (partial RoPE). Both options thread cleanly through `apply_rope`.

    `v_weight=None` enables the K=V-tied path used by Gemma 4
    full_attention layers (which don't ship a v_proj). When None,
    `k_weight` is used for both K and V — there is no separate V
    weight, consistent with how the model was trained.

    `rotary_dim=None` rotates the full head_dim (sliding behavior).
    Pass an integer < head_dim for partial RoPE (full layers).

    `rope_type` is forwarded to `apply_rope`: "default" for sliding
    layers, "proportional" for Gemma 4 full-attention layers (see
    `ModelConfig.rope_type_for_layer`).

    `window_size=None` attends over the whole cache; an integer limits
    attention to the last `window_size` cached tokens (the cache itself
    is not trimmed).

    `attn_scale=None` falls back to the standard `1/sqrt(head_dim)`
    scaling. Gemma 4 — which applies q_norm + k_norm to bring Q/K
    rows to unit RMS — passes 1.0 (no extra scaling), per
    `mlx_lm.models.gemma4_text:Attention.__init__` line 203:
    `self.scale = 1.0`. Step 26E.

    Step 26E also makes V pick up `RMSNormNoScale` (a plain RMS
    normalization with no learnable weight) when `k_norm_weight` is
    provided — the Gemma family ships `v_norm` alongside q/k norms
    for stability, and APEX previously skipped it (V was either raw
    projected or aliased to post-k-norm K).
    """
    # Project Q and K always. V either gets its own projection or,
    # for the `attention_k_eq_v` path, is derived from `k_weight`
    # below (un-normed K projection, taken before RoPE).
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
    #
    # gemma_unit_offset=False: this NVFP4 checkpoint stores norm
    # weights in the PLAIN convention (q_norm γ ≈ 1.02, already
    # full-scale), NOT the Gemma 1/2 delta-from-1 convention. mlx's
    # nn.RMSNorm applies `x_normed * weight` with no +1; matching it
    # is what brings the per-layer state to ~1.0 cosine vs the
    # reference. Flipping these to True re-breaks generation.
    if q_norm_weight is not None:
        q = rms_norm(q, q_norm_weight, eps=rms_eps, gemma_unit_offset=False)
    if k_norm_weight is not None:
        k = rms_norm(k, k_norm_weight, eps=rms_eps, gemma_unit_offset=False)

    # K=V-tied path: V picks up the K projection (alias) but NOT
    # k_norm or RoPE. Reference applies its own v_norm (RMSNormNoScale)
    # below.
    if v is None:
        # `linear(hidden, k_weight)` is what `k` was BEFORE k_norm;
        # we need to re-project (or capture pre-norm K) to get the
        # un-normalized V. Cheapest path: project hidden through
        # k_weight again. The duplicate matmul is the price of
        # keeping k post-norm separate from v pre-norm.
        v = linear(hidden, k_weight).reshape(1, num_kv_heads, 1, head_dim)

    # Step 26E: V gets `RMSNormNoScale` — a plain RMS normalization
    # with no learnable scale weight — when q_norm/k_norm are
    # configured. (mlx_lm/models/gemma4_text.py:217 wires
    # `RMSNormNoScale(self.head_dim, eps=...)` whenever has_kv.)
    # When q/k norms are absent (Mixtral / Qwen / synthetic), v_norm
    # is skipped.
    if k_norm_weight is not None:
        v = rms_norm(v, weight=None, eps=rms_eps)

    # RoPE at this token's position in the sequence — the cache's
    # current length IS the next position (0-indexed). At position 0
    # RoPE is identity, which keeps step 16's smoke math correct
    # when wired through this code path with an empty cache.
    # `rotary_dim` is None for sliding (full rotation) or
    # `head_dim * partial_rotary_factor` for full (partial rotation).
    #
    # Q and K both get RoPE so the q·k dot product carries positional
    # information. V deliberately does NOT — that's the standard
    # transformer convention. In the K=V-tied path, `v` was derived
    # above from the un-rotated K projection and only `q` and `k` are
    # rotated below, so V keeps its pre-RoPE values (correct semantic;
    # only K rotates).
    position_offset = kv_cache.length(layer_id)
    q = apply_rope(
        q, position_offset=position_offset,
        theta=rope_theta, rotary_dim=rotary_dim, rope_type=rope_type,
    )
    k = apply_rope(
        k, position_offset=position_offset,
        theta=rope_theta, rotary_dim=rotary_dim, rope_type=rope_type,
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
    # Step 26E: `attn_scale=None` falls back to `1/sqrt(head_dim)` (the
    # standard transformer scale). Gemma 4 passes 1.0 explicitly
    # because its q_norm + k_norm already bring Q/K rows to unit RMS,
    # so the additional 1/sqrt(d) would over-suppress attention scores.
    # `np.sqrt(int)` returns numpy.float64; dividing an fp32 score
    # array by an fp64 scalar promotes the result to fp64 — and that
    # promotion then propagates through softmax, the v matmul, and
    # the o_weight projection, so the *layer's output* becomes fp64.
    # Layer L+1's hidden then carries fp64 into its KV cache, which
    # silently doubles the per-layer KV bytes (visible in
    # `kv_snapshot.per_layer_bytes`). Cast the divisor to fp32 to
    # keep the whole attention path in single precision.
    scale = (
        np.float32(attn_scale)
        if attn_scale is not None
        else (np.float32(head_dim) ** -0.5)
    )
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
    math without managing a cache. Don't pass `kv_cache` or `layer_id`
    here (they would collide with the ones this wrapper supplies);
    call `attention_decode_step` directly when you need a real cache.
    Every other keyword, `rope_theta` included, is forwarded as-is.
    """
    return attention_decode_step(
        hidden,
        **kwargs,
        kv_cache=KVCache(num_layers=1),
        layer_id=0,
    )
