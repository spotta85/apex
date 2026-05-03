"""Decoder layer — single canonical implementation, two convenience wrappers.

`decoder_decode_step` is the canonical implementation: sandwich-norm
Gemma block (input_norm → attention → post_attn_norm → +residual →
pre_ffn_norm → MoE → post_ffn_norm → +residual) with attention
flowing through `attention_decode_step` (KVCache + RoPE).

`single_token_decoder_layer` is a thin wrapper that constructs a
fresh 1-layer KVCache and calls `decoder_decode_step` at layer 0 —
RoPE position 0 is identity, so it remains mathematically equivalent
to the pre-step-17 "no cache" path used by step 16D's smoke.

Why one implementation: the two surfaces previously duplicated the
entire sandwich-norm + MoE composition (everything but which
attention function to call). One body keeps the residual structure
in one place, where future tweaks (sandwich-norm variants, dense
MLP shared expert) can be made once.

What is intentionally NOT here:
  - Dense-MLP shared-expert path. Real Gemma 4 layers add a dense
    MLP output to the routed-MoE output before the post feedforward
    norm; we only run the routed MoE.
  - The `_1` / `_2` parallel-path norm variants the real model
    carries (one set per parallel feedforward path).
  - Sliding vs full attention switching. Layer 0 is sliding (head_dim=256,
    has v_proj). Full-attention layers (5/11/17/23/29) tie K=V and
    use head_dim=512 — out of scope for now.
"""

from apex.attention import attention_decode_step
from apex.kv_cache import KVCache
from apex.ops import rms_norm


def decoder_decode_step(
    hidden,
    *,
    norms: dict,
    attn_kwargs: dict,
    moe_call,
    kv_cache,
    layer_id: int,
    rms_eps: float = 1e-6,
):
    """Run one decode step of the sandwich-norm decoder block.

    `norms`: dict with keys {input, post_attn, pre_ffn, post_ffn},
    each a 1-D ndarray of shape (hidden_dim,) — the four BF16 norm
    weights for this layer.

    `attn_kwargs`: keyword arguments to `attention_decode_step`
    *except* `kv_cache` and `layer_id`, which are threaded through
    here so callers can't accidentally pass two layer ids that
    disagree.

    `moe_call`: callable `hidden -> hidden` running this layer's MoE
    forward. Closure over the layer id, router, expert manager, and
    backend.
    """
    residual = hidden
    h = rms_norm(hidden, norms["input"], eps=rms_eps, gemma_unit_offset=True)
    h = attention_decode_step(
        h, **attn_kwargs, kv_cache=kv_cache, layer_id=layer_id,
    )
    h = rms_norm(h, norms["post_attn"], eps=rms_eps, gemma_unit_offset=True)
    hidden = residual + h

    residual = hidden
    h = rms_norm(hidden, norms["pre_ffn"], eps=rms_eps, gemma_unit_offset=True)
    h = moe_call(h)
    h = rms_norm(h, norms["post_ffn"], eps=rms_eps, gemma_unit_offset=True)
    return residual + h


def single_token_decoder_layer(
    hidden,
    *,
    norms: dict,
    attn_kwargs: dict,
    moe_call,
    rms_eps: float = 1e-6,
):
    """Position-0, no-cache wrapper over `decoder_decode_step`.

    Constructs a fresh `KVCache(num_layers=1)`, calls
    `decoder_decode_step` at layer 0, and drops the cache. RoPE at
    position 0 is identity, so this is equivalent to the pre-step-17
    "no cache, no RoPE" decoder block — matching the step 16D smoke
    contract (finite + correctly shaped output).

    Use this for one-off decoder smokes / tests; use
    `decoder_decode_step` directly when you have a real cache to
    thread across multiple decode steps.
    """
    return decoder_decode_step(
        hidden,
        norms=norms,
        attn_kwargs=attn_kwargs,
        moe_call=moe_call,
        kv_cache=KVCache(num_layers=1),
        layer_id=0,
        rms_eps=rms_eps,
    )
