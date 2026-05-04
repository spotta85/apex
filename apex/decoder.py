"""Decoder layer — single canonical implementation, two convenience wrappers.

`decoder_decode_step` is the canonical implementation: sandwich-norm
Gemma block (input_norm → attention → post_attn_norm → +residual →
pre_ffn_norm → {MoE [+ dense MLP]} → post_ffn_norm → +residual) with
attention flowing through `attention_decode_step` (KVCache + RoPE).

`single_token_decoder_layer` is a thin wrapper that constructs a
fresh 1-layer KVCache and calls `decoder_decode_step` at layer 0 —
RoPE position 0 is identity, so it remains mathematically equivalent
to the pre-step-17 "no cache" path used by step 16D's smoke.

Why one implementation: the two surfaces previously duplicated the
entire sandwich-norm + MoE composition (everything but which
attention function to call). One body keeps the residual structure
in one place, where future tweaks (sandwich-norm variants, dense
MLP shared expert) can be made once.

Step 24B added an optional `dense_mlp_call` parallel branch: when
provided, both the routed MoE and the dense MLP consume the same
`pre_ffn_norm(hidden)` and their outputs are summed before the
shared `post_ffn_norm + residual`. Default `None` reproduces the
pre-step-24 routed-only behavior byte-for-byte.

What is intentionally NOT here:
  - The `_1` / `_2` parallel-path norm variants the real model
    carries (one set per parallel feedforward path). Routed and
    dense paths share `pre_ffn` / `post_ffn` in the standard
    interpretation; if quality stays bad after 24B, those variants
    are the next thing to investigate.
  - Sliding vs full attention switching is owned by
    `attention_decode_step` (step 23 wired the per-layer kwargs).
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
    dense_mlp_call=None,
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

    `dense_mlp_call`: optional callable `hidden -> hidden` running this
    layer's parallel dense FFN (Gemma 4 ships one alongside the routed
    MoE). When provided, both branches consume the same
    `pre_ffn_norm(hidden)` and their outputs are summed before the
    shared `post_ffn_norm + residual`. None reproduces the pre-step-24
    routed-only behavior.
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
    moe_out = moe_call(h)
    if dense_mlp_call is not None:
        # Parallel branches share `pre_ffn` input AND `post_ffn` output;
        # summing before the post-norm matches the standard interpretation
        # of Gemma 4's parallel feedforward block.
        moe_out = moe_out + dense_mlp_call(h)
    h = rms_norm(moe_out, norms["post_ffn"], eps=rms_eps, gemma_unit_offset=True)
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
