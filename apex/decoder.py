"""Decoder layer — single canonical implementation, two convenience wrappers.

`decoder_decode_step` is the canonical implementation: sandwich-norm
Gemma block (input_norm → attention → post_attn_norm → +residual →
pre_ffn_norm → {MoE [+ dense MLP]} → post_ffn_norm → +residual) with
attention flowing through `attention_decode_step` (KVCache + RoPE).
That single-FFN-norm shape is the fallback for layouts without the
`_1`/`_2` norms; Gemma 4's two-branch FFN is spelled out under
"Step 26B" below.

Every RMSNorm in this block is the plain form
`x / sqrt(mean(x²) + eps) * weight` (`gemma_unit_offset=False`, no
`+1`): Gemma 4's norm weights are stored full-scale, and the Gemma 1/2
delta-from-1 convention does not apply to them.

`single_token_decoder_layer` is a thin wrapper that constructs a
fresh 1-layer KVCache and calls `decoder_decode_step` at layer 0 —
RoPE position 0 is identity, so it remains mathematically equivalent
to the pre-step-17 "no cache" path used by step 16D's smoke.

Why one implementation: the two surfaces previously duplicated the
entire sandwich-norm + MoE composition (everything but which
attention function to call). One body keeps the residual structure
in one place, where future tweaks (sandwich-norm variants, dense
MLP shared expert) can be made once.

Step 24B added an optional `dense_mlp_call` parallel branch. In the
fallback path (no `_1`/`_2` norms) both the routed MoE and the dense
MLP consume the same `pre_ffn_norm(hidden)` and their outputs are
summed before the shared `post_ffn_norm + residual`. Default `None`
means no dense branch (routed-only).

Step 25A added an optional `trace` dict: when provided, gets
populated with raw intermediate tensors (post-attn-norm output, MoE
output, dense MLP output, and in the parallel path the router's ids
and weights) so a caller — typically `MoEModel.forward_with_trace` —
can compute norms / top-k signatures externally for parity
comparison. None ⇒ no tracing work.

Step 26B activated the per-branch norm variants. When `norms` carries
the optional `pre_ffn_2` / `post_ffn_1` / `post_ffn_2` keys (Gemma 4
ships them on disk and the loader surfaces them automatically), the
feedforward block runs as a true parallel architecture:

    h_dense  = post_ffn_1(mlp(pre_ffn(hidden)))
    ids, w   = route(hidden)              # router reads the RAW hidden
    h_routed = post_ffn_2(experts(pre_ffn_2(hidden), ids, w))
    h        = post_ffn(h_dense + h_routed)
    return (residual + h) * layer_scalar

Each branch gets its own input + output norm: the dense branch uses
the unsuffixed `pre_ffn` with `post_ffn_1`, the routed branch uses
`pre_ffn_2` with `post_ffn_2`, and `post_ffn` (no suffix) acts as the
combined output norm on the sum before the residual. Layouts without
the variants (Mixtral, Qwen, synthetic) take the fallback path — both
branches share `pre_ffn` and the unsuffixed `post_ffn` only.
`layer_scalar` (one bf16 scalar per layer in the Gemma 4 checkpoint)
multiplies the layer's output after the residual add; None skips it.

What is intentionally NOT here:
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
    moe_call=None,
    kv_cache,
    layer_id: int,
    rms_eps: float = 1e-6,
    dense_mlp_call=None,
    trace: dict | None = None,
    route_call=None,
    experts_call=None,
    layer_scalar=None,
):
    """Run one decode step of the sandwich-norm decoder block.

    `norms`: dict with keys {input, post_attn, pre_ffn, post_ffn},
    each a 1-D ndarray of shape (hidden_dim,) — the four BF16 norm
    weights for this layer. Step 26B added optional `pre_ffn_2`,
    `post_ffn_1`, `post_ffn_2` for the Gemma 4 parallel-branch path.

    `attn_kwargs`: keyword arguments to `attention_decode_step`
    *except* `kv_cache` and `layer_id`, which are threaded through
    here so callers can't accidentally pass two layer ids that
    disagree.

    Two FFN-block call surfaces, picked by which kwargs are provided:

      - **`moe_call(h) -> hidden`**: combined route + experts in one
        callable. Used by every pre-step-26B caller (synthetic tests,
        Mixtral / Qwen demos, the smoke scripts). With this surface
        the decoder routes on the post-pre-ffn-norm hidden.
      - **`route_call(h) -> (ids, weights)` + `experts_call(h, ids,
        weights) -> tensor`**: split surface needed by Gemma 4's
        actual architecture. The router consumes the RAW post-attn
        hidden (NOT the pre-ffn-norm version), while experts consume
        the pre_ffn_2-normalized hidden. Step 26B added this surface
        because the previous moe_call shape couldn't express the
        different inputs the two operations want.

    `dense_mlp_call`: optional callable `hidden -> hidden` running this
    layer's parallel dense FFN (Gemma 4 ships one alongside the routed
    MoE). Required when running the parallel-arch path.

    `layer_scalar`: optional shape-(1,) bf16/fp32 multiplier applied
    to the layer's final output (post-residual-add). Gemma 4 layers
    ship one of these per layer. None ⇒ no multiplier (back-compat).

    `trace`: optional dict that, when provided, gets populated in place
    with the raw intermediate tensors useful for parity / debugging:
    `attn_out` (post-attn-norm), `moe_out` (raw routed MoE output),
    `dense_out` (raw dense MLP output, only when `dense_mlp_call` is
    set), plus `route_ids` / `route_weights` on the parallel-arch path.
    None ⇒ no allocation, no work. The dict is mutated rather than
    returned so the caller can pass a fresh dict per layer.
    """
    # --- Attention sub-block (unchanged across all paths) ----------
    # All rms_norm calls below pass gemma_unit_offset=False: this
    # Gemma 4 NVFP4 checkpoint stores norm weights already centered at
    # full scale (γ ≈ 1+), matching mlx's plain `x_normed * weight`.
    # The Gemma 1/2 delta-from-1 (+1 offset) convention does NOT apply
    # here — using it inflates every norm and compounds into gibberish.
    residual = hidden
    h = rms_norm(hidden, norms["input"], eps=rms_eps, gemma_unit_offset=False)
    h = attention_decode_step(
        h, **attn_kwargs, kv_cache=kv_cache, layer_id=layer_id,
    )
    h = rms_norm(h, norms["post_attn"], eps=rms_eps, gemma_unit_offset=False)
    if trace is not None:
        trace["attn_out"] = h
    hidden = residual + h

    residual = hidden

    # Step 26B: parallel-branch architecture (matches mlx-lm /
    # transformers Gemma 4 reference). Activated only when the layout
    # provides the `_1` / `_2` norm variants AND the caller supplies
    # the split route_call + experts_call (so the router can consume
    # the raw post-attn hidden while experts consume the pre_ffn_2
    # normalized version — the previous combined moe_call couldn't
    # express that).
    use_parallel_arch = (
        dense_mlp_call is not None
        and route_call is not None
        and experts_call is not None
        and "pre_ffn_2" in norms
        and "post_ffn_1" in norms
        and "post_ffn_2" in norms
    )

    if use_parallel_arch:
        # Dense branch: uses the UNSUFFIXED pre_ffn + post_ffn_1.
        # (Reference: mlx_lm/models/gemma4_text.py lines 354-356,
        # `h1 = self.mlp(self.pre_feedforward_layernorm(h))` then
        # `h1 = self.post_feedforward_layernorm_1(h1)`.)
        h_dense_in = rms_norm(
            hidden, norms["pre_ffn"],
            eps=rms_eps, gemma_unit_offset=False,
        )
        dense_out = dense_mlp_call(h_dense_in)
        if trace is not None:
            trace["dense_out"] = dense_out
        h_dense = rms_norm(
            dense_out, norms["post_ffn_1"],
            eps=rms_eps, gemma_unit_offset=False,
        )

        # Routed branch: uses the `_2` variants. Critical detail —
        # router runs on RAW `hidden` (the post-attn residual stream),
        # NOT on the pre_ffn_2-normalized version. This is the bug
        # the pre-step-26B code path silently had: `moe_call(h_norm)`
        # forced both router and experts to consume the same
        # normalized input, but the reference routes on the unnormalized
        # residual stream.
        ids, weights = route_call(hidden)
        if trace is not None:
            trace["route_ids"] = ids
            trace["route_weights"] = weights
        h_routed_in = rms_norm(
            hidden, norms["pre_ffn_2"],
            eps=rms_eps, gemma_unit_offset=False,
        )
        routed_out = experts_call(h_routed_in, ids, weights)
        if trace is not None:
            trace["moe_out"] = routed_out
        h_routed = rms_norm(
            routed_out, norms["post_ffn_2"],
            eps=rms_eps, gemma_unit_offset=False,
        )

        # Combined output norm on the sum, then residual add.
        h = rms_norm(
            h_dense + h_routed, norms["post_ffn"],
            eps=rms_eps, gemma_unit_offset=False,
        )
        result = residual + h

    else:
        # Pre-step-26B fallback path: synthetic / Mixtral / Qwen / any
        # caller still passing only `moe_call`. Both branches share
        # the unsuffixed pre / post norms; routing is done by moe_call
        # on the pre-ffn-normalized hidden (the inherited behavior
        # from steps 17D + 24B).
        if moe_call is None:
            raise ValueError(
                "decoder_decode_step requires either moe_call (back-compat) "
                "or route_call + experts_call + dense_mlp_call + "
                "_1/_2 norms (parallel-arch path)"
            )
        h = rms_norm(
            hidden, norms["pre_ffn"], eps=rms_eps, gemma_unit_offset=False,
        )
        moe_out = moe_call(h)
        if trace is not None:
            trace["moe_out"] = moe_out
        if dense_mlp_call is not None:
            dense_out = dense_mlp_call(h)
            if trace is not None:
                trace["dense_out"] = dense_out
            moe_out = moe_out + dense_out
        h = rms_norm(
            moe_out, norms["post_ffn"], eps=rms_eps, gemma_unit_offset=False,
        )
        result = residual + h

    # Step 26B: per-layer scalar multiplier (Gemma 4 ships
    # `layer_scalar` of shape (1,) per layer; reference applies it
    # to the final layer output AFTER the FFN-block residual add).
    # Layouts without the tensor pass None ⇒ no-op.
    if layer_scalar is not None:
        result = result * layer_scalar

    return result


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
