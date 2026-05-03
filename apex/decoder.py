"""Single-token decoder layer: norms + attention + residual + MoE + residual.

One callable, `single_token_decoder_layer`, that composes the
building blocks from steps 16A-16C into the standard Gemma-4-style
"sandwich norm" decoder block:

  residual = hidden
  hidden = input_layernorm(hidden)
  hidden = single_token_attention(hidden, ...)
  hidden = post_attention_layernorm(hidden)
  hidden = residual + hidden                    # attention residual

  residual = hidden
  hidden = pre_feedforward_layernorm(hidden)
  hidden = moe_call(hidden)                     # routed MoE block
  hidden = post_feedforward_layernorm(hidden)
  hidden = residual + hidden                    # feedforward residual

`moe_call` is a callable `hidden_state -> hidden_state` so the same
function works whether the MoE step comes from an in-process
ExpertManager + run_moe_layer pipeline (the smoke / real-Gemma path)
or a hand-supplied closure (synthetic tests).

What this is NOT (intentional gaps for now):
  - No KV cache and no RoPE — single-token at position 0 makes both
    moot. Multi-token / generation-loop versions land in 16E and
    follow-on steps.
  - No dense-MLP shared expert. Gemma 4 layers carry both routed
    experts AND a dense `mlp.{gate,up,down}_proj` block (and extra
    `_1`/`_2` norms for the parallel paths). Driving both would
    change the residual sum; supporting it is a follow-on step.
  - No sliding-window vs full-attention switching: layer 0 is
    sliding (head_dim=256, has v_proj). Full-attention layers
    (5/11/17/23/29) tie K=V and use head_dim=512; the smoke targets
    sliding only.
"""

import numpy as np

from apex.attention import single_token_attention
from apex.ops import rms_norm


def single_token_decoder_layer(
    hidden,
    *,
    norms: dict,
    attn_kwargs: dict,
    moe_call,
    rms_eps: float = 1e-6,
):
    """Run one decoder layer's forward pass for a single token.

    `norms`: dict with keys {input, post_attn, pre_ffn, post_ffn},
    each a 1-D ndarray of shape (hidden_dim,) — the four BF16 norm
    weights for this layer.

    `attn_kwargs`: dict of keyword arguments to `single_token_attention`
    (everything except `hidden`).

    `moe_call`: callable `hidden -> hidden` running this layer's MoE
    forward. Closure over the layer id, router, expert manager, and
    backend.

    Returns hidden ndarray with the same shape as input.
    """
    # Attention sub-block. Sandwich norm: pre-norm before attention,
    # post-norm on the attention output, then residual add.
    residual = hidden
    h = rms_norm(hidden, norms["input"], eps=rms_eps, gemma_unit_offset=True)
    h = single_token_attention(h, **attn_kwargs)
    h = rms_norm(h, norms["post_attn"], eps=rms_eps, gemma_unit_offset=True)
    hidden = residual + h

    # Feedforward sub-block. Same sandwich-norm structure with the
    # MoE block in place of the dense MLP.
    residual = hidden
    h = rms_norm(hidden, norms["pre_ffn"], eps=rms_eps, gemma_unit_offset=True)
    h = moe_call(h)
    h = rms_norm(h, norms["post_ffn"], eps=rms_eps, gemma_unit_offset=True)
    return residual + h
