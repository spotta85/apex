"""Step 17D demo: multi-token generation on real Gemma 4 with KV cache.

Pipeline (decode-only, "simple loop first" per the step 17 spec):

    prompt = [tokens...]                       # hardcoded ids
    for token in prompt:
        embed → decoder_decode_step → final_norm → lm_head → argmax
        cache grows by 1 per layer
    # the last prefill argmax is the first generated token
    for _ in range(N - 1):
        feed last generated token through the same decode path
        argmax → next token

Prefill is deliberately the same code path as decode — call
`decoder_decode_step` for every prompt token. This is O(prompt_len)
through the cache + O(layers × generated_count) thereafter, with
the simplicity of one function instead of two. A real prefill that
batches the prompt into a single multi-token attention with causal
mask is a follow-on step.

Memory:
  - Embedding table dequant: ~3 GB (one-time)
  - Per-layer KV cache: (1, num_kv_heads=8, seq, head_dim=256) × 2
    tensors × 4 bytes = 16 KB per token per layer. With one layer
    and a 100-token sequence that's 1.6 MB — negligible.

Run:
    uv run python scripts/demo_generate_with_kv.py
        [--path /path/to/checkpoint] [--num-layers 1]
        [--prompt 2,235,1,...] [--generate 5]
        [--top-k 8] [--max-memory-mb 512]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Make the apex package importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apex.backends.fake_backend import FakeBackend
from apex.decoder import decoder_decode_step
from apex.dequant import DequantizingExpertStore
from apex.expert_cache import ExpertCache
from apex.expert_manager import ExpertManager
from apex.generation import embed_token, final_logits, sample_argmax
from apex.kv_cache import KVCache
from apex.metrics import Metrics
from apex.moe_layer import run_moe_layer
from apex.ops import rms_norm
from apex.real_router import MultiLayerRealRouter
from apex.checkpoint_loaders import (
    load_attention_weights,
    load_embedding,
    load_final_norm,
    load_layer_norms,
    read_text_config,
)
from scripts._real_moe_helpers import (
    add_real_moe_args,
    detect_real_moe_layout,
    nvfp4_matmul_warnings,
    quantized_store_for_layout,
)


# Same cap as the 16E demo: layers 5+ are full_attention with K=V tied
# and head_dim=512 — not yet supported.
_MAX_SLIDING_LAYERS = 5


def _parse_prompt(prompt_str: str) -> list:
    return [int(tok) for tok in prompt_str.split(",") if tok.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    add_real_moe_args(parser, default_max_memory_mb=512)
    parser.add_argument(
        "--num-layers",
        type=int,
        default=1,
        help=(
            f"how many decoder layers (default 1; max {_MAX_SLIDING_LAYERS} — "
            "layer 5+ uses full_attention with K=V tied, not supported)"
        ),
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="2",
        help=(
            "comma-separated token ids for the prompt (default '2' = "
            "Gemma 4 BOS only)"
        ),
    )
    parser.add_argument(
        "--generate",
        type=int,
        default=3,
        help="how many tokens to generate after the prompt (default 3)",
    )
    args = parser.parse_args()

    if args.num_layers > _MAX_SLIDING_LAYERS:
        raise SystemExit(
            f"--num-layers {args.num_layers} > {_MAX_SLIDING_LAYERS}: "
            "Gemma 4 layer 5+ uses full_attention with K=V tied; this "
            "demo only supports the sliding-attention layers (0-4)."
        )

    prompt_ids = _parse_prompt(args.prompt)
    if not prompt_ids:
        raise SystemExit("--prompt must contain at least one token id")

    path = Path(args.path)
    if not path.is_dir():
        raise SystemExit(f"checkpoint not found: {path}")

    layout_name, layout = detect_real_moe_layout(path)
    text_cfg = read_text_config(path)
    print(f"layout: {layout_name}")
    print(
        f"hidden_size={text_cfg['hidden_size']} "
        f"vocab_size={text_cfg['vocab_size']} "
        f"final_softcap={text_cfg.get('final_logit_softcapping')}"
    )
    print(f"prompt: {prompt_ids}")
    print(f"prefill tokens: {len(prompt_ids)}  generate: {args.generate}")
    print()

    # Load big tensors once.
    print("loading + dequantizing embedding table (a few seconds)...")
    embedding = load_embedding(path, layout)
    final_norm_w = load_final_norm(path, layout)

    # Per-layer attn weights + norms get loaded once and reused
    # across both prefill and decode steps. Each layer's KV cache
    # grows monotonically — that's the whole point.
    per_layer_attn = []
    per_layer_norms = []
    for layer_id in range(args.num_layers):
        per_layer_attn.append(load_attention_weights(path, layout, layer_id))
        per_layer_norms.append(load_layer_norms(path, layout, layer_id))

    # MoE plumbing — same as the 16D/16E demos.
    metrics = Metrics()
    cache = ExpertCache(
        max_memory_mb=args.max_memory_mb, verbose=False, metrics=metrics,
    )
    quant_store = quantized_store_for_layout(path, layout)
    store = DequantizingExpertStore(quant_store, verbose=False)
    manager = ExpertManager(store=store, cache=cache, metrics=metrics)
    backend = FakeBackend()
    router = MultiLayerRealRouter(path, layout, top_k=args.top_k)

    # KV cache lives across all prefill + generate steps.
    kv = KVCache(num_layers=args.num_layers)

    def _decode(token_id: int) -> int:
        """Push one token through embed → N decoder layers → lm_head → argmax."""
        hidden = embed_token(embedding, token_id)
        for layer_id in range(args.num_layers):
            attn_weights = per_layer_attn[layer_id]
            norms = per_layer_norms[layer_id]
            attn_kwargs = dict(
                q_weight=attn_weights["q_weight"],
                k_weight=attn_weights["k_weight"],
                v_weight=attn_weights["v_weight"],
                o_weight=attn_weights["o_weight"],
                num_heads=text_cfg["num_attention_heads"],
                num_kv_heads=text_cfg["num_key_value_heads"],
                head_dim=text_cfg["head_dim"],
                q_norm_weight=attn_weights["q_norm"],
                k_norm_weight=attn_weights["k_norm"],
                rms_eps=text_cfg["rms_norm_eps"],
                rope_theta=10000.0,             # sliding_attention default
            )

            def moe_call(h, lid=layer_id):
                with nvfp4_matmul_warnings():
                    ids, weights = router.route(lid, h)
                    return run_moe_layer(
                        h, ids, weights,
                        lambda eid: manager.ensure_loaded(lid, eid),
                        backend,
                    )

            with nvfp4_matmul_warnings():
                hidden = decoder_decode_step(
                    hidden,
                    norms=norms,
                    attn_kwargs=attn_kwargs,
                    moe_call=moe_call,
                    kv_cache=kv,
                    layer_id=layer_id,
                    rms_eps=text_cfg["rms_norm_eps"],
                )

        hidden = rms_norm(
            hidden, final_norm_w,
            eps=text_cfg["rms_norm_eps"], gemma_unit_offset=True,
        )
        with nvfp4_matmul_warnings():
            logits = final_logits(
                hidden, embedding,
                softcap=text_cfg.get("final_logit_softcapping"),
            )
        assert np.all(np.isfinite(logits)), "logits must be finite"
        return sample_argmax(logits)

    # Prefill. The argmax of the LAST prompt token is the first
    # generated token — same convention HF generation loops use.
    print("=== prefill ===")
    last_pred = None
    for prompt_token in prompt_ids:
        last_pred = _decode(prompt_token)
        print(
            f"  token {prompt_token} → next-id {last_pred}  "
            f"kv_cache length: {kv.length(0)}  "
            f"kv_cache bytes: {kv.total_bytes}"
        )

    # Decode loop. Feed the previous prediction; argmax becomes the
    # next prediction; repeat.
    print()
    print("=== generate ===")
    generated = [last_pred]
    for step in range(args.generate - 1):
        last_pred = _decode(last_pred)
        generated.append(last_pred)
        print(
            f"  generating token {step + 2}  "
            f"kv_cache length: {kv.length(0)}  "
            f"next-id {last_pred}"
        )

    print()
    print(f"prompt:    {prompt_ids}")
    print(f"generated: {generated}")
    print(
        f"final kv_cache length: {kv.length(0)}  "
        f"bytes: {kv.total_bytes}  "
        f"({kv.total_bytes / (1024 * 1024):.2f} MB)"
    )
    print()
    print("[ok] real multi-token generation passed")


if __name__ == "__main__":
    main()
