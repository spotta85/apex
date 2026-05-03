"""Step 16E demo: one-token forward + argmax on real Gemma 4.

Pipeline:
    token_id (hardcoded)
        → embedding lookup
        → N decoder layers (apex.decoder.single_token_decoder_layer)
        → final RMSNorm
        → tied lm_head (hidden @ embed.T) + final_logit_softcapping
        → argmax over vocab

Token I/O:
    Tokenizer integration is intentionally NOT done here — neither
    `tokenizers` nor `transformers` is in the project deps. Pass
    `--token-id N` to feed a known token (Gemma 4 BOS = 2 by default)
    and the script prints the predicted next id. To decode to a
    string, install `tokenizers` and run:

        from tokenizers import Tokenizer
        tok = Tokenizer.from_file("<checkpoint>/tokenizer.json")
        print(tok.decode([next_id]))

Layer count:
    Default `--num-layers 1` runs only layer 0 (sliding_attention,
    head_dim=256 with v_proj). Layer 5 is the first full_attention
    layer and ties K=V — the decoder doesn't yet handle that
    heterogeneity, so going past 5 layers will fail at load time.
    For a smoke (correctness = finite), one layer is enough to
    exercise the full embedding + decoder + lm_head path.

Run:
    uv run python scripts/demo_one_token_generate.py
        [--path /path/to/checkpoint] [--num-layers 1] [--token-id 2]
        [--top-k 8] [--max-memory-mb 512]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Make the apex package importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apex.backends.fake_backend import FakeBackend
from apex.decoder import single_token_decoder_layer
from apex.dequant import DequantizingExpertStore
from apex.expert_cache import ExpertCache
from apex.expert_manager import ExpertManager
from apex.generation import embed_token, final_logits, sample_argmax
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


# Layers 0-4 are sliding_attention (per text_config.layer_types). Layer
# 5 is the first full_attention layer, which ties K=V and uses
# head_dim=512 — supporting it is a follow-on step.
_MAX_SLIDING_LAYERS = 5


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    add_real_moe_args(parser, default_max_memory_mb=512)
    parser.add_argument(
        "--num-layers",
        type=int,
        default=1,
        help=(
            f"how many decoder layers to drive (default 1; max "
            f"{_MAX_SLIDING_LAYERS} — beyond is full_attention, not "
            "yet supported)"
        ),
    )
    parser.add_argument(
        "--token-id",
        type=int,
        default=2,
        help="input token id (default 2 = Gemma 4 BOS)",
    )
    args = parser.parse_args()

    if args.num_layers > _MAX_SLIDING_LAYERS:
        raise SystemExit(
            f"--num-layers {args.num_layers} > {_MAX_SLIDING_LAYERS}: "
            "Gemma 4 layer 5+ uses full_attention with K=V tied; this "
            "demo only supports the sliding-attention layers (0-4)."
        )

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

    # Embedding dequant is the dominant memory cost (~3 GB on Gemma 4).
    # Loaded once, reused for both the input lookup and the tied lm_head.
    print("loading + dequantizing embedding table (a few seconds)...")
    embedding = load_embedding(path, layout)
    print(f"embedding shape: {embedding.shape}")
    final_norm_w = load_final_norm(path, layout)
    print(f"final_norm shape: {final_norm_w.shape}")

    # MoE plumbing (router + manager + cache) — same as the decoder smoke.
    metrics = Metrics()
    cache = ExpertCache(
        max_memory_mb=args.max_memory_mb, verbose=False, metrics=metrics,
    )
    quant_store = quantized_store_for_layout(path, layout)
    store = DequantizingExpertStore(quant_store, verbose=False)
    manager = ExpertManager(store=store, cache=cache, metrics=metrics)
    backend = FakeBackend()
    router = MultiLayerRealRouter(path, layout, top_k=args.top_k)

    # Embedding lookup → starting hidden state.
    print(f"input token id: {args.token_id}")
    hidden = embed_token(embedding, args.token_id)
    assert hidden.shape == (1, text_cfg["hidden_size"]), hidden.shape

    # N decoder layers. Per-layer: load attn + norms (lazy-cached
    # across layers since each one's weights differ), compose attention
    # + MoE through `single_token_decoder_layer`.
    for layer_id in range(args.num_layers):
        attn_weights = load_attention_weights(path, layout, layer_id)
        norms = load_layer_norms(path, layout, layer_id)
        attn_kwargs = dict(
            q_weight=attn_weights["q_weight"], k_weight=attn_weights["k_weight"],
            v_weight=attn_weights["v_weight"], o_weight=attn_weights["o_weight"],
            num_heads=text_cfg["num_attention_heads"],
            num_kv_heads=text_cfg["num_key_value_heads"],
            head_dim=text_cfg["head_dim"],
            q_norm_weight=attn_weights["q_norm"],
            k_norm_weight=attn_weights["k_norm"],
            rms_eps=text_cfg["rms_norm_eps"],
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
            hidden = single_token_decoder_layer(
                hidden,
                norms=norms,
                attn_kwargs=attn_kwargs,
                moe_call=moe_call,
                rms_eps=text_cfg["rms_norm_eps"],
            )
        finite = bool(np.all(np.isfinite(hidden)))
        print(
            f"layer {layer_id} output finite: {finite}  "
            f"min={float(hidden.min()):.4f} max={float(hidden.max()):.4f}"
        )

    # Final RMSNorm (Gemma `+1` convention) before lm_head.
    hidden = rms_norm(
        hidden, final_norm_w,
        eps=text_cfg["rms_norm_eps"], gemma_unit_offset=True,
    )
    print(f"final_norm output finite: {bool(np.all(np.isfinite(hidden)))}")

    # lm_head via tied embedding + Gemma softcap + argmax.
    softcap = text_cfg.get("final_logit_softcapping")
    with nvfp4_matmul_warnings():
        logits = final_logits(hidden, embedding, softcap=softcap)
    assert logits.shape == (1, text_cfg["vocab_size"]), logits.shape
    assert np.all(np.isfinite(logits)), "logits must be finite"
    next_id = sample_argmax(logits)
    print(f"finite logits: True  shape={logits.shape}")
    print(f"next token id: {next_id}")
    print("decoded token: <tokenizer not loaded — token id only>")
    print()
    print("[ok] real one-token generation passed")


if __name__ == "__main__":
    main()
