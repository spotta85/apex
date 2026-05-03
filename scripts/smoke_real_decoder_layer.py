"""Step 16D smoke: full decoder layer 0 (norms + attention + MoE + residuals).

Loads layer 0's attention weights (NVFP4_U32), router weight, all four
sandwich-norm weights (BF16), and runs a single token through the
sandwich-norm decoder block in `apex.decoder.single_token_decoder_layer`.

What gets exercised:
  - The 16C single_token_attention path on real attention weights.
  - The existing real-checkpoint MoE path (router + DequantizingExpertStore
    + ExpertCache + run_moe_layer) — same machinery the demo_real_runtime
    script uses, just plugged into the decoder block as a `moe_call`.
  - Sandwich-norm composition (input → attn → post_attn_norm → +resid,
    pre_ffn_norm → moe → post_ffn_norm → +resid).

What does NOT match HF Gemma 4 numerically (acceptable for smoke):
  - The dense-MLP shared-expert path is skipped. Real Gemma 4 layers
    add a dense MLP output to the routed-MoE output before the post
    feedforward norm; we only run the routed MoE.
  - RoPE / KV cache absent — fine for single-token at position 0.
  - The `_1` / `_2` parallel-path norm variants are unused.

Output contract (smoke = finite + correctly shaped):
    running real decoder layer 0
    attention output finite: true
    moe output finite: true
    layer output shape: (1, 2816)

Run:
    uv run python scripts/smoke_real_decoder_layer.py
        [--path /path/to/checkpoint] [--layer 0] [--seed 0]
        [--max-memory-mb 512]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Make the apex package importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apex.attention import single_token_attention
from apex.backends.fake_backend import FakeBackend
from apex.decoder import single_token_decoder_layer
from apex.dequant import DequantizingExpertStore
from apex.expert_cache import ExpertCache
from apex.expert_manager import ExpertManager
from apex.metrics import Metrics
from apex.moe_layer import run_moe_layer
from apex.real_router import MultiLayerRealRouter
from apex.checkpoint_loaders import (
    load_attention_weights,
    load_layer_norms,
    read_text_config,
)
from scripts._real_moe_helpers import (
    add_real_moe_args,
    detect_real_moe_layout,
    nvfp4_matmul_warnings,
    quantized_store_for_layout,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    add_real_moe_args(parser, default_max_memory_mb=512)
    args = parser.parse_args()

    path = Path(args.path)
    if not path.is_dir():
        raise SystemExit(f"checkpoint not found: {path}")

    layout_name, layout = detect_real_moe_layout(path)
    text_cfg = read_text_config(path)
    print(f"layout: {layout_name}")

    # MoE plumbing: same as demo_real_runtime, but used as a callable
    # rather than the per-layer loop in apex.runtime.Runtime.
    metrics = Metrics()
    cache = ExpertCache(
        max_memory_mb=args.max_memory_mb, verbose=False, metrics=metrics,
    )
    quant_store = quantized_store_for_layout(path, layout)
    store = DequantizingExpertStore(quant_store, verbose=False)
    manager = ExpertManager(store=store, cache=cache, metrics=metrics)
    backend = FakeBackend()
    router = MultiLayerRealRouter(path, layout, top_k=args.top_k)

    # Pre-load attention + per-layer norms so the decoder closure has
    # everything in memory before the first compute.
    attn_weights = load_attention_weights(path, layout, args.layer)
    norms = load_layer_norms(path, layout, args.layer)

    print(
        f"running real decoder layer {args.layer}: "
        f"hidden_size={text_cfg['hidden_size']} "
        f"top_k={args.top_k} num_experts={text_cfg['num_experts']}"
    )

    rng = np.random.default_rng(args.seed)
    hidden = rng.standard_normal((1, text_cfg["hidden_size"])).astype(np.float32)

    # Quick standalone sanity prints — match the spec's intermediate
    # checkpoints. Doing them here rather than inside the decoder
    # function keeps the compose path clean.
    with nvfp4_matmul_warnings():
        attn_only = single_token_attention(
            hidden,
            q_weight=attn_weights["q_weight"], k_weight=attn_weights["k_weight"],
            v_weight=attn_weights["v_weight"], o_weight=attn_weights["o_weight"],
            num_heads=text_cfg["num_attention_heads"],
            num_kv_heads=text_cfg["num_key_value_heads"],
            head_dim=text_cfg["head_dim"],
            q_norm_weight=attn_weights["q_norm"],
            k_norm_weight=attn_weights["k_norm"],
            rms_eps=text_cfg["rms_norm_eps"],
        )
    print(f"attention output finite: {bool(np.all(np.isfinite(attn_only)))}")

    def moe_call(h):
        with nvfp4_matmul_warnings():
            ids, weights = router.route(args.layer, h)
            expert_loader = lambda eid: manager.ensure_loaded(args.layer, eid)
            return run_moe_layer(h, ids, weights, expert_loader, backend)

    # MoE-only sanity: same hidden through the routed-MoE block alone.
    moe_only = moe_call(hidden)
    print(f"moe output finite: {bool(np.all(np.isfinite(moe_only)))}")

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

    with nvfp4_matmul_warnings():
        out = single_token_decoder_layer(
            hidden,
            norms=norms,
            attn_kwargs=attn_kwargs,
            moe_call=moe_call,
            rms_eps=text_cfg["rms_norm_eps"],
        )

    assert out.shape == (1, text_cfg["hidden_size"]), out.shape
    assert np.all(np.isfinite(out)), "decoder layer output must be finite"
    print(f"layer output shape: {out.shape}")
    print(
        f"finite: True  min={float(out.min()):.4f}  max={float(out.max()):.4f}"
    )
    print()
    print("[ok] real decoder layer smoke passed")


if __name__ == "__main__":
    main()
