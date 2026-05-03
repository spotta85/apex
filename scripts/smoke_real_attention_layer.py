"""Step 16C smoke: single-token attention on real Gemma layer 0.

Loads the four NVFP4_U32 attention projections + BF16 q_norm/k_norm
for layer 0 of Gemma 4 26B-A4B-IT NVFP4, runs one forward token
through `apex.attention.single_token_attention`, asserts the output is
shape (1, hidden_size) and finite.

What this proves:
  - The 16A layout names resolve against real shards.
  - The 16B ops compose correctly through the attention math.
  - NVFP4_U32 dequant + BF16 norms feed into the same compute graph
    that 16D will reuse with KV cache + RoPE.

What this does NOT prove (intentionally):
  - Numerical agreement vs HF transformers Gemma. Single-token softmax
    is trivially 1.0, so the output equals `linear(v_reshape, o_w)` —
    correctness here means "wiring works", not "matches reference".
  - Multi-token / RoPE / KV-cache behavior — those are 16D.

Run:
    uv run python scripts/smoke_real_attention_layer.py
        [--path /path/to/checkpoint]
        [--layer 0] [--seed 0]

Layer 0 is sliding_attention with head_dim=256. Pointing `--layer` at
a full_attention layer (5/11/17/23/29) will fail because those layers
tie K=V (no v_proj on disk) and use head_dim=512; supporting them is
a 16D concern.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Make the apex package importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apex.attention import single_token_attention
from scripts._real_moe_helpers import (
    add_real_moe_args,
    detect_real_moe_layout,
    load_attention_weights,
    nvfp4_matmul_warnings,
    read_text_config,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    add_real_moe_args(parser, include_max_memory=False)
    args = parser.parse_args()

    path = Path(args.path)
    if not path.is_dir():
        raise SystemExit(f"checkpoint not found: {path}")

    layout_name, layout = detect_real_moe_layout(path)
    text_cfg = read_text_config(path)
    print(f"layout: {layout_name}")
    print(
        f"hidden_size={text_cfg['hidden_size']} "
        f"num_attention_heads={text_cfg['num_attention_heads']} "
        f"num_kv_heads={text_cfg['num_key_value_heads']} "
        f"head_dim={text_cfg['head_dim']}"
    )

    weights = load_attention_weights(path, layout, args.layer)
    print(
        f"loaded layer {args.layer} attention weights "
        f"(q={weights['q_weight'].shape} k={weights['k_weight'].shape} "
        f"v={weights['v_weight'].shape} o={weights['o_weight'].shape})"
    )

    rng = np.random.default_rng(args.seed)
    hidden = rng.standard_normal((1, text_cfg["hidden_size"])).astype(np.float32)

    with nvfp4_matmul_warnings():
        out = single_token_attention(
            hidden,
            q_weight=weights["q_weight"],
            k_weight=weights["k_weight"],
            v_weight=weights["v_weight"],
            o_weight=weights["o_weight"],
            num_heads=text_cfg["num_attention_heads"],
            num_kv_heads=text_cfg["num_key_value_heads"],
            head_dim=text_cfg["head_dim"],
            q_norm_weight=weights["q_norm"],
            k_norm_weight=weights["k_norm"],
            rms_eps=text_cfg["rms_norm_eps"],
        )

    # Hard contract: the wiring must produce a finite, correctly-shaped
    # vector. If either fails the smoke is meaningless and we want a
    # non-zero exit, not a printed warning.
    assert out.shape == (1, text_cfg["hidden_size"]), out.shape
    assert np.all(np.isfinite(out)), "attention output must be finite"
    print(f"output shape: {out.shape}")
    print(f"finite: True  min={out.min():.4f}  max={out.max():.4f}")
    print()
    print("[ok] real attention smoke passed")


if __name__ == "__main__":
    main()
