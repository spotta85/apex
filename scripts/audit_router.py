"""Step 24A audit: print router decisions for one layer + a fixed hidden.

Used to eyeball that `router.scale` and `router.per_expert_scale` are
having a sensible effect — the unscaled top-k vs the scaled top-k
should differ on real Gemma 4, and the scaled `per_expert_scale`
distribution should look at least vaguely peaky (not all 1.0).

Run:
    uv run python scripts/audit_router.py \
        --path /Users/spotta/Desktop/models/gemma4/gemma-4-26b-a4b-it-nvfp4 \
        --layer 0
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Make the apex package importable when run directly from a clone.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apex.real_router import MultiLayerRealRouter
from apex.weight_layout import get_layout
from scripts._real_moe_helpers import detect_real_moe_layout


_TOP_K = 8


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--path", required=True,
        help="checkpoint directory",
    )
    parser.add_argument(
        "--layer", type=int, default=0,
        help="which layer's router to audit (default 0)",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="seed for the random hidden vector (default 0)",
    )
    args = parser.parse_args()

    path = Path(args.path)
    layout_name, layout = detect_real_moe_layout(path)
    print(f"checkpoint: {path}")
    print(f"layout:     {layout_name}")
    print(f"layer:      {args.layer}")

    router = MultiLayerRealRouter(path, layout, top_k=_TOP_K)
    router._load_for_layer(args.layer)  # populate caches

    has_hidden_scale = router._hidden_scales.get(args.layer) is not None
    has_pe_scale = router._per_expert_scales.get(args.layer) is not None
    print(f"router.scale loaded:           {has_hidden_scale}")
    print(f"router.per_expert_scale loaded:{has_pe_scale}")

    # Print quick stats on the per-expert scale distribution so the
    # operator can see whether it's a peaky biaser or near-uniform.
    if has_pe_scale:
        pes = router._per_expert_scales[args.layer]
        print(f"per_expert_scale stats: min={pes.min():.3f}, "
              f"max={pes.max():.3f}, mean={pes.mean():.3f}, std={pes.std():.3f}")

    # Same hidden through scaled vs unscaled router. Random fp32 with a
    # fixed seed for reproducibility — not from a real prompt; the
    # purpose is to see whether the scales change routing decisions.
    rng = np.random.default_rng(args.seed)
    num_experts, hidden_dim = router.shape_for_layer(args.layer)
    hidden = rng.standard_normal((1, hidden_dim)).astype(np.float32)

    ids_scaled, w_scaled = router.route(args.layer, hidden)
    print()
    print(f"=== top-{_TOP_K} with scaling (production path) ===")
    for rank, (eid, w) in enumerate(zip(ids_scaled, w_scaled), 1):
        print(f"  {rank:2d}. expert {eid:>3d}  weight {w:.4f}")

    # Patch out the scales for a side-by-side comparison.
    saved_h = router._hidden_scales[args.layer]
    saved_pe = router._per_expert_scales[args.layer]
    router._hidden_scales[args.layer] = None
    router._per_expert_scales[args.layer] = None
    ids_unscaled, w_unscaled = router.route(args.layer, hidden)
    router._hidden_scales[args.layer] = saved_h
    router._per_expert_scales[args.layer] = saved_pe

    print()
    print(f"=== top-{_TOP_K} WITHOUT scaling (pre-step-24 path) ===")
    for rank, (eid, w) in enumerate(zip(ids_unscaled, w_unscaled), 1):
        print(f"  {rank:2d}. expert {eid:>3d}  weight {w:.4f}")

    print()
    overlap = set(ids_scaled) & set(ids_unscaled)
    print(f"top-{_TOP_K} overlap (set intersection): {len(overlap)}/{_TOP_K}")
    if ids_scaled == ids_unscaled:
        print("  ! identical — scaling is having no observable effect")
    else:
        print("  ok: scaling changes the routing decision")

    print()
    print("[ok] router audit complete")


if __name__ == "__main__":
    main()
