"""Step 11 smoke test: two real MoE layers in sequence on Gemma 4 NVFP4.

Step 9 proved one real layer routes + dequantizes + computes; step 10
proved the cache reuses experts across runs of the same layer. This
script proves the third leg: the same cache + manager handle multiple
layers in one pass without confusing them.

What we exercise:

  - Layer 0 router → top-8 experts → cache misses → SwiGLU → out_0.
  - out_0 is fed into layer 1's router → top-8 experts → cache misses
    → SwiGLU → out_1. (No attention / norm / residual between the two
    layers — this is a smoke test for *plumbing*, not numerical
    correctness vs HF; the real model would interleave attention and
    LayerNorm here.)
  - Re-run the same two-layer pass: every expert is now a cache hit.
  - Cache is sized to hold all 16 experts (8 per layer × 2 layers) so
    no eviction kicks in mid-pass and tricks the second-pass hit count.

Wins this locks in:
  - Cache keyed on (layer, expert) tuples actually disambiguates by
    layer (not just expert id) — otherwise layer 1 expert 0 would
    collide with layer 0 expert 0.
  - The cache + manager work fine being driven by run_moe_layer per
    layer; nothing in those primitives is layer-specific.

Run:
    uv run python scripts/smoke_real_moe_two_layer.py
        [--path /path/to/Gemma-4-26B-A4B-NVFP4]
        [--top-k 8] [--seed 0] [--max-memory-mb 512]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Make the apex package importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apex.backends.fake_backend import FakeBackend
from apex.dequant import DequantizingExpertStore
from apex.expert_cache import ExpertCache
from apex.expert_manager import ExpertManager
from apex.metrics import Metrics
from apex.moe_layer import route, run_moe_layer
from scripts._real_moe_helpers import (
    add_real_moe_args,
    detect_real_moe_layout,
    load_router,
    nvfp4_matmul_warnings,
    quantized_store_for_layout,
)


def main():
    args = _parse_args()
    path = Path(args.path)
    if not path.is_dir():
        raise SystemExit(
            f"checkpoint directory not found: {path}\n"
            f"download from https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4 "
            f"or pass --path"
        )

    layout_name, layout = detect_real_moe_layout(path)
    print(f"[layout] {layout_name}")

    # --- 1. Pre-load both layer routers (BF16 → fp32 inside the loader).
    # Done up front so we can establish hidden_dim before the loop and
    # so router-load timing isn't tangled with per-layer compute timing.
    routers = {}
    for layer_id in args.layers:
        routers[layer_id], num_experts, hidden_dim = load_router(path, layout, layer_id)
        if layer_id == args.layers[0]:
            first_num_experts, first_hidden_dim = num_experts, hidden_dim
    print(f"[load] {len(routers)} routers loaded "
          f"(num_experts={first_num_experts}, hidden_dim={first_hidden_dim})")

    # --- 2. One shared cache + manager for both layers. Budget sized
    # for top_k * num_layers fp32 experts (~24 MB each on Gemma 4 ⇒
    # ~384 MB for 16; default 512 MB has slack).
    quant_store = quantized_store_for_layout(path, layout)
    store = DequantizingExpertStore(quant_store, verbose=False)
    metrics = Metrics()
    cache = ExpertCache(max_memory_mb=args.max_memory_mb, verbose=False, metrics=metrics)
    manager = ExpertManager(store=store, cache=cache, metrics=metrics)
    backend = FakeBackend()

    # --- 3. Seeded fake hidden state (reproducible routing).
    rng = np.random.default_rng(args.seed)
    hidden_initial = rng.standard_normal((1, first_hidden_dim)).astype(np.float32)

    # --- 4. First pass: thread hidden through the layers; each layer's
    # router decides which experts to load on demand.
    print("\n=== first pass (cold cache) ===")
    selections_per_layer, out_first = _two_layer_pass(
        hidden_initial, args.layers, routers, manager, backend, args.top_k,
    )

    expected_misses = args.top_k * len(args.layers)
    assert metrics.cache_misses == expected_misses, (
        f"first pass: expected {expected_misses} misses, got {metrics.cache_misses}"
    )
    assert metrics.cache_hits == 0
    assert metrics.evictions == 0, (
        f"first pass evicted {metrics.evictions} experts — bump --max-memory-mb"
    )
    assert out_first.shape == (1, first_hidden_dim)
    assert np.all(np.isfinite(out_first))
    print(f"[ok] first pass: {metrics.cache_misses} misses, "
          f"{metrics.cache_hits} hits, {metrics.evictions} evictions")

    # --- 5. Second pass: same hidden ⇒ same routing decisions per layer
    # ⇒ every expert is a cache hit. Per-layer selections are the same
    # by construction (same hidden, same router, no cache state changes
    # routing) so we discard them here.
    print("\n=== second pass (warm cache) ===")
    metrics_before = (metrics.cache_misses, metrics.cache_hits)
    _, out_second = _two_layer_pass(
        hidden_initial, args.layers, routers, manager, backend, args.top_k,
    )

    new_misses = metrics.cache_misses - metrics_before[0]
    new_hits = metrics.cache_hits - metrics_before[1]
    assert new_misses == 0, f"second pass should re-load nothing, got {new_misses} misses"
    assert new_hits == expected_misses, (
        f"second pass should hit on every expert, got {new_hits} hits"
    )
    assert metrics.evictions == 0
    np.testing.assert_array_equal(out_first, out_second)
    print(f"[ok] second pass: 0 misses, {new_hits} hits, "
          f"{metrics.evictions} evictions")

    # --- 6. Sanity: layer 0 and layer 1 select genuinely different
    # expert sets (otherwise the test would degenerate to "single-layer
    # cache test, run twice"). On real Gemma 4 with random hidden the
    # two routings overlap by chance only — this asserts the test is
    # measuring what it claims.
    if len(args.layers) >= 2:
        first_two_unique = set(selections_per_layer[0]) | set(selections_per_layer[1])
        assert len(first_two_unique) > args.top_k, (
            "layer 0 and layer 1 routings appear identical — the cache "
            "size assertion above is degenerate. Investigate."
        )
        print(f"[ok] layer 0 + layer 1 picked {len(first_two_unique)} distinct experts")

    print("\n[ok] two-layer real MoE smoke test passed")


# --- helpers ---------------------------------------------------------------

def _two_layer_pass(hidden, layers, routers, manager, backend, top_k):
    """Run all layers in sequence; return (per-layer selections, final hidden)."""
    selections = []
    for layer_id in layers:
        # NVFP4 weights have a wide dynamic range; transient over/underflow
        # during partial sums is benign (final accum is finite). Same
        # filter pattern as scripts/smoke_real_moe_layer.py.
        with nvfp4_matmul_warnings():
            ids, weights = route(hidden, routers[layer_id], top_k=top_k)
        selected = ids[0].tolist()
        routing_weights = weights[0]
        selections.append(selected)

        expert_loader = lambda eid, lid=layer_id: manager.ensure_loaded(lid, eid)
        t0 = time.perf_counter()
        with nvfp4_matmul_warnings():
            out = run_moe_layer(hidden, selected, routing_weights, expert_loader, backend)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"  layer {layer_id}: selected={selected[:4]}... "
              f"finite={bool(np.all(np.isfinite(out)))} "
              f"std={float(out.std()):.3f} ({elapsed_ms:.0f} ms)")
        # Thread the output forward as the next layer's hidden — this
        # is what the real model does between MoE blocks (modulo the
        # attention + LayerNorm + residual we're skipping).
        hidden = out
    return selections, hidden


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    add_real_moe_args(p, include_layer=False, default_max_memory_mb=512)
    p.add_argument("--layers", type=int, nargs="+", default=[0, 1],
                   help="layer indices to drive in sequence (default: 0 1)")
    return p.parse_args()


if __name__ == "__main__":
    main()
