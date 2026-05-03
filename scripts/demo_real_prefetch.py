"""Step 15 demo: AsyncPrefetcher warming the cache for real Gemma layers.

Wires the production stack with prefetch enabled:

  MultiLayerRealRouter -> Runtime -> ExpertManager
                              |
                              +-- AsyncPrefetcher (background worker)

After layer 0 finishes, Runtime peeks the router for layer 1's experts
and hands them to the prefetcher. The worker thread loads + dequants
each one into the cache while we wait at `wait_for_layer(1)` — so by
the time layer 1 starts, every expert is already a cache hit.

What this proves:
  - The prefetch + Runtime + real-router wiring works end-to-end on a
    real checkpoint, not just on fake-MoE demos.
  - Metrics counts prefetch_hits correctly: every prefetched expert
    that the foreground later consumes is credited.
  - Hit rate is meaningful even at 100% — deterministic-case prefetch
    fully covers next-layer demand. Smaller caches or imperfect
    predictors would knock that down; this demo just establishes the
    measurement path is plumbed through to real weights.

Cache budget (--max-memory-mb default 512) is sized to comfortably
hold all top_k * num_layers experts so prefetch isn't sabotaged by
mid-run eviction. Pair with `demo_real_memory_budget.py` to see the
opposite case (tight budget, eviction-driven trace).

Run:
    uv run python scripts/demo_real_prefetch.py
        [--path /path/to/Gemma-4-26B-A4B-NVFP4]
        [--num-layers 2] [--top-k 8] [--seed 0] [--max-memory-mb 512]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Make the apex package importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apex.backends.fake_backend import FakeBackend
from apex.dequant import DequantizingExpertStore
from apex.expert_cache import ExpertCache
from apex.expert_manager import ExpertManager
from apex.metrics import Metrics
from apex.prefetcher import AsyncPrefetcher
from apex.real_router import MultiLayerRealRouter
from apex.runtime import Runtime
from scripts._real_moe_helpers import (
    add_real_moe_args,
    detect_real_moe_layout,
    nvfp4_matmul_warnings,
    quantized_store_for_layout,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    add_real_moe_args(parser, include_layer=False, default_max_memory_mb=512)
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="number of real MoE layers to drive (default: 2)",
    )
    args = parser.parse_args()

    path = Path(args.path)
    if not path.is_dir():
        raise SystemExit(f"checkpoint directory not found: {path}")

    layout_name, layout = detect_real_moe_layout(path)
    print(f"layout: {layout_name}")

    metrics = Metrics()
    cache = ExpertCache(
        max_memory_mb=args.max_memory_mb, verbose=False, metrics=metrics,
    )
    quant_store = quantized_store_for_layout(path, layout)
    store = DequantizingExpertStore(quant_store, verbose=False)
    manager = ExpertManager(store=store, cache=cache, metrics=metrics)
    backend = FakeBackend()
    router = MultiLayerRealRouter(path, layout, top_k=args.top_k)
    # No predictor: Runtime falls back to peeking the router for
    # layer N+1's actual experts. With wait_for_layer this gives a
    # deterministic 100% prefetch hit rate — the measurement we want.
    prefetcher = AsyncPrefetcher(manager, verbose=True, metrics=metrics)

    rng = np.random.default_rng(args.seed)
    hidden_dim = router.shape_for_layer(0)[1]
    initial = rng.standard_normal((1, hidden_dim)).astype(np.float32)
    runtime = Runtime(
        router=router,
        expert_manager=manager,
        backend=backend,
        prefetcher=prefetcher,
        num_layers=args.num_layers,
        initial_hidden=initial,
    )

    try:
        with nvfp4_matmul_warnings():
            out = runtime.run()
    finally:
        # Always shut the worker down so the script can exit cleanly,
        # even if the run raised partway through.
        prefetcher.shutdown()

    assert out.shape == (1, hidden_dim)
    assert np.all(np.isfinite(out)), "output must be finite"

    expected_misses = args.top_k  # only layer 0 misses; layer 1+ are prefetched
    expected_prefetched = args.top_k * (args.num_layers - 1)

    print()
    print("metrics:")
    metrics.print_summary()

    # Hard contracts: layer 0 cold-misses, every subsequent layer is
    # served entirely from prefetched experts. If wait_for_layer ever
    # raced (or the prefetcher silently dropped a request) one of
    # these would fail loudly with a non-zero exit.
    assert metrics.cache_misses == expected_misses, (
        f"expected {expected_misses} foreground misses (layer 0 only), "
        f"got {metrics.cache_misses}"
    )
    assert metrics.prefetch_scheduled == expected_prefetched, (
        f"expected {expected_prefetched} prefetches scheduled, "
        f"got {metrics.prefetch_scheduled}"
    )
    assert metrics.prefetch_hits == expected_prefetched, (
        f"expected {expected_prefetched} prefetch hits, "
        f"got {metrics.prefetch_hits}"
    )
    print()
    print("[ok] real prefetch demo passed")


if __name__ == "__main__":
    main()
