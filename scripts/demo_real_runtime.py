"""Step 12 demo: real Gemma 4 MoE layers driven through Runtime."""

import argparse
from pathlib import Path
import sys

import numpy as np

# Make the apex package importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apex.backends.fake_backend import FakeBackend
from apex.dequant import DequantizingExpertStore
from apex.expert_cache import ExpertCache
from apex.expert_manager import ExpertManager
from apex.metrics import Metrics
from apex.real_router import MultiLayerRealRouter
from apex.runtime import Runtime
from apex.weight_layout import available_layouts, get_layout
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
    parser.add_argument(
        "--layout",
        choices=available_layouts(),
        default=None,
        help="checkpoint layout; omitted means auto-detect Gemma layouts",
    )
    args = parser.parse_args()

    path = Path(args.path)
    if not path.is_dir():
        raise SystemExit(f"checkpoint directory not found: {path}")

    layout_name, layout = (args.layout, get_layout(args.layout)) if args.layout else (
        detect_real_moe_layout(path)
    )
    print(f"layout: {layout_name}")
    router = MultiLayerRealRouter(path, layout, top_k=args.top_k)
    quant_store = quantized_store_for_layout(path, layout)
    store = DequantizingExpertStore(quant_store, verbose=False)
    metrics = Metrics()
    cache = ExpertCache(max_memory_mb=args.max_memory_mb, verbose=False, metrics=metrics)
    manager = ExpertManager(store=store, cache=cache, metrics=metrics)
    backend = FakeBackend()

    rng = np.random.default_rng(args.seed)
    hidden_dim = router.shape_for_layer(0)[1]
    initial = rng.standard_normal((1, hidden_dim)).astype(np.float32)
    runtime = Runtime(
        router=router,
        expert_manager=manager,
        backend=backend,
        num_layers=args.num_layers,
        initial_hidden=initial,
    )
    expected = args.top_k * args.num_layers

    print("=== run 1 (cold cache) ===")
    with nvfp4_matmul_warnings():
        out_first = runtime.run()
    assert metrics.cache_misses == expected
    assert metrics.cache_hits == 0
    assert metrics.evictions == 0
    assert out_first.shape == (1, hidden_dim)
    assert np.all(np.isfinite(out_first))
    print(f"shape={out_first.shape} finite=True misses={metrics.cache_misses}")

    print("\n=== run 2 (warm cache) ===")
    misses_before = metrics.cache_misses
    hits_before = metrics.cache_hits
    with nvfp4_matmul_warnings():
        out_second = runtime.run()
    new_misses = metrics.cache_misses - misses_before
    new_hits = metrics.cache_hits - hits_before
    assert new_misses == 0, f"second run re-loaded {new_misses} experts"
    assert new_hits == expected
    assert metrics.evictions == 0
    np.testing.assert_array_equal(out_first, out_second)
    print(f"shape={out_second.shape} finite=True hits={new_hits}")

    print("\n[ok] Runtime drives real Gemma layers end-to-end")


if __name__ == "__main__":
    main()
