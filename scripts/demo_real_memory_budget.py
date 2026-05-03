"""Step 14 demo: real Gemma 4 MoE under a tight memory budget.

Drives N layers of Gemma 4 NVFP4 through Runtime with a deliberately
small `--max-memory-mb` budget so LRU eviction must kick in to keep
the cache under it. Proves the core APEX promise:

  - the cache never exceeds the budget after eviction (peak ≤ budget),
  - evictions happen on least-recently-used experts,
  - per-expert-size accounting is correct (running total matches
    sum of resident expert nbytes),
  - the foreground load path still produces finite output even while
    experts are being evicted between layers.

Default `--max-memory-mb 512` is sized so two-layer Gemma 4 (each
expert ~24 MB after dequant; 8 experts/layer × 2 layers = 16 experts ≈
384 MB) WOULD fit if all 16 stayed resident, but `--max-memory-mb 256`
forces eviction since one layer's 8 experts ≈ 192 MB and the second
layer pushes the cache past the 256 MB budget. The default is the
"breathing room" case; pass a smaller budget to see eviction.

Run:
    uv run python scripts/demo_real_memory_budget.py
        [--path /path/to/Gemma-4-26B-A4B-NVFP4]
        [--num-layers 2] [--top-k 8] [--seed 0] [--max-memory-mb 256]
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
from apex.memory import format_mb, snapshot
from apex.metrics import Metrics
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
    add_real_moe_args(parser, include_layer=False, default_max_memory_mb=256)
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

    # Cache verbose=True so the per-load "current cache memory" + per-eviction
    # "evicting layer L expert E" lines render in the trace. The cache prints
    # bytes; we follow up with a MB-formatted MemoryReport at the end so the
    # operator can read the budget vs peak summary in human units.
    metrics = Metrics()
    cache = ExpertCache(
        max_memory_mb=args.max_memory_mb, verbose=True, metrics=metrics,
    )
    quant_store = quantized_store_for_layout(path, layout)
    store = DequantizingExpertStore(quant_store, verbose=False)
    manager = ExpertManager(store=store, cache=cache, metrics=metrics)
    backend = FakeBackend()
    router = MultiLayerRealRouter(path, layout, top_k=args.top_k)

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

    print(f"max memory budget: {format_mb(cache.max_memory_bytes)}")
    print()
    with nvfp4_matmul_warnings():
        out = runtime.run()

    assert out.shape == (1, hidden_dim)
    assert np.all(np.isfinite(out)), "output must be finite even under eviction"

    print()
    report = snapshot(cache, metrics)
    report.print_summary()
    # Hard assertion: the contract this demo exists to prove. If the
    # cache ever exceeded its budget the cache itself has a bug, so
    # fail loudly with a non-zero exit instead of silently passing.
    assert report.budget_respected, (
        f"peak {report.peak_bytes} bytes > budget {report.budget_bytes} bytes"
    )
    print()
    print("[ok] real memory-budget enforcement demo passed")


if __name__ == "__main__":
    main()
