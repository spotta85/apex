"""Step 10 demo: real MoE cache integration on Gemma 4 NVFP4.

Proves APEX's core product value (bounded-memory expert reuse) on
real weights, not fake ones:

  - Load BF16 router for layer 0; route a fake hidden state through
    the *real* gate to pick the actual top-8 experts the model would
    use for that token.
  - First run: every expert is a cache miss → load NVFP4 bundle from
    the sharded checkpoint → dequantize → put in cache → run SwiGLU.
  - Second run with the SAME hidden state: every expert is a cache
    hit → SwiGLU runs straight off the cached fp32 bundle. Dequant
    fires zero times. Wall-clock drops accordingly.

The piece that makes this work without runtime / cache changes is
``apex.dequant.DequantizingExpertStore``: it wraps the quantized
store so the bundle the cache *holds* is already fp32, not packed
NVFP4. That way ``ExpertManager.ensure_loaded`` is the only seam
``run_moe_layer``'s loader hook touches — exactly as the user spec'd:

    expert_loader = lambda eid: manager.ensure_loaded(layer_id, eid)

Run:
    uv run python scripts/demo_real_moe_cache.py
        [--path /path/to/Gemma-4-26B-A4B-NVFP4]
        [--layer 0] [--top-k 8] [--seed 0] [--max-memory-mb 256]
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
    print(f"layout: {layout_name}")

    # --- 1. Resolve router shard + load BF16 router as fp32. The
    # router is in the model's `ignore` list so it's NOT NVFP4-quantized;
    # safetensors_loader auto-converts BF16 to fp32 on read.
    router, _, hidden_dim = load_router(path, layout, args.layer)
    print(f"real layer {args.layer}")

    # --- 2. Seeded fake hidden state so routing is reproducible across
    # runs (the cached experts on second run match the first run's
    # selection — otherwise we'd get all-misses both times).
    rng = np.random.default_rng(args.seed)
    hidden = rng.standard_normal((1, hidden_dim)).astype(np.float32)
    with nvfp4_matmul_warnings():
        expert_ids, weights = route(hidden, router, top_k=args.top_k)
    selected = expert_ids[0].tolist()
    routing_weights = weights[0]
    print(f"router selected experts: {selected}")
    print()

    # --- 3. Wire the cache stack: NVFP4 store → dequantizing wrapper →
    # cache + manager. Cache budget is sized for ~8 dequantized experts
    # (each ~24 MB at fp32 for Gemma 4's intermediate=704, hidden=2816).
    quant_store = quantized_store_for_layout(path, layout)
    store = DequantizingExpertStore(quant_store, verbose=True)
    metrics = Metrics()
    # verbose=False on the cache: the per-put "current cache memory" lines
    # are noise here; we want the cache hit/miss + dequant trace clean.
    cache = ExpertCache(max_memory_mb=args.max_memory_mb, verbose=False, metrics=metrics)
    backend = FakeBackend()
    manager = ExpertManager(store=store, cache=cache, metrics=metrics)

    # --- 4. Two runs with the same hidden state. First run = misses;
    # second run = hits. The expert_loader is the loader hook spec'd in
    # the step 10 brief: pure delegation to ensure_loaded so the cache
    # is the choke point.
    def expert_loader(eid):
        return manager.ensure_loaded(args.layer, eid)

    print("first run:")
    out_first, t_first = _time_layer(
        hidden, selected, routing_weights, expert_loader, backend
    )
    _print_run_summary(out_first, t_first)
    print()

    print("second run:")
    out_second, t_second = _time_layer(
        hidden, selected, routing_weights, expert_loader, backend
    )
    _print_run_summary(out_second, t_second)
    print()

    # --- 5. Metrics + speedup. The user's bar: at least 8 cache hits
    # and finite output in both runs. The speedup line shows the
    # qualitative win — usually 5-10x once disk + dequant are skipped.
    print("metrics:")
    print(f"cache hits: {metrics.cache_hits}")
    print(f"cache misses: {metrics.cache_misses}")
    print(f"evictions: {metrics.evictions}")
    print(f"speedup (first / second): {t_first / max(t_second, 1e-9):.1f}x "
          f"({t_first * 1000:.0f} ms → {t_second * 1000:.0f} ms)")

    # --- 6. Assertions: the demo asserts its own contract so a
    # regression (e.g. cache eviction silently kicking in because the
    # budget was set too tight) fails the script with a non-zero exit.
    assert metrics.cache_misses == args.top_k, (
        f"first run should have {args.top_k} misses, got {metrics.cache_misses}"
    )
    assert metrics.cache_hits == args.top_k, (
        f"second run should have {args.top_k} hits, got {metrics.cache_hits}"
    )
    assert metrics.evictions == 0, (
        f"no evictions expected (cache budget {args.max_memory_mb} MB), "
        f"got {metrics.evictions}"
    )
    assert np.all(np.isfinite(out_first)) and np.all(np.isfinite(out_second)), (
        "both run outputs must be finite"
    )
    # Same input + same cached weights ⇒ second run should be bit-exact
    # to first run. Catches any accidental non-determinism in the cache
    # path (e.g. mutating a cached array in place).
    np.testing.assert_array_equal(out_first, out_second)
    print()
    print("[ok] real MoE cache integration demo passed")


# --- helpers ---------------------------------------------------------------

def _time_layer(hidden, selected, routing_weights, expert_loader, backend):
    """Run one MoE layer, return (output, wall_seconds)."""
    t0 = time.perf_counter()
    # Same warning filter as smoke_real_moe_layer.py: real NVFP4 weights
    # span a wide dynamic range and trip transient overflow warnings
    # during partial sums even though the final output is finite.
    with nvfp4_matmul_warnings():
        out = run_moe_layer(hidden, selected, routing_weights, expert_loader, backend)
    return out, time.perf_counter() - t0


def _print_run_summary(out, wall_seconds: float) -> None:
    print(f"output shape: {out.shape}")
    print(f"finite: {bool(np.all(np.isfinite(out)))}")
    print(f"wall: {wall_seconds * 1000:.0f} ms")


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    add_real_moe_args(p)
    return p.parse_args()


if __name__ == "__main__":
    main()
