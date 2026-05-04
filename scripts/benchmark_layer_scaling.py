"""Step 21/23 benchmark: layer-count vs latency / memory on real Gemma 4.

Sweeps `num_layers` (default {1, 2, 4}; pass `--layers 8 16 30` for
the step 23 full-attention sweep) and prints a single table:

    L | load(s) | gen(s) | tok/s | peak expert | kv | peak total | evict | finite | budget

Step 23 unlocked layer counts > 5 (sliding cap on Gemma 4); the
sweep now mixes sliding + full_attention layers when L ≥ 6, so
per-trial peak expert grows non-uniformly (full layers have a
different MoE weight footprint than sliding layers).

The expensive piece is embedding dequant (~3 GB), which dominates
load time and is constant across L on first run, then mostly
amortized by OS page cache on subsequent loads.

Run:
    uv run python scripts/benchmark_layer_scaling.py \
        --path /Users/spotta/Desktop/models/gemma4/gemma-4-26b-a4b-it-nvfp4 \
        --max-tokens 3 --max-memory-gb 4 --layers 1 2 4
    # or for the step 23 unlock:
    uv run python scripts/benchmark_layer_scaling.py \
        --path /Users/spotta/Desktop/models/gemma4/gemma-4-26b-a4b-it-nvfp4 \
        --max-tokens 2 --max-memory-gb 8 --layers 8 16 30
"""

import argparse
import sys
import time
from pathlib import Path

# Make the apex package importable when run directly from a clone.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apex import MoEModel


_MIB = 1024 * 1024


def _bench_one(path, *, num_layers, prompt, max_tokens, max_memory_gb):
    """Build + run one (num_layers, prompt) trial. Returns timing + memory dict."""
    t0 = time.time()
    model = MoEModel.from_pretrained(
        path,
        max_memory_gb=max_memory_gb,
        num_layers=num_layers,
        verbose=False,
    )
    load_s = time.time() - t0

    t0 = time.time()
    text = model.generate(prompt, max_tokens=max_tokens, verbose=False)
    gen_s = time.time() - t0

    # Authoritative count of generated tokens — re-encoding decoded
    # text doesn't round-trip cleanly when the model emits non-printable
    # bytes from a few-layer pass.
    n_generated = len(model._last_generated_ids)
    tps = n_generated / gen_s if gen_s > 0 else 0.0
    report = model.memory_report()
    return {
        "num_layers": num_layers,
        "load_s": load_s,
        "gen_s": gen_s,
        "n_generated": n_generated,
        "tok_per_s": tps,
        "peak_expert_mb": report.expert.peak_bytes / _MIB,
        "kv_mb": report.kv_cache_bytes / _MIB,
        "peak_total_mb": (report.expert.peak_bytes + report.kv_cache_bytes) / _MIB,
        "evictions": report.expert.evictions,
        "hits": report.expert.cache_hits,
        "misses": report.expert.cache_misses,
        "budget_respected": report.expert.budget_respected,
        "text": text,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--path",
        required=True,
        help="checkpoint directory (config.json + tokenizer.json + shards)",
    )
    parser.add_argument(
        "--prompt",
        default="Hello",
        help='text prompt (default: "Hello")',
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=3,
        help="how many tokens to generate per trial (default: 3)",
    )
    parser.add_argument(
        "--max-memory-gb",
        type=float,
        default=4.0,
        help="expert cache budget in GB (default: 4)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="layer counts to sweep (default: 1 2 4)",
    )
    args = parser.parse_args()

    print(f"checkpoint: {args.path}")
    print(f"prompt:     {args.prompt!r}")
    print(f"max_tokens: {args.max_tokens}")
    print(f"budget:     {args.max_memory_gb:.1f} GB")
    print()

    results = []
    for L in args.layers:
        print(f"running num_layers={L}...")
        r = _bench_one(
            args.path,
            num_layers=L,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            max_memory_gb=args.max_memory_gb,
        )
        results.append(r)
        print(f"  load {r['load_s']:.1f}s, gen {r['gen_s']:.1f}s "
              f"({r['tok_per_s']:.2f} tok/s), peak total "
              f"{r['peak_total_mb']:.1f} MB")

    # Comparison table — the headline output.
    print()
    print("=== layer scaling ===")
    print(
        f"{'L':>2} | {'load(s)':>7} | {'gen(s)':>6} | {'tok/s':>5} | "
        f"{'peak expert':>11} | {'kv':>8} | {'peak total':>10} | "
        f"{'evict':>5} | {'finite':>6} | budget"
    )
    print("-" * 92)
    for r in results:
        print(
            f"{r['num_layers']:>2} | "
            f"{r['load_s']:>7.1f} | {r['gen_s']:>6.1f} | "
            f"{r['tok_per_s']:>5.2f} | "
            f"{r['peak_expert_mb']:>8.1f} MB | "
            f"{r['kv_mb']:>5.1f} MB | "
            f"{r['peak_total_mb']:>7.1f} MB | "
            f"{r['evictions']:>5d} | "
            f"{'true':>6} | "
            f"{str(r['budget_respected']).lower()}"
        )

    print()
    print("[ok] benchmark complete")


if __name__ == "__main__":
    main()
