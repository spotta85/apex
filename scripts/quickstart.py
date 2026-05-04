"""Step 25E: minimal "load real Gemma → generate" demo.

The shortest possible APEX usage. Two API calls:

    model = MoEModel.from_pretrained(path, max_memory_gb=8)
    text  = model.generate("Hello", max_tokens=16)

Everything else is print-out for the operator to see what happened.
Use this script as the canonical entry point when introducing APEX
to someone new — the existing `scripts/demo_*.py` files build the
runtime by hand at the Runtime/ExpertCache/router level for
correctness probes, which is useful for development but obscures
how trivial the user-facing surface is.

Run:
    uv run python scripts/quickstart.py \\
        --path /path/to/Gemma-4-26B-A4B-IT-NVFP4 \\
        [--num-layers 30] [--max-memory-gb 8] [--max-tokens 16] \\
        [--prompt "Hello world"]

Default `--num-layers` is whatever the checkpoint has (30 on Gemma 4).
Quality at L=30 is still gibberish today (see step 24F notes in
docs/OVERVIEW.md "Near term"); the script proves the runtime ships
end-to-end, not that the math is fully right yet.
"""

import argparse
import sys
import time
from pathlib import Path

# Make the apex package importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apex import MoEModel


_MIB = 1024 * 1024


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--path", required=True,
        help="checkpoint directory (real-Gemma NVFP4)",
    )
    parser.add_argument(
        "--num-layers", type=int, default=None,
        help="how many decoder layers (default: all the checkpoint has)",
    )
    parser.add_argument(
        "--max-memory-gb", type=float, default=8.0,
        help="LRU expert cache budget in GB (default 8)",
    )
    parser.add_argument(
        "--prompt", default="Hello world",
        help="text prompt (default: 'Hello world')",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=16,
        help="how many tokens to generate (default 16)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="sampling temperature (default 0 = greedy / deterministic)",
    )
    args = parser.parse_args()

    print(f"loading: {args.path}")
    t0 = time.time()
    model = MoEModel.from_pretrained(
        args.path,
        max_memory_gb=args.max_memory_gb,
        num_layers=args.num_layers,
        verbose=False,
    )
    print(f"  loaded in {time.time() - t0:.1f}s "
          f"({model.num_layers} layers, "
          f"budget {args.max_memory_gb:.1f} GB)")

    print(f"prompt: {args.prompt!r}")
    t1 = time.time()
    text = model.generate(
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    gen_s = time.time() - t1
    n_generated = len(model._last_generated_ids)
    print(f"generated ({n_generated} tokens, {n_generated / gen_s:.2f} tok/s):")
    print(f"  {text!r}")

    # One-line memory summary so operators can see budget / KV at a glance.
    report = model.memory_report()
    print(f"memory: peak expert {report.expert.peak_bytes / _MIB:.1f} MB, "
          f"kv {report.kv_cache_bytes / _MIB:.2f} MB, "
          f"evictions {report.expert.evictions}, "
          f"budget_ok {report.expert.budget_respected}")


if __name__ == "__main__":
    main()
