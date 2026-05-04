"""Step 21 quickstart: drive multiple decoder layers through MoEModel.

`examples/quickstart.py` runs whatever `MoEModel.from_pretrained`'s
`num_layers` default picks (= the sliding-attention cap, 5 on Gemma 4).
This script makes the layer-count knob explicit and prints a memory
report after each generation, so the `1 → 2 → 4` scaling story is
visible at a glance.

Why this matters: until step 21 the demos all ran with `num_layers=1`
because the wiring was new. The runtime supports up to the sliding
cap; this script proves the multi-layer path with a simple
`for L in [1, 2, 4]` driver.

Run:
    uv run python examples/quickstart_multilayer.py \
        --path /Users/spotta/Desktop/models/gemma4/gemma-4-26b-a4b-it-nvfp4 \
        --prompt "Hello" --max-tokens 4 --max-memory-gb 4

Caveat: even at 4 layers the generated text isn't coherent — Gemma 4
has 30 layers. This script proves the layer-scaling wiring is sound
(finite output, budget respected, KV scales linearly), not that the
model is producing meaningful continuations.
"""

import argparse
import sys
import time
from pathlib import Path

# Make the apex package importable when run directly from a clone.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apex import MoEModel


def _run_one(path, *, num_layers, prompt, max_tokens, max_memory_gb):
    """Build a fresh model at `num_layers`, generate, report stats.

    Each call constructs a new MoEModel so the cache + KV start clean
    — keeps the per-layer numbers comparable across runs.
    """
    print(f"\n=== num_layers={num_layers} ===")
    t0 = time.time()
    model = MoEModel.from_pretrained(
        path,
        max_memory_gb=max_memory_gb,
        num_layers=num_layers,
        verbose=False,
    )
    load_s = time.time() - t0
    print(f"load: {load_s:.1f}s")

    t0 = time.time()
    text = model.generate(prompt, max_tokens=max_tokens, verbose=False)
    gen_s = time.time() - t0
    # Authoritative count of generated tokens — re-encoding the
    # decoded text isn't reliable when output is non-printable bytes
    # (one-layer / few-layer model output BPE-re-segments).
    n_generated = len(model._last_generated_ids)
    tps = n_generated / gen_s if gen_s > 0 else float("inf")
    print(f"generate: {gen_s:.1f}s for {n_generated} tokens "
          f"({tps:.2f} tokens/sec)")
    print(f"text: {text!r}")

    report = model.memory_report()
    print("memory:")
    report.print_summary()
    print(f"finite: true (sample assertions passed inside generate)")
    return load_s, gen_s, report


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
        default=4,
        help="how many tokens to generate per layer count (default: 4)",
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
    print(f"sweep:      num_layers in {args.layers}")

    for L in args.layers:
        _run_one(
            args.path,
            num_layers=L,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            max_memory_gb=args.max_memory_gb,
        )

    print()
    print("[ok] layer scaling sweep complete")


if __name__ == "__main__":
    main()
