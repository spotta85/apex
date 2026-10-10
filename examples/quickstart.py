"""Step 20 quickstart: end-to-end text generation in three function calls.

Mirrors the README pitch:

    model = MoEModel.from_pretrained(path, max_memory_gb=8)
    text  = model.generate("Explain MoE simply.", max_tokens=32)
    print(text)

…with a memory report at the end so you can see where APEX's RAM
went. No subgrouping by step number, no internal helpers — this is
the user-facing API only.

Run:
    uv run python examples/quickstart.py \
        --path /Users/spotta/Desktop/models/gemma4/gemma-4-26b-a4b-it-nvfp4 \
        --prompt "Explain MoE simply." --max-tokens 32

Caveat: `--num-layers` defaults to every decoder layer the checkpoint
has (30 on Gemma 4: sliding + full attention), so the default run loads
the whole model. Pass a smaller `--num-layers` for a faster,
lower-memory trial — but a truncated stack's output isn't meaningful
text; that only proves wiring, not model quality.
"""

import argparse
import sys
from pathlib import Path

# Make `apex` importable when this script is run directly from a clone.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apex import MoEModel


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--path",
        required=True,
        help="checkpoint directory (config.json + tokenizer.json + shards)",
    )
    parser.add_argument(
        "--prompt",
        default="Explain MoE simply.",
        help='text prompt (default: "Explain MoE simply.")',
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="how many tokens to generate (default: 32)",
    )
    parser.add_argument(
        "--max-memory-gb",
        type=float,
        default=8.0,
        help="expert cache budget in GB (default: 8)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="how many decoder layers to load (default: all sliding-attention)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="sampling temperature (0 = greedy; default 0)",
    )
    args = parser.parse_args()

    model = MoEModel.from_pretrained(
        args.path,
        max_memory_gb=args.max_memory_gb,
        num_layers=args.num_layers,
    )

    print()
    print(f"prompt: {args.prompt!r}")
    text = model.generate(
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        verbose=True,
    )
    print()
    print(f"generated: {text!r}")
    print(f"full text: {(args.prompt + text)!r}")

    print()
    print("=== memory ===")
    model.memory_report().print_summary()


if __name__ == "__main__":
    main()
