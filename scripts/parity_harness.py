"""Step 25A: APEX-side parity harness.

Loads real Gemma 4 at a chosen depth, runs `MoEModel.forward_with_trace`
over a fixed prompt list, and dumps the captured per-layer trace plus
the top-K final logits to a JSON file.

The captured signature per prompt:
  - tokenized input ids
  - per layer: router top-k expert ids + weights, attn-out norm,
    MoE-out norm, dense-MLP-out norm (when applicable),
    post-layer-hidden norm
  - final top-K logits as [(token_id, logit_value), ...]

This is the **APEX half** of the eventual cross-runtime parity test.
The other half — loading the same NVFP4 checkpoint into a reference
runtime (HF transformers via the `compressed-tensors` plugin, or
mlx-lm) — is a follow-up. With both halves dumping the same schema,
diffing the two JSONs at the first layer where the norms diverge
localizes a math drift to a specific operator instead of leaving it
buried in a 30-layer end-to-end output.

Run:
    uv run python scripts/parity_harness.py \\
        --path /Users/spotta/Desktop/models/gemma4/gemma-4-26b-a4b-it-nvfp4 \\
        --num-layers 8 --max-memory-gb 8 \\
        --prompts "The capital of France is" "2 + 2 =" \\
        --out traces/parity_apex_l8.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Make the apex package importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apex import MoEModel


_DEFAULT_PROMPTS = (
    "The capital of France is",
    "2 + 2 =",
    "Once upon a time",
    "def fibonacci(n):",
)


def _capture(model, prompt: str, top_k: int) -> dict:
    """Run forward_with_trace and shape it for the JSON dump."""
    t0 = time.time()
    _, trace = model.forward_with_trace(prompt, top_k_logits=top_k)
    trace["forward_seconds"] = time.time() - t0
    return trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--path", required=True,
        help="checkpoint directory (real-Gemma NVFP4)",
    )
    parser.add_argument(
        "--num-layers", type=int, default=8,
        help="how many decoder layers to drive (default 8)",
    )
    parser.add_argument(
        "--max-memory-gb", type=float, default=8.0,
        help="LRU expert cache budget in GB (default 8)",
    )
    parser.add_argument(
        "--prompts", nargs="+", default=list(_DEFAULT_PROMPTS),
        help="prompts to capture (default: 4 canonical)",
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="how many top-K final logits to capture per prompt",
    )
    parser.add_argument(
        "--out", required=True,
        help="output JSON path (parent dir created if missing)",
    )
    args = parser.parse_args()

    print(f"loading model: layers={args.num_layers}, "
          f"budget={args.max_memory_gb} GB")
    model = MoEModel.from_pretrained(
        args.path,
        max_memory_gb=args.max_memory_gb,
        num_layers=args.num_layers,
        verbose=False,
    )

    captures = {}
    for prompt in args.prompts:
        print(f"  capturing: {prompt!r}")
        captures[prompt] = _capture(model, prompt, args.top_k)

    out = {
        "checkpoint_path": str(args.path),
        "num_layers": args.num_layers,
        "max_memory_gb": args.max_memory_gb,
        "top_k_logits": args.top_k,
        "prompts": captures,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
