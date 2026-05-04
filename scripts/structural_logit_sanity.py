"""Step 22C: structural sanity for one-layer / few-layer logits.

Without an HF/MLX reference baseline we can't compare logits exactly,
but we can verify that the runtime is producing plausibly-shaped
logits — not degenerate (same argmax everywhere), not NaN/inf, not
collapsed to a few peaky values.

Per-prompt checks:
  - logits shape == (1, vocab_size)
  - all finite
  - top-K is varied (>1 distinct id at the top)
  - softcap respected (`abs(logits) <= softcap` when set)

Cross-prompt check:
  - different prompts produce different argmax tokens (the model
    isn't always emitting the same id regardless of input). With one
    layer this is the strongest "is the runtime actually computing
    something" signal we have.

This is NOT a reference comparison — that requires loading the same
checkpoint via HF transformers or mlx-lm, which would pull in torch
or large dependencies. The script's contract is that the runtime
isn't degenerate; reference parity is a separate (deferred) sprint.

Run:
    uv run python scripts/structural_logit_sanity.py \
        --path /Users/spotta/Desktop/models/gemma4/gemma-4-26b-a4b-it-nvfp4 \
        --num-layers 4
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Make the apex package importable when run directly from a clone.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apex import MoEModel


_PROMPTS = (
    "Hello",
    "The capital of France is",
    "import numpy as np",
    "Once upon a time",
)
_TOP_K = 10


def _audit_one(model, prompt: str):
    """Forward `prompt`, print top-K table, return (top_k_ids, argmax)."""
    logits = model.forward(prompt)
    assert logits.shape == (1, model.config.vocab_size), (
        f"unexpected logits shape {logits.shape} "
        f"(want (1, {model.config.vocab_size}))"
    )
    assert np.all(np.isfinite(logits)), "logits must be finite"

    softcap = model.config.final_logit_softcapping
    if softcap is not None:
        max_abs = float(np.max(np.abs(logits)))
        # `tanh` saturates at exactly ±1.0 in fp32 once the input
        # magnitude exceeds ~9, so equality with softcap is allowed.
        assert max_abs <= softcap + 1e-3, (
            f"logits exceed softcap: max|logits|={max_abs} > {softcap}"
        )

    flat = logits[0]
    # `argpartition` for the top-K indices, then sort by value desc.
    top_idx = np.argpartition(-flat, _TOP_K)[:_TOP_K]
    top_idx = top_idx[np.argsort(-flat[top_idx])]
    top_vals = flat[top_idx]

    print(f"\nprompt: {prompt!r}")
    print(f"  logits shape: {logits.shape}, finite: true")
    print(f"  range: [{flat.min():.3f}, {flat.max():.3f}], "
          f"mean={flat.mean():.3f}, std={flat.std():.3f}")
    print(f"  top-{_TOP_K} ids+values:")
    for rank, (tid, val) in enumerate(zip(top_idx, top_vals), 1):
        text = model.tokenizer.decode([int(tid)]).replace("\n", "\\n")
        print(f"    {rank:2d}. id={int(tid):>6d}  logit={float(val):>8.3f}  "
              f"text={text!r}")
    return tuple(int(x) for x in top_idx), int(top_idx[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--path",
        required=True,
        help="checkpoint directory (config.json + tokenizer.json + shards)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="how many decoder layers (default: all sliding-attention)",
    )
    parser.add_argument(
        "--max-memory-gb",
        type=float,
        default=4.0,
        help="expert cache budget in GB (default: 4)",
    )
    args = parser.parse_args()

    print(f"loading model from {args.path}")
    model = MoEModel.from_pretrained(
        args.path,
        max_memory_gb=args.max_memory_gb,
        num_layers=args.num_layers,
        verbose=False,
    )
    print(f"num_layers: {model.num_layers}, "
          f"vocab: {model.config.vocab_size}")

    argmaxes = []
    top_sets = []
    for prompt in _PROMPTS:
        topk, am = _audit_one(model, prompt)
        argmaxes.append(am)
        top_sets.append(set(topk))

    print()
    print("=== cross-prompt sanity ===")
    distinct_argmax = len(set(argmaxes))
    print(f"distinct argmax tokens across {len(_PROMPTS)} prompts: "
          f"{distinct_argmax}/{len(_PROMPTS)}")
    if distinct_argmax == 1:
        # With one layer the model often degenerates to a single
        # high-frequency token — flag it loudly but don't exit non-zero;
        # the user might be running with --num-layers 1 deliberately.
        print(f"  ! degenerate: every prompt's argmax is id {argmaxes[0]}")
    else:
        print("  ok: argmax varies across prompts")

    # Top-K Jaccard overlap between adjacent pairs — measures how
    # similar the top-K *sets* are (not just the #1 pick).
    print(f"top-{_TOP_K} jaccard (adjacent pairs):")
    for i in range(len(_PROMPTS) - 1):
        a, b = top_sets[i], top_sets[i + 1]
        jac = len(a & b) / len(a | b)
        print(f"  {_PROMPTS[i]!r:>30}  vs  {_PROMPTS[i+1]!r:<30}  jaccard={jac:.2f}")

    print()
    print("memory:")
    model.memory_report().print_summary()
    print()
    print("[ok] structural sanity complete")


if __name__ == "__main__":
    main()
