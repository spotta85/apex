"""Step 24F: per-prompt × per-L quality benchmark on real Gemma 4.

Drives the user-facing `MoEModel.generate_chat` over a fixed prompt
list at each layer count in the sweep, capturing per-trial:

  - generated text + token ids
  - whether the loop hit EOS (or ran to max_tokens)
  - tokens/second
  - peak expert MB, kv MB, evictions
  - finite-logit assertion
  - simple repetition score (longest run of the same token id)

The point isn't a single pass/fail bit — it's an honest report of
what shows up at each depth so the operator can see how router
scaling + dense MLP + sliding-window changed observable behavior
versus pre-step-24.

Prompts go through the chat template by default. This checkpoint is
instruction-tuned and 4-bit quantized; outside its chat template it
degenerates even in the mlx_lm reference (`"The capital of France is"`
→ `"作用作用作用…"` there too), so raw-completion output says nothing
about whether the runtime is correct. `--raw` keeps the old behaviour
for comparing against earlier runs.

Only the full-depth row is a quality signal: a model truncated to 8 or
16 layers emits noise by construction. The shallower rows are there
for the speed / memory scaling columns.

Run:
    uv run python scripts/benchmark_quality.py \\
        --path /Users/spotta/Desktop/models/gemma4/gemma-4-26b-a4b-it-nvfp4 \\
        --max-memory-gb 10 --layers 8 16 30
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Make the apex package importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apex import MoEModel


# User turns, rendered through the checkpoint's chat template.
_CHAT_PROMPTS = (
    "What is the capital of France?",
    "What is 2+2?",
    "Write a haiku about the ocean.",
    "Write a Python function that returns the nth Fibonacci number.",
)

# Raw-completion prompts (`--raw`): no chat template, just BOS + text.
_RAW_PROMPTS = (
    "The capital of France is",
    "2 + 2 =",
    "Once upon a time",
    "def fibonacci(n):",
)

_MIB = 1024 * 1024
_DEFAULT_PATHS = (
    Path("/Users/spotta/Desktop/models/Gemma-4-26B-A4B-NVFP4"),
    Path("/Users/spotta/Desktop/models/gemma4/Gemma-4-26B-A4B-NVFP4"),
    Path("/Users/spotta/Desktop/models/gemma4/gemma-4-26b-a4b-it-nvfp4"),
)


def _default_model_path() -> str:
    """Return the first known local checkpoint path that exists."""
    for path in _DEFAULT_PATHS:
        if path.is_dir():
            return str(path)
    return str(_DEFAULT_PATHS[0])


def _longest_repeat_run(ids: list) -> int:
    """Length of the longest contiguous run of identical ids in `ids`.

    A simple `repetition score` proxy — high values often mean the
    model has degenerated into a loop (still finite, but bad output).
    """
    if not ids:
        return 0
    best = 1
    cur = 1
    prev = ids[0]
    for x in ids[1:]:
        if x == prev:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
            prev = x
    return best


def _bench_one(model, prompt: str, max_tokens: int, raw: bool):
    """Run one (model, prompt) trial. Returns a per-trial dict."""
    t0 = time.time()
    if raw:
        text = model.generate(prompt, max_tokens=max_tokens, verbose=False)
    else:
        text = model.generate_chat(
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens, verbose=False,
        )
    gen_s = time.time() - t0
    ids = model._last_generated_ids
    n_generated = len(ids)
    eos_set = set(model.config.eos_token_ids)
    hit_eos = any(i in eos_set for i in ids)
    # `generate` asserts finite logits per step, so reaching here means
    # the run was finite. Capture it explicitly anyway for the report.
    tps = n_generated / gen_s if gen_s > 0 else 0.0
    return {
        "prompt": prompt,
        "text": text,
        "ids": list(ids),
        "n_generated": n_generated,
        "hit_eos": hit_eos,
        "gen_s": gen_s,
        "tok_per_s": tps,
        "longest_run": _longest_repeat_run(list(ids)),
    }


def _bench_layer(path, num_layers: int, max_memory_gb: float, max_tokens: int,
                 backend: str = "numpy", raw: bool = False):
    """Build a model at `num_layers` and run the full prompt sweep."""
    print(f"\n=== num_layers={num_layers} ===")
    t0 = time.time()
    model = MoEModel.from_pretrained(
        path,
        max_memory_gb=max_memory_gb,
        num_layers=num_layers,
        backend=backend,
        verbose=False,
    )
    load_s = time.time() - t0
    print(f"load: {load_s:.1f}s")

    rows = []
    for prompt in (_RAW_PROMPTS if raw else _CHAT_PROMPTS):
        rows.append(_bench_one(model, prompt, max_tokens, raw))

    report = model.memory_report()
    print(f"memory: peak expert {report.expert.peak_bytes / _MIB:.1f} MB, "
          f"kv {report.kv_cache_bytes / _MIB:.2f} MB, "
          f"evictions {report.expert.evictions}, "
          f"budget_ok {report.expert.budget_respected}")

    print(f"  {'prompt':<66}{'tok':>4}{'tok/s':>7}{'eos':>4}"
          f"{'maxrun':>7}  text")
    for r in rows:
        print(f"  {r['prompt']!r:<66}{r['n_generated']:>4d}"
              f"{r['tok_per_s']:>7.2f}{('Y' if r['hit_eos'] else '-'):>4}"
              f"{r['longest_run']:>7d}  {r['text']!r}")
    return load_s, rows, report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--path",
        default=_default_model_path(),
        help="checkpoint directory (default: first known local Gemma path)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=32,
        help="how many tokens to generate per prompt (default 32)",
    )
    parser.add_argument(
        "--raw", action="store_true",
        help="raw completion (no chat template) with the pre-chat prompt "
             "list; the model degenerates there even in the reference",
    )
    parser.add_argument(
        "--max-memory-gb", type=float, default=10.0,
        help="expert cache budget in GB (default 10)",
    )
    parser.add_argument(
        "--layers", type=int, nargs="+", default=[8, 16, 30],
        help="layer counts to sweep (default 8 16 30)",
    )
    parser.add_argument(
        "--backend", choices=("numpy", "mlx"), default="numpy",
        help="expert compute backend (default numpy; mlx = Apple-Silicon GPU)",
    )
    args = parser.parse_args()
    path = Path(args.path)
    if not path.is_dir():
        parser.error(f"checkpoint directory not found: {path}")
    args.path = str(path)

    print(f"checkpoint: {args.path}")
    print(f"prompts:    {len(_RAW_PROMPTS if args.raw else _CHAT_PROMPTS)} "
          f"({'raw completion' if args.raw else 'chat template'})")
    print(f"max_tokens: {args.max_tokens}")
    print(f"budget:     {args.max_memory_gb:.1f} GB")
    print(f"layers:     {args.layers}")
    print(f"backend:    {args.backend}")

    for L in args.layers:
        _bench_layer(
            args.path, L, args.max_memory_gb, args.max_tokens, args.backend,
            args.raw,
        )

    print()
    print("[ok] quality benchmark complete")


if __name__ == "__main__":
    main()
