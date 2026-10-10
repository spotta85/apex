"""Interactive chat runner for the APEX MoEModel API.

Run from the repo root:

    uv run python tests/chat/chat_with_model.py \
        --path /Users/spotta/Desktop/models/Gemma-4-26B-A4B-NVFP4 \
        --num-layers 8 --max-memory-gb 8

Use `exit` or `quit` to stop.
"""

import argparse
import contextlib
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from apex import MoEModel


DEFAULT_PATHS = (
    Path("/Users/spotta/Desktop/models/Gemma-4-26B-A4B-NVFP4"),
    Path("/Users/spotta/Desktop/models/gemma4/Gemma-4-26B-A4B-NVFP4"),
    Path("/Users/spotta/Desktop/models/gemma4/gemma-4-26b-a4b-it-nvfp4"),
)
DEFAULT_STATUS_PATH = Path(__file__).resolve().parent / "status.txt"


def default_model_path() -> str:
    """Return the first known local checkpoint path that exists."""
    for path in DEFAULT_PATHS:
        if path.is_dir():
            return str(path)
    return str(DEFAULT_PATHS[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with a local APEX model.")
    parser.add_argument(
        "--path",
        default=default_model_path(),
        help="checkpoint directory (default: first known local Gemma path)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=8,
        help="decoder layers to load (default: 8; use 30 for full model)",
    )
    parser.add_argument(
        "--max-memory-gb",
        type=float,
        default=4.0,
        help="LRU expert cache budget in GB (default: 4). Holds a hot "
             "subset of experts and evicts the rest; total RSS is this "
             "+ ~13 GB fixed overhead. Don't set huge.",
    )
    parser.add_argument(
        "--backend",
        choices=("mlx", "numpy"),
        default="mlx",
        help="expert compute backend (default: mlx, the fast path on "
             "Apple Silicon; numpy for the portable dense-fp32 path)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="max response tokens per turn (default: 128)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="sampling temperature (default: 0 = greedy)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print model loading and per-token generation traces",
    )
    parser.add_argument(
        "--status-path",
        default=str(DEFAULT_STATUS_PATH),
        help="where cache/load status lines are written",
    )
    args = parser.parse_args()
    status_path = Path(args.status_path)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text("")

    print(f"loading model from {args.path}")
    model = MoEModel.from_pretrained(
        args.path,
        max_memory_gb=args.max_memory_gb,
        num_layers=args.num_layers,
        backend=args.backend,
        verbose=args.verbose,
    )
    print("ready. type 'exit' or 'quit' to stop.")
    print(f"status log: {status_path}")

    messages = []
    turn = 0
    while True:
        user_text = input("\nyou: ").strip()
        if user_text.lower() in {"exit", "quit"}:
            break
        if not user_text:
            continue

        messages.append({"role": "user", "content": user_text})
        turn += 1
        with status_path.open("a") as status_file:
            status_file.write(f"\n=== turn {turn} ===\n")
            status_file.flush()
            with contextlib.redirect_stdout(status_file):
                reply = model.generate_chat(
                    messages,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    verbose=args.verbose,
                )
        print(f"\nmodel: {reply}")
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
