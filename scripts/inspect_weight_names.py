"""Print every tensor name in a checkpoint, sorted, one per line.

Usage:
    uv run python scripts/inspect_weight_names.py <path>

Where <path> is one of:
  - a single .safetensors file
  - a directory containing model.safetensors.index.json
    (real HF checkpoints are typically sharded across many files)
  - a single .gguf file

Manual escape hatch for figuring out which apex.weight_layout class
matches an unknown checkpoint: run it on a real download, eyeball the
patterns, then map them through the appropriate layout.

Exits 0 on success, 1 on any error (bad path, missing index file,
unsupported format). Errors go to stderr so output piping stays clean.

The actual name-listing logic lives in apex/checkpoint_index.py so
scripts/validate_layout.py can share it.
"""

import sys
from pathlib import Path

# Make the apex package importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apex.checkpoint_index import list_tensor_names


def main(argv) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <path>", file=sys.stderr)
        return 1
    path = Path(argv[1])
    if not path.exists():
        print(f"path does not exist: {path}", file=sys.stderr)
        return 1
    try:
        names = list_tensor_names(path)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    for name in names:
        print(name)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
