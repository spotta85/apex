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
patterns, then map them through the appropriate layout (or write a
new one if none fit).

Exits 0 on success, 1 on any error (bad path, missing index file,
unsupported format). Errors go to stderr so output piping stays clean.
"""

import json
import sys
from pathlib import Path

# Make the apex package importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def list_safetensors_keys(path: Path) -> list:
    """Return all tensor keys in a single .safetensors file, sorted."""
    from safetensors import safe_open

    with safe_open(str(path), framework="numpy") as f:
        return sorted(f.keys())


def list_sharded_safetensors_keys(directory: Path) -> list:
    """Return the union of all tensor keys across a sharded safetensors dir.

    Reads the model.safetensors.index.json (HF's standard sharding
    metadata file). The 'weight_map' field is a dict mapping every
    tensor name to the shard file that contains it; we only need the
    keys.
    """
    indexes = list(directory.glob("*.safetensors.index.json"))
    if not indexes:
        raise FileNotFoundError(
            f"no *.safetensors.index.json in {directory} — is this really a "
            f"sharded-safetensors directory?"
        )
    if len(indexes) > 1:
        # Shouldn't happen with HF checkpoints; surface the ambiguity
        # rather than silently picking one.
        raise ValueError(
            f"multiple safetensors index files in {directory}: {indexes}"
        )
    with open(indexes[0]) as f:
        index = json.load(f)
    weight_map = index.get("weight_map")
    if weight_map is None:
        raise ValueError(
            f"{indexes[0]} is missing the required 'weight_map' field"
        )
    return sorted(weight_map)


def list_gguf_keys(path: Path) -> list:
    """Return all tensor keys in a .gguf file, sorted."""
    from apex.gguf_tensor_index import GGUFTensorIndex

    return sorted(GGUFTensorIndex(str(path)).names())


def inspect(path: Path) -> list:
    """Dispatch on path shape; return sorted tensor names."""
    if path.is_dir():
        return list_sharded_safetensors_keys(path)
    if path.suffix == ".safetensors":
        return list_safetensors_keys(path)
    if path.suffix == ".gguf":
        return list_gguf_keys(path)
    raise ValueError(
        f"expected a .safetensors file, .gguf file, or directory containing "
        f"model.safetensors.index.json; got: {path}"
    )


def main(argv) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <path>", file=sys.stderr)
        return 1
    path = Path(argv[1])
    if not path.exists():
        print(f"path does not exist: {path}", file=sys.stderr)
        return 1
    try:
        names = inspect(path)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    for name in names:
        print(name)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
