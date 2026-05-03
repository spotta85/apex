"""Read tensor inventory from a checkpoint without loading any tensor data.

Three checkpoint shapes are supported, dispatched by the input path:

  - single .safetensors file
  - directory containing model.safetensors.index.json (sharded safetensors)
  - single .gguf file

Two lightweight queries:

  - list_tensor_names(path)       -> sorted list[str]
  - get_tensor_shape(path, name)  -> tuple[int, ...]

Both only touch metadata — never the bulk tensor bytes — so they're
safe to run against multi-GB checkpoints. For the sharded safetensors
case we read the index file once to find the shard mapping, then open
the right shard for shape queries.

Shared backend for scripts/inspect_weight_names.py (full inventory)
and scripts/validate_layout.py (named-existence + shape consistency).
"""

import json
from pathlib import Path


def list_tensor_names(path) -> list:
    """Return a sorted list of every tensor name in `path`.

    Raises ValueError for unsupported file types and FileNotFoundError /
    KeyError for malformed sharded-safetensors directories.
    """
    p = Path(path)
    if p.is_dir():
        return _list_sharded_safetensors_keys(p)
    if p.suffix == ".safetensors":
        return _list_safetensors_keys(p)
    if p.suffix == ".gguf":
        return _list_gguf_keys(p)
    raise ValueError(_unsupported_path_message(p))


def get_tensor_shape(path, name) -> tuple:
    """Return one tensor's shape (numpy ordering) without loading its data.

    Raises KeyError if `name` is not present in the checkpoint.
    """
    p = Path(path)
    if p.is_dir():
        return _sharded_safetensors_shape(p, name)
    if p.suffix == ".safetensors":
        return _safetensors_shape(p, name)
    if p.suffix == ".gguf":
        return _gguf_shape(p, name)
    raise ValueError(_unsupported_path_message(p))


# --- safetensors helpers ---------------------------------------------------

def _list_safetensors_keys(path: Path) -> list:
    from safetensors import safe_open

    with safe_open(str(path), framework="numpy") as f:
        return sorted(f.keys())


def _safetensors_shape(path: Path, name: str) -> tuple:
    from safetensors import safe_open

    with safe_open(str(path), framework="numpy") as f:
        # get_slice exposes shape without materializing the tensor.
        return tuple(f.get_slice(name).get_shape())


def read_safetensors_index(directory) -> dict:
    """Return the parsed `weight_map` dict from <dir>/*.safetensors.index.json.

    Public so other loaders (e.g. the quantized expert store) can
    follow the same shard mapping without re-implementing the lookup.
    Errors are FileNotFoundError (no index) / ValueError (multiple
    indexes or malformed index).
    """
    directory = Path(directory)
    indexes = list(directory.glob("*.safetensors.index.json"))
    if not indexes:
        raise FileNotFoundError(
            f"no *.safetensors.index.json in {directory} — is this really a "
            f"sharded-safetensors directory?"
        )
    if len(indexes) > 1:
        # Shouldn't happen with HF checkpoints; surface ambiguity loudly.
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
    return weight_map


def _list_sharded_safetensors_keys(directory: Path) -> list:
    return sorted(read_safetensors_index(directory))


def _sharded_safetensors_shape(directory: Path, name: str) -> tuple:
    weight_map = read_safetensors_index(directory)
    if name not in weight_map:
        raise KeyError(
            f"tensor {name!r} not found in {directory}'s safetensors index"
        )
    shard_path = directory / weight_map[name]
    return _safetensors_shape(shard_path, name)


# --- gguf helpers ----------------------------------------------------------

def _list_gguf_keys(path: Path) -> list:
    from apex.gguf_tensor_index import GGUFTensorIndex

    return sorted(GGUFTensorIndex(str(path)).names())


def _gguf_shape(path: Path, name: str) -> tuple:
    from apex.gguf_tensor_index import GGUFTensorIndex

    return tuple(GGUFTensorIndex(str(path)).metadata(name)["shape"])


# --- shared error message --------------------------------------------------

def _unsupported_path_message(path: Path) -> str:
    return (
        f"expected a .safetensors file, .gguf file, or directory containing "
        f"model.safetensors.index.json; got: {path}"
    )
