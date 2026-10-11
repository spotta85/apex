"""Repack the Gemma 4 safetensors shards into the engine's on-disk layout.

The checkpoint ships as three safetensors files: a JSON header followed by
a blob of tensor bytes in whatever order the exporter chose, unaligned, and
with a vision tower we never use. The engine wants two things instead:

  core.bin
      Every text-tower tensor that is not an expert (embeddings, final norm,
      and per layer: the seven norms, layer_scalar, attention q/k/v/o, the
      dense MLP, the router). Each tensor starts on a 16 KB page boundary so
      the whole file can be mmap'd once and any tensor handed to Metal as a
      no-copy buffer.

  layer_00.bin .. layer_29.bin
      That layer's 128 experts at a fixed page-aligned stride. Expert e is
      the byte range [e * stride, (e + 1) * stride). Inside a slot the six
      pieces sit at fixed offsets in a fixed order: gate weight, gate scales,
      up weight, up scales, down weight, down scales. Loading an expert is
      one pread at a computed offset, straight into a GPU-visible buffer.

  manifest.json
      Where everything landed: dtype, shape, offset, size, and a SHA-256 per
      core tensor and per expert slot, plus the model's text config and
      quantization settings so the loader can assert on them.

Nothing is decoded. The packed 4-bit values and their FP8 scales are copied
byte for byte; the NVFP4 decoder lives in the Metal kernels.

Usage (from tools/):
    uv run python repack.py            # pack into <checkpoint>/../packed
    uv run python repack.py verify     # re-hash the packed files
"""

import argparse
import hashlib
import json
import struct
from contextlib import ExitStack
from pathlib import Path

PAGE_SIZE = 16_384          # macOS page size; Metal no-copy buffers need this alignment
EXPERTS = 128
NUM_LAYERS = 30

CKPT = Path("/Users/spotta/Desktop/models/gemma4/gemma-4-26b-a4b-it-nvfp4")
PREFIX = "language_model.model."   # text tower only; vision_tower / embed_vision are dropped

# Order of the six pieces inside one expert slot. Fixed forever; C++ hardcodes it.
EXPERT_PIECES = (
    "gate_proj.weight", "gate_proj.scales",
    "up_proj.weight",   "up_proj.scales",
    "down_proj.weight", "down_proj.scales",
)


def align_to_page(size):
    """Round up to the next 16 KB boundary."""
    return (size + PAGE_SIZE - 1) // PAGE_SIZE * PAGE_SIZE


# A safetensors file:
# ┌─────────────┬────────────────────────┬─────────────────────────┐
# │ Header size │ JSON header            │ Raw tensor data         │
# │ 8 bytes     │ N bytes                │ Remaining bytes         │
# └─────────────┴────────────────────────┴─────────────────────────┘
# Each header entry's data_offsets are relative to the start of the raw data.

def read_headers(ckpt):
    """name -> {dtype, shape, file, start, end} for every tensor across all shards."""
    tensors = {}
    shards = sorted(ckpt.glob("*.safetensors"))
    if not shards:
        raise ValueError(f"No safetensors files in {ckpt}")
    for path in shards:
        with path.open("rb") as file:
            header_size = struct.unpack("<Q", file.read(8))[0]
            header = json.loads(file.read(header_size))
        data_start = 8 + header_size
        for name, info in header.items():
            if name == "__metadata__":
                continue
            if name in tensors:
                raise ValueError(f"Duplicate tensor across shards: {name}")
            start, end = info["data_offsets"]
            tensors[name] = {
                "dtype": info["dtype"],
                "shape": info["shape"],
                "file": path,
                "start": data_start + start,
                "end": data_start + end,
            }
    return tensors


def read_bytes(tensor, start=None, end=None):
    """Raw bytes of a tensor, or of a sub-range [start, end) relative to it."""
    lo = tensor["start"] + (start or 0)
    hi = tensor["start"] + end if end is not None else tensor["end"]
    with tensor["file"].open("rb") as file:
        file.seek(lo)
        data = file.read(hi - lo)
    if len(data) != hi - lo:
        raise ValueError(f"Truncated read from {tensor['file'].name}")
    return data


# --------------------------------------------------------------------
# core.bin
# --------------------------------------------------------------------

def core_order(name):
    """embed_tokens first, then the final norm, then layers in numeric order."""
    parts = name.removeprefix(PREFIX).split(".")
    if parts[0] == "embed_tokens":
        return (0, 0, name)
    if parts[0] == "norm":
        return (1, 0, name)
    return (2, int(parts[1]), name)


def write_core(headers, out_dir):
    """Write every non-expert text-tower tensor, page-aligned, to core.bin."""
    entries = []
    names = [n for n in headers if n.startswith(PREFIX) and ".experts." not in n]
    with (out_dir / "core.bin").open("wb") as output:
        for name in sorted(names, key=core_order):
            tensor = headers[name]
            offset = align_to_page(output.tell())
            output.seek(offset)
            data = read_bytes(tensor)
            output.write(data)
            entries.append({
                "name": name.removeprefix(PREFIX),
                "dtype": tensor["dtype"],
                "shape": tensor["shape"],
                "offset": offset,
                "nbytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            })
        # Pad the tail so the file length is a page multiple (mmap-friendly).
        output.truncate(align_to_page(output.tell()))
    return entries


# --------------------------------------------------------------------
# layer_NN.bin
# --------------------------------------------------------------------

def expert_layout(headers, layer):
    """(pieces, stride) for one layer.

    Each expert tensor is stacked [128, R, C] and stored row-major, so
    expert e of a piece is the contiguous byte range
    [e * piece_bytes, (e + 1) * piece_bytes) within that tensor.
    """
    pieces = []
    offset = 0
    for name in EXPERT_PIECES:
        tensor = headers[f"{PREFIX}layers.{layer}.experts.switch_glu.{name}"]
        total = tensor["end"] - tensor["start"]
        if tensor["shape"][0] != EXPERTS or total % EXPERTS:
            raise ValueError(f"Unexpected expert tensor shape: layer {layer} {name}")
        piece_bytes = total // EXPERTS
        pieces.append({
            "name": name,
            "dtype": tensor["dtype"],
            "shape": tensor["shape"][1:],
            "offset": offset,          # within the slot
            "nbytes": piece_bytes,
            "tensor": tensor,
        })
        offset += piece_bytes
    return pieces, align_to_page(offset)


def write_layer(headers, layer, out_dir):
    """Write one layer's 128 experts at a fixed stride. Returns the manifest entry."""
    pieces, stride = expert_layout(headers, layer)
    hashes = []
    with (out_dir / f"layer_{layer:02d}.bin").open("wb") as output:
        output.truncate(EXPERTS * stride)      # preallocate; the per-slot pad stays zero
        for expert in range(EXPERTS):
            digest = hashlib.sha256()
            for piece in pieces:
                data = read_bytes(piece["tensor"],
                                  expert * piece["nbytes"], (expert + 1) * piece["nbytes"])
                output.seek(expert * stride + piece["offset"])
                output.write(data)
                digest.update(data)
            hashes.append(digest.hexdigest())
    return {
        "layer": layer,
        "file": f"layer_{layer:02d}.bin",
        "size": EXPERTS * stride,
        "num_experts": EXPERTS,
        "stride": stride,
        "pieces": [{k: v for k, v in p.items() if k != "tensor"} for p in pieces],
        "expert_sha256": hashes,
    }


# --------------------------------------------------------------------
# verification (optoinal)
# --------------------------------------------------------------------

def verify(out_dir, manifest):
    """Re-hash every core tensor and every expert slot against the manifest."""
    def check(file, offset, size, expected, label):
        file.seek(offset)
        if hashlib.sha256(file.read(size)).hexdigest() != expected:
            raise ValueError(f"Hash mismatch: {label}")

    core = manifest["core"]
    with (out_dir / core["file"]).open("rb") as file:
        if (out_dir / core["file"]).stat().st_size != core["size"]:
            raise ValueError("core.bin size mismatch")
        for t in core["tensors"]:
            check(file, t["offset"], t["nbytes"], t["sha256"], t["name"])
    print("verified core.bin", flush=True)

    for entry in manifest["layers"]:
        path = out_dir / entry["file"]
        if path.stat().st_size != entry["size"]:
            raise ValueError(f"size mismatch: {path.name}")
        slot_bytes = sum(p["nbytes"] for p in entry["pieces"])   # pieces are contiguous
        with path.open("rb") as file:
            for expert, expected in enumerate(entry["expert_sha256"]):
                check(file, expert * entry["stride"], slot_bytes, expected,
                      f"{path.name} expert {expert}")
        print(f"verified {path.name}", flush=True)


# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("cmd", nargs="?", choices=("pack", "verify"), default="pack")
    parser.add_argument("--ckpt", type=Path, default=CKPT)
    parser.add_argument("--out", type=Path, default=CKPT.parent / "packed")
    args = parser.parse_args()
    out_dir = args.out
    manifest_path = out_dir / "manifest.json"

    if args.cmd == "verify":
        verify(out_dir, json.loads(manifest_path.read_text()))
        return

    config = json.loads((args.ckpt / "config.json").read_text())
    text_config = config["text_config"]
    if text_config["num_hidden_layers"] != NUM_LAYERS or text_config["num_experts"] != EXPERTS:
        raise ValueError(f"Expected {NUM_LAYERS} layers and {EXPERTS} experts")

    headers = read_headers(args.ckpt)
    # Refuse to run if the expert tensors are not exactly the set we know how
    # to lay out; anything else would be silently dropped from the output.
    expected = {f"{PREFIX}layers.{l}.experts.switch_glu.{p}"
                for l in range(NUM_LAYERS) for p in EXPERT_PIECES}
    actual = {n for n in headers if n.startswith(PREFIX) and ".experts." in n}
    if actual != expected:
        raise ValueError("Unexpected expert tensor set; refusing to pack")

    # Fresh directory only, so a failed run cannot leave a stale manifest.
    out_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "version": 1,
        "alignment": PAGE_SIZE,
        "text_config": text_config,
        "quantization": config.get("quantization"),
        "core": {"file": "core.bin", "tensors": write_core(headers, out_dir)},
        "layers": [],
    }
    manifest["core"]["size"] = (out_dir / "core.bin").stat().st_size
    print("wrote core.bin", flush=True)
    for layer in range(NUM_LAYERS):
        manifest["layers"].append(write_layer(headers, layer, out_dir))
        print(f"wrote layer_{layer:02d}.bin", flush=True)

    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
