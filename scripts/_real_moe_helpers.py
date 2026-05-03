"""Shared bootstrapping for real-Gemma smoke and demo scripts."""

import argparse
import contextlib
from pathlib import Path

import numpy as np

from apex.checkpoint_index import read_safetensors_index
from apex.safetensors_loader import load_tensor


DEFAULT_GEMMA_PATH = "/Users/spotta/Desktop/models/Gemma-4-26B-A4B-NVFP4"
_WEIGHT_MAP_CACHE = {}


@contextlib.contextmanager
def nvfp4_matmul_warnings():
    """Silence transient real-NVFP4 matmul warnings; callers assert finiteness."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        yield


def load_router(path, layout, layer_id):
    """Return (router_weight_fp32, num_experts, hidden_dim) for one layer."""
    path = Path(path)
    name = layout.gate_weight(layer_id)
    cache_key = path.resolve()
    if cache_key not in _WEIGHT_MAP_CACHE:
        _WEIGHT_MAP_CACHE[cache_key] = read_safetensors_index(cache_key)
    weight_map = _WEIGHT_MAP_CACHE[cache_key]
    router = load_tensor(path / weight_map[name], name)
    num_experts, hidden_dim = router.shape
    return router, num_experts, hidden_dim


def add_real_moe_args(
    parser: argparse.ArgumentParser,
    *,
    include_layer: bool = True,
    include_max_memory: bool = True,
    default_max_memory_mb: float = 256,
):
    """Register the standard real-Gemma CLI flags."""
    parser.add_argument(
        "--path",
        default=DEFAULT_GEMMA_PATH,
        help=f"checkpoint directory (default: {DEFAULT_GEMMA_PATH})",
    )
    if include_layer:
        parser.add_argument(
            "--layer",
            type=int,
            default=0,
            help="layer index to drive end-to-end (default: 0)",
        )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="experts per layer (default: 8, matches Gemma 4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for the fake hidden state (default: 0)",
    )
    if include_max_memory:
        parser.add_argument(
            "--max-memory-mb",
            type=float,
            default=default_max_memory_mb,
            help=f"cache budget in MB (default: {default_max_memory_mb})",
        )
