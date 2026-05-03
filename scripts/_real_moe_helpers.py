"""Shared bootstrapping for real-Gemma smoke and demo scripts."""

import argparse
import contextlib
from pathlib import Path

import numpy as np

from apex.checkpoint_index import read_safetensors_index


_GEMMA_PATH_CANDIDATES = (
    Path("/Users/spotta/Desktop/models/Gemma-4-26B-A4B-NVFP4"),
    Path("/Users/spotta/Desktop/models/gemma4/Gemma-4-26B-A4B-NVFP4"),
)
DEFAULT_GEMMA_PATH = str(
    next((path for path in _GEMMA_PATH_CANDIDATES if path.is_dir()), _GEMMA_PATH_CANDIDATES[0])
)
_WEIGHT_MAP_CACHE = {}


@contextlib.contextmanager
def nvfp4_matmul_warnings():
    """Silence transient real-NVFP4 matmul warnings; callers assert finiteness."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        yield


def load_router(path, layout, layer_id):
    """Return (router_weight_fp32, num_experts, hidden_dim) for one layer."""
    from apex.real_router import MultiLayerRealRouter

    router_loader = MultiLayerRealRouter(path, layout, top_k=1)
    router = router_loader.weight_for_layer(layer_id)
    num_experts, hidden_dim = router.shape
    return router, num_experts, hidden_dim


def detect_real_moe_layout(path):
    """Return a supported Gemma layout for a sharded checkpoint path."""
    from apex.weight_layout import get_layout

    path = Path(path)
    cache_key = path.resolve()
    if cache_key not in _WEIGHT_MAP_CACHE:
        _WEIGHT_MAP_CACHE[cache_key] = read_safetensors_index(cache_key)
    weight_map = _WEIGHT_MAP_CACHE[cache_key]
    for name in ("gemma-moe", "gemma-hf-moe"):
        layout = get_layout(name)
        if layout.gate_weight(0) in weight_map:
            return name, layout
    raise SystemExit(f"could not detect a supported Gemma MoE layout for {path}")


def quantized_store_for_layout(path, layout):
    """Return the quantized expert store matching `layout`."""
    from apex.gemma_hf_store import GemmaHFQuantizedExpertStore
    from apex.quantized_safetensors_store import QuantizedSafetensorsExpertStore

    if getattr(layout, "stacked_experts", False):
        return GemmaHFQuantizedExpertStore(path, layout=layout)
    return QuantizedSafetensorsExpertStore(path, layout=layout)


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
