"""Shared bootstrapping for real-Gemma smoke and demo scripts."""

import argparse
import contextlib
import json
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


def read_text_config(path) -> dict:
    """Return the `text_config` dict from a Gemma 4 checkpoint's config.json.

    Gemma 4 is multimodal (Gemma4ForConditionalGeneration), so the
    language-model hyperparams (num_attention_heads, head_dim,
    rms_norm_eps, etc.) are nested under `text_config` rather than
    sitting at the top level. Pull them out once here so callers
    don't repeat the parse.
    """
    with open(Path(path) / "config.json") as f:
        cfg = json.load(f)
    text = cfg.get("text_config")
    if text is None:
        raise ValueError(f"{path}/config.json has no text_config field")
    return text


def load_layer_norms(path, layout, layer_id: int) -> dict:
    """Load layer L's four BF16 norm weights as fp32 ndarrays.

    Returns {input, post_attn, pre_ffn, post_ffn}. Skips Gemma 4's
    `_1` / `_2` variants (those belong to parallel feedforward paths
    we don't drive yet — dense MLP alongside the routed MoE).
    """
    from apex.safetensors_loader import load_tensor

    path = Path(path)
    weight_map = read_safetensors_index(path)

    def _load(name: str):
        return load_tensor(path / weight_map[name], name)

    return {
        "input": _load(layout.input_layernorm(layer_id)),
        "post_attn": _load(layout.post_attention_layernorm(layer_id)),
        "pre_ffn": _load(layout.pre_feedforward_layernorm(layer_id)),
        "post_ffn": _load(layout.post_feedforward_layernorm(layer_id)),
    }


def load_attention_weights(path, layout, layer_id: int) -> dict:
    """Load + dequantize layer `layer_id`'s q/k/v/o + q_norm/k_norm.

    Returns a dict with keys {q_weight, k_weight, v_weight, o_weight,
    q_norm, k_norm}. Each *_weight is fp32 from NVFP4_U32 dequant
    (mlx-community lowercase Gemma layout); each *_norm is fp32 from
    BF16. Mirrors `apex.real_router.MultiLayerRealRouter._load_router_weight`
    (the same `QuantizedTensor` + `dequantize` pattern, just over four
    projections instead of one).
    """
    from apex.dequant import dequantize
    from apex.safetensors_loader import load_tensor
    from apex.tensor_types import QuantizedTensor

    path = Path(path)
    weight_map = read_safetensors_index(path)

    def _load(name: str):
        return load_tensor(path / weight_map[name], name)

    def _load_proj(names: dict):
        data = _load(names["weight"])
        tensor = QuantizedTensor(
            data=data,
            dtype="NVFP4_U32",
            shape=tuple(data.shape),
            weight_scale=_load(names["scales"]),
        )
        return dequantize(tensor, name=names["weight"])

    return {
        "q_weight": _load_proj(layout.attn_q_proj_tensors(layer_id)),
        "k_weight": _load_proj(layout.attn_k_proj_tensors(layer_id)),
        "v_weight": _load_proj(layout.attn_v_proj_tensors(layer_id)),
        "o_weight": _load_proj(layout.attn_o_proj_tensors(layer_id)),
        "q_norm": _load(layout.attn_q_norm(layer_id)),
        "k_norm": _load(layout.attn_k_norm(layer_id)),
    }


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
