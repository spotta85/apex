"""High-level loaders for non-MoE Gemma 4 model components.

These are the "give me the dequantized fp32 array for this part of
the model" functions. They sit one layer above:

  - apex.safetensors_loader (raw tensor I/O, BF16/F8 decoding)
  - apex.checkpoint_index   (sharded-safetensors metadata)
  - apex.dequant            (NVFP4 / NVFP4_U32 / AFFINE_UINT8_U32)
  - apex.tensor_types       (QuantizedTensor wrapper)

…and are consumed by the smoke / generation scripts and the
real-Gemma tests. Pre-cleanup these lived in `scripts/_real_moe_helpers`
which forced tests to import from `scripts.` to reach them; the
loaders are real library functionality and belong here.

What's here:

  - read_text_config(path)                   → dict
  - load_embedding(path, layout)             → fp32 (vocab, hidden)
  - load_final_norm(path, layout)            → fp32 (hidden,)
  - load_layer_norms(path, layout, layer_id) → dict of four norms
  - load_attention_weights(path, layout, layer_id) → dict of q/k/v/o + qknorm

What's NOT here:
  - Per-expert MoE loading (use apex.quantized_safetensors_store /
    apex.gemma_hf_store / apex.dequant.DequantizingExpertStore)
  - Per-layer router loading (apex.real_router.MultiLayerRealRouter)
  - CLI argparse boilerplate (scripts/_real_moe_helpers.add_real_moe_args)
  - Layout auto-detection (scripts/_real_moe_helpers.detect_real_moe_layout)
"""

import json
from pathlib import Path

import numpy as np

from apex.checkpoint_index import read_safetensors_index
from apex.dequant import dequantize
from apex.safetensors_loader import load_tensor
from apex.tensor_types import QuantizedTensor


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


def load_embedding(path, layout):
    """Load + dequantize the token embedding table to fp32.

    Returns (vocab_size, hidden_size). On Gemma 4 26B-A4B-IT NVFP4
    the embedding is NVFP4_U32-packed (262144 × 352 packed cols + a
    U8 scales sidecar); the dequantized fp32 table is ~3 GB. Caller
    holds it for both the input-token row lookup and the tied lm_head
    matmul.

    Falls back to a plain `np.asarray(..., float32)` cast if the layout
    doesn't expose a `scales` sidecar (e.g. the FakeMoELayout's
    synthetic checkpoint, if it ever grows one).
    """
    path = Path(path)
    weight_map = read_safetensors_index(path)
    names = layout.embedding_tensors()
    data = load_tensor(path / weight_map[names["weight"]], names["weight"])

    if "scales" in names:
        scales = load_tensor(
            path / weight_map[names["scales"]], names["scales"],
        )
        tensor = QuantizedTensor(
            data=data,
            dtype="NVFP4_U32",
            shape=tuple(data.shape),
            weight_scale=scales,
        )
        return dequantize(tensor, name=names["weight"])
    return np.asarray(data, dtype=np.float32)


def load_final_norm(path, layout):
    """Load the model-final RMSNorm weight (BF16 → fp32) by layout name."""
    path = Path(path)
    weight_map = read_safetensors_index(path)
    name = layout.final_norm()
    return load_tensor(path / weight_map[name], name)


def load_layer_norms(path, layout, layer_id: int) -> dict:
    """Load layer L's four BF16 norm weights as fp32 ndarrays.

    Returns {input, post_attn, pre_ffn, post_ffn}. Skips Gemma 4's
    `_1` / `_2` variants (those belong to parallel feedforward paths
    we don't drive yet — dense MLP alongside the routed MoE).
    """
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
