"""High-level loaders for non-MoE Gemma 4 model components.

These are the "give me the dequantized fp32 array for this part of
the model" functions. They sit one layer above:

  - apex.safetensors_loader (raw tensor I/O, BF16/F8 decoding)
  - apex.checkpoint_index   (sharded-safetensors metadata)
  - apex.dequant            (NVFP4 / NVFP4_U32 / AFFINE_UINT8_U32)
  - apex.tensor_types       (QuantizedTensor wrapper)

…and are consumed by `MoEModel.from_pretrained` (apex/model.py), the
smoke / generation scripts and the real-Gemma tests. Pre-cleanup these lived in `scripts/_real_moe_helpers`
which forced tests to import from `scripts.` to reach them; the
loaders are real library functionality and belong here.

What's here:

  - read_text_config(path)                   → dict
  - load_embedding(path, layout)             → fp32 (vocab, hidden)
  - load_final_norm(path, layout)            → fp32 (hidden,)
  - load_layer_norms(path, layout, layer_id) → dict of norms (the four
                                  sandwich norms, plus the Gemma 4
                                  `_1`/`_2` variants and `layer_scalar`
                                  when the layout has them)
  - load_attention_weights(path, layout, layer_id, *, k_eq_v=False)
                                             → dict of q/k/v/o + qknorm
  - load_dense_mlp_weights(path, layout, layer_id)
                                             → dict of gate/up/down, or None

What's NOT here:
  - Per-expert MoE loading (use apex.quantized_safetensors_store /
    apex.gemma_hf_store / apex.dequant.DequantizingExpertStore)
  - Per-layer router loading (apex.real_router.MultiLayerRealRouter)
  - CLI argparse boilerplate (scripts/_real_moe_helpers.add_real_moe_args)
  - Layout auto-detection (scripts/_real_moe_helpers.detect_real_moe_layout)

Step 24B added `load_dense_mlp_weights` for Gemma 4's per-layer dense
MLP (the parallel branch alongside the routed MoE). Same dequant
pipeline as `load_attention_weights`, over three projections; the
encoding (8-bit affine or NVFP4) is read from the checkpoint.
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
    """Load layer L's BF16 norm weights as fp32 ndarrays.

    Always returns {input, post_attn, pre_ffn, post_ffn}. When the
    layout exposes Gemma 4's parallel-branch variants (step 26B), the
    dict additionally carries:

      - `pre_ffn_2`   — pre-norm for the routed MoE branch
      - `post_ffn_1`  — post-norm for the dense MLP branch
      - `post_ffn_2`  — post-norm for the routed MoE branch

    (The dense branch uses the unsuffixed `pre_ffn` as its pre-norm;
    `post_ffn` then normalises the sum of both branches.) It also
    carries `layer_scalar` — the shape-(1,) per-layer output
    multiplier — when the layout exposes one.

    Layouts without the variants (Mixtral, Qwen, synthetic) return
    None from the optional accessors; the keys are simply absent
    from the returned dict, so `decoder_decode_step` falls back to
    the shared `pre_ffn` / `post_ffn` norm structure (no per-branch
    norms).
    """
    path = Path(path)
    weight_map = read_safetensors_index(path)

    def _load(name: str):
        return load_tensor(path / weight_map[name], name)

    norms = {
        "input": _load(layout.input_layernorm(layer_id)),
        "post_attn": _load(layout.post_attention_layernorm(layer_id)),
        "pre_ffn": _load(layout.pre_feedforward_layernorm(layer_id)),
        "post_ffn": _load(layout.post_feedforward_layernorm(layer_id)),
    }

    # Step 26B parallel-branch variants. Each accessor returns None
    # for layouts that don't ship the tensor; in that case we omit
    # the key rather than carrying a None through (the decoder uses
    # `if k in norms` to decide which architecture to run).
    pre_2_name = layout.pre_feedforward_layernorm_2(layer_id)
    if pre_2_name is not None:
        norms["pre_ffn_2"] = _load(pre_2_name)
    post_1_name = layout.post_feedforward_layernorm_1(layer_id)
    if post_1_name is not None:
        norms["post_ffn_1"] = _load(post_1_name)
    post_2_name = layout.post_feedforward_layernorm_2(layer_id)
    if post_2_name is not None:
        norms["post_ffn_2"] = _load(post_2_name)

    # Step 26B: per-layer scalar (Gemma 4 ships shape-(1,) bf16 per
    # layer; reference applies it as a multiplier on the layer's
    # final output). Stored on the same dict for plumbing convenience.
    scalar_name = layout.layer_scalar(layer_id)
    if scalar_name is not None:
        norms["layer_scalar"] = _load(scalar_name)

    return norms


def load_attention_weights(path, layout, layer_id: int, *, k_eq_v: bool = False) -> dict:
    """Load + dequantize layer `layer_id`'s q/k/v/o + q_norm/k_norm.

    Returns a dict with keys {q_weight, k_weight, v_weight, o_weight,
    q_norm, k_norm}. Each *_weight is fp32 from NVFP4_U32 dequant
    (mlx-community lowercase Gemma layout); each *_norm is fp32 from
    BF16. Mirrors `apex.real_router.MultiLayerRealRouter._load_router_weight`
    (the same `QuantizedTensor` + `dequantize` pattern, just over four
    projections instead of one).

    `k_eq_v=True` (step 23) is for Gemma 4 full_attention layers where
    V is tied to K and there is no `v_proj` on disk. The returned dict
    sets `v_weight=None` — `attention_decode_step` interprets that as
    "use `k_weight` for V too" (V comes from the un-normed, pre-RoPE K
    projection, then gets a plain RMS norm when a k_norm is loaded).
    Sliding-attention layers always pass
    `k_eq_v=False` (the default) since they do ship a v_proj.
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
        "v_weight": (
            None if k_eq_v else _load_proj(layout.attn_v_proj_tensors(layer_id))
        ),
        "o_weight": _load_proj(layout.attn_o_proj_tensors(layer_id)),
        "q_norm": _load(layout.attn_q_norm(layer_id)),
        "k_norm": _load(layout.attn_k_norm(layer_id)),
    }


def load_dense_mlp_weights(path, layout, layer_id: int) -> dict | None:
    """Load + dequantize layer `layer_id`'s dense MLP gate/up/down projections.

    Returns `{gate_weight, up_weight, down_weight}` (each fp32,
    dequantized) when the layout exposes a dense MLP, or `None` when it
    doesn't (e.g. synthetic / Mixtral / Qwen layouts that only have
    routed experts).

    Two on-disk encodings are supported, picked per projection from
    what the checkpoint actually contains — mlx-community has shipped
    both under the same repo name:
      - AFFINE_UINT8_U32 (weight + bf16 scales + bf16 biases), the
        original upload;
      - NVFP4_U32 (weight + E4M3 scales, no biases), the 2026-07
        re-upload — same encoding as the attention projections.

    On Gemma 4 the dense MLP runs PARALLEL to the routed MoE inside a
    single decoder layer: the dense branch is
    `pre_ffn → dense MLP → post_ffn_1`, the routed branch has its own
    `pre_ffn_2` / `post_ffn_2` (and routes on the raw hidden), and the
    two outputs are summed before the shared `post_ffn` and the
    residual add (see `apex.decoder`). Loading the dense MLP closes a
    real capacity gap that step 23 left open.

    Memory note: per layer this is ~71 MB at hidden=2816 / intermediate=2112
    (3 fp32 matrices). Across 30 layers that's ~2.1 GB of NON-EVICTABLE
    weights — operators running deep models need to budget for this on
    top of the LRU expert cache budget.
    """
    gate_names = layout.dense_mlp_gate_proj(layer_id)
    if gate_names is None:
        return None
    up_names = layout.dense_mlp_up_proj(layer_id)
    down_names = layout.dense_mlp_down_proj(layer_id)
    if up_names is None or down_names is None:
        # Defensive: if the layout exposes one of the three but not the
        # others, refuse rather than silently running with partial weights.
        raise ValueError(
            f"layout exposes some but not all dense MLP projections for "
            f"layer {layer_id}; expected gate + up + down or none"
        )

    path = Path(path)
    weight_map = read_safetensors_index(path)

    def _load(name: str):
        return load_tensor(path / weight_map[name], name)

    def _load_proj(names: dict):
        # The encoding belongs to the checkpoint, not the layout. MLX's
        # affine mode is the only one that writes a `.biases` sidecar
        # (NVFP4 has scales only), so its presence tells the two apart
        # without consulting config.json's per-module quantization map.
        data = _load(names["weight"])
        if names["biases"] in weight_map:
            tensor = QuantizedTensor(
                data=data,
                dtype="AFFINE_UINT8_U32",
                shape=tuple(data.shape),
                weight_scale=_load(names["scales"]),
                weight_scale_2=_load(names["biases"]),
            )
        else:
            tensor = QuantizedTensor(
                data=data,
                dtype="NVFP4_U32",
                shape=tuple(data.shape),
                weight_scale=_load(names["scales"]),
            )
        return dequantize(tensor, name=names["weight"])

    return {
        "gate_weight": _load_proj(gate_names),
        "up_weight": _load_proj(up_names),
        "down_weight": _load_proj(down_names),
    }
