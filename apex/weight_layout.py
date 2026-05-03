"""Tensor-name conventions for MoE checkpoints.

A layout is the seam between APEX's logical names ("gate_proj" /
"up_proj" / "down_proj") and the actual tensor name strings inside a
specific checkpoint format. The store adapter
(`SafetensorsExpertStore`, `GGUFExpertStore`, ...) asks the layout
what string to look up; the layout returns it. Same store code,
different model, different file — only the layout class changes.

Each layout implements five methods:

  - gate_weight(layer_id)                  -> str
  - expert_gate_proj(layer_id, expert_id)  -> str
  - expert_up_proj(layer_id, expert_id)    -> str
  - expert_down_proj(layer_id, expert_id)  -> str
  - expert_tensors(layer_id, expert_id)    -> {"gate_proj": str,
                                                "up_proj":   str,
                                                "down_proj": str}

The bundle method is shared via `_MoELayoutBase` — every layout
returns the same three keys (only the values change), so the store
code can build an `ExpertWeights(**layout.expert_tensors(...))`-style
bundle uniformly across models.

Layouts shipped here:
  - FakeMoELayout    APEX's own fake checkpoints (used by demos / tests)
  - MixtralLayout    mistralai/Mixtral-8x7B-v0.1
  - QwenMoELayout    Qwen/Qwen1.5-MoE-A2.7B
  - GemmaMoELayout   NVIDIA-style Gemma 4 NVFP4 tensor names
  - GemmaHFMoELayout mlx-community lowercase Gemma 4 checkpoint names

Use `get_layout(name)` to look one up by short name.
"""


class _MoELayoutBase:
    """Shared `expert_tensors` so each subclass only writes 4 string methods.

    Not exported. Subclasses must implement `gate_weight`,
    `expert_gate_proj`, `expert_up_proj`, `expert_down_proj`.
    """

    def expert_tensors(self, layer_id: int, expert_id: int) -> dict:
        """Return all three projection tensor names for one expert.

        Stores call this to build an ExpertWeights bundle uniformly:

            names = layout.expert_tensors(layer, expert)
            ExpertWeights(
                gate_proj=load(names["gate_proj"]),
                up_proj=load(names["up_proj"]),
                down_proj=load(names["down_proj"]),
            )

        The keys are stable across layouts; only the values (full
        tensor names) change with the underlying naming scheme.
        """
        return {
            "gate_proj": self.expert_gate_proj(layer_id, expert_id),
            "up_proj": self.expert_up_proj(layer_id, expert_id),
            "down_proj": self.expert_down_proj(layer_id, expert_id),
        }


class FakeMoELayout(_MoELayoutBase):
    """Layout used by APEX's scripts/make_fake_*_moe.py setup scripts.

    Tensor names look like:
      layers.<L>.gate.weight
      layers.<L>.experts.<E>.gate_proj.weight
      layers.<L>.experts.<E>.up_proj.weight
      layers.<L>.experts.<E>.down_proj.weight
    """

    def gate_weight(self, layer_id: int) -> str:
        return f"layers.{layer_id}.gate.weight"

    def expert_gate_proj(self, layer_id: int, expert_id: int) -> str:
        return f"layers.{layer_id}.experts.{expert_id}.gate_proj.weight"

    def expert_up_proj(self, layer_id: int, expert_id: int) -> str:
        return f"layers.{layer_id}.experts.{expert_id}.up_proj.weight"

    def expert_down_proj(self, layer_id: int, expert_id: int) -> str:
        return f"layers.{layer_id}.experts.{expert_id}.down_proj.weight"


class MixtralLayout(_MoELayoutBase):
    """mistralai/Mixtral-8x7B-v0.1 (32 layers x 8 experts, per-expert tensors).

    NOTE: Mixtral's w1/w2/w3 don't follow the gate/up/down ordering
    you might expect from the names alone:
      w1 = gate_proj  (silu input)
      w2 = down_proj  (output projection — yes, the *middle* one)
      w3 = up_proj    (multiplied with silu(gate_proj))
    Get the mapping wrong and the math runs without crashing but
    produces silently incorrect numerical output.

    HF model card: https://huggingface.co/mistralai/Mixtral-8x7B-v0.1
    """

    def gate_weight(self, layer_id: int) -> str:
        return f"model.layers.{layer_id}.block_sparse_moe.gate.weight"

    def expert_gate_proj(self, layer_id: int, expert_id: int) -> str:
        return f"model.layers.{layer_id}.block_sparse_moe.experts.{expert_id}.w1.weight"

    def expert_up_proj(self, layer_id: int, expert_id: int) -> str:
        return f"model.layers.{layer_id}.block_sparse_moe.experts.{expert_id}.w3.weight"

    def expert_down_proj(self, layer_id: int, expert_id: int) -> str:
        return f"model.layers.{layer_id}.block_sparse_moe.experts.{expert_id}.w2.weight"


class QwenMoELayout(_MoELayoutBase):
    """Qwen/Qwen1.5-MoE-A2.7B (24 layers x 60 routed experts + 1 shared/layer).

    Each layer also has `model.layers.{L}.mlp.shared_expert.{proj}.weight`
    (an always-on expert in addition to the routed ones). APEX has no
    shared-expert concept yet, so those tensors are ignored. Adding
    support is a runtime + cache change, not just a naming one — out
    of scope for this layout.

    HF model card: https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B
    """

    def gate_weight(self, layer_id: int) -> str:
        return f"model.layers.{layer_id}.mlp.gate.weight"

    def expert_gate_proj(self, layer_id: int, expert_id: int) -> str:
        return f"model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.weight"

    def expert_up_proj(self, layer_id: int, expert_id: int) -> str:
        return f"model.layers.{layer_id}.mlp.experts.{expert_id}.up_proj.weight"

    def expert_down_proj(self, layer_id: int, expert_id: int) -> str:
        return f"model.layers.{layer_id}.mlp.experts.{expert_id}.down_proj.weight"


class GemmaMoELayout(_MoELayoutBase):
    """google/gemma-4-26B-A4B (30 layers x 128 experts, top-k=8).

    Verified end-to-end against the NVIDIA NVFP4 variant
    (nvidia/Gemma-4-26B-A4B-NVFP4) via scripts/validate_layout.py:
    every name produced here exists in the real safetensors index.

    Notable differences vs Mixtral/Qwen:

      1. The `model.language_model.` prefix exists because
         Gemma4ForConditionalGeneration is a multimodal wrapper
         (language model + vision tower); MoE experts live under the
         language sub-module.
      2. There is no `mlp` or `block_sparse_moe` intermediate prefix —
         experts hang directly off the layer.
      3. The router weight is `router.proj.weight` (not `gate.weight`
         like Qwen/Mixtral). The model also stores `router.scale` and
         `router.per_expert_scale` per layer that the basic
         gate/expert API doesn't surface yet — they're scoring scales,
         not routing weights, and APEX's routers don't consume them.
      4. Each MoE layer ALSO has a dense `mlp.{gate,up,down}_proj` —
         a "shared expert" / always-on FFN that runs alongside the
         routed experts. APEX has no shared-expert concept yet, so
         those tensors are loaded by neither this layout nor any
         store. Adding shared-expert support is a runtime concern.

    The NVFP4-quantized variant adds companion tensors per weight:
      - .input_scale / .weight_scale / .weight_scale_2
    Each `.weight` itself is a packed-int8 tensor, not float32 —
    APEX's existing stores will load it but the contents are not
    SwiGLU-ready until dequantized. Dequantization is out of scope
    for this layout (which only handles tensor naming).

    HF model card: https://huggingface.co/google/gemma-4-26B-A4B-it
    NVFP4 variant:  https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4
    """

    def gate_weight(self, layer_id: int) -> str:
        return f"model.language_model.layers.{layer_id}.router.proj.weight"

    def expert_gate_proj(self, layer_id: int, expert_id: int) -> str:
        return f"model.language_model.layers.{layer_id}.experts.{expert_id}.gate_proj.weight"

    def expert_up_proj(self, layer_id: int, expert_id: int) -> str:
        return f"model.language_model.layers.{layer_id}.experts.{expert_id}.up_proj.weight"

    def expert_down_proj(self, layer_id: int, expert_id: int) -> str:
        return f"model.language_model.layers.{layer_id}.experts.{expert_id}.down_proj.weight"


class GemmaHFMoELayout(_MoELayoutBase):
    """mlx-community/gemma-4-26b-a4b-it-nvfp4 stacked expert layout.

    This is deliberately separate from `GemmaMoELayout`: the local
    lowercase checkpoint uses different names and stores every layer's
    routed experts in one stacked `experts.switch_glu.*` tensor rather
    than one tensor per expert. Stores that support this layout must
    slice axis 0 by expert id before dequantizing.
    """

    stacked_experts = True
    num_experts = 128
    expert_pack_factor = 8
    router_pack_factor = 4

    def gate_weight(self, layer_id: int) -> str:
        return f"language_model.model.layers.{layer_id}.router.proj.weight"

    def router_tensors(self, layer_id: int) -> dict:
        prefix = f"language_model.model.layers.{layer_id}.router.proj"
        return {
            "weight": f"{prefix}.weight",
            "scales": f"{prefix}.scales",
            "biases": f"{prefix}.biases",
        }

    def expert_gate_proj(self, layer_id: int, expert_id: int) -> str:
        return f"language_model.model.layers.{layer_id}.experts.switch_glu.gate_proj.weight"

    def expert_up_proj(self, layer_id: int, expert_id: int) -> str:
        return f"language_model.model.layers.{layer_id}.experts.switch_glu.up_proj.weight"

    def expert_down_proj(self, layer_id: int, expert_id: int) -> str:
        return f"language_model.model.layers.{layer_id}.experts.switch_glu.down_proj.weight"


# Registry: short name -> layout class. Private because extension is
# class-level (just import the class), not registration-level.
_LAYOUTS = {
    "fake": FakeMoELayout,
    "gemma-hf-moe": GemmaHFMoELayout,
    "mixtral": MixtralLayout,
    "qwen-moe": QwenMoELayout,
    "gemma-moe": GemmaMoELayout,
}


def get_layout(name: str):
    """Look up a layout class by short name and return a fresh instance.

    Use this to swap layouts in factories or scripts without importing
    the class directly:

        store = SafetensorsExpertStore(path, layout=get_layout("mixtral"))

    Raises KeyError (with the available names in the message) if the
    lookup fails — explicit and actionable, instead of letting None
    propagate and explode somewhere downstream.
    """
    try:
        cls = _LAYOUTS[name]
    except KeyError:
        raise KeyError(
            f"unknown layout: {name!r}; known: {sorted(_LAYOUTS)}"
        )
    return cls()


def available_layouts() -> list:
    """Return all registered layout short names, sorted.

    Used by argparse `choices=...` in scripts so the CLI advertises
    valid names without importing the private registry directly.
    """
    return sorted(_LAYOUTS)
