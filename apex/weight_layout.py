"""Tensor-name conventions for MoE checkpoints.

A layout is the seam between APEX's logical names ("gate_proj" /
"up_proj" / "down_proj") and the actual tensor name strings inside a
specific checkpoint format. The store adapter
(`SafetensorsExpertStore`, `GGUFExpertStore`, ...) asks the layout
what string to look up; the layout returns it. Same store code,
different model, different file — only the layout class changes.

Every layout implements the MoE-core five (gate + per-expert
projections + bundle helper). Layouts that drive a *full* decoder
(attention, norms, embeddings, lm_head) optionally implement the
non-MoE methods listed below — these are intentionally optional so
adapters that only care about MoE wiring (Mixtral, Qwen) don't have to
fabricate names they've never validated.

MoE core (every layout):

  - gate_weight(layer_id)                  -> str
  - expert_gate_proj(layer_id, expert_id)  -> str
  - expert_up_proj(layer_id, expert_id)    -> str
  - expert_down_proj(layer_id, expert_id)  -> str
  - expert_tensors(layer_id, expert_id)    -> {"gate_proj": str,
                                                "up_proj":   str,
                                                "down_proj": str}

Optional non-MoE methods (implemented by layouts that drive full
decoder layers — currently FakeMoELayout, GemmaMoELayout,
GemmaHFMoELayout):

  - attn_q_proj_tensors(layer_id) -> {"weight": str, "scales"?: str}
  - attn_k_proj_tensors(layer_id) -> {"weight": str, "scales"?: str}
  - attn_v_proj_tensors(layer_id) -> {"weight": str, "scales"?: str}
  - attn_o_proj_tensors(layer_id) -> {"weight": str, "scales"?: str}
  - attn_q_norm(layer_id)         -> str
  - attn_k_norm(layer_id)         -> str
  - input_layernorm(layer_id)     -> str
  - post_attention_layernorm(layer_id) -> str
  - pre_feedforward_layernorm(layer_id)  -> str
  - post_feedforward_layernorm(layer_id) -> str
  - embedding_tensors()           -> {"weight": str, "scales"?: str}
  - final_norm()                  -> str
  - lm_head_tensors()             -> dict | None  (None ⇒ tied to embeddings)

Quantized-weight projections (NVFP4_U32 family) return both the base
`weight` name and a `scales` sidecar name; pure BF16 single-tensor
norms return a plain `str`. Callers that need to know whether to
expect a sidecar should test `"scales" in tensors` rather than
introspect the dtype.

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

    # --- Optional step-24 router stabilizers ---------------------------
    # Default to None so layouts that don't carry these tensors (every
    # synthetic / Mixtral / Qwen layout) keep working untouched. Real
    # Gemma 4 overrides both in GemmaHFMoELayout.

    def router_scale(self, layer_id: int) -> str | None:
        """Tensor name for `router.scale` (per-feature, shape (hidden,))."""
        return None

    def router_per_expert_scale(self, layer_id: int) -> str | None:
        """Tensor name for `router.per_expert_scale` (shape (num_experts,))."""
        return None

    # --- Optional step-24B dense MLP (parallel to routed experts) -------
    # Returns a `{weight, scales, biases}` dict per projection — same
    # shape contract as `router_tensors` so the loader can reuse the
    # existing AFFINE_UINT8_U32 dequant pipeline. Default None for
    # layouts that don't carry a dense MLP branch.

    def dense_mlp_gate_proj(self, layer_id: int) -> dict | None:
        return None

    def dense_mlp_up_proj(self, layer_id: int) -> dict | None:
        return None

    def dense_mlp_down_proj(self, layer_id: int) -> dict | None:
        return None

    # --- Optional step-26B parallel-branch norm variants ---------------
    # Real Gemma 4 ships TWO pre-feedforward norms and THREE post-
    # feedforward norms per layer. Mapping (per `mlx_lm.models.gemma4_text`
    # `DecoderLayer.__call__`, lines 353-369):
    #   - `pre_feedforward_layernorm`     → DENSE-branch input norm
    #   - `pre_feedforward_layernorm_2`   → ROUTED-branch input norm
    #   - `post_feedforward_layernorm_1`  → DENSE-branch output norm
    #   - `post_feedforward_layernorm_2`  → ROUTED-branch output norm
    #   - `post_feedforward_layernorm`    → output norm on the summed
    #                                       branches (before residual)
    # Default None so synthetic / Mixtral / Qwen layouts (no parallel
    # branches) skip them and keep the pre-step-26B routed-only
    # decoder behavior. Real Gemma layouts override.

    def pre_feedforward_layernorm_2(self, layer_id: int) -> str | None:
        return None

    def post_feedforward_layernorm_1(self, layer_id: int) -> str | None:
        return None

    def post_feedforward_layernorm_2(self, layer_id: int) -> str | None:
        return None

    # --- Optional step-26B per-layer scalar -----------------------------
    # Gemma 4 ships a shape-(1,) bf16 `layer_scalar` per decoder layer.
    # The reference (`mlx_lm.models.gemma4_text:DecoderLayer.__call__`
    # lines 386-387) multiplies the layer's final output by this scalar
    # AFTER the FFN-block residual add. Default None for layouts that
    # don't carry it.
    def layer_scalar(self, layer_id: int) -> str | None:
        return None


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

    # --- Optional non-MoE methods --------------------------------------
    # FakeMoELayout's synthetic checkpoints don't actually contain
    # these tensors; the names exist so layout-shape tests can exercise
    # the contract without having to depend on a real checkpoint.

    def attn_q_proj_tensors(self, layer_id: int) -> dict:
        return {"weight": f"layers.{layer_id}.attn.q_proj.weight"}

    def attn_k_proj_tensors(self, layer_id: int) -> dict:
        return {"weight": f"layers.{layer_id}.attn.k_proj.weight"}

    def attn_v_proj_tensors(self, layer_id: int) -> dict:
        return {"weight": f"layers.{layer_id}.attn.v_proj.weight"}

    def attn_o_proj_tensors(self, layer_id: int) -> dict:
        return {"weight": f"layers.{layer_id}.attn.o_proj.weight"}

    def attn_q_norm(self, layer_id: int) -> str:
        return f"layers.{layer_id}.attn.q_norm.weight"

    def attn_k_norm(self, layer_id: int) -> str:
        return f"layers.{layer_id}.attn.k_norm.weight"

    def input_layernorm(self, layer_id: int) -> str:
        return f"layers.{layer_id}.input_layernorm.weight"

    def post_attention_layernorm(self, layer_id: int) -> str:
        return f"layers.{layer_id}.post_attention_layernorm.weight"

    def pre_feedforward_layernorm(self, layer_id: int) -> str:
        return f"layers.{layer_id}.pre_feedforward_layernorm.weight"

    def post_feedforward_layernorm(self, layer_id: int) -> str:
        return f"layers.{layer_id}.post_feedforward_layernorm.weight"

    def embedding_tensors(self) -> dict:
        return {"weight": "embed_tokens.weight"}

    def final_norm(self) -> str:
        return "final_norm.weight"

    def lm_head_tensors(self) -> dict:
        # Synthetic fake doesn't tie embeddings — every name distinct
        # so tests that rely on lm_head being its own tensor still work.
        return {"weight": "lm_head.weight"}


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

    # --- Optional non-MoE methods --------------------------------------
    # Names follow the NVIDIA-style `model.language_model.` prefix.
    # Quantized projections expect a `.scales` sidecar in the NVFP4
    # variant (and a `.weight_scale_2`/`.input_scale` pair, but the
    # store layer queries the base name + reads sidecars by suffix
    # already — we surface only the two the layout name layer cares
    # about, matching how MoE projections work today).

    def attn_q_proj_tensors(self, layer_id: int) -> dict:
        prefix = f"model.language_model.layers.{layer_id}.self_attn.q_proj"
        return {"weight": f"{prefix}.weight", "scales": f"{prefix}.weight_scale"}

    def attn_k_proj_tensors(self, layer_id: int) -> dict:
        prefix = f"model.language_model.layers.{layer_id}.self_attn.k_proj"
        return {"weight": f"{prefix}.weight", "scales": f"{prefix}.weight_scale"}

    def attn_v_proj_tensors(self, layer_id: int) -> dict:
        prefix = f"model.language_model.layers.{layer_id}.self_attn.v_proj"
        return {"weight": f"{prefix}.weight", "scales": f"{prefix}.weight_scale"}

    def attn_o_proj_tensors(self, layer_id: int) -> dict:
        prefix = f"model.language_model.layers.{layer_id}.self_attn.o_proj"
        return {"weight": f"{prefix}.weight", "scales": f"{prefix}.weight_scale"}

    def attn_q_norm(self, layer_id: int) -> str:
        return f"model.language_model.layers.{layer_id}.self_attn.q_norm.weight"

    def attn_k_norm(self, layer_id: int) -> str:
        return f"model.language_model.layers.{layer_id}.self_attn.k_norm.weight"

    def input_layernorm(self, layer_id: int) -> str:
        return f"model.language_model.layers.{layer_id}.input_layernorm.weight"

    def post_attention_layernorm(self, layer_id: int) -> str:
        return f"model.language_model.layers.{layer_id}.post_attention_layernorm.weight"

    def pre_feedforward_layernorm(self, layer_id: int) -> str:
        return f"model.language_model.layers.{layer_id}.pre_feedforward_layernorm.weight"

    def post_feedforward_layernorm(self, layer_id: int) -> str:
        return f"model.language_model.layers.{layer_id}.post_feedforward_layernorm.weight"

    # Step 26B: parallel-branch norm variants. Same Gemma 4 weights
    # as GemmaHFMoELayout, just under a different name prefix.
    def pre_feedforward_layernorm_2(self, layer_id: int) -> str:
        return f"model.language_model.layers.{layer_id}.pre_feedforward_layernorm_2.weight"

    def post_feedforward_layernorm_1(self, layer_id: int) -> str:
        return f"model.language_model.layers.{layer_id}.post_feedforward_layernorm_1.weight"

    def post_feedforward_layernorm_2(self, layer_id: int) -> str:
        return f"model.language_model.layers.{layer_id}.post_feedforward_layernorm_2.weight"

    def layer_scalar(self, layer_id: int) -> str:
        return f"model.language_model.layers.{layer_id}.layer_scalar"

    def embedding_tensors(self) -> dict:
        prefix = "model.language_model.embed_tokens"
        return {"weight": f"{prefix}.weight", "scales": f"{prefix}.weight_scale"}

    def final_norm(self) -> str:
        return "model.language_model.norm.weight"

    def lm_head_tensors(self):
        # Gemma 4 ties lm_head to embed_tokens (tie_word_embeddings=True
        # in the model config). Callers reuse the embedding weight
        # transposed instead of looking up a separate tensor.
        return None


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

    def router_scale(self, layer_id: int) -> str:
        # bf16 vector of shape (hidden_size,). Multiplied element-wise
        # into hidden BEFORE the router projection (step 24A).
        return f"language_model.model.layers.{layer_id}.router.scale"

    def router_per_expert_scale(self, layer_id: int) -> str:
        # bf16 vector of shape (num_experts,). Multiplied element-wise
        # into projected logits BEFORE top-k (step 24A).
        return f"language_model.model.layers.{layer_id}.router.per_expert_scale"

    def dense_mlp_gate_proj(self, layer_id: int) -> dict:
        prefix = f"language_model.model.layers.{layer_id}.mlp.gate_proj"
        return {
            "weight": f"{prefix}.weight",
            "scales": f"{prefix}.scales",
            "biases": f"{prefix}.biases",
        }

    def dense_mlp_up_proj(self, layer_id: int) -> dict:
        prefix = f"language_model.model.layers.{layer_id}.mlp.up_proj"
        return {
            "weight": f"{prefix}.weight",
            "scales": f"{prefix}.scales",
            "biases": f"{prefix}.biases",
        }

    def dense_mlp_down_proj(self, layer_id: int) -> dict:
        prefix = f"language_model.model.layers.{layer_id}.mlp.down_proj"
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

    # --- Optional non-MoE methods --------------------------------------
    # Verified against the local mlx-community
    # gemma-4-26b-a4b-it-nvfp4 safetensors index. NVFP4_U32 quantized
    # projections pair `.weight` with a `.scales` sidecar; norms are
    # plain BF16 single tensors.

    def attn_q_proj_tensors(self, layer_id: int) -> dict:
        prefix = f"language_model.model.layers.{layer_id}.self_attn.q_proj"
        return {"weight": f"{prefix}.weight", "scales": f"{prefix}.scales"}

    def attn_k_proj_tensors(self, layer_id: int) -> dict:
        prefix = f"language_model.model.layers.{layer_id}.self_attn.k_proj"
        return {"weight": f"{prefix}.weight", "scales": f"{prefix}.scales"}

    def attn_v_proj_tensors(self, layer_id: int) -> dict:
        prefix = f"language_model.model.layers.{layer_id}.self_attn.v_proj"
        return {"weight": f"{prefix}.weight", "scales": f"{prefix}.scales"}

    def attn_o_proj_tensors(self, layer_id: int) -> dict:
        prefix = f"language_model.model.layers.{layer_id}.self_attn.o_proj"
        return {"weight": f"{prefix}.weight", "scales": f"{prefix}.scales"}

    def attn_q_norm(self, layer_id: int) -> str:
        return f"language_model.model.layers.{layer_id}.self_attn.q_norm.weight"

    def attn_k_norm(self, layer_id: int) -> str:
        return f"language_model.model.layers.{layer_id}.self_attn.k_norm.weight"

    def input_layernorm(self, layer_id: int) -> str:
        return f"language_model.model.layers.{layer_id}.input_layernorm.weight"

    def post_attention_layernorm(self, layer_id: int) -> str:
        return f"language_model.model.layers.{layer_id}.post_attention_layernorm.weight"

    def pre_feedforward_layernorm(self, layer_id: int) -> str:
        return f"language_model.model.layers.{layer_id}.pre_feedforward_layernorm.weight"

    def post_feedforward_layernorm(self, layer_id: int) -> str:
        return f"language_model.model.layers.{layer_id}.post_feedforward_layernorm.weight"

    # Step 26B: parallel-branch norm variants. Mapping per the
    # `mlx_lm/models/gemma4_text.py:DecoderLayer.__call__` reference.
    # Dense branch uses the unsuffixed pre + `_1` post; routed
    # branch uses `_2` for both; the unsuffixed post is the combined
    # output norm.
    def pre_feedforward_layernorm_2(self, layer_id: int) -> str:
        return f"language_model.model.layers.{layer_id}.pre_feedforward_layernorm_2.weight"

    def post_feedforward_layernorm_1(self, layer_id: int) -> str:
        return f"language_model.model.layers.{layer_id}.post_feedforward_layernorm_1.weight"

    def post_feedforward_layernorm_2(self, layer_id: int) -> str:
        return f"language_model.model.layers.{layer_id}.post_feedforward_layernorm_2.weight"

    def layer_scalar(self, layer_id: int) -> str:
        return f"language_model.model.layers.{layer_id}.layer_scalar"

    def embedding_tensors(self) -> dict:
        prefix = "language_model.model.embed_tokens"
        return {"weight": f"{prefix}.weight", "scales": f"{prefix}.scales"}

    def final_norm(self) -> str:
        return "language_model.model.norm.weight"

    def lm_head_tensors(self):
        # tie_word_embeddings=True in this checkpoint's config — there
        # is no lm_head.weight on disk; callers reuse embed_tokens.
        return None


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
