"""Tests for the layout adapters (FakeMoELayout + real-model layouts).

These tests verify the *strings* each layout emits — not that any
particular string actually matches a real downloaded checkpoint.
End-to-end checkpoint compatibility is a separate concern, manual via
scripts/inspect_weight_names.py once a real download is at hand.
This distinction matters most for `GemmaMoELayout.gate_weight()`,
which is a best-guess string locked in as a contract here without
any claim that it's been validated against a real Gemma 4 checkpoint.
"""

import pytest

from apex.weight_layout import (
    FakeMoELayout,
    GemmaHFMoELayout,
    GemmaMoELayout,
    MixtralLayout,
    QwenMoELayout,
)


# --- FakeMoELayout ---------------------------------------------------------

def test_fake_gate_weight_format():
    layout = FakeMoELayout()
    assert layout.gate_weight(0) == "layers.0.gate.weight"
    assert layout.gate_weight(12) == "layers.12.gate.weight"


def test_fake_expert_projection_formats():
    layout = FakeMoELayout()
    assert layout.expert_gate_proj(0, 1) == "layers.0.experts.1.gate_proj.weight"
    assert layout.expert_up_proj(0, 1) == "layers.0.experts.1.up_proj.weight"
    assert layout.expert_down_proj(0, 1) == "layers.0.experts.1.down_proj.weight"


def test_fake_expert_tensors_returns_all_three_projections():
    # The bundle helper is what stores will call — it must return all
    # three projection names keyed by canonical short names.
    layout = FakeMoELayout()
    assert layout.expert_tensors(0, 1) == {
        "gate_proj": "layers.0.experts.1.gate_proj.weight",
        "up_proj": "layers.0.experts.1.up_proj.weight",
        "down_proj": "layers.0.experts.1.down_proj.weight",
    }


def test_fake_expert_and_gate_names_are_distinct():
    # Sanity: gate name shouldn't collide with any of the per-expert
    # projection names. Distinct namespaces inside the same file.
    layout = FakeMoELayout()
    gate = layout.gate_weight(0)
    for expert_id in range(10):
        assert gate not in layout.expert_tensors(0, expert_id).values()


def test_fake_idempotent():
    layout = FakeMoELayout()
    assert layout.expert_gate_proj(2, 3) == layout.expert_gate_proj(2, 3)
    assert layout.gate_weight(5) == layout.gate_weight(5)
    assert layout.expert_tensors(2, 3) == layout.expert_tensors(2, 3)


# --- MixtralLayout ---------------------------------------------------------
# w1 = gate_proj, w2 = down_proj, w3 = up_proj. Get this wrong and the
# math runs but produces silently incorrect output, so it's worth
# pinning explicitly in tests.

def test_mixtral_gate_weight_format():
    assert (
        MixtralLayout().gate_weight(0)
        == "model.layers.0.block_sparse_moe.gate.weight"
    )


def test_mixtral_expert_projection_formats():
    layout = MixtralLayout()
    base = "model.layers.0.block_sparse_moe.experts.1"
    assert layout.expert_gate_proj(0, 1) == f"{base}.w1.weight"
    assert layout.expert_up_proj(0, 1) == f"{base}.w3.weight"
    assert layout.expert_down_proj(0, 1) == f"{base}.w2.weight"


def test_mixtral_expert_tensors_keys_are_canonical():
    # Even though the underlying names are w1/w2/w3, the bundle keys
    # stay the canonical "gate_proj"/"up_proj"/"down_proj" — that's
    # the contract every store relies on.
    names = MixtralLayout().expert_tensors(0, 1)
    assert set(names) == {"gate_proj", "up_proj", "down_proj"}
    assert names["gate_proj"].endswith(".w1.weight")
    assert names["up_proj"].endswith(".w3.weight")
    assert names["down_proj"].endswith(".w2.weight")


# --- QwenMoELayout ---------------------------------------------------------

def test_qwen_gate_weight_format():
    assert QwenMoELayout().gate_weight(0) == "model.layers.0.mlp.gate.weight"


def test_qwen_expert_projection_formats():
    layout = QwenMoELayout()
    base = "model.layers.0.mlp.experts.1"
    assert layout.expert_gate_proj(0, 1) == f"{base}.gate_proj.weight"
    assert layout.expert_up_proj(0, 1) == f"{base}.up_proj.weight"
    assert layout.expert_down_proj(0, 1) == f"{base}.down_proj.weight"


def test_qwen_expert_tensors_returns_canonical_keys():
    names = QwenMoELayout().expert_tensors(0, 1)
    assert set(names) == {"gate_proj", "up_proj", "down_proj"}


# --- GemmaMoELayout --------------------------------------------------------
# The `model.language_model.` prefix and the missing `mlp` /
# `block_sparse_moe` middle prefix are the two notable shape changes
# vs Mixtral / Qwen. gate_weight is verified against the NVIDIA NVFP4
# checkpoint (`router.proj.weight`, not `gate.weight`).

def test_gemma_gate_weight_format():
    # Verified against the real nvidia/Gemma-4-26B-A4B-NVFP4 safetensors
    # index via scripts/validate_layout.py. The name "router.proj.weight"
    # is intentional (Gemma 4 uses 'router' rather than 'gate').
    assert (
        GemmaMoELayout().gate_weight(0)
        == "model.language_model.layers.0.router.proj.weight"
    )


def test_gemma_expert_projection_formats():
    layout = GemmaMoELayout()
    base = "model.language_model.layers.0.experts.1"
    assert layout.expert_gate_proj(0, 1) == f"{base}.gate_proj.weight"
    assert layout.expert_up_proj(0, 1) == f"{base}.up_proj.weight"
    assert layout.expert_down_proj(0, 1) == f"{base}.down_proj.weight"


def test_gemma_no_intermediate_mlp_prefix():
    # Unlike Mixtral / Qwen, Gemma 4 hangs experts directly off the
    # layer — no "mlp." or "block_sparse_moe." in between. Pin that.
    name = GemmaMoELayout().expert_gate_proj(5, 7)
    assert ".mlp." not in name
    assert ".block_sparse_moe." not in name
    assert "model.language_model.layers.5.experts.7." in name


def test_gemma_expert_tensors_returns_canonical_keys():
    names = GemmaMoELayout().expert_tensors(0, 1)
    assert set(names) == {"gate_proj", "up_proj", "down_proj"}


# --- GemmaHFMoELayout ------------------------------------------------------

def test_gemma_hf_gate_weight_format():
    assert (
        GemmaHFMoELayout().gate_weight(0)
        == "language_model.model.layers.0.router.proj.weight"
    )


def test_gemma_hf_stacked_expert_projection_formats():
    layout = GemmaHFMoELayout()
    base = "language_model.model.layers.0.experts.switch_glu"
    assert layout.expert_gate_proj(0, 7) == f"{base}.gate_proj.weight"
    assert layout.expert_up_proj(0, 7) == f"{base}.up_proj.weight"
    assert layout.expert_down_proj(0, 7) == f"{base}.down_proj.weight"


def test_gemma_hf_router_sidecar_formats():
    assert GemmaHFMoELayout().router_tensors(3) == {
        "weight": "language_model.model.layers.3.router.proj.weight",
        "scales": "language_model.model.layers.3.router.proj.scales",
        "biases": "language_model.model.layers.3.router.proj.biases",
    }


def test_gemma_hf_marks_stacked_experts():
    layout = GemmaHFMoELayout()
    assert layout.stacked_experts is True
    assert layout.num_experts == 128
    assert layout.expert_pack_factor == 8
    assert layout.router_pack_factor == 4


# --- Cross-layout sanity ---------------------------------------------------

@pytest.mark.parametrize(
    "layout_cls",
    [FakeMoELayout, MixtralLayout, QwenMoELayout, GemmaMoELayout, GemmaHFMoELayout],
)
def test_all_layouts_share_expert_tensors_contract(layout_cls):
    # Every layout must return the same three keys from expert_tensors.
    # Stores depend on this.
    layout = layout_cls()
    names = layout.expert_tensors(0, 1)
    assert set(names) == {"gate_proj", "up_proj", "down_proj"}
    # And each value must be the same string the corresponding
    # individual method returned.
    assert names["gate_proj"] == layout.expert_gate_proj(0, 1)
    assert names["up_proj"] == layout.expert_up_proj(0, 1)
    assert names["down_proj"] == layout.expert_down_proj(0, 1)


# --- Optional non-MoE methods ---------------------------------------------
# These are present on layouts that drive a *full* decoder. Verified
# against literal tensor names in the local Gemma 4 26B-A4B-IT NVFP4
# safetensors index (mlx-community lowercase variant) so the GemmaHF
# expectations are not aspirational. FakeMoELayout names are
# synthetic-only and exercise the contract without touching disk.

def test_gemma_hf_attn_q_proj_includes_scales_sidecar():
    layout = GemmaHFMoELayout()
    assert layout.attn_q_proj_tensors(0) == {
        "weight": "language_model.model.layers.0.self_attn.q_proj.weight",
        "scales": "language_model.model.layers.0.self_attn.q_proj.scales",
    }


def test_gemma_hf_attn_kvo_proj_format():
    layout = GemmaHFMoELayout()
    base = "language_model.model.layers.5.self_attn"
    assert layout.attn_k_proj_tensors(5) == {
        "weight": f"{base}.k_proj.weight", "scales": f"{base}.k_proj.scales",
    }
    assert layout.attn_v_proj_tensors(5) == {
        "weight": f"{base}.v_proj.weight", "scales": f"{base}.v_proj.scales",
    }
    assert layout.attn_o_proj_tensors(5) == {
        "weight": f"{base}.o_proj.weight", "scales": f"{base}.o_proj.scales",
    }


def test_gemma_hf_attn_qk_norm_format():
    layout = GemmaHFMoELayout()
    base = "language_model.model.layers.7.self_attn"
    assert layout.attn_q_norm(7) == f"{base}.q_norm.weight"
    assert layout.attn_k_norm(7) == f"{base}.k_norm.weight"


def test_gemma_hf_layernorm_formats():
    layout = GemmaHFMoELayout()
    assert (
        layout.input_layernorm(0)
        == "language_model.model.layers.0.input_layernorm.weight"
    )
    assert (
        layout.post_attention_layernorm(29)
        == "language_model.model.layers.29.post_attention_layernorm.weight"
    )
    assert (
        layout.pre_feedforward_layernorm(0)
        == "language_model.model.layers.0.pre_feedforward_layernorm.weight"
    )
    assert (
        layout.post_feedforward_layernorm(11)
        == "language_model.model.layers.11.post_feedforward_layernorm.weight"
    )


def test_gemma_hf_embedding_and_final_norm():
    layout = GemmaHFMoELayout()
    assert layout.embedding_tensors() == {
        "weight": "language_model.model.embed_tokens.weight",
        "scales": "language_model.model.embed_tokens.scales",
    }
    assert layout.final_norm() == "language_model.model.norm.weight"


def test_gemma_hf_lm_head_is_tied_to_embeddings():
    # tie_word_embeddings=True in the local checkpoint config; there is
    # literally no lm_head.weight on disk. None signals callers to
    # reuse the embedding tensor transposed.
    assert GemmaHFMoELayout().lm_head_tensors() is None


def test_gemma_attn_q_proj_uses_nvidia_prefix():
    # NVIDIA NVFP4 variant carries the `model.language_model.` prefix
    # and uses `weight_scale` (matching MoE-expert sidecars), not
    # `scales`. Names taken from the model card; not validated locally
    # but pinned so the next person who downloads it has a starting
    # point that follows the established contract.
    layout = GemmaMoELayout()
    assert layout.attn_q_proj_tensors(0) == {
        "weight": "model.language_model.layers.0.self_attn.q_proj.weight",
        "scales": "model.language_model.layers.0.self_attn.q_proj.weight_scale",
    }


def test_gemma_lm_head_is_tied_to_embeddings():
    # tie_word_embeddings=True for the published Gemma 4 weights too.
    assert GemmaMoELayout().lm_head_tensors() is None


def test_fake_layout_attn_names_are_synthetic():
    # Exercising the contract surface — no real safetensors involved.
    layout = FakeMoELayout()
    assert layout.attn_q_proj_tensors(0) == {"weight": "layers.0.attn.q_proj.weight"}
    assert layout.attn_k_proj_tensors(2) == {"weight": "layers.2.attn.k_proj.weight"}
    assert layout.attn_v_proj_tensors(0) == {"weight": "layers.0.attn.v_proj.weight"}
    assert layout.attn_o_proj_tensors(0) == {"weight": "layers.0.attn.o_proj.weight"}


def test_fake_layout_norms_and_embeddings():
    layout = FakeMoELayout()
    assert layout.input_layernorm(3) == "layers.3.input_layernorm.weight"
    assert (
        layout.post_attention_layernorm(0)
        == "layers.0.post_attention_layernorm.weight"
    )
    assert (
        layout.pre_feedforward_layernorm(0)
        == "layers.0.pre_feedforward_layernorm.weight"
    )
    assert (
        layout.post_feedforward_layernorm(2)
        == "layers.2.post_feedforward_layernorm.weight"
    )
    assert layout.attn_q_norm(0) == "layers.0.attn.q_norm.weight"
    assert layout.attn_k_norm(0) == "layers.0.attn.k_norm.weight"
    assert layout.embedding_tensors() == {"weight": "embed_tokens.weight"}
    assert layout.final_norm() == "final_norm.weight"
    # FakeMoELayout deliberately exposes a real lm_head (no tying) so
    # the non-Gemma codepath gets exercised somewhere.
    assert layout.lm_head_tensors() == {"weight": "lm_head.weight"}


def test_mixtral_does_not_implement_non_moe_methods():
    # Sanity: layouts we haven't validated end-to-end shouldn't make
    # up names. validate_layout.py uses hasattr() to skip these.
    layout = MixtralLayout()
    assert not hasattr(layout, "attn_q_proj_tensors")
    assert not hasattr(layout, "input_layernorm")
    assert not hasattr(layout, "embedding_tensors")
    assert not hasattr(layout, "lm_head_tensors")


def test_qwen_does_not_implement_non_moe_methods():
    layout = QwenMoELayout()
    assert not hasattr(layout, "attn_q_proj_tensors")
    assert not hasattr(layout, "input_layernorm")
    assert not hasattr(layout, "embedding_tensors")
    assert not hasattr(layout, "lm_head_tensors")
