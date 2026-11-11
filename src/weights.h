/// @file weights.h
/// @brief Resolve every core tensor the forward pass needs into plain structs, once, at load.
/// After this nothing in the per-token path looks a tensor up by name.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include "model_info.h"
#include "ops.h"

namespace apex {

/// Router tensors of one layer (all in core.bin). proj is 8-bit affine, group 64.
struct RouterWeights {
    size_t scale;            // BF16 [hidden]   router.scale
    size_t per_expert_scale; // BF16 [experts]  router.per_expert_scale
    size_t proj_weight;      // U32  [experts, hidden/4]
    size_t proj_scales;      // BF16 [experts, hidden/64]
    size_t proj_biases;      // BF16 [experts, hidden/64]
    uint32_t group_size;     // 64
};

/// Everything one decoder layer reads from core.bin, as byte offsets into the core buffer.
struct LayerWeights {
    AttentionShape shape;
    AttentionWeights attn;
    size_t input_norm;      // input_layernorm
    size_t post_attn_norm;  // post_attention_layernorm
    size_t pre_ffn_norm;    // pre_feedforward_layernorm   (dense MLP input)
    size_t pre_ffn_norm_2;  // pre_feedforward_layernorm_2 (experts input)
    size_t post_ffn_norm;   // post_feedforward_layernorm  (on dense + moe)
    size_t post_ffn_norm_1; // post_feedforward_layernorm_1 (on dense_out)
    size_t post_ffn_norm_2; // post_feedforward_layernorm_2 (on moe_out)
    size_t layer_scalar;    // BF16 [1]
    Nvfp4Matrix mlp_gate, mlp_up, mlp_down;
    RouterWeights router;
};

/// All core weights plus the expert slot layout.
struct ModelWeights {
    Buffer *core = nullptr;
    Nvfp4Matrix embed;  // embed_tokens, also the tied output head [vocab, hidden]
    size_t final_norm;  // norm.weight
    std::vector<LayerWeights> layers;
    ExpertLayout expert_layout; // piece offsets inside one expert slot
};

/// GPU description of an NVFP4 matrix from its .weight and .scales tensors in core.bin.
inline Nvfp4Matrix quantMatrix(const ModelInfo &info, Buffer *core, const std::string &name) {
    const TensorInfo &w = info.getTensor(name + ".weight"); // uint32 [rows, K/8]
    const TensorInfo &s = info.getTensor(name + ".scales"); // uint8  [rows, K/16]
    return Nvfp4Matrix{core, w.offset, core, s.offset, uint32_t(w.shape[0]),
                       uint32_t(w.shape[1] * 8)}; // 8 fp4 codes per uint32
}

/// One layer's attention weights. Full layers have no v_proj.
inline AttentionWeights attentionWeights(const ModelInfo &info, Buffer *core, int layer) {
    const std::string p = "layers." + std::to_string(layer) + ".self_attn.";
    AttentionWeights w;
    w.q = quantMatrix(info, core, p + "q_proj");
    w.k = quantMatrix(info, core, p + "k_proj");
    w.has_v = info.core.count(p + "v_proj.weight") > 0;
    w.v = w.has_v ? quantMatrix(info, core, p + "v_proj") : w.k;
    w.o = quantMatrix(info, core, p + "o_proj");
    w.norms = core;
    w.q_norm_offset = info.getTensor(p + "q_norm.weight").offset;
    w.k_norm_offset = info.getTensor(p + "k_norm.weight").offset;
    return w;
}

/// Expert slot layout from the manifest's piece table.
inline ExpertLayout expertLayout(const ModelInfo &info) {
    auto off = [&](const char *name) {
        auto it = info.expert_pieces.find(name);
        if (it == info.expert_pieces.end())
            throw std::runtime_error(std::string("expert piece missing: ") + name);
        return uint32_t(it->second.offset);
    };
    ExpertLayout l;
    l.stride = uint32_t(info.expert_slot_size);
    l.gate_w = off("gate_proj.weight");
    l.gate_s = off("gate_proj.scales");
    l.up_w = off("up_proj.weight");
    l.up_s = off("up_proj.scales");
    l.down_w = off("down_proj.weight");
    l.down_s = off("down_proj.scales");
    return l;
}

/// Resolve every tensor once. core is the GPU buffer wrapping core.bin.
inline ModelWeights resolveWeights(const ModelInfo &info, Buffer *core) {
    ModelWeights mw;
    mw.core = core;
    mw.embed = quantMatrix(info, core, "embed_tokens");
    mw.final_norm = info.getTensor("norm.weight").offset;
    mw.expert_layout = expertLayout(info);
    const ModelConfig &c = info.config;
    for (int l = 0; l < c.num_hidden_layers; ++l) {
        const std::string p = "layers." + std::to_string(l) + ".";
        auto off = [&](const std::string &name) { return info.getTensor(p + name).offset; };
        LayerWeights w;
        w.shape = attentionShape(c, l);
        w.attn = attentionWeights(info, core, l);
        w.input_norm = off("input_layernorm.weight");
        w.post_attn_norm = off("post_attention_layernorm.weight");
        w.pre_ffn_norm = off("pre_feedforward_layernorm.weight");
        w.pre_ffn_norm_2 = off("pre_feedforward_layernorm_2.weight");
        w.post_ffn_norm = off("post_feedforward_layernorm.weight");
        w.post_ffn_norm_1 = off("post_feedforward_layernorm_1.weight");
        w.post_ffn_norm_2 = off("post_feedforward_layernorm_2.weight");
        w.layer_scalar = off("layer_scalar");
        w.mlp_gate = quantMatrix(info, core, p + "mlp.gate_proj");
        w.mlp_up = quantMatrix(info, core, p + "mlp.up_proj");
        w.mlp_down = quantMatrix(info, core, p + "mlp.down_proj");
        w.router.scale = off("router.scale");
        w.router.per_expert_scale = off("router.per_expert_scale");
        w.router.proj_weight = off("router.proj.weight");
        w.router.proj_scales = off("router.proj.scales");
        w.router.proj_biases = off("router.proj.biases");
        // group = hidden / scales-per-row (2816 / 44 = 64)
        w.router.group_size =
            uint32_t(c.hidden_size / info.getTensor(p + "router.proj.scales").shape[1]);
        mw.layers.push_back(w);
    }
    return mw;
}

} // namespace apex
