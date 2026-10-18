/// @file model_info.h
/// @brief Types to help us read packed binary files. 
#pragma once 

#include <cstdint>  
#include <cstdio>
#include <string>  
#include <vector>   
#include <map>
#include <stdexcept>
#include <fstream>            
#include <utility>
#include <nlohmann/json.hpp> 
namespace apex {

// core.bin
//          tensor A                   tensor B                   tensor C
//     ┌──────────────┬─────────┬──────────────────┬─────────┬──────────────┬─────┐
//     │ weight bytes │ padding │   weight bytes   │ padding │ weight bytes │ ... │
//     └──────────────┴─────────┴──────────────────┴─────────┴──────────────┴─────┘
//     ↑                        ↑
//     A.offset                 B.offset
//     │← A.nbytes →│           │←── B.nbytes ───→│

// layer_NN.bin
//     ┌──────────────────┬──────────────────┬──────────────────┬─────┐
//     │     Expert 0     │     Expert 1     │     Expert 2     │ ... │
//     └──────────────────┴──────────────────┴──────────────────┴─────┘
//     │←──── stride ────→│

// Exper slot layout 
// ┌────────┬────────┬────────┬────────┬────────┬────────┬─────────┐
// │ gate W │ gate S │  up W  │  up S  │ down W │ down S │ padding │
// └────────┴────────┴────────┴────────┴────────┴────────┴─────────┘
// ↑                          ↑
// slot start                 slot start + piece.offset

// │←────────────────── expert_stride ────────────────────────→│

// Simplified layout; pieces can also have padding between them.


/// Describes one tensor inside core.bin, data extracted from manifest.json
struct TensorInfo {
    std::string dtype;                // "BF16" or "U32".
    std::vector<std::int64_t> shape;  // 
    std::uint64_t offset = 0;         // where tensor starts in bytes
    std::uint64_t nbytes = 0;       // in bytes
};

/// Describes one tensor inside an expert's storage slot (ex: gate W)
struct ExpertPiece {
    std::string name;                 
    std::string dtype;               
    std::vector<std::int64_t> shape; 
    std::uint64_t offset = 0;        
    std::uint64_t nbytes = 0;         
};

// Model config values from teh manifest. 
struct ModelConfig {
    int hidden_size = 0;            // 2816
    int intermediate_size = 0;      // 2112  (dense MLP)
    int moe_intermediate_size = 0;  // 704   (per expert)
    int num_attention_heads = 0;    // 16
    int num_key_value_heads = 0;    // 8
    int head_dim = 0;               // 256
    int num_hidden_layers = 0;      // 30
    int num_experts = 0;            // 128
    int top_k_experts = 0;          // 8
    int vocab_size = 0;             // 262144
    int sliding_window = 0;         // 1024
    double rms_norm_eps = 0;        // 1e-6
    double final_logit_softcapping = 0;  // 30.0
    std::vector<std::string> layer_types;  // "sliding_attention" | "full_attention"

    // attention and rope stuff. 
    int global_head_dim = 0;               // 512  (full-attention layers)
    int num_global_key_value_heads = 0;    // 2    (full-attention layers)
    bool attention_k_eq_v = false;         // true: full layers have no v_proj
    double rope_theta_sliding = 0;         // 1e4
    double rope_theta_full = 0;            // 1e6
    double partial_rotary_factor_full = 0; // 0.25
};

/// @brief Contains info about the model itself including model config and location of tensors (core and experts)
struct ModelInfo {
    std::string dir;
    ModelConfig config;

    int quant_group_size = 0;  // 64 or 128
    int quant_bits = 0;
    std::string quant_mode; // quantization format. 

    std::uint64_t core_size = 0;  

    std::map<std::string, TensorInfo> core;  // name -> info

    std::uint64_t expert_slot_size = 0;  // size of one expert slot in bytes
    std::map<std::string, ExpertPiece> expert_pieces;  // Tensor layout within each expert slot. name -> info

    const TensorInfo& getTensor(const std::string& name) const {
        auto it = core.find(name); // returns an iterator for whatever reason.
        if (it == core.end()) {
            throw std::runtime_error("tensor not found: " + name);
        }
        return it->second; 
    }

    static ModelInfo load(const std::string& dir);  // load manifest.json from disk

    /// Returns the path to a packed expert layer file.
    std::string layer_file(int layer) const {
        char filename[32];
        std::snprintf(filename, sizeof filename, "/layer_%02d.bin", layer);
        return dir + filename;
    }
};

/// @brief Reads in the manifest.json and converts metadata into ModelInfo Config, and TensorInfo and ExpertPeices
/// @param dir location of manifest.json file. 
/// @return ModelInfo Object with model config, shared tensor info with offsets, and expert peices info. 
inline ModelInfo ModelInfo::load(const std::string& dir) {
    std::ifstream in(dir + "/manifest.json");

    if (!in) {
        throw std::runtime_error(
            "ModelInfo: cannot open " + dir + "/manifest.json");
    }

    // Parse the file. 
    const auto j = nlohmann::json::parse(in);

    ModelInfo model_info;
    model_info.dir = dir;

    const auto& tc = j.at("text_config");

    // Update config. 
    ModelConfig& c = model_info.config;

    c.hidden_size           = tc.at("hidden_size").get<int>();
    c.intermediate_size     = tc.at("intermediate_size").get<int>();
    c.moe_intermediate_size = tc.at("moe_intermediate_size").get<int>();
    c.num_attention_heads   = tc.at("num_attention_heads").get<int>();
    c.num_key_value_heads   = tc.at("num_key_value_heads").get<int>();
    c.head_dim              = tc.at("head_dim").get<int>();
    c.num_hidden_layers     = tc.at("num_hidden_layers").get<int>();
    c.num_experts           = tc.at("num_experts").get<int>();
    c.top_k_experts         = tc.at("top_k_experts").get<int>();
    c.vocab_size            = tc.at("vocab_size").get<int>();
    c.sliding_window        = tc.at("sliding_window").get<int>();
    c.rms_norm_eps          = tc.at("rms_norm_eps").get<double>();
    c.final_logit_softcapping = tc.at("final_logit_softcapping").get<double>();
    c.layer_types           = tc.at("layer_types").get<std::vector<std::string>>();
    // attentoin and rope stuff. 
    c.global_head_dim = tc.at("global_head_dim").get<int>();
    c.num_global_key_value_heads = tc.at("num_global_key_value_heads").get<int>();
    c.attention_k_eq_v = tc.at("attention_k_eq_v").get<bool>();
    const auto &rp = tc.at("rope_parameters");
    c.rope_theta_sliding = rp.at("sliding_attention").at("rope_theta").get<double>();
    c.rope_theta_full = rp.at("full_attention").at("rope_theta").get<double>();
    c.partial_rotary_factor_full =
    rp.at("full_attention").at("partial_rotary_factor").get<double>();

    // Read how the weights were quantized.
    const auto& q = j.at("quantization");
    model_info.quant_group_size = q.at("group_size").get<int>();
    model_info.quant_bits = q.at("bits").get<int>();
    model_info.quant_mode = q.at("mode").get<std::string>();

    // Read the expected core.bin file size.
    const auto& core = j.at("core");
    model_info.core_size = core.at("size").get<std::uint64_t>();

    // Each element of "tensors" describes one tensor in core.bin.
    for (const auto& t : core.at("tensors")) {
        TensorInfo info;
        info.dtype = t.at("dtype").get<std::string>();
        info.shape = t.at("shape").get<std::vector<std::int64_t>>();
        info.offset = t.at("offset").get<std::uint64_t>();
        info.nbytes = t.at("nbytes").get<std::uint64_t>();

        // Store its metadata under its tensor name.
        model_info.core.emplace(t.at("name").get<std::string>(), std::move(info));
    }


    // All expert layers share the same slot layout; read it from layer 0.
    const auto& layer = j.at("layers").at(0);
    model_info.expert_slot_size = layer.at("stride").get<std::uint64_t>();

    for (const auto& p : layer.at("pieces")) {
        ExpertPiece piece;
        piece.name = p.at("name").get<std::string>();
        piece.dtype = p.at("dtype").get<std::string>();
        piece.shape = p.at("shape").get<std::vector<std::int64_t>>();
        piece.offset = p.at("offset").get<std::uint64_t>();
        piece.nbytes = p.at("nbytes").get<std::uint64_t>();

        // Save the key before moving the piece, which also moves its name.
        const std::string name = piece.name;
        model_info.expert_pieces.emplace(name, std::move(piece));
    }

    return model_info;
}
    
}  // namespace apex
