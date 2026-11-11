/// @file model.h
/// @brief The Gemma 4 forward pass: weights resolved once, scratch and KV cache owned, and
/// forward(ids, n, pos0) that runs n tokens at positions pos0 .. pos0 + n - 1.
///
/// Per call, everything is queued into ONE command buffer (resident mode) and the only thing
/// the CPU reads back is the argmax token id per position. Routing never leaves the GPU: the
/// router writes (slot, weight) lists that the expert kernels read directly.
///
/// Dispatches per layer (11):
///   1  matmul_nvfp4            q, k, v projections in one dispatch (q, k on full layers)
///   2  qkv_norm_rope_kv_f16    q/k/v head norms + RoPE; K and V land in the KV cache
///   3  attention_kv_f16        online softmax over the cached keys
///      (the _kv_f32 variants when ModelOptions::kv_dtype is F32)
///   4  matmul_nvfp4            o_proj
///   5  attn_residual_norms_f32 h += norm(o); x_mlp = norm(h, pre_ffn); x_moe = norm(h, pre_ffn_2)
///   6  gate_up_geglu_nvfp4     dense gate + up + GeGLU
///   7  matmul_nvfp4            dense down
///   8  router_f32              norm + affine matvec + top-8 + softmax + per_expert_scale
///   9  expert_gate_up_nvfp4    gate + up + GeGLU for the 8 routed experts of every token
///  10  expert_down_nvfp4       down + weighted sum over the 8 experts
///  11  layer_tail_f32          post-FFN norms, residual, layer_scalar, next layer's input norm
/// plus embed_norm_nvfp4 before layer 0 and, when logits are wanted, the head matmul (with the
/// softcap) and argmax_f32 after layer 29. The final norm is the last layer tail's "next norm".
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "backend/backend.h"
#include "experts.h"
#include "file.h"
#include "kv_cache.h"
#include "model_info.h"
#include "ops.h"
#include "profiling/timer.h"
#include "weights.h"

namespace apex {

struct ModelOptions {
    ExpertResidency residency = ExpertResidency::Resident;
    uint32_t max_context = 4096; // longest sequence the KV cache holds
    uint32_t max_tokens = 64;    // most tokens one forward() call takes (scratch size)
    KvDtype kv_dtype = KvDtype::F16;
    // Sliding-layer rings hold window + max(KV_RING_MIN_SLACK, max_tokens) positions, so any
    // forward() batch and any rollback of up to that many speculative tokens is legal.
};

class Model {
  public:
    Model(Backend &backend, const ModelInfo &info, const ModelOptions &options)
        : backend_(backend), info_(info), config_(info.config), options_(options), ops_(backend) {
        if (options.max_tokens == 0 || options.max_context == 0)
            throw std::runtime_error("Model: max_tokens and max_context must be > 0");
        core_ = mapFile(info.dir + "/core.bin", "core_weights");
        if (core_.size != info.core_size) throw std::runtime_error("core.bin size != manifest");
        core_buffer_ = backend.wrapNoCopy(core_.data, core_.size);
        weights_ = resolveWeights(info, core_buffer_);
        experts_ = makeExpertSource(options.residency, backend, info);
        KvCacheOptions kv;
        kv.max_context = options.max_context;
        kv.slack = std::max(KV_RING_MIN_SLACK, options.max_tokens);
        kv.dtype = options.kv_dtype;
        cache_ = std::make_unique<KvCacheSet>(backend, config_, kv);
        allocateScratch();
        eps_ = float(config_.rms_norm_eps);
    }

    ~Model() {
        for (Buffer *b : scratch_) backend_.release(b);
        cache_.reset();
        experts_.reset();
        backend_.release(core_buffer_); // GPU lets go of core.bin before it is unmapped
        unmapFile(core_);
    }
    Model(const Model &) = delete;
    Model &operator=(const Model &) = delete;

    /// @brief Run n tokens at positions pos0 .. pos0 + n - 1 (n <= max_tokens).
    /// pos0 may be at most the current cache length; writing before the end overwrites (that
    /// is how rejected speculative tokens are replaced). Afterwards the cache length is
    /// pos0 + n.
    /// @param want_logits false skips the output head (intermediate prefill chunks).
    /// @return argmax token id per position (n entries), or nullptr if !want_logits. Valid
    /// until the next call.
    const uint32_t *forward(const uint32_t *ids, uint32_t n, uint32_t pos0,
                            bool want_logits = true) {
        if (n == 0 || n > options_.max_tokens)
            throw std::runtime_error("forward: 1 <= n <= max_tokens");
        cache_->checkWrite(pos0, n);
        Timer timer("forward");
        auto *id_dst = static_cast<uint32_t *>(backend_.contents(ids_));
        for (uint32_t t = 0; t < n; ++t) {
            if (ids[t] >= uint32_t(config_.vocab_size))
                throw std::runtime_error("forward: token id out of range");
            id_dst[t] = ids[t];
        }

        const uint32_t H = uint32_t(config_.hidden_size);
        ops_.embedNorm(ids_, weights_.embed, core_buffer_, weights_.layers[0].input_norm, h_, x_,
                       n, eps_);
        const bool layer_sync = experts_->needsLayerSync();
        for (int l = 0; l < config_.num_hidden_layers; ++l) {
            Buffer *experts = experts_->acquire(l);
            encodeLayer(l, experts, n, pos0, H);
            if (layer_sync) {
                // Streamed mode only: the layer file can be dropped once its work is done.
                backend_.sync();
                experts_->release(l);
                if (layer_hook) layer_hook(l, n);
            }
        }
        if (want_logits) {
            ops_.matmulNvfp4(x_, weights_.embed, logits_, n,
                             float(config_.final_logit_softcapping));
            ops_.argmax(logits_, argmax_, n, uint32_t(config_.vocab_size));
        }
        backend_.sync(); // the one wait per call in resident mode
        cache_->commitWrite(pos0, n);
        return want_logits ? static_cast<const uint32_t *>(backend_.contents(argmax_)) : nullptr;
    }

    /// Cache length (tokens whose keys and values are stored).
    uint32_t length() const { return cache_->length(); }
    /// Drop positions >= new_length (rejected speculative tokens); at most slack + 1 at a time
    /// once a sliding ring has wrapped.
    void truncate(uint32_t new_length) { cache_->truncate(new_length); }

    // --- State reachable from outside forward() (for multi-token prediction later) ---
    /// Final hidden state of the last call, the input to the final norm: [n, hidden] f32.
    Buffer *finalHidden() const { return h_; }
    /// norm(finalHidden, norm.weight) of the last call: [n, hidden] f32.
    Buffer *finalNormed() const { return x_; }
    /// Logits of the last call with want_logits: [n, vocab] f32, after the softcap.
    Buffer *logits() const { return logits_; }
    /// Routes of the most recently encoded layer: [n, top_k] ExpertRoute.
    Buffer *routes() const { return routes_; }
    /// KV cache of one layer (29 is the last full-attention layer, 28 the last sliding one).
    const KvCache &kvCache(int layer) const { return cache_->layer(layer); }
    const KvCacheSet &kvCaches() const { return *cache_; }

    const ModelOptions &options() const { return options_; }
    const ModelInfo &info() const { return info_; }

    /// Streamed mode only (where each layer is synced): called after layer l completed, with
    /// the layer's outputs readable in finalHidden() / routes(). Used by the parity test.
    std::function<void(int layer, uint32_t n)> layer_hook;

  private:
    /// Queue one decoder layer (11 dispatches, see the file comment).
    void encodeLayer(int l, Buffer *experts, uint32_t n, uint32_t pos0, uint32_t H) {
        const LayerWeights &w = weights_.layers[size_t(l)];
        const bool last = l + 1 == config_.num_hidden_layers;
        const size_t next_norm =
            last ? weights_.final_norm : weights_.layers[size_t(l) + 1].input_norm;
        Buffer *core = core_buffer_;

        ops_.attentionBlock(x_, w.attn, w.shape, cache_->layer(l), attn_, o_out_, n, pos0, eps_);
        ops_.attnResidualNorms(o_out_, h_, core, w.post_attn_norm, w.pre_ffn_norm,
                               w.pre_ffn_norm_2, x_mlp_, x_moe_, n, H, eps_);
        ops_.denseMlp(x_mlp_, w.mlp_gate, w.mlp_up, w.mlp_down, dense_act_, dense_out_, n);
        const RouterWeights &r = w.router;
        ops_.router(h_, core, r.scale, r.proj_weight, r.proj_scales, r.proj_biases,
                    r.per_expert_scale, routes_, counts_, routerParams(config_, r.group_size), n);
        ops_.experts(x_moe_, experts, routes_, counts_, expert_act_, moe_out_,
                     expertParams(config_, weights_.expert_layout, n));
        ops_.layerTail(dense_out_, moe_out_, h_, core, w.post_ffn_norm_1, w.post_ffn_norm_2,
                       w.post_ffn_norm, w.layer_scalar, next_norm, x_, n, H, eps_);
    }

    void allocateScratch() {
        const size_t M = options_.max_tokens;
        const size_t H = size_t(config_.hidden_size);
        const size_t q_rows = size_t(config_.num_attention_heads) *
                              size_t(std::max(config_.head_dim, config_.global_head_dim));
        const size_t kv_rows =
            size_t(std::max(config_.num_key_value_heads * config_.head_dim,
                            config_.num_global_key_value_heads * config_.global_head_dim));
        const size_t K = size_t(config_.top_k_experts);
        auto f32 = [&](size_t count) { return alloc(count * sizeof(float)); };
        ids_ = alloc(M * sizeof(uint32_t));
        h_ = f32(M * H);
        x_ = f32(M * H);
        attn_.q = f32(M * q_rows);
        attn_.k = f32(M * kv_rows);
        attn_.v = f32(M * kv_rows);
        attn_.o = f32(M * q_rows);
        o_out_ = f32(M * H);
        x_mlp_ = f32(M * H);
        x_moe_ = f32(M * H);
        dense_act_ = f32(M * size_t(config_.intermediate_size));
        dense_out_ = f32(M * H);
        routes_ = alloc(M * K * sizeof(ExpertRoute));
        counts_ = alloc(M * sizeof(uint32_t));
        expert_act_ = f32(M * K * size_t(config_.moe_intermediate_size));
        moe_out_ = f32(M * H);
        logits_ = f32(M * size_t(config_.vocab_size));
        argmax_ = alloc(M * sizeof(uint32_t));
    }
    Buffer *alloc(size_t bytes) {
        Buffer *b = backend_.allocate(bytes, "activations");
        scratch_.push_back(b);
        return b;
    }

    Backend &backend_;
    const ModelInfo &info_;
    const ModelConfig &config_;
    ModelOptions options_;
    Ops ops_;
    float eps_ = 0;

    MappedFile core_;
    Buffer *core_buffer_ = nullptr;
    ModelWeights weights_;
    std::unique_ptr<ExpertSource> experts_;
    std::unique_ptr<KvCacheSet> cache_;

    // Scratch, sized for max_tokens rows. h_ is the residual stream; x_ is always the normed
    // input of the next consumer (attention, or the head after the last layer).
    std::vector<Buffer *> scratch_;
    Buffer *ids_ = nullptr, *h_ = nullptr, *x_ = nullptr;
    AttentionScratch attn_{};
    Buffer *o_out_ = nullptr, *x_mlp_ = nullptr, *x_moe_ = nullptr;
    Buffer *dense_act_ = nullptr, *dense_out_ = nullptr;
    Buffer *routes_ = nullptr, *counts_ = nullptr, *expert_act_ = nullptr, *moe_out_ = nullptr;
    Buffer *logits_ = nullptr, *argmax_ = nullptr;
};

} // namespace apex
