/// @file ops.h
/// @brief Host-side launchers for every kernel. Each launcher knows its kernel's grid and
/// argument order; the engine and the tests call these instead of Backend::run directly.
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include "backend/backend.h"
#include "model_info.h"
#include "../shaders/shared.h"

namespace apex {
// BTW weights have an offset: all core weights live in one buffer.

// An NVFP4 matrix [w_rows, hidden_dim] in GPU memory: codes and scales may live in the
// same buffer (core.bin, expert slots) at different offsets.
struct Nvfp4Matrix {
    Buffer *weights;
    size_t weights_offset; // uint32 [w_rows, hidden_dim/8]
    Buffer *scales;
    size_t scales_offset; // uint8  [w_rows, hidden_dim/16]
    uint32_t w_rows;
    uint32_t hidden_dim; // K (2816 for most matrices)
};

// A view of a buffer at a byte offset; used for weight vectors and cache slices.
struct BufferRef {
    Buffer *buffer = nullptr;
    size_t offset = 0;
};

inline uint32_t ceilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

// Per-layer attention geometry, derived from ModelConfig and the layer type.
struct AttentionShape {
    uint32_t q_heads;    // 16
    uint32_t kv_heads;   // 8 sliding, 2 full
    uint32_t head_dim;   // 256 sliding, 512 full
    uint32_t rope_pairs; // 128 sliding, 64 full
    float rope_base;     // 1e4 sliding, 1e6 full
    uint32_t window;     // 1024 sliding, 0 full
};

inline AttentionShape attentionShape(const ModelConfig &c, int layer) {
    const bool full = c.layer_types[layer] == "full_attention";
    AttentionShape s;
    s.q_heads = uint32_t(c.num_attention_heads);
    s.kv_heads = uint32_t(full ? c.num_global_key_value_heads : c.num_key_value_heads);
    s.head_dim = uint32_t(full ? c.global_head_dim : c.head_dim);
    s.rope_pairs = uint32_t(full ? (c.global_head_dim * c.partial_rotary_factor_full) / 2
                                 : c.head_dim / 2);
    s.rope_base = float(full ? c.rope_theta_full : c.rope_theta_sliding);
    s.window = uint32_t(full ? 0 : c.sliding_window);
    return s;
}

// One layer's attention weights. v is unused when has_v is false (full layers: v_raw = k_raw).
struct AttentionWeights {
    Nvfp4Matrix q, k, v, o;
    bool has_v;
    Buffer *norms;
    size_t q_norm_offset, k_norm_offset; // BF16 [head_dim] each, in the core buffer
};

// Element type of the KV cache. Math is float either way; F16 halves the cache's memory and
// read bandwidth at the cost of rounding the stored keys and values.
enum class KvDtype { F32, F16 };
inline size_t kvElementBytes(KvDtype d) { return d == KvDtype::F16 ? 2 : 4; }

// One layer's KV cache: K and V are [kv_heads][capacity][head_dim] of `dtype`.
// Position p lives in slot p % capacity. A full-attention layer's capacity covers the longest
// sequence (no wrap); a sliding layer is a ring of window + slack slots.
struct KvCache {
    Buffer *k = nullptr;
    Buffer *v = nullptr;
    uint32_t capacity = 0;
    KvDtype dtype = KvDtype::F32;
};

/// Throws unless `tokens` tokens can be written at pos0 and attended without overwriting a key
/// one of them still needs: a linear cache must hold pos0 + tokens positions; a ring must keep
/// window - 1 older keys for the first token while the last one is written, i.e.
/// tokens <= capacity - window + 1.
inline void checkKvWrite(const KvCache &cache, uint32_t window, uint32_t pos0, uint32_t tokens) {
    const bool wraps = pos0 + tokens > cache.capacity;
    if (!wraps) return;
    if (window == 0 || window > cache.capacity || tokens > cache.capacity - window + 1)
        throw std::runtime_error("KV cache: write of " + std::to_string(tokens) +
                                 " tokens at " + std::to_string(pos0) +
                                 " does not fit (capacity " + std::to_string(cache.capacity) +
                                 ", window " + std::to_string(window) + ")");
}

// Scratch for one attention call, sized for the largest layer (head_dim 512).
struct AttentionScratch {
    Buffer *q; // [tokens, 16, hd]  projection, then normed+roped in place
    Buffer *k; // [tokens, kv, hd]  projection
    Buffer *v; // [tokens, kv, hd]  projection (sliding layers only)
    Buffer *o; // [tokens, 16, hd]  attention output
};

/// @brief Every kernel handle, resolved once from the backend, plus their launchers.
/// Construct one per backend at load time; launching is then free of string lookups.
class Ops {
  public:
    explicit Ops(Backend &backend)
        : backend_(backend), rms_norm_(backend.kernel("rms_norm_f32")),
          matmul_(backend.kernel("matmul_nvfp4")),
          gate_up_geglu_(backend.kernel("gate_up_geglu_nvfp4")),
          qkv_norm_rope_{backend.kernel("qkv_norm_rope_kv_f32"),
                         backend.kernel("qkv_norm_rope_kv_f16")},
          attention_{backend.kernel("attention_kv_f32"), backend.kernel("attention_kv_f16")}, router_(backend.kernel("router_f32")),
          expert_gate_up_(backend.kernel("expert_gate_up_nvfp4")),
          expert_down_(backend.kernel("expert_down_nvfp4")),
          embed_norm_(backend.kernel("embed_norm_nvfp4")),
          attn_residual_(backend.kernel("attn_residual_norms_f32")),
          layer_tail_(backend.kernel("layer_tail_f32")), argmax_(backend.kernel("argmax_f32")) {}

    Backend &backend() { return backend_; }

    // Queue RMSNorm for float32 rows using BF16 weights.
    void rmsNorm(Buffer *input, Buffer *weights, size_t weight_offset, Buffer *result,
                 uint32_t rows, uint32_t hidden_dim, float eps) {
        constexpr uint32_t THREADS_PER_GROUP = 256;
        backend_.run(rms_norm_,
                     {Arg::buf(input), Arg::buf(weights, weight_offset), Arg::buf(result),
                      Arg::val(hidden_dim), Arg::val(eps)},
                     Dim3{rows}, Dim3{THREADS_PER_GROUP});
    }

    /// @brief Up to three NVFP4 matrices that share one input, in one dispatch.
    /// mats[m] reads input [tokens, K] and writes outputs[m] [tokens, mats[m].w_rows].
    /// softcap > 0 applies tanh(y / softcap) * softcap (the output head).
    void matmulNvfp4(Buffer *input, size_t input_offset, const Nvfp4Matrix *mats,
                     const BufferRef *outputs, uint32_t count, uint32_t tokens,
                     float softcap = 0.0f) {
        if (count < 1 || count > 3) throw std::runtime_error("matmulNvfp4: 1..3 matrices");
        MatmulParams p{};
        p.in_dim = mats[0].hidden_dim;
        p.tokens = tokens;
        p.softcap = softcap;
        uint32_t rows[3] = {0, 0, 0};
        uint32_t total_rows = 0;
        for (uint32_t m = 0; m < count; ++m) {
            if (mats[m].hidden_dim != p.in_dim || mats[m].w_rows % APEX_MATMUL_ROWS_PER_GROUP)
                throw std::runtime_error("matmulNvfp4: matrices must share K, rows % 8 == 0");
            rows[m] = mats[m].w_rows;
            total_rows += rows[m];
        }
        p.rows0 = rows[0];
        p.rows1 = rows[1];
        p.rows2 = rows[2];
        // Unused slots are bound to matrix 0; the kernel never reads them (rows_m == 0).
        auto mat = [&](uint32_t m) { return mats[m < count ? m : 0]; };
        auto out = [&](uint32_t m) { return outputs[m < count ? m : 0]; };
        const Nvfp4Matrix a = mat(0), b = mat(1), c = mat(2);
        const BufferRef oa = out(0), ob = out(1), oc = out(2);
        backend_.run(matmul_,
                     {Arg::buf(input, input_offset), Arg::buf(a.weights, a.weights_offset),
                      Arg::buf(a.scales, a.scales_offset), Arg::buf(oa.buffer, oa.offset),
                      Arg::buf(b.weights, b.weights_offset), Arg::buf(b.scales, b.scales_offset),
                      Arg::buf(ob.buffer, ob.offset), Arg::buf(c.weights, c.weights_offset),
                      Arg::buf(c.scales, c.scales_offset), Arg::buf(oc.buffer, oc.offset),
                      Arg::val(p)},
                     Dim3{total_rows / APEX_MATMUL_ROWS_PER_GROUP,
                          ceilDiv(tokens, APEX_MATMUL_TOKEN_TILE)},
                     Dim3{APEX_MATMUL_ROWS_PER_GROUP * 32});
    }

    /// One matrix: output [tokens, W.w_rows] = input [tokens, K] . W^T.
    void matmulNvfp4(Buffer *input, const Nvfp4Matrix &W, Buffer *output, uint32_t tokens,
                     float softcap = 0.0f) {
        const BufferRef out{output, 0};
        matmulNvfp4(input, 0, &W, &out, 1, tokens, softcap);
    }

    /// act [tokens, rows] = gelu_tanh(gate . x) * (up . x), one dispatch.
    void gateUpGeglu(Buffer *input, const Nvfp4Matrix &gate, const Nvfp4Matrix &up,
                     Buffer *output, uint32_t tokens) {
        const GateUpParams p{gate.hidden_dim, gate.w_rows, tokens};
        backend_.run(gate_up_geglu_,
                     {Arg::buf(input), Arg::buf(gate.weights, gate.weights_offset),
                      Arg::buf(gate.scales, gate.scales_offset),
                      Arg::buf(up.weights, up.weights_offset), Arg::buf(up.scales, up.scales_offset),
                      Arg::buf(output), Arg::val(p)},
                     Dim3{ceilDiv(gate.w_rows, APEX_MATMUL_ROWS_PER_GROUP),
                          ceilDiv(tokens, APEX_MATMUL_TOKEN_TILE)},
                     Dim3{APEX_MATMUL_ROWS_PER_GROUP * 32});
    }

    // Dense MLP: two dispatches.
    /// input → gate+up+GeGLU (fused) → act → down → output
    void denseMlp(Buffer *input, const Nvfp4Matrix &gate_weights, const Nvfp4Matrix &up_weights,
                  const Nvfp4Matrix &down_weights, Buffer *act_tmp, Buffer *output,
                  uint32_t tokens) {
        gateUpGeglu(input, gate_weights, up_weights, act_tmp, tokens);
        matmulNvfp4(act_tmp, down_weights, output, tokens);
    }

    /// @brief q/K/V per-head norm + RoPE in one dispatch; K and V go straight into the cache
    /// at positions pos0 .. pos0 + tokens - 1 (slots pos % capacity). v_raw is the k
    /// projection on full layers.
    void qkvNormRope(Buffer *q, Buffer *k_raw, Buffer *v_raw, const AttentionWeights &w,
                     const AttentionShape &s, const KvCache &cache, uint32_t tokens,
                     uint32_t pos0, float eps) {
        checkKvWrite(cache, s.window, pos0, tokens);
        const QkvNormRopeParams p{s.q_heads, s.kv_heads, s.head_dim, s.rope_pairs,
                                  s.rope_base, pos0,      cache.capacity, eps};
        backend_.run(qkv_norm_rope_[cache.dtype == KvDtype::F16],
                     {Arg::buf(q), Arg::buf(k_raw), Arg::buf(v_raw),
                      Arg::buf(w.norms, w.q_norm_offset), Arg::buf(w.norms, w.k_norm_offset),
                      Arg::buf(cache.k), Arg::buf(cache.v), Arg::val(p)},
                     Dim3{tokens, s.q_heads + 2 * s.kv_heads}, Dim3{APEX_HEAD_NORM_THREADS});
    }

    /// Queue the softmax-over-keys for every (token, q head).
    void attention(Buffer *q, const KvCache &cache, Buffer *output, const AttentionShape &s,
                   uint32_t tokens, uint32_t pos0) {
        const AttentionParams p{s.q_heads, s.kv_heads, s.head_dim, cache.capacity,
                                pos0,      s.window,   tokens};
        backend_.run(attention_[cache.dtype == KvDtype::F16],
                     {Arg::buf(q), Arg::buf(cache.k), Arg::buf(cache.v), Arg::buf(output),
                      Arg::val(p)},
                     Dim3{tokens, s.q_heads}, Dim3{APEX_ATTN_THREADS});
    }

    /// @brief Whole attention block for `tokens` tokens starting at position pos0, 4 dispatches:
    ///   x → q/k/v proj (one dispatch) → norms + RoPE (K,V into the cache) → attention → o_proj
    /// output is [tokens, hidden], before post_attention_layernorm.
    void attentionBlock(Buffer *x, const AttentionWeights &w, const AttentionShape &s,
                        const KvCache &cache, const AttentionScratch &tmp, Buffer *output,
                        uint32_t tokens, uint32_t pos0, float eps) {
        const Nvfp4Matrix mats[3] = {w.q, w.k, w.v};
        const BufferRef outs[3] = {{tmp.q, 0}, {tmp.k, 0}, {tmp.v, 0}};
        matmulNvfp4(x, 0, mats, outs, w.has_v ? 3 : 2, tokens);
        qkvNormRope(tmp.q, tmp.k, w.has_v ? tmp.v : tmp.k, w, s, cache, tokens, pos0, eps);
        attention(tmp.q, cache, tmp.o, s, tokens, pos0);
        matmulNvfp4(tmp.o, w.o, output, tokens);
    }

    /// @brief Fused router for `tokens` rows of h: writes routes [tokens, top_k] (slot = expert
    /// id, weight) and counts [tokens], all on the GPU.
    /// core holds the router tensors; offsets come from RouterWeights (weights.h).
    void router(Buffer *h, Buffer *core, size_t scale, size_t proj_weight, size_t proj_scales,
                size_t proj_biases, size_t per_expert_scale, Buffer *routes, Buffer *counts,
                const RouterParams &p, uint32_t tokens) {
        if (p.hidden > APEX_MAX_HIDDEN || p.experts > APEX_MAX_EXPERTS || p.top_k > 32 ||
            p.top_k > APEX_MAX_TOP_K || p.hidden % p.group_size || p.group_size % 4)
            throw std::runtime_error("router: shape outside the kernel's limits");
        backend_.run(router_,
                     {Arg::buf(h), Arg::buf(core, scale), Arg::buf(core, proj_weight),
                      Arg::buf(core, proj_scales), Arg::buf(core, proj_biases),
                      Arg::buf(core, per_expert_scale), Arg::buf(routes), Arg::buf(counts),
                      Arg::val(p)},
                     Dim3{tokens}, Dim3{APEX_ROUTER_THREADS});
    }

    /// @brief Routed experts, two dispatches:
    ///   gate+up+GeGLU for all of each token's routed slots → act [tokens, max_k, inter]
    ///   down + weighted sum over the slots                → output [tokens, in_dim]
    /// experts is the layer's slot buffer; routes/counts are what the router wrote (or a
    /// remapped copy when slots are not expert ids).
    void experts(Buffer *input, Buffer *experts, Buffer *routes, Buffer *counts, Buffer *act,
                 Buffer *output, const ExpertParams &p) {
        if (p.inter % APEX_MATMUL_ROWS_PER_GROUP || p.in_dim % APEX_MATMUL_ROWS_PER_GROUP)
            throw std::runtime_error("experts: sizes must be multiples of 8 rows");
        backend_.run(expert_gate_up_,
                     {Arg::buf(input), Arg::buf(experts), Arg::buf(routes), Arg::buf(counts),
                      Arg::buf(act), Arg::val(p)},
                     Dim3{p.inter / APEX_MATMUL_ROWS_PER_GROUP, p.max_k, p.tokens},
                     Dim3{APEX_MATMUL_ROWS_PER_GROUP * 32});
        backend_.run(expert_down_,
                     {Arg::buf(act), Arg::buf(experts), Arg::buf(routes), Arg::buf(counts),
                      Arg::buf(output), Arg::val(p)},
                     Dim3{p.in_dim / APEX_MATMUL_ROWS_PER_GROUP, p.tokens},
                     Dim3{APEX_MATMUL_ROWS_PER_GROUP * 32});
    }

    /// h [tokens, hidden] = embed_tokens[ids] * sqrt(hidden); x = norm(h, norm_w).
    /// ids is a GPU buffer of uint32 token ids.
    void embedNorm(Buffer *ids, const Nvfp4Matrix &embed, Buffer *norm_w, size_t norm_offset,
                   Buffer *h, Buffer *x, uint32_t tokens, float eps) {
        checkHidden(embed.hidden_dim);
        const RowParams p{embed.hidden_dim, eps, std::sqrt(float(embed.hidden_dim)), tokens};
        backend_.run(embed_norm_,
                     {Arg::buf(ids), Arg::buf(embed.weights, embed.weights_offset),
                      Arg::buf(embed.scales, embed.scales_offset), Arg::buf(norm_w, norm_offset),
                      Arg::buf(h), Arg::buf(x), Arg::val(p)},
                     Dim3{tokens}, Dim3{APEX_ROW_THREADS});
    }

    /// h += norm(o, post_attn); x_mlp = norm(h, pre_ffn); x_moe = norm(h, pre_ffn_2).
    /// The three weights are offsets into `norms` (the core buffer).
    void attnResidualNorms(Buffer *o, Buffer *h, Buffer *norms, size_t post_attn,
                           size_t pre_ffn, size_t pre_ffn_2, Buffer *x_mlp, Buffer *x_moe,
                           uint32_t tokens, uint32_t hidden, float eps) {
        checkHidden(hidden);
        const RowParams p{hidden, eps, 0.0f, tokens};
        backend_.run(attn_residual_,
                     {Arg::buf(o), Arg::buf(h), Arg::buf(norms, post_attn),
                      Arg::buf(norms, pre_ffn), Arg::buf(norms, pre_ffn_2), Arg::buf(x_mlp),
                      Arg::buf(x_moe), Arg::val(p)},
                     Dim3{tokens}, Dim3{APEX_ROW_THREADS});
    }

    /// f = norm(dense, post_1) + norm(moe, post_2); h = (h + norm(f, post)) * scalar;
    /// x = norm(h, next_norm). All weights are offsets into `norms` (the core buffer).
    void layerTail(Buffer *dense, Buffer *moe, Buffer *h, Buffer *norms, size_t post_1,
                   size_t post_2, size_t post, size_t layer_scalar, size_t next_norm, Buffer *x,
                   uint32_t tokens, uint32_t hidden, float eps) {
        checkHidden(hidden);
        const RowParams p{hidden, eps, 0.0f, tokens};
        backend_.run(layer_tail_,
                     {Arg::buf(dense), Arg::buf(moe), Arg::buf(h), Arg::buf(norms, post_1),
                      Arg::buf(norms, post_2), Arg::buf(norms, post),
                      Arg::buf(norms, layer_scalar), Arg::buf(norms, next_norm), Arg::buf(x),
                      Arg::val(p)},
                     Dim3{tokens}, Dim3{APEX_ROW_THREADS});
    }

    /// out[t] = argmax(logits[t, :cols]) as uint32, lowest index on ties.
    void argmax(Buffer *logits, Buffer *out, uint32_t tokens, uint32_t cols) {
        const ArgmaxParams p{cols};
        backend_.run(argmax_, {Arg::buf(logits), Arg::buf(out), Arg::val(p)}, Dim3{tokens},
                     Dim3{APEX_ARGMAX_THREADS});
    }

  private:
    static void checkHidden(uint32_t hidden) {
        if (hidden > APEX_MAX_HIDDEN || hidden % 16)
            throw std::runtime_error("row kernel: hidden size outside APEX_MAX_HIDDEN");
    }

    Backend &backend_;
    Kernel *rms_norm_;
    Kernel *matmul_;
    Kernel *gate_up_geglu_;
    Kernel *qkv_norm_rope_[2]; // [KvDtype::F32, KvDtype::F16]
    Kernel *attention_[2];
    Kernel *router_;
    Kernel *expert_gate_up_;
    Kernel *expert_down_;
    Kernel *embed_norm_;
    Kernel *attn_residual_;
    Kernel *layer_tail_;
    Kernel *argmax_;
};

} // namespace apex
