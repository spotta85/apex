#include <metal_stdlib>
#include "quant.h"
#include "reduce.h"
#include "shared.h"
using namespace metal;

// Row kernels: one threadgroup per token row of the hidden state, APEX_ROW_THREADS threads.
// Each fuses a chain of element-wise work and RMSNorms that would otherwise be several
// dispatches with full-row round trips through memory.
//
// Every norm here is the checkpoint's plain RMSNorm, no +1 offset:
//     norm(x, w) = (x * rsqrt(mean(x^2) + eps)) * w
//
// Thread i of the row owns elements i, i + THREADS, ...; it writes and later re-reads only its
// own elements of the threadgroup row buffers, so the only barriers needed are the ones
// inside the reductions.

constant constexpr uint THREADS = APEX_ROW_THREADS;
constant constexpr uint SIMDS = THREADS / 32;

// Embedding gather + the first layer's input norm.
//   h[t] = dequant(embed_tokens[id_t]) * sqrt(hidden)     (embed_tokens is NVFP4)
//   x[t] = norm(h[t], layers.0.input_layernorm)
kernel void embed_norm_nvfp4(device const uint *ids [[buffer(0)]],        // [T]
                             device const uint *embed_w [[buffer(1)]],    // [vocab, hidden/8]
                             device const uchar *embed_s [[buffer(2)]],   // [vocab, hidden/16]
                             device const ushort *norm_w [[buffer(3)]],   // BF16 [hidden]
                             device float *h [[buffer(4)]],               // [T, hidden]
                             device float *x [[buffer(5)]],               // [T, hidden]
                             constant RowParams &p [[buffer(6)]],
                             uint token [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint simd_id [[simdgroup_index_in_threadgroup]]) {
    threadgroup float row[APEX_MAX_HIDDEN];
    threadgroup float scratch[32];
    const uint id = ids[token];
    const uint groups = p.hidden / NVFP4_GROUP;
    embed_w += ulong(id) * (p.hidden / NVFP4_PER_WORD);
    embed_s += ulong(id) * groups;
    h += token * p.hidden;
    x += token * p.hidden;

    // Thread g decodes group g (16 values); the products e2m1 * e4m3 are exact in float.
    float ss = 0.0f;
    for (uint g = tid; g < groups; g += THREADS) {
        float w[NVFP4_GROUP];
        decodeGroup(embed_w[2 * g], embed_w[2 * g + 1], w);
        const float scale = e4m3ToFloat(embed_s[g]);
        for (uint j = 0; j < NVFP4_GROUP; ++j) {
            const float v = (w[j] * scale) * p.scale;
            row[g * NVFP4_GROUP + j] = v;
            h[g * NVFP4_GROUP + j] = v;
            ss += v * v;
        }
    }
    // The barrier inside the reduction also publishes row[] (written per group, read strided).
    ss = threadgroupSum(ss, scratch, lane, simd_id, SIMDS);
    const float inv = rsqrt(ss / float(p.hidden) + p.eps);
    for (uint i = tid; i < p.hidden; i += THREADS) {
        x[i] = (row[i] * inv) * bf16ToFloat(norm_w[i]);
    }
}

// After the attention block (o is the o_proj output, before post_attention_layernorm):
//   h     = h + norm(o, post_attention_layernorm)        (in place)
//   x_mlp = norm(h, pre_feedforward_layernorm)           dense MLP input
//   x_moe = norm(h, pre_feedforward_layernorm_2)         experts input
// The router reads the updated raw h itself.
kernel void attn_residual_norms_f32(device const float *o [[buffer(0)]],          // [T, hidden]
                                    device float *h [[buffer(1)]],                // [T, hidden]
                                    device const ushort *w_post_attn [[buffer(2)]],
                                    device const ushort *w_pre_ffn [[buffer(3)]],
                                    device const ushort *w_pre_ffn_2 [[buffer(4)]],
                                    device float *x_mlp [[buffer(5)]],
                                    device float *x_moe [[buffer(6)]],
                                    constant RowParams &p [[buffer(7)]],
                                    uint token [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint simd_id [[simdgroup_index_in_threadgroup]]) {
    threadgroup float row[APEX_MAX_HIDDEN];
    threadgroup float scratch[32];
    const ulong base = ulong(token) * p.hidden;
    o += base;
    h += base;
    x_mlp += base;
    x_moe += base;

    float ss = 0.0f;
    for (uint i = tid; i < p.hidden; i += THREADS) ss += o[i] * o[i];
    ss = threadgroupSum(ss, scratch, lane, simd_id, SIMDS);
    const float inv_o = rsqrt(ss / float(p.hidden) + p.eps);

    float ss_h = 0.0f;
    for (uint i = tid; i < p.hidden; i += THREADS) {
        const float v = h[i] + (o[i] * inv_o) * bf16ToFloat(w_post_attn[i]);
        row[i] = v;
        h[i] = v;
        ss_h += v * v;
    }
    ss_h = threadgroupSum(ss_h, scratch, lane, simd_id, SIMDS);
    const float inv_h = rsqrt(ss_h / float(p.hidden) + p.eps);
    for (uint i = tid; i < p.hidden; i += THREADS) {
        const float n = row[i] * inv_h;
        x_mlp[i] = n * bf16ToFloat(w_pre_ffn[i]);
        x_moe[i] = n * bf16ToFloat(w_pre_ffn_2[i]);
    }
}

// End of a decoder layer:
//   f = norm(dense_out, post_ffn_norm_1) + norm(moe_out, post_ffn_norm_2)
//   h = (h + norm(f, post_ffn_norm)) * layer_scalar                      (in place)
//   x = norm(h, next_norm)
// next_norm is the next layer's input_layernorm, or the final `norm` after the last layer, so
// x is always what the next consumer (attention or the output head) reads.
kernel void layer_tail_f32(device const float *dense [[buffer(0)]],   // [T, hidden]
                           device const float *moe [[buffer(1)]],     // [T, hidden]
                           device float *h [[buffer(2)]],             // [T, hidden]
                           device const ushort *w_post_1 [[buffer(3)]],
                           device const ushort *w_post_2 [[buffer(4)]],
                           device const ushort *w_post [[buffer(5)]],
                           device const ushort *layer_scalar [[buffer(6)]], // BF16 [1]
                           device const ushort *next_norm [[buffer(7)]],
                           device float *x [[buffer(8)]],             // [T, hidden]
                           constant RowParams &p [[buffer(9)]],
                           uint token [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint simd_id [[simdgroup_index_in_threadgroup]]) {
    threadgroup float row[APEX_MAX_HIDDEN];
    threadgroup float scratch[32];
    const ulong base = ulong(token) * p.hidden;
    dense += base;
    moe += base;
    h += base;
    x += base;

    float ss_d = 0.0f, ss_m = 0.0f;
    for (uint i = tid; i < p.hidden; i += THREADS) {
        ss_d += dense[i] * dense[i];
        ss_m += moe[i] * moe[i];
    }
    ss_d = threadgroupSum(ss_d, scratch, lane, simd_id, SIMDS);
    ss_m = threadgroupSum(ss_m, scratch, lane, simd_id, SIMDS);
    const float inv_d = rsqrt(ss_d / float(p.hidden) + p.eps);
    const float inv_m = rsqrt(ss_m / float(p.hidden) + p.eps);

    float ss_f = 0.0f;
    for (uint i = tid; i < p.hidden; i += THREADS) {
        const float f = (dense[i] * inv_d) * bf16ToFloat(w_post_1[i]) +
                        (moe[i] * inv_m) * bf16ToFloat(w_post_2[i]);
        row[i] = f;
        ss_f += f * f;
    }
    ss_f = threadgroupSum(ss_f, scratch, lane, simd_id, SIMDS);
    const float inv_f = rsqrt(ss_f / float(p.hidden) + p.eps);

    const float scalar = bf16ToFloat(layer_scalar[0]);
    float ss_h = 0.0f;
    for (uint i = tid; i < p.hidden; i += THREADS) {
        const float v = (h[i] + (row[i] * inv_f) * bf16ToFloat(w_post[i])) * scalar;
        row[i] = v;
        h[i] = v;
        ss_h += v * v;
    }
    ss_h = threadgroupSum(ss_h, scratch, lane, simd_id, SIMDS);
    const float inv_h = rsqrt(ss_h / float(p.hidden) + p.eps);
    for (uint i = tid; i < p.hidden; i += THREADS) {
        x[i] = (row[i] * inv_h) * bf16ToFloat(next_norm[i]);
    }
}
