#include <metal_stdlib>
#include "quant.h"
#include "reduce.h"
#include "shared.h"
using namespace metal;

// MoE router, fully fused: one threadgroup per token does (mlx_lm gemma4_text.Router)
//
//   x      = rms_norm(h, router.scale * hidden^-0.5)      h is the RAW post-attention hidden
//   scores = proj(x)                                       8-bit affine, group 64, BF16 scales/biases
//   ids    = top_k(scores)                                 k = 8
//   w      = softmax(scores[ids]) * per_expert_scale[ids]
//
// and writes the token's (slot, weight) list and count straight into GPU buffers, so the
// expert kernels can read them without the CPU ever seeing the routing. In resident mode the
// slot of an expert is its id.
//
// Affine 8-bit: each uint32 of proj.weight holds four codes, element 0 in the lowest byte.
// Element i of row e dequantizes to q * scales[e][i/64] + biases[e][i/64]. Like mlx's qmv,
// each lane accumulates sum(q*x) and sum(x) over the 4 elements of one word and applies
// scale and bias per word: result += scale * sum(q*x) + bias * sum(x).
kernel void router_f32(device const float *h [[buffer(0)]],             // [T, hidden]
                       device const ushort *scale [[buffer(1)]],         // BF16 [hidden]
                       device const uint *proj_w [[buffer(2)]],          // [experts, hidden/4]
                       device const ushort *proj_scales [[buffer(3)]],   // BF16 [experts, hidden/64]
                       device const ushort *proj_biases [[buffer(4)]],   // BF16 [experts, hidden/64]
                       device const ushort *per_expert [[buffer(5)]],    // BF16 [experts]
                       device ExpertRoute *routes [[buffer(6)]],         // [T, top_k]
                       device uint *counts [[buffer(7)]],                // [T]
                       constant RouterParams &p [[buffer(8)]],
                       uint token [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint simd_id [[simdgroup_index_in_threadgroup]]) {
    constexpr uint THREADS = APEX_ROUTER_THREADS;
    constexpr uint SIMDS = THREADS / 32;
    threadgroup float x[APEX_MAX_HIDDEN];
    threadgroup float scores[APEX_MAX_EXPERTS];
    threadgroup float scratch[32];

    h += token * p.hidden;

    // 1. RMSNorm with the effective weight scale * root_scale (computed in float, as mlx does
    //    `self.scale * self._root_size` before calling rms_norm).
    float ss = 0.0f;
    for (uint i = tid; i < p.hidden; i += THREADS) ss += h[i] * h[i];
    ss = threadgroupSum(ss, scratch, lane, simd_id, SIMDS);
    const float inv = rsqrt(ss / float(p.hidden) + p.eps);
    for (uint i = tid; i < p.hidden; i += THREADS) {
        const float w = bf16ToFloat(scale[i]) * p.root_scale;
        x[i] = (h[i] * inv) * w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. Scores: one simdgroup per expert row. Lane l reads words l, l + 32, ... of the row,
    //    so the simdgroup reads 128 contiguous bytes per step and x as consecutive float4s.
    const uint words = p.hidden / 4;
    const uint words_per_group = p.group_size / 4;
    const uint groups = p.hidden / p.group_size;
    threadgroup const float4 *x4 = reinterpret_cast<threadgroup const float4 *>(x);
    for (uint e = simd_id; e < p.experts; e += SIMDS) {
        device const uint *row = proj_w + e * words;
        device const ushort *srow = proj_scales + e * groups;
        device const ushort *brow = proj_biases + e * groups;
        float partial = 0.0f;
        for (uint wi = lane; wi < words; wi += 32) {
            const uint q = row[wi];
            const float4 xv = x4[wi];
            const float qdot = float(q & 0xFF) * xv.x + float((q >> 8) & 0xFF) * xv.y +
                               float((q >> 16) & 0xFF) * xv.z + float(q >> 24) * xv.w;
            const float xsum = xv.x + xv.y + xv.z + xv.w;
            const uint g = wi / words_per_group;
            partial += bf16ToFloat(srow[g]) * qdot + bf16ToFloat(brow[g]) * xsum;
        }
        const float s = simd_sum(partial);
        if (lane == 0) scores[e] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. Top-k, softmax, per_expert_scale: one simdgroup. Lane l holds experts l, l+32, l+64,
    //    l+96. Each round takes the max (lowest id on a tie) and removes it; round k's winner
    //    is kept by lane k, so after the loop lanes 0..k-1 hold the picks in score order.
    if (simd_id != 0) return; // no barriers below
    constexpr uint PER_LANE = APEX_MAX_EXPERTS / 32;
    float v[PER_LANE];
    for (uint j = 0; j < PER_LANE; ++j) {
        const uint e = lane + 32 * j;
        v[j] = e < p.experts ? scores[e] : -INFINITY;
    }
    float my_score = -INFINITY;
    uint my_id = 0;
    for (uint k = 0; k < p.top_k; ++k) {
        float best = -INFINITY;
        uint best_id = 0xFFFFFFFFu;
        for (uint j = 0; j < PER_LANE; ++j) {
            if (v[j] > best) { // strict: the lower j (lower id) wins inside a lane
                best = v[j];
                best_id = lane + 32 * j;
            }
        }
        const float m = simd_max(best);
        const uint id = simd_min(best == m ? best_id : 0xFFFFFFFFu);
        for (uint j = 0; j < PER_LANE; ++j) {
            if (lane + 32 * j == id) v[j] = -INFINITY;
        }
        if (lane == k) {
            my_score = m;
            my_id = id;
        }
    }
    const bool active = lane < p.top_k;
    const float top = simd_max(active ? my_score : -INFINITY);
    const float e = active ? exp(my_score - top) : 0.0f;
    const float sum = simd_sum(e);
    if (active) {
        const float weight = e * (1.0f / sum) * bf16ToFloat(per_expert[my_id]);
        routes[token * p.top_k + lane] = ExpertRoute{my_id, weight};
    }
    if (lane == 0) counts[token] = p.top_k;
}
