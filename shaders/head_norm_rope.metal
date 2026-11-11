#include <metal_stdlib>
#include "quant.h"
#include "reduce.h"
#include "shared.h"
using namespace metal;

// Per-head RMSNorm + RoPE for q, K and V of one layer, in ONE dispatch.
//
// Grid: x = token, y = head slot in [0, q_heads + 2 * kv_heads):
//   [0, q_heads)                    q head:  norm with q_norm, RoPE, written back in place
//   [q_heads, q_heads + kv_heads)   K head:  norm with k_norm, RoPE, written into the K cache
//   [q_heads + kv_heads, ...)       V head:  norm without weight, no RoPE, into the V cache
//
// Cache layout: [kv_heads][capacity][head_dim]; token t lands at slot pos0 + t.
//
// RoPE (mlx traditional=False) pairs element i with i + head_dim/2 and rotates the first
// rope_pairs pairs by angle = pos * base^(-2i / head_dim). On full-attention layers only 64
// of the 256 pairs rotate but the exponent still uses the full head_dim (verified in B6_3).
//
// Reading and writing q in place is safe: each threadgroup reads its own (token, head) row
// into threadgroup memory before writing it back.
kernel void qkv_norm_rope_f32(device float *q [[buffer(0)]],             // [T, q_heads, hd]
                              device const float *k_raw [[buffer(1)]],   // [T, kv_heads, hd]
                              device const float *v_raw [[buffer(2)]],   // [T, kv_heads, hd]
                              device const ushort *q_norm [[buffer(3)]], // BF16 [hd]
                              device const ushort *k_norm [[buffer(4)]], // BF16 [hd]
                              device float *k_cache [[buffer(5)]],
                              device float *v_cache [[buffer(6)]],
                              constant QkvNormRopeParams &p [[buffer(7)]],
                              uint2 tg [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint simd_id [[simdgroup_index_in_threadgroup]]) {
    constexpr uint THREADS = APEX_HEAD_NORM_THREADS;
    const uint token = tg.x;
    const uint slot = tg.y;
    const uint hd = p.head_dim;
    const uint pos = p.pos0 + token;

    // Pick the row this threadgroup normalizes and where it goes.
    device const float *input;
    device float *output;
    device const ushort *weight = nullptr; // nullptr: plain RMSNorm (v_norm)
    bool rope = true;
    if (slot < p.q_heads) {
        input = q + (token * p.q_heads + slot) * hd;
        output = q + (token * p.q_heads + slot) * hd;
        weight = q_norm;
    } else if (slot < p.q_heads + p.kv_heads) {
        const uint kv = slot - p.q_heads;
        input = k_raw + (token * p.kv_heads + kv) * hd;
        output = k_cache + (kv * p.capacity + pos) * hd;
        weight = k_norm;
    } else {
        const uint kv = slot - p.q_heads - p.kv_heads;
        input = v_raw + (token * p.kv_heads + kv) * hd;
        output = v_cache + (kv * p.capacity + pos) * hd;
        rope = false;
    }

    // RMSNorm over head_dim.
    threadgroup float scratch[32];
    float ss = 0.0f;
    for (uint i = tid; i < hd; i += THREADS) {
        ss += input[i] * input[i];
    }
    ss = threadgroupSum(ss, scratch, lane, simd_id, THREADS / 32);
    const float inv = rsqrt(ss / float(hd) + p.eps);

    // Normalized row staged in threadgroup memory: RoPE pairs (i, i + half) are owned by
    // different threads.
    threadgroup float row[APEX_MAX_HEAD_DIM];
    for (uint i = tid; i < hd; i += THREADS) {
        const float x = input[i] * inv;
        row[i] = weight ? x * bf16ToFloat(weight[i]) : x;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (rope) {
        const uint half_dim = hd / 2;
        for (uint i = tid; i < p.rope_pairs; i += THREADS) {
            const float angle = float(pos) * pow(p.rope_base, -2.0f * float(i) / float(hd));
            const float c = cos(angle);
            const float s = sin(angle);
            const float a = row[i];
            const float b = row[half_dim + i];
            row[i] = a * c - b * s;
            row[half_dim + i] = a * s + b * c;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = tid; i < hd; i += THREADS) {
        output[i] = row[i];
    }
}
