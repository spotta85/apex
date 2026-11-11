#include <metal_stdlib>
#include "shared.h"
using namespace metal;

// Greedy sampling: index of the largest logit of each token row; ties go to the lowest index
// (numpy argmax). One threadgroup of APEX_ARGMAX_THREADS threads per row.
//
// Thread i scans columns i, i + THREADS, ... (coalesced across the threadgroup) keeping the
// first maximum it sees, then simdgroups and finally the threadgroup combine (max value,
// then lowest index among the lanes holding it).
kernel void argmax_f32(device const float *logits [[buffer(0)]], // [T, cols]
                       device uint *out [[buffer(1)]],           // [T]
                       constant ArgmaxParams &p [[buffer(2)]],
                       uint token [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint simd_id [[simdgroup_index_in_threadgroup]]) {
    constexpr uint THREADS = APEX_ARGMAX_THREADS;
    constexpr uint SIMDS = THREADS / 32;
    logits += ulong(token) * p.cols;

    float best = -INFINITY;
    uint best_i = 0xFFFFFFFFu;
    for (uint i = tid; i < p.cols; i += THREADS) {
        const float v = logits[i];
        if (v > best) { // strict: keep the first (lowest) index of equal values
            best = v;
            best_i = i;
        }
    }

    // Combine (value, index) pairs: max value, lowest index among equals.
    float m = simd_max(best);
    uint mi = simd_min(best == m ? best_i : 0xFFFFFFFFu);

    threadgroup float values[SIMDS];
    threadgroup uint indices[SIMDS];
    if (lane == 0) {
        values[simd_id] = m;
        indices[simd_id] = mi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
        const float v = lane < SIMDS ? values[lane] : -INFINITY;
        const uint vi = lane < SIMDS ? indices[lane] : 0xFFFFFFFFu;
        const float top = simd_max(v);
        const uint top_i = simd_min(v == top ? vi : 0xFFFFFFFFu);
        if (lane == 0) out[token] = top_i;
    }
}
