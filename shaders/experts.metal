#include <metal_stdlib>
#include "quant.h"
#include "shared.h"
using namespace metal;

// Routed experts, two dispatches per layer:
//
//   expert_gate_up_nvfp4  act[t, k, r] = gelu_tanh(gate_s . x_t) * (up_s . x_t)
//                         for every token t and each of its routed slots s = routes[t, k].slot
//   expert_down_nvfp4     moe[t, r]    = sum_k routes[t, k].weight * (down_s[r] . act[t, k])
//
// The experts of one layer live in one buffer of fixed-size slots; slot s starts at byte
// s * layout.stride and the six pieces sit at the layout's offsets (from the manifest). The
// kernels only ever see slot indices, so the same code serves the resident engine (slot ==
// expert id, the buffer is the whole layer file) and a small disk-backed slot cache.
//
// Experts are per token, so there is no cross-token weight reuse here: each (token, expert)
// pair reads its expert's rows once. Lane layout is the matmul's: one simdgroup per output
// row, lanes take 16-weight groups round-robin.

constant constexpr uint ROWS_PER_GROUP = APEX_MATMUL_ROWS_PER_GROUP;
constant constexpr uint LANES = 32;

// Start of slot `slot` in the layer's expert buffer.
inline device const uchar *slotBase(device const uchar *experts, uint slot, uint stride) {
    return experts + ulong(slot) * ulong(stride);
}

// Grid: x = row groups of the intermediate size (704 / 8 = 88), y = route index k, z = token.
kernel void expert_gate_up_nvfp4(device const float *input [[buffer(0)]],        // [T, in_dim]
                                 device const uchar *experts [[buffer(1)]],      // slot buffer
                                 device const ExpertRoute *routes [[buffer(2)]], // [T, max_k]
                                 device const uint *counts [[buffer(3)]],        // [T]
                                 device float *act [[buffer(4)]],                // [T, max_k, inter]
                                 constant ExpertParams &p [[buffer(5)]],
                                 uint3 tg [[threadgroup_position_in_grid]],
                                 uint simd_id [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    const uint token = tg.z;
    const uint k = tg.y;
    const uint row = tg.x * ROWS_PER_GROUP + simd_id;
    if (k >= counts[token] || row >= p.inter) {
        return;
    }
    const uint slot = routes[token * p.max_k + k].slot;
    device const uchar *base = slotBase(experts, slot, p.layout.stride);
    const uint groups = p.in_dim / NVFP4_GROUP;
    const uint row_words = p.in_dim / NVFP4_PER_WORD;
    device const uint *gate_w =
        reinterpret_cast<device const uint *>(base + p.layout.gate_w) + row * row_words;
    device const uint *up_w =
        reinterpret_cast<device const uint *>(base + p.layout.up_w) + row * row_words;
    device const uchar *gate_s = base + p.layout.gate_s + row * groups;
    device const uchar *up_s = base + p.layout.up_s + row * groups;
    input += token * p.in_dim;

    float acc_g = 0.0f, acc_u = 0.0f;
    for (uint g = lane; g < groups; g += LANES) {
        float wg[NVFP4_GROUP], wu[NVFP4_GROUP];
        decodeGroup(gate_w[2 * g], gate_w[2 * g + 1], wg);
        decodeGroup(up_w[2 * g], up_w[2 * g + 1], wu);
        device const float *x = input + g * NVFP4_GROUP;
        acc_g += e4m3ToFloat(gate_s[g]) * dotGroup(wg, x);
        acc_u += e4m3ToFloat(up_s[g]) * dotGroup(wu, x);
    }
    const float gate = simd_sum(acc_g);
    const float up = simd_sum(acc_u);
    if (lane == 0) {
        act[(token * p.max_k + k) * p.inter + row] = gegluTanh(gate, up);
    }
}

// Grid: x = row groups of the hidden size (2816 / 8 = 352), y = token.
// Each simdgroup owns one output row and walks the token's routed experts: per expert it
// reduces the row's dot product fully (simd_sum), then adds weight * y. That is the
// reference's order: y_k = down_k(act_k) first, then sum_k w_k * y_k.
kernel void expert_down_nvfp4(device const float *act [[buffer(0)]],            // [T, max_k, inter]
                              device const uchar *experts [[buffer(1)]],
                              device const ExpertRoute *routes [[buffer(2)]],
                              device const uint *counts [[buffer(3)]],
                              device float *output [[buffer(4)]],               // [T, in_dim]
                              constant ExpertParams &p [[buffer(5)]],
                              uint2 tg [[threadgroup_position_in_grid]],
                              uint simd_id [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
    const uint token = tg.y;
    const uint row = tg.x * ROWS_PER_GROUP + simd_id;
    if (row >= p.in_dim) {
        return;
    }
    const uint n = counts[token];
    const uint groups = p.inter / NVFP4_GROUP;    // 44
    const uint row_words = p.inter / NVFP4_PER_WORD;

    float total = 0.0f;
    for (uint k = 0; k < n; ++k) {
        const ExpertRoute route = routes[token * p.max_k + k];
        device const uchar *base = slotBase(experts, route.slot, p.layout.stride);
        device const uint *down_w =
            reinterpret_cast<device const uint *>(base + p.layout.down_w) + row * row_words;
        device const uchar *down_s = base + p.layout.down_s + row * groups;
        device const float *a = act + (token * p.max_k + k) * p.inter;
        float partial = 0.0f;
        for (uint g = lane; g < groups; g += LANES) {
            float w[NVFP4_GROUP];
            decodeGroup(down_w[2 * g], down_w[2 * g + 1], w);
            partial += e4m3ToFloat(down_s[g]) * dotGroup(w, a + g * NVFP4_GROUP);
        }
        total += route.weight * simd_sum(partial);
    }
    if (lane == 0) {
        output[token * p.in_dim + row] = total;
    }
}
