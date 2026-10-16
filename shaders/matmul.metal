#include "quant.h"

// Multiply an NVFP4 weight matrix by each token's input vector.
kernel void matmul_nvfp4(device const float *input [[buffer(0)]], // asdfs
                         device const uint *weights [[buffer(1)]],
                         device const uchar *scales [[buffer(2)]], // asdfsdfw
                         device float *output [[buffer(3)]],
                         constant uint &hidden_dim [[buffer(4)]], 
                         constant uint &w_rows [[buffer(5)]],
                         uint2 tg [[threadgroup_position_in_grid]],
                         uint simd_id [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {

    constexpr uint SIMD_PER_GROUP = 8;
    const uint row = tg.x * SIMD_PER_GROUP + simd_id; // each SIMD does 1 row of W
    if (row >= w_rows)
        return;
    const uint token = tg.y; // which token in the input (row in input)

    constexpr uint PACKED_WEIGHTS_PER_INT = 8;
    constexpr uint WEIGHTS_PER_SCALE = 16;
    const uint scale_count = hidden_dim / WEIGHTS_PER_SCALE;

    input += token * hidden_dim; // go to correct row in the input matrix.
    weights += row * (hidden_dim / PACKED_WEIGHTS_PER_INT); // hidden_dim / PAC... = num_cols of weights tensor
    scales += row * scale_count;

    constexpr uint LANES_PER_SIMD = 32;

    float dot = 0.0f;
    for (uint i = lane; i < scale_count; i += LANES_PER_SIMD) {
        const uint lo = weights[2 * i]; // packed vals from W
        const uint hi = weights[2 * i + 1];
        device const float *input_group = input + i * WEIGHTS_PER_SCALE; // input vals (16)

        constexpr uint BITS_PER_WEIGHT = 4;
        constexpr uint WEIGHT_MASK = 0xF; // (16)
        float partial = 0.0f;

        // convert and add each packed value from 2 ints.
        for (uint j = 0; j < PACKED_WEIGHTS_PER_INT; j++) {
            const uint shift = BITS_PER_WEIGHT * j;
            partial += e2m1ToFloat((lo >> shift) & WEIGHT_MASK) * input_group[j];
            partial += e2m1ToFloat((hi >> shift) & WEIGHT_MASK) * input_group[PACKED_WEIGHTS_PER_INT + j];
        }
        dot += e4m3ToFloat(scales[i]) * partial; // multiply sum by scale.
    }
    // sum up for all simd (this whole row in W)
    dot = simd_sum(dot);
    if (lane == 0)
            output[(token * w_rows) + row] = dot; // token * w_rows = row of output

}