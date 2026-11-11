#include <cmath>
#include <metal_compute.h>
#include <metal_stdlib>
using namespace metal;

// for each new token t:
//   for each query head h:
//     kv = h / 2
//     pos = pos0 + t
//     for each key j in 0..pos:            # 1. score
//       s[j] = dot(Q[t,h], K[kv][j])
//     w = softmax(s)                         # 2. normalize
//     out[t,h] = sum_j  w[j] * V[kv][j]      # 3. blend


// Number of keys processed together in each iteration.
constant constexpr uint CHUNK = 256;

/// Return maxiumum across 256 threads.
static float goupMax(float value, uint lane, uint simd_id, threadgroup float *scratch) {
    // sum withing lanes of simd group
    value = simd_max(value);
    if (lane == 0)
        scratch[simd_id] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // sum across simd groups (8)
    if (simd_id == 0) {
        float maximum = lane < 8 ? scratch[lane] : -INFINITY;
        maximum = simd_max(maximum);
        if (lane == 0)
            scratch[0] = maximum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float result = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}

// Return the sum across 256 threads using 8 shared partial results.
// uses the same llgic as above. 
static float groupSum(float value, uint lane, uint simd_id, threadgroup float *scratch) {
    value = simd_sum(value);
    if (lane == 0)
        scratch[simd_id] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float sum = lane < 8 ? scratch[lane] : 0.0f;
        sum = simd_sum(sum);
        if (lane == 0)
            scratch[0] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float result = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}

// Compute attention for each token and query head using cached K/V.
kernel void attention_f32(
    device const float *q [[buffer(0)]],      // [tokens, q_heads, head_dim]
    device const float *keys [[buffer(1)]],   // [kv_heads, capacity, head_dim] (conatains cached values)
    device const float *values [[buffer(2)]], // [kv_heads, capacity, head_dim] (contains cached values)
    device float *output [[buffer(3)]],       // [tokens, q_heads, head_dim]
    constant uint &q_heads [[buffer(4)]],     // num of query heads
    constant uint &kv_heads [[buffer(5)]],    // num kv heads
    constant uint &head_dim [[buffer(6)]], constant uint &capacity [[buffer(7)]],
    constant uint &pos0 [[buffer(8)]],   // absolute position of the first query token.
    constant uint &window [[buffer(9)]], // 1024 for sliding, 0 for normal
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]) {

    // Each group: read one Q vector, scan visible K/V pos in chunks and produces one output vecotr [head_dim]



    
    
}