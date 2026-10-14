

#include <metal_stdlib>
using namespace metal;
// rms_norm_f32 goes here (B6 stage 1).

// Normalize each input row and multiply by the BF16 weights.
//
// Buffer in memory
//       0      ┌──────────────────────┐
//              │ Token 0: 4096 floats │
//    4096      ├──────────────────────┤
//              │ Token 1: 4096 floats │
//    8192      ├──────────────────────┤
//              │ Token 2: 4096 floats │
//   12288      ├──────────────────────┤
//              │ Token 3: 4096 floats │
//              └──────────────────────┘
// GPU Dispatch
// │
// ├── Threadgroup 0 ── Processes Token 0
// │   │
// │   ├── SIMD Group 0(32 threads)
// │   │   ├── Thread 0(lane 0)
// │   │   ├── Thread 1(lane 1)
// │   │   ├── ...
// │   │   └── Thread 31(lane 31)
// │   │
// │   │
// │   ├── SIMD Group 2(Threads 64–95)
// │   ├── SIMD Group 3(Threads 96–127)...
// │
// ├── Threadgroup 1-3 ── Processes Token 1-3 ....
// │   └── 8 SIMD groups × 32 threads

kernel void rms_norm_f32(device const float *input [[buffer(0)]],    // Input tokens. [rows, n]
                      device const ushort *weights [[buffer(1)]], // RMS norm weights [n]
                      device float *result [[buffer(2)]],         // result buffer  [rows, n]
                      constant uint &hidden_dim [[buffer(3)]],    // hidden dim
                      constant float &eps [[buffer(4)]],
                      uint token [[threadgroup_position_in_grid]], // row for this grup
                      uint tid [[thread_index_in_threadgroup]],
                      uint group_size [[threads_per_threadgroup]], // num of threads ex: 256
                      uint lane [[thread_index_in_simdgroup]],
                      uint simd_id [[simdgroup_index_in_threadgroup]]) {

    // Point to the right offset in the input block (token row)
    input += token * hidden_dim;
    result += token * hidden_dim;

    // Calculate sum of squares in parallel
    float sum_squares = 0.0f; // private for every thread. 
    for (uint i = tid; i < hidden_dim; i += group_size) { // strided access
        sum_squares += input[i] * input[i];
    }
    sum_squares = simd_sum(sum_squares);    // sum across simd group (warp)
    threadgroup float partial[32];  // shared mem
    if (lane == 0) {
        partial[simd_id] = sum_squares;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup); // now we have simd group sums.

    // simd sums -> full sum done by simd 0.
    if (simd_id == 0) {
        uint nsimd = (group_size + 31) / 32;
        float v = (lane < nsimd) ? partial[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) {
            partial[0] = v; // only one thread updates mem
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv = rsqrt(partial[0] / float(hidden_dim) + eps);

    // normalize each val and apply weights.
    for (uint i = tid; i < hidden_dim; i += group_size) {
        float weight = as_type<float>(uint(weights[i]) << 16);
        result [i] = input[i] * inv * weight;
    }
}
