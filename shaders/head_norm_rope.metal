#include <metal_stdlib>
using namespace metal;

// Normalize each head, optionally apply RoPE, and write to the output.
kernel void head_norm_rope_f32(device const float *input [[buffer(0)]], // [T, heads, head_dim]
                               device const ushort *weight [[buffer(1)]], // [head_dim]
                               device float *output [[buffer(2)]], // 
                               constant uint &heads [[buffer(3)]],    //
                               constant uint &head_dim [[buffer(4)]], //
                               constant uint &has_weight [[buffer(5)]],
                               constant uint &rope_pairs [[buffer(6)]], // pairs to rotate (128 or 64 or 0)
                               constant float &rope_base [[buffer(7)]], // rot_freq_base
                               constant uint &abs_pos [[buffer(8)]],
                               constant uint &out_token_stride [[buffer(9)]],
                               constant uint &out_head_stride [[buffer(10)]],
                               constant float &eps [[buffer(11)]],
                               uint2 group [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint simd_id [[simdgroup_index_in_threadgroup]]) {

    const uint token = group.x;
    const uint head = group.y;

    input += (token * heads + head) * head_dim;
    output += token * out_token_stride + head * out_head_stride;

    float ss = 0.0f;
    for (uint i = tid; i < head_dim; i += 256) {
        ss += input[i]* input[i];
    }
    ss = simd_sum(ss);
    threadgroup float partial [8]; // sum up every simd group in tg
    if (lane == 0) {
        partial[simd_id] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
        float v = (lane < 8) ? partial[lane] : 0;
        v = simd_sum(v); // use 8 lanes to sum up values from partial
        if (lane == 0) partial[0] = v; // put total sum in 0
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = rsqrt(partial[0] / float(head_dim) + eps);

    threadgroup float row[512]; // shared threadgroup memory
    for (uint i = tid; i < head_dim; i += 256) { // 256 threads in tg
        const float w = has_weight ? as_type<float>(uint(weight[i]) << 16) : 1.0f;
        row[i] = input[i] * inv * w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // now we have the row in shared threadgroup memory and we need to do rotation. 
    const uint half_dim = head_dim / 2;
    // Rotation angle is per token and per pair. Each thread does one pair
    if (tid < rope_pairs) {
        const float pos = float(abs_pos + token);
        // frequency  = inner then we mutliply by pos for angle.
        const float angle = pos * pow(rope_base, -2.0f * float(tid) / float(head_dim));

        const float c = cos(angle);
        const float s = sin(angle);
        const float a = row[tid];
        const float b = row[half_dim + tid];

        row[tid] = a * c - b * s;
        row[half_dim + tid] = a * s + b * c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < head_dim; i += 256) {
        output[i] = row[i];
    }
}
