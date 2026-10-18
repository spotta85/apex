#pragma once
#include "backend/backend.h"

namespace apex {
// BTW weights has an offset all weights use one buffer.

// An NVFP4 matrix [w_rows, hidden_dim] in GPU memory: codes and scales may live in the
// same buffer (core.bin, expert slots) at different offsets.
struct Nvfp4Matrix {
    Buffer *weights;
    size_t weights_offset; // uint32 [w_rows, hidden_dim/8]
    Buffer *scales;
    size_t scales_offset; // uint8  [w_rows, hidden_dim/16]
    uint32_t w_rows;
    uint32_t hidden_dim; // 2816
};
// Queue RMSNorm for float32 rows using BF16 weights.
inline void rmsNorm(Backend &backend, Buffer *input, Buffer *weights, size_t weight_offset,
                    Buffer *result, uint32_t rows, uint32_t hidden_dim, float eps) {

    constexpr uint32_t THREADS_PER_GROUP = 256;
    backend.run("rms_norm_f32",
                {Arg::buf(input), Arg::buf(weights, weight_offset), Arg::buf(result),
                 Arg::val(hidden_dim), Arg::val(eps)},
                Dim3{rows}, Dim3{THREADS_PER_GROUP});
}

/// @brief Runs MatmulNvfp4 kernel with a packed weights.
/// @param backend pointer to backend running this.
/// @param input Pointer to input values.
/// @param W Nvfp4Matrix with weights and scales for quantized & packed matmul.
/// @param output
/// @param tokens number of tokens.
inline void matmulNvfp4(Backend &backend, Buffer *input, const Nvfp4Matrix &W, Buffer *output,
                        uint32_t tokens) {
    constexpr uint32_t SIMD_PER_GROUP = 8; // btw one simd does one row of W.
    constexpr uint32_t THREADS_PER_GROUP = 256;
    // each
    const uint32_t row_groups = (W.w_rows + SIMD_PER_GROUP - 1) / SIMD_PER_GROUP;
    backend.run("matmul_nvfp4",
                {Arg::buf(input), Arg::buf(W.weights, W.weights_offset),
                 Arg::buf(W.scales, W.scales_offset), Arg::buf(output), Arg::val(W.hidden_dim),
                 Arg::val(W.w_rows)},
                Dim3{row_groups, tokens},
                Dim3{THREADS_PER_GROUP}); // x = row-groups and y = tokens.
}

// Queue GELU(gate) * up for every element.
inline void geglu(Backend &backend, Buffer *gate, Buffer *up, Buffer *output, uint32_t count) {
    constexpr uint32_t THREADS_PER_GROUP = 256;
    const uint32_t groups = (count + THREADS_PER_GROUP - 1) / THREADS_PER_GROUP;

    backend.run("geglu_f32", {Arg::buf(gate), Arg::buf(up), Arg::buf(output), Arg::val(count)},
                Dim3{groups}, Dim3{THREADS_PER_GROUP});
}

// Calls backend.run with the proper configurations given the inputs to do a dense mlp layer.
/// input → gate projection → gate_tmp ─┐
/// ----------------------------------- GeGLU → gate_tmp → down → output
/// input → up projection   → up_tmp ───┘
inline void denseMlp(Backend &backend, Buffer *input, const Nvfp4Matrix &gate_weights,
                     const Nvfp4Matrix &up_weights, const Nvfp4Matrix &down_weights,
                     Buffer *gate_tmp, Buffer *up_tmp, Buffer *output, uint32_t tokens) {

    matmulNvfp4(backend, input, gate_weights, gate_tmp, tokens);
    matmulNvfp4(backend, input, up_weights, up_tmp, tokens);
    geglu(backend, gate_tmp, up_tmp, gate_tmp, tokens * gate_weights.w_rows);
    matmulNvfp4(backend, gate_tmp, down_weights, output, tokens);
}

/// @brief Queue per-head RMSNorm, optional learned scaling, and optional RoPE.
/// @param backend
/// @param input
/// @param weights // learned BF16 RMSNorm weights shaped [head_dim]
/// @param weights_offset
/// @param has_weights if RMS norm applies learned nrom weights (none for V)
/// @param output
/// @param output_offset
/// @param tokens
/// @param heads
/// @param head_dim
/// @param rope_pairs // number of dimension pairs to rotate.
/// @param rope_base
/// @param abs_pos
/// @param out_token_stride Distance in float elements between consecutive output tokens.
/// @param out_head_stride Distance in float elements between consecutive output heads.
/// @param eps
inline void headNormRope(Backend &backend, 
    Buffer *input, 
    Buffer *weights, 
    size_t weights_offset,
                         bool has_weights, 
                         Buffer *output, 
                         size_t output_offset, 
                         uint32_t tokens,
                         uint32_t heads, 
                         uint32_t head_dim, 
                         uint32_t rope_pairs,  float rope_base,
                         uint32_t abs_pos, 
                         uint32_t out_token_stride, uint32_t out_head_stride,
                         float eps) {

    constexpr uint32_t THREADS_PER_GROUP = 256;
    const uint32_t weight_flag = has_weights ? 1u : 0u;

    backend.run("head_norm_rope_f32",
                {
                    Arg::buf(input),
                    Arg::buf(weights, weights_offset),
                    Arg::buf(output, output_offset),
                    Arg::val(heads),
                    Arg::val(head_dim),
                    Arg::val(weight_flag),
                    Arg::val(rope_pairs),
                    Arg::val(rope_base),
                    Arg::val(abs_pos),
                    Arg::val(out_token_stride),
                    Arg::val(out_head_stride),
                    Arg::val(eps),
                },
                Dim3{tokens, heads}, Dim3{THREADS_PER_GROUP});
}

} // namespace apex