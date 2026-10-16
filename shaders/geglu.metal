#include <metal_stdlib>
using namespace metal;

/// Apply GeLU and then multiply with up projection.
/// Simple: each thread does one calculation. 
kernel void geglu_f32(device const float *gate [[buffer(0)]], // [T, intermediate_dim]
                      device const float *up [[buffer(1)]],
                      device float *output [[buffer(2)]],
                      constant uint &count [[buffer(3)]], // total num of elements (T * 2112)
                      uint index [[thread_position_in_grid]]) {
    if (index >= count)
        return;

    // GeLU Calculation.
    constexpr float SQRT_TWO_OVER_PI = 0.7978845608028654f;
    constexpr float GELU_CUBIC_COEFF = 0.044715f;

    const float g = gate[index];
    const float t = tanh(SQRT_TWO_OVER_PI * (g + GELU_CUBIC_COEFF * g * g * g));
    const float activated = 0.5f * g * (1.0f + t);

    // multiply by up prjoection
    output[index] = activated * up[index];
}