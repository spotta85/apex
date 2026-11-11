// Decode helpers for the quantized formats in the checkpoint.
// Included by every kernel that reads weights; keep it free of kernels.
#pragma once
#include <metal_stdlib>
using namespace metal;

// E2M1 magnitudes indexed by the low 3 bits; bit 3 is the sign.
constant float E2M1_TABLE[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

inline float e2m1ToFloat(uint code) { // code in [0, 16)
    const float v = E2M1_TABLE[code & 7];
    return (code & 8) ? -v : v;
}

// FP8 E4M3 (fn variant): sign | exponent(4, bias 7) | mantissa(3).
// e == 0 is subnormal: m/8 * 2^-6. 0x7F / 0xFF are NaN; we never store them.
inline float e4m3ToFloat(uint b) {
    const uint e = (b >> 3) & 0xF;
    const uint m = b & 7;
    const float v =
        (e == 0) ? float(m) * (1.0f / 512.0f) : ldexp(1.0f + float(m) * 0.125f, int(e) - 7);
    return (b & 0x80) ? -v : v;
}

// BF16 bits -> float: BF16 is the top half of an IEEE float.
inline float bf16ToFloat(ushort bits) { return as_type<float>(uint(bits) << 16); }

// ---------------------------------------------------------------------------------------
// NVFP4 group helpers. A group is 16 consecutive weights of one row: two uint32 words of
// 4-bit codes (element 0 in the lowest nibble of the first word) and one E4M3 scale.
// ---------------------------------------------------------------------------------------

constant constexpr uint NVFP4_GROUP = 16;   // weights per scale
constant constexpr uint NVFP4_PER_WORD = 8; // 4-bit codes per uint32

// Decode the 16 codes of one group (without the scale) into w[0..15].
inline void decodeGroup(uint lo, uint hi, thread float *w) {
    for (uint j = 0; j < NVFP4_PER_WORD; ++j) {
        w[j] = e2m1ToFloat((lo >> (4 * j)) & 0xF);
        w[NVFP4_PER_WORD + j] = e2m1ToFloat((hi >> (4 * j)) & 0xF);
    }
}

// Dot of one decoded group with 16 inputs, before the scale. x must be 16-byte aligned.
// Summation order (element j, then element 8 + j) is the order the first matmul kernel used,
// which matched the reference to ~1e-7; keep it so every NVFP4 kernel rounds the same way.
inline float dotGroup(thread const float *w, device const float *x) {
    device const float4 *x4 = reinterpret_cast<device const float4 *>(x);
    const float4 a = x4[0], b = x4[1], c = x4[2], d = x4[3];
    const float xs[16] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w,
                          c.x, c.y, c.z, c.w, d.x, d.y, d.z, d.w};
    float partial = 0.0f;
    for (uint j = 0; j < NVFP4_PER_WORD; ++j) {
        partial += w[j] * xs[j];
        partial += w[NVFP4_PER_WORD + j] * xs[NVFP4_PER_WORD + j];
    }
    return partial;
}

// gelu_tanh(g) * u, the GeGLU used by the dense MLP and the experts (mlx nn.gelu_approx).
inline float gegluTanh(float g, float u) {
    constexpr float SQRT_TWO_OVER_PI = 0.7978845608028654f;
    constexpr float GELU_CUBIC_COEFF = 0.044715f;
    const float t = tanh(SQRT_TWO_OVER_PI * (g + GELU_CUBIC_COEFF * g * g * g));
    return 0.5f * g * (1.0f + t) * u;
}
