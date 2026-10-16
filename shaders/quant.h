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
