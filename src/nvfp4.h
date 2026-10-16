// CPU reference decoder for the checkpoint's NVFP4 weights.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace apex {

// Decode a 4-bit E2M1 code, with the sign in bit 3.
inline float e2m1ToFloat(uint32_t code) {
    static constexpr float table[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float value = table[code & 7];
    return (code & 8) ? -value : value;
}

// Decode a finite FP8 E4M3 scale; the checkpoint never stores NaN codes.
inline float e4m3ToFloat(uint8_t byte) {
    const uint32_t exponent = (byte >> 3) & 0xF;
    const uint32_t mantissa = byte & 7;
    const float value = exponent == 0
                            ? float(mantissa) * (1.0f / 512.0f)
                            : std::ldexp(1.0f + float(mantissa) * 0.125f, int(exponent) - 7);
    return (byte & 0x80) ? -value : value;
}

// Decode one row into output. input_size must be a multiple of 16.
inline void decodeRowNvfp4(const uint32_t* codes, const uint8_t* scales, float* output,
                          size_t input_size) {
    constexpr size_t WEIGHTS_PER_WORD = 8;
    constexpr size_t WEIGHTS_PER_SCALE = 16;
    constexpr uint32_t BITS_PER_WEIGHT = 4;
    constexpr uint32_t WEIGHT_MASK = 0xF;

    for (size_t group = 0; group < input_size / WEIGHTS_PER_SCALE; ++group) {
        const float scale = e4m3ToFloat(scales[group]);
        for (size_t i = 0; i < WEIGHTS_PER_SCALE; ++i) {
            const uint32_t word = codes[2 * group + i / WEIGHTS_PER_WORD];
            const uint32_t code = (word >> (BITS_PER_WEIGHT * (i % WEIGHTS_PER_WORD))) & WEIGHT_MASK;
            output[group * WEIGHTS_PER_SCALE + i] = e2m1ToFloat(code) * scale;
        }
    }
}

} // namespace apex
