
// CPU reference math shared by the parity tests. Slow and simple on purpose.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>
#include "../nvfp4.h"

// Normalize one token vector on the CPU using BF16 weights.
inline void rmsNormCpu(const float *x, const uint16_t *w_bf16, float *y, size_t n, float eps) {
    double ss = 0;
    for (size_t i = 0; i < n; ++i)
        ss += double(x[i]) * x[i];
    float inv = 1.0f / std::sqrt(float(ss / n) + eps);
    for (size_t i = 0; i < n; ++i) {
        uint32_t bits = uint32_t(w_bf16[i]) << 16;
        float wf;
        std::memcpy(&wf, &bits, 4);
        y[i] = x[i] * inv * wf;
    }
}

// max |a-b| and relative L2 ||a-b|| / ||b||
struct Diff {
    double max_abs = 0, rel_l2 = 0;
};
inline Diff diff(const float *a, const float *b, size_t n) {
    double num = 0, den = 0;
    Diff d;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            return {std::numeric_limits<double>::infinity(),
                    std::numeric_limits<double>::infinity()};
        }
        double e = double(a[i]) - b[i];
        num += e * e;
        den += double(b[i]) * b[i];
        d.max_abs = std::max(d.max_abs, std::fabs(e));
    }
    d.rel_l2 = den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
    return d;
}

// output[t, :] = W · input[t, :] for an NVFP4 matrix W [rows, input_size].
// Decodes each weight row once, then dots it with every token.
inline void matvecNvfp4Cpu(const uint32_t *codes, const uint8_t *scales, const float *input,
                           float *output, size_t tokens, size_t rows, size_t input_size) {
    std::vector<float> row(input_size);
    for (size_t r = 0; r < rows; ++r) {
        apex::decodeRowNvfp4(codes + r * (input_size / 8), scales + r * (input_size / 16),
                             row.data(), input_size);
        for (size_t t = 0; t < tokens; ++t) {
            double sum = 0;
            for (size_t i = 0; i < input_size; ++i)
                sum += double(row[i]) * input[t * input_size + i];
            output[t * rows + r] = float(sum);
        }
    }
}

// GELU, tanh approximation (same formula as geglu.metal).
inline float geluTanh(float g) {
    return 0.5f * g * (1.0f + std::tanh(0.7978845608028654f * (g + 0.044715f * g * g * g)));
}
