
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

// BF16 bits -> float.
inline float bf16Cpu(uint16_t bits) {
    const uint32_t u = uint32_t(bits) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// Normalize one token vector on the CPU using BF16 weights (nullptr: weight 1.0).
// Reads all of x before writing y, so x == y (in place) is fine.
inline void rmsNormCpu(const float *x, const uint16_t *w_bf16, float *y, size_t n, float eps) {
    double ss = 0;
    for (size_t i = 0; i < n; ++i)
        ss += double(x[i]) * x[i];
    float inv = 1.0f / std::sqrt(float(ss / n) + eps);
    for (size_t i = 0; i < n; ++i) {
        y[i] = w_bf16 ? x[i] * inv * bf16Cpu(w_bf16[i]) : x[i] * inv;
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

// RoPE on one head vector, mlx traditional=False pairing (i, i + half).
inline void ropeCpu(float *x, size_t head_dim, size_t pairs, float base, uint32_t pos) {
    const size_t half = head_dim / 2;
    for (size_t i = 0; i < pairs; ++i) {
        const float angle = float(pos) * std::pow(base, -2.0f * float(i) / float(head_dim));
        const float c = std::cos(angle), s = std::sin(angle);
        const float a = x[i], b = x[half + i];
        x[i] = a * c - b * s;
        x[half + i] = a * s + b * c;
    }
}

// softmax_j(q·K_j) V_j for one query over keys first..last (scale 1.0). Double accumulation.
// K and V are [positions][head_dim] for one kv head.
inline void attentionRowCpu(const float *q, const float *K, const float *V, float *out,
                            size_t head_dim, size_t first, size_t last) {
    std::vector<double> score(last - first + 1);
    double m = -INFINITY;
    for (size_t j = first; j <= last; ++j) {
        double s = 0;
        for (size_t d = 0; d < head_dim; ++d)
            s += double(q[d]) * K[j * head_dim + d];
        score[j - first] = s;
        m = std::max(m, s);
    }
    double l = 0;
    for (double &s : score) {
        s = std::exp(s - m);
        l += s;
    }
    for (size_t d = 0; d < head_dim; ++d) {
        double acc = 0;
        for (size_t j = first; j <= last; ++j)
            acc += score[j - first] * V[j * head_dim + d];
        out[d] = float(acc / l);
    }
}

// Per-head norms + RoPE + causal (optionally windowed) attention on raw projections, for a
// whole sequence at positions 0..tokens-1. q_raw [T, nq, hd], k_raw/v_raw [T, nkv, hd],
// output o [T, nq, hd]. q_norm/k_norm may be nullptr (weight 1.0).
inline void attentionCoreCpu(const float *q_raw, const float *k_raw, const float *v_raw,
                             const uint16_t *q_norm, const uint16_t *k_norm, float *o,
                             size_t tokens, size_t nq, size_t nkv, size_t hd, size_t rope_pairs,
                             float rope_base, size_t window, float eps) {
    std::vector<float> q(q_raw, q_raw + tokens * nq * hd);
    std::vector<float> K(nkv * tokens * hd), V(nkv * tokens * hd); // [nkv][T][hd]
    for (size_t t = 0; t < tokens; ++t) {
        for (size_t h = 0; h < nq; ++h) {
            float *row = q.data() + (t * nq + h) * hd;
            rmsNormCpu(row, q_norm, row, hd, eps);
            ropeCpu(row, hd, rope_pairs, rope_base, uint32_t(t));
        }
        for (size_t h = 0; h < nkv; ++h) {
            float *k = K.data() + (h * tokens + t) * hd;
            rmsNormCpu(k_raw + (t * nkv + h) * hd, k_norm, k, hd, eps);
            ropeCpu(k, hd, rope_pairs, rope_base, uint32_t(t));
            rmsNormCpu(v_raw + (t * nkv + h) * hd, nullptr, V.data() + (h * tokens + t) * hd, hd,
                       eps);
        }
    }
    for (size_t t = 0; t < tokens; ++t) {
        const size_t first = (window && t + 1 > window) ? t + 1 - window : 0;
        for (size_t h = 0; h < nq; ++h) {
            const size_t kvh = h / (nq / nkv);
            attentionRowCpu(q.data() + (t * nq + h) * hd, K.data() + kvh * tokens * hd,
                            V.data() + kvh * tokens * hd, o + (t * nq + h) * hd, hd, first, t);
        }
    }
}

// Deterministic pseudo-random floats in [-1, 1) for synthetic tests.
struct Lcg {
    uint64_t state;
    explicit Lcg(uint64_t seed) : state(seed * 6364136223846793005ull + 1442695040888963407ull) {}
    float next() {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        return float((state >> 40) & 0xFFFFFF) / float(1 << 23) - 1.0f;
    }
};
