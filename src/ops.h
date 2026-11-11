/// @file ops.h
/// @brief Host-side launchers for every kernel. Each launcher knows its kernel's grid and
/// argument order; the engine and the tests call these instead of Backend::run directly.
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include "backend/backend.h"
#include "../shaders/shared.h"

namespace apex {
// BTW weights have an offset: all core weights live in one buffer.

// An NVFP4 matrix [w_rows, hidden_dim] in GPU memory: codes and scales may live in the
// same buffer (core.bin, expert slots) at different offsets.
struct Nvfp4Matrix {
    Buffer *weights;
    size_t weights_offset; // uint32 [w_rows, hidden_dim/8]
    Buffer *scales;
    size_t scales_offset; // uint8  [w_rows, hidden_dim/16]
    uint32_t w_rows;
    uint32_t hidden_dim; // K (2816 for most matrices)
};

// A view of a buffer at a byte offset; used for weight vectors and cache slices.
struct BufferRef {
    Buffer *buffer = nullptr;
    size_t offset = 0;
};

inline uint32_t ceilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

/// @brief Every kernel handle, resolved once from the backend, plus their launchers.
/// Construct one per backend at load time; launching is then free of string lookups.
class Ops {
  public:
    explicit Ops(Backend &backend)
        : backend_(backend), rms_norm_(backend.kernel("rms_norm_f32")),
          matmul_(backend.kernel("matmul_nvfp4")),
          gate_up_geglu_(backend.kernel("gate_up_geglu_nvfp4")) {}

    Backend &backend() { return backend_; }

    // Queue RMSNorm for float32 rows using BF16 weights.
    void rmsNorm(Buffer *input, Buffer *weights, size_t weight_offset, Buffer *result,
                 uint32_t rows, uint32_t hidden_dim, float eps) {
        constexpr uint32_t THREADS_PER_GROUP = 256;
        backend_.run(rms_norm_,
                     {Arg::buf(input), Arg::buf(weights, weight_offset), Arg::buf(result),
                      Arg::val(hidden_dim), Arg::val(eps)},
                     Dim3{rows}, Dim3{THREADS_PER_GROUP});
    }

    /// @brief Up to three NVFP4 matrices that share one input, in one dispatch.
    /// mats[m] reads input [tokens, K] and writes outputs[m] [tokens, mats[m].w_rows].
    /// softcap > 0 applies tanh(y / softcap) * softcap (the output head).
    void matmulNvfp4(Buffer *input, size_t input_offset, const Nvfp4Matrix *mats,
                     const BufferRef *outputs, uint32_t count, uint32_t tokens,
                     float softcap = 0.0f) {
        if (count < 1 || count > 3) throw std::runtime_error("matmulNvfp4: 1..3 matrices");
        MatmulParams p{};
        p.in_dim = mats[0].hidden_dim;
        p.tokens = tokens;
        p.softcap = softcap;
        uint32_t rows[3] = {0, 0, 0};
        uint32_t total_rows = 0;
        for (uint32_t m = 0; m < count; ++m) {
            if (mats[m].hidden_dim != p.in_dim || mats[m].w_rows % APEX_MATMUL_ROWS_PER_GROUP)
                throw std::runtime_error("matmulNvfp4: matrices must share K, rows % 8 == 0");
            rows[m] = mats[m].w_rows;
            total_rows += rows[m];
        }
        p.rows0 = rows[0];
        p.rows1 = rows[1];
        p.rows2 = rows[2];
        // Unused slots are bound to matrix 0; the kernel never reads them (rows_m == 0).
        auto mat = [&](uint32_t m) { return mats[m < count ? m : 0]; };
        auto out = [&](uint32_t m) { return outputs[m < count ? m : 0]; };
        const Nvfp4Matrix a = mat(0), b = mat(1), c = mat(2);
        const BufferRef oa = out(0), ob = out(1), oc = out(2);
        backend_.run(matmul_,
                     {Arg::buf(input, input_offset), Arg::buf(a.weights, a.weights_offset),
                      Arg::buf(a.scales, a.scales_offset), Arg::buf(oa.buffer, oa.offset),
                      Arg::buf(b.weights, b.weights_offset), Arg::buf(b.scales, b.scales_offset),
                      Arg::buf(ob.buffer, ob.offset), Arg::buf(c.weights, c.weights_offset),
                      Arg::buf(c.scales, c.scales_offset), Arg::buf(oc.buffer, oc.offset),
                      Arg::val(p)},
                     Dim3{total_rows / APEX_MATMUL_ROWS_PER_GROUP,
                          ceilDiv(tokens, APEX_MATMUL_TOKEN_TILE)},
                     Dim3{APEX_MATMUL_ROWS_PER_GROUP * 32});
    }

    /// One matrix: output [tokens, W.w_rows] = input [tokens, K] . W^T.
    void matmulNvfp4(Buffer *input, const Nvfp4Matrix &W, Buffer *output, uint32_t tokens,
                     float softcap = 0.0f) {
        const BufferRef out{output, 0};
        matmulNvfp4(input, 0, &W, &out, 1, tokens, softcap);
    }

    /// act [tokens, rows] = gelu_tanh(gate . x) * (up . x), one dispatch.
    void gateUpGeglu(Buffer *input, const Nvfp4Matrix &gate, const Nvfp4Matrix &up,
                     Buffer *output, uint32_t tokens) {
        const GateUpParams p{gate.hidden_dim, gate.w_rows, tokens};
        backend_.run(gate_up_geglu_,
                     {Arg::buf(input), Arg::buf(gate.weights, gate.weights_offset),
                      Arg::buf(gate.scales, gate.scales_offset),
                      Arg::buf(up.weights, up.weights_offset), Arg::buf(up.scales, up.scales_offset),
                      Arg::buf(output), Arg::val(p)},
                     Dim3{ceilDiv(gate.w_rows, APEX_MATMUL_ROWS_PER_GROUP),
                          ceilDiv(tokens, APEX_MATMUL_TOKEN_TILE)},
                     Dim3{APEX_MATMUL_ROWS_PER_GROUP * 32});
    }

    // Dense MLP: two dispatches.
    /// input → gate+up+GeGLU (fused) → act → down → output
    void denseMlp(Buffer *input, const Nvfp4Matrix &gate_weights, const Nvfp4Matrix &up_weights,
                  const Nvfp4Matrix &down_weights, Buffer *act_tmp, Buffer *output,
                  uint32_t tokens) {
        gateUpGeglu(input, gate_weights, up_weights, act_tmp, tokens);
        matmulNvfp4(act_tmp, down_weights, output, tokens);
    }

  private:
    Backend &backend_;
    Kernel *rms_norm_;
    Kernel *matmul_;
    Kernel *gate_up_geglu_;
};

} // namespace apex
