# B6 stage 2: the NVFP4 matvec and the dense MLP

Stage 1 built the dispatch machinery and proved it on RMSNorm. Stage 2 builds
the kernel the whole engine is made of: a matrix-vector product over NVFP4
weights. Everything heavy in Gemma 4 is this one kernel with different
shapes: q/k/v/o projections (stage 3), the dense MLP (this stage), the 8
expert MLPs per token (stage 5), and the output head (stage 6). The expert
files that section C streams from disk are consumed by this same kernel, so
its memory layout assumptions are the layout `repack.py` already wrote.

Verified against the `dense_out` dump for all 30 layers.

## What the dense MLP computes

From the stage 1 layer math, the piece this stage owns:

```
x         = norm(h, pre_feedforward_layernorm)         h = residual + attn_out
gate      = W_gate x        [2112]                      W_gate: [2112, 2816]
up        = W_up   x        [2112]                      W_up:   [2112, 2816]
act       = gelu_tanh(gate) * up                        elementwise
dense_out = W_down act      [2816]                      W_down: [2816, 2112]
```

`gelu_tanh` is `nn.gelu_approx` in mlx (verified in `activations.py`):

```
gelu_tanh(g) = 0.5 * g * (1 + tanh( sqrt(2/pi) * (g + 0.044715 * g^3) ))
```

`sqrt(2/pi) = 0.7978845608028654`. The reference computes the MLP from the
`pre_feedforward_layernorm` output, so our test does too: residual add on
the CPU (stage 6 puts it on the GPU), norm on the GPU (already verified),
MLP on the GPU, compare with `dense_out[l]`.

## The NVFP4 format, exactly as this checkpoint stores it

Verified bit for bit against `mx.dequantize(..., mode="nvfp4")` on
`layers.0.mlp.gate_proj` from the packed `core.bin` (max abs diff 0.0).

A matrix `W` of shape `[rows, K]` is stored as two tensors:

| Tensor | dtype | shape | meaning |
|---|---|---|---|
| `name.weight` | U32 | `[rows, K/8]` | 4-bit E2M1 codes, 8 per uint32, element `i` of the uint32 in bits `4i..4i+3` (element 0 in the 4 **least** significant bits) |
| `name.scales` | U8 | `[rows, K/16]` | one FP8 E4M3 scale per group of 16 consecutive elements of a row |

Decoding element `k` of row `r`:

```
code   = (weight[r][k/8] >> (4 * (k % 8))) & 0xF
scale  = e4m3(scales[r][k/16])
W[r,k] = e2m1(code) * scale
```

There is **no bias** (unlike the affine mode the router uses) and **no
per-tensor global scale**: the checkpoint has no `global_scale` tensors, and
the manifest records only `.weight` and `.scales` for every NVFP4 matrix.

**E2M1** (4 bits: sign, 2-bit exponent, 1-bit mantissa). Only 8 magnitudes
exist, so decode is a table lookup:

| code & 7 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| value | 0 | 0.5 | 1 | 1.5 | 2 | 3 | 4 | 6 |

Bit 3 is the sign.

**E4M3** (8 bits: sign, 4-bit exponent with bias 7, 3-bit mantissa), the
"fn" variant: no infinities, `0x7F`/`0xFF` are NaN.

```
e = (b >> 3) & 0xF,  m = b & 7
e == 0:  value = m / 8 * 2^-6  = m / 512          (subnormal)
e  > 0:  value = (1 + m/8) * 2^(e - 7)
bit 7 set: negate
```

The scales in this checkpoint are tiny (bytes 2..22 in layer 0's gate_proj,
values 0.0039 .. 0.055), so many are subnormal. Get the subnormal branch
right or whole groups decode to zero.

Shapes in this model, from the manifest:

| matrix | rows | K | `.weight` shape | `.scales` shape |
|---|---|---|---|---|
| `mlp.gate_proj`, `mlp.up_proj` | 2112 | 2816 | `[2112, 352]` | `[2112, 176]` |
| `mlp.down_proj` | 2816 | 2112 | `[2816, 264]` | `[2816, 132]` |
| expert `gate_proj`, `up_proj` (stage 5) | 704 | 2816 | `[704, 352]` | `[704, 176]` |
| expert `down_proj` (stage 5) | 2816 | 704 | `[2816, 88]` | `[2816, 44]` |
| `self_attn.q_proj` (stage 3) | 4096 | 2816 | `[4096, 352]` | `[4096, 176]` |
| `embed_tokens` (stage 6 head) | 262144 | 2816 | `[262144, 352]` | `[262144, 176]` |

One kernel, parameterized by `rows` and `K`, serves all of them. `K` is
always a multiple of 16 here; the kernel asserts nothing else.

## Kernel design

`y[t, r] = sum_k W[r, k] * x[t, k]` for every token `t` and output row `r`.

- **One simdgroup per output row.** The 32 lanes split the row's `K/16`
  groups round-robin (`g = lane, lane+32, ...`). Each lane reads two
  uint32 (16 codes), one scale byte, and 16 floats of `x`, accumulates the
  group's dot product, multiplies by the scale once, and adds to its
  running sum. `simd_sum` at the end; lane 0 writes `y`.
- **8 rows per threadgroup** (256 threads = 8 simdgroups). Threadgroup `x`
  picks the row block; threadgroup `y` picks the token. This is the first
  use of `Dim3` with two axes.
- Weight bytes are read exactly once per token. That is the decode-time
  budget: ~3 MB per dense MLP matrix, ~3.4 MB per expert slot. The
  kernel is memory bound; at decode (one token) nothing else matters.
- Multiplying by the scale once per group instead of once per element
  halves the multiplies and matches how mlx accumulates (scale times
  the group sum), which keeps us within float rounding of the reference.

What this kernel does **not** do yet, by design: reuse weight bytes across
tokens (prefill wants a tile of `x` in threadgroup memory, D15), fuse
gate and up into one pass (D12), or anything with `half`. Correctness
first; D11 measures it, D12-D15 change it.

## Files

| File | Change |
|---|---|
| `shaders/quant.h` | new: `e2m1ToFloat`, `e4m3ToFloat` shared by every quantized kernel |
| `shaders/matvec.metal` | new: `matvec_nvfp4` |
| `shaders/geglu.metal` | new: `geglu_f32` |
| `cmake/MetalShaders.cmake` | add a `DEPENDS` list so editing `quant.h` rebuilds the `.air` files |
| `CMakeLists.txt` | list the two new shaders and `quant.h`; add `src/tests/test_mlp.cpp` |
| `src/nvfp4.h` | new: CPU decoder (the reference) |
| `src/ops.h` | add `Nvfp4Matrix`, `matvecNvfp4`, `geglu`, `denseMlp` |
| `src/tests/cpu_ref.h` | new: `rmsNormCpu`, `Diff`, `diff` moved out of `test_norm.cpp` so both tests share them |
| `src/tests/test_norm.cpp` | include `cpu_ref.h`, delete the local copies |
| `src/tests/test_mlp.cpp` | new: `run_test_mlp` |
| `src/main.cpp` | `test-mlp <packed_dir> <dump_dir>` subcommand |

## 1. `shaders/quant.h`

```metal
// Decode helpers for the quantized formats in the checkpoint.
// Included by every kernel that reads weights; keep it free of kernels.
#pragma once
#include <metal_stdlib>
using namespace metal;

// E2M1 magnitudes indexed by the low 3 bits; bit 3 is the sign.
constant float E2M1_TABLE[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

inline float e2m1ToFloat(uint code) {           // code in [0, 16)
    const float v = E2M1_TABLE[code & 7];
    return (code & 8) ? -v : v;
}

// FP8 E4M3 (fn variant): sign | exponent(4, bias 7) | mantissa(3).
// e == 0 is subnormal: m/8 * 2^-6. 0x7F / 0xFF are NaN; we never store them.
inline float e4m3ToFloat(uint b) {
    const uint e = (b >> 3) & 0xF;
    const uint m = b & 7;
    const float v = (e == 0) ? float(m) * (1.0f / 512.0f)
                             : ldexp(1.0f + float(m) * 0.125f, int(e) - 7);
    return (b & 0x80) ? -v : v;
}
```

`ldexp` and the subnormal multiply are exact in float, so the GPU and CPU
decoders agree bit for bit.

## 2. `shaders/matvec.metal`

```metal
#include <metal_stdlib>
#include "quant.h"
using namespace metal;

// y[t, row] = sum_k W[row, k] * x[t, k]
//
// W is NVFP4 [out_rows, K]: codes as uint32 [out_rows, K/8], scales as
// uint8 [out_rows, K/16]. One simdgroup per output row, 8 rows per
// threadgroup, grid.y = token.
//
// Threadgroup (tx, ty)
// ├── simdgroup 0 -> row tx*8 + 0   lanes stride over the K/16 groups
// ├── simdgroup 1 -> row tx*8 + 1
// ├── ...
// └── simdgroup 7 -> row tx*8 + 7   all for token ty

constant constexpr uint ROWS_PER_TG = 8;   // 256 threads / 32 lanes

kernel void matvec_nvfp4(device const float* x        [[buffer(0)]],   // [tokens, K]
                         device const uint*  w        [[buffer(1)]],   // [out_rows, K/8]
                         device const uchar* s        [[buffer(2)]],   // [out_rows, K/16]
                         device float*       y        [[buffer(3)]],   // [tokens, out_rows]
                         constant uint&      K        [[buffer(4)]],
                         constant uint&      out_rows [[buffer(5)]],
                         uint2 tg      [[threadgroup_position_in_grid]],
                         uint  simd_id [[simdgroup_index_in_threadgroup]],
                         uint  lane    [[thread_index_in_simdgroup]]) {
    const uint row = tg.x * ROWS_PER_TG + simd_id;
    // Every lane of a simdgroup shares `row`, so the whole simdgroup leaves
    // together and the simd_sum below is still executed uniformly.
    if (row >= out_rows) return;

    const uint token  = tg.y;
    const uint groups = K / 16;
    x += token * K;
    w += row * (K / 8);
    s += row * groups;

    float acc = 0.0f;
    for (uint g = lane; g < groups; g += 32) {
        const uint lo = w[2 * g];          // codes 0..7 of the group
        const uint hi = w[2 * g + 1];      // codes 8..15
        device const float* xg = x + g * 16;
        float part = 0.0f;
        for (uint i = 0; i < 8; ++i) part += e2m1ToFloat((lo >> (4 * i)) & 0xF) * xg[i];
        for (uint i = 0; i < 8; ++i) part += e2m1ToFloat((hi >> (4 * i)) & 0xF) * xg[8 + i];
        acc += e4m3ToFloat(s[g]) * part;   // scale once per group
    }
    acc = simd_sum(acc);
    if (lane == 0) y[token * out_rows + row] = acc;
}
```

Why the early `return` is safe: Metal requires `simd_sum` to be reached by
the whole simdgroup. All 32 lanes compute the same `row`, so they either
all return or all continue.

## 3. `shaders/geglu.metal`

```metal
#include <metal_stdlib>
using namespace metal;

// out[i] = gelu_tanh(gate[i]) * up[i]   (mlx nn.gelu_approx)
kernel void geglu_f32(device const float* gate [[buffer(0)]],
                      device const float* up   [[buffer(1)]],
                      device float*       out  [[buffer(2)]],
                      constant uint&      n    [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    const float g = gate[i];
    const float t = tanh(0.7978845608028654f * (g + 0.044715f * g * g * g));
    out[i] = 0.5f * g * (1.0f + t) * up[i];
}
```

`-fno-fast-math` is what makes `tanh` here the precise one.

## 4. Build changes

`cmake/MetalShaders.cmake`: add a `DEPENDS` argument so a header edit
rebuilds every `.air`:

```cmake
# add_metal_library(<target> OUTPUT <name.metallib> SOURCES a.metal ... [DEPENDS h1 h2 ...])
function(add_metal_library target)
  cmake_parse_arguments(ARG "" "OUTPUT" "SOURCES;DEPENDS" ${ARGN})
  set(deps)
  foreach(d ${ARG_DEPENDS})
    list(APPEND deps ${CMAKE_CURRENT_SOURCE_DIR}/${d})
  endforeach()
  set(airs)
  foreach(src ${ARG_SOURCES})
    get_filename_component(name ${src} NAME_WE)
    set(air ${CMAKE_CURRENT_BINARY_DIR}/${name}.air)
    add_custom_command(OUTPUT ${air}
      COMMAND xcrun -sdk macosx metal -O2 -fno-fast-math -c ${CMAKE_CURRENT_SOURCE_DIR}/${src} -o ${air}
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${src} ${deps}
      COMMENT "metal ${src}")
    list(APPEND airs ${air})
  endforeach()
  ...unchanged...
```

`CMakeLists.txt`:

```cmake
add_metal_library(apex_shaders OUTPUT apex.metallib
  SOURCES shaders/norm.metal shaders/matvec.metal shaders/geglu.metal
  DEPENDS shaders/quant.h)

add_executable(apex src/main.cpp src/metal_impl.cpp src/backend/metal_backend.cpp
  src/tests/test_norm.cpp src/tests/test_mlp.cpp)
```

The `metal` compiler resolves `#include "quant.h"` relative to the
`.metal` file, so nothing else is needed.

## 5. `src/nvfp4.h` (CPU reference decoder)

The CPU side exists to be an independent second opinion on the GPU kernel,
written from the format description rather than from the shader. Keep it
simple and obviously right.

```cpp
// Purpose: CPU decoder for NVFP4 weights, the reference the GPU kernel is
// checked against. Same format as shaders/quant.h; see docs/B6_2.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace apex {

inline float e2m1ToFloat(uint32_t code) {
    static constexpr float table[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float v = table[code & 7];
    return (code & 8) ? -v : v;
}

inline float e4m3ToFloat(uint8_t b) {
    const uint32_t e = (b >> 3) & 0xF;
    const uint32_t m = b & 7;
    const float v = (e == 0) ? float(m) * (1.0f / 512.0f)
                             : std::ldexp(1.0f + float(m) * 0.125f, int(e) - 7);
    return (b & 0x80) ? -v : v;
}

// Decode one row of an NVFP4 matrix: k codes packed 8 per uint32, one
// scale byte per 16 codes.
inline void decodeRowNvfp4(const uint32_t* codes, const uint8_t* scales, float* out, size_t k) {
    for (size_t g = 0; g < k / 16; ++g) {
        const float scale = e4m3ToFloat(scales[g]);
        for (size_t i = 0; i < 16; ++i) {
            const uint32_t word = codes[2 * g + i / 8];
            const uint32_t code = (word >> (4 * (i % 8))) & 0xF;
            out[g * 16 + i] = e2m1ToFloat(code) * scale;
        }
    }
}

}  // namespace apex
```

## 6. `src/ops.h` additions

```cpp
#include <cstddef>
#include "backend/backend.h"

namespace apex {

// An NVFP4 matrix [rows, k] in GPU memory: codes and scales may live in the
// same buffer (core.bin, expert slots) at different offsets.
struct Nvfp4Matrix {
    Buffer* codes;  size_t codes_offset;   // uint32 [rows, k/8]
    Buffer* scales; size_t scales_offset;  // uint8  [rows, k/16]
    uint32_t rows;
    uint32_t k;
};

// y[t, :] = W x[t, :] for every token.  x: [tokens, k] f32, y: [tokens, rows] f32.
inline void matvecNvfp4(Backend& backend, Buffer* x, const Nvfp4Matrix& W, Buffer* y,
                        uint32_t tokens) {
    constexpr uint32_t ROWS_PER_GROUP = 8;   // 256 threads = 8 simdgroups, one row each
    const uint32_t row_groups = (W.rows + ROWS_PER_GROUP - 1) / ROWS_PER_GROUP;
    backend.run("matvec_nvfp4",
                {Arg::buf(x), Arg::buf(W.codes, W.codes_offset),
                 Arg::buf(W.scales, W.scales_offset), Arg::buf(y),
                 Arg::val(W.k), Arg::val(W.rows)},
                Dim3{row_groups, tokens}, Dim3{256});
}

// out[i] = gelu_tanh(gate[i]) * up[i] over n elements.
inline void geglu(Backend& backend, Buffer* gate, Buffer* up, Buffer* out, uint32_t n) {
    backend.run("geglu_f32", {Arg::buf(gate), Arg::buf(up), Arg::buf(out), Arg::val(n)},
                Dim3{(n + 255) / 256}, Dim3{256});
}

// Dense MLP: y = W_down( gelu_tanh(W_gate x) * W_up x ).
// gate_tmp and up_tmp are [tokens, intermediate] f32 scratch; act is written
// over gate_tmp. Four runs, one command buffer; the caller syncs.
inline void denseMlp(Backend& backend, Buffer* x, const Nvfp4Matrix& W_gate,
                     const Nvfp4Matrix& W_up, const Nvfp4Matrix& W_down,
                     Buffer* gate_tmp, Buffer* up_tmp, Buffer* y, uint32_t tokens) {
    matvecNvfp4(backend, x, W_gate, gate_tmp, tokens);
    matvecNvfp4(backend, x, W_up,   up_tmp,   tokens);
    geglu(backend, gate_tmp, up_tmp, gate_tmp, tokens * W_gate.rows);
    matvecNvfp4(backend, gate_tmp, W_down, y, tokens);
}

}  // namespace apex
```

`Arg::val` stores a pointer and `run` copies the bytes immediately via
`setBytes`, so passing `W.k` and `row_groups` (locals of this call) is fine.

Writing `act` over `gate_tmp` is safe: `geglu_f32` reads `gate[i]` before
it writes `out[i]`, and no two threads touch the same `i`.

## 7. `src/tests/cpu_ref.h`

Move `rmsNormCpu`, `Diff`, and `diff` out of `test_norm.cpp` into this
header unchanged, mark them `inline`, and include it from both tests.
Add one more helper:

```cpp
// y[t, :] = W x[t, :] on the CPU, decoding each weight row once.
inline void matvecNvfp4Cpu(const uint32_t* codes, const uint8_t* scales, const float* x,
                           float* y, size_t tokens, size_t rows, size_t k) {
    std::vector<float> row(k);
    for (size_t r = 0; r < rows; ++r) {
        decodeRowNvfp4(codes + r * (k / 8), scales + r * (k / 16), row.data(), k);
        for (size_t t = 0; t < tokens; ++t) {
            double acc = 0;
            for (size_t i = 0; i < k; ++i) acc += double(row[i]) * x[t * k + i];
            y[t * rows + r] = float(acc);
        }
    }
}

inline float geluTanh(float g) {
    return 0.5f * g * (1.0f + std::tanh(0.7978845608028654f * (g + 0.044715f * g * g * g)));
}
```

Per layer the CPU side is 3 × 2112 × 2816 × 20 ≈ 357 M multiply-adds; in
the Release build that is well under a second per layer.

## 8. `src/tests/test_mlp.cpp`

```cpp
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "../backend/metal_backend.h"
#include "../core_weights.h"
#include "../npy.h"
#include "../nvfp4.h"
#include "../ops.h"
#include "cpu_ref.h"

// Build the GPU-side description of an NVFP4 matrix from its two core tensors.
static apex::Nvfp4Matrix quantMatrix(apex::CoreWeights& core, const std::string& name) {
    apex::TensorView w = core.tensor(name + ".weight");
    apex::TensorView s = core.tensor(name + ".scales");
    const auto& shape = w.info->shape;                 // [rows, k/8]
    return {w.buffer, w.offset, s.buffer, s.offset,
            static_cast<uint32_t>(shape[0]), static_cast<uint32_t>(shape[1] * 8)};
}

// Compare each layer's dense MLP output against CPU and Python references.
int run_test_mlp(const std::string& packed_dir, const std::string& dump_dir) {
    apex::Manifest m = apex::Manifest::load(packed_dir);
    apex::MetalBackend backend(APEX_METALLIB);
    apex::CoreWeights core(m, backend);

    apex::Npy embed     = apex::loadNpy(dump_dir + "/embed.npy");       // (P, H)
    apex::Npy attn_out  = apex::loadNpy(dump_dir + "/attn_out.npy");    // (L, P, H)
    apex::Npy layer_out = apex::loadNpy(dump_dir + "/layer_out.npy");   // (L, P, H)
    apex::Npy dense_out = apex::loadNpy(dump_dir + "/dense_out.npy");   // (L, P, H)

    const int64_t layer_count = m.config.num_hidden_layers;
    const int64_t hidden      = m.config.hidden_size;        // 2816
    const int64_t inter       = m.config.intermediate_size;  // 2112
    const int64_t tokens      = embed.shape[0];
    const std::vector<int64_t> stage_shape{layer_count, tokens, hidden};
    if (embed.shape != std::vector<int64_t>{tokens, hidden} || attn_out.shape != stage_shape ||
        layer_out.shape != stage_shape || dense_out.shape != stage_shape) {
        throw std::runtime_error("test-mlp: dump shapes do not match model");
    }

    const size_t hidden_count = size_t(tokens) * size_t(hidden);
    const size_t inter_count  = size_t(tokens) * size_t(inter);
    const float eps = static_cast<float>(m.config.rms_norm_eps);

    apex::Buffer* h_buf    = backend.allocate(hidden_count * 4, "activations");
    apex::Buffer* x_buf    = backend.allocate(hidden_count * 4, "activations");
    apex::Buffer* gate_buf = backend.allocate(inter_count * 4,  "activations");
    apex::Buffer* up_buf   = backend.allocate(inter_count * 4,  "activations");
    apex::Buffer* y_buf    = backend.allocate(hidden_count * 4, "activations");

    std::vector<float> h(hidden_count), x_cpu(hidden_count), gate_cpu(inter_count),
                       up_cpu(inter_count), y_cpu(hidden_count);

    int failures = 0;
    for (int64_t layer = 0; layer < layer_count; ++layer) {
        // h = residual + attn_out: residual is the layer input.
        const float* residual = layer == 0 ? embed.f32()
                                           : layer_out.f32() + size_t(layer - 1) * hidden_count;
        const float* a = attn_out.f32() + size_t(layer) * hidden_count;
        for (size_t i = 0; i < hidden_count; ++i) h[i] = residual[i] + a[i];
        const float* expected = dense_out.f32() + size_t(layer) * hidden_count;

        const std::string prefix = "layers." + std::to_string(layer) + ".";
        apex::TensorView norm_w = core.tensor(prefix + "pre_feedforward_layernorm.weight");
        apex::Nvfp4Matrix W_gate = quantMatrix(core, prefix + "mlp.gate_proj");
        apex::Nvfp4Matrix W_up   = quantMatrix(core, prefix + "mlp.up_proj");
        apex::Nvfp4Matrix W_down = quantMatrix(core, prefix + "mlp.down_proj");

        // GPU: norm, then the MLP, all in one command buffer.
        std::memcpy(backend.contents(h_buf), h.data(), hidden_count * 4);
        apex::rmsNorm(backend, h_buf, norm_w.buffer, norm_w.offset, x_buf,
                      uint32_t(tokens), uint32_t(hidden), eps);
        apex::denseMlp(backend, x_buf, W_gate, W_up, W_down, gate_buf, up_buf, y_buf,
                       uint32_t(tokens));
        backend.sync();

        // CPU reference from the same h, independent of the GPU path.
        const uint16_t* norm_bf16 =
            static_cast<const uint16_t*>(core.cpu(prefix + "pre_feedforward_layernorm.weight"));
        for (int64_t t = 0; t < tokens; ++t)
            rmsNormCpu(h.data() + t * hidden, norm_bf16, x_cpu.data() + t * hidden, hidden, eps);
        auto codes  = [&](const std::string& n) { return static_cast<const uint32_t*>(core.cpu(n + ".weight")); };
        auto scales = [&](const std::string& n) { return static_cast<const uint8_t*>(core.cpu(n + ".scales")); };
        matvecNvfp4Cpu(codes(prefix + "mlp.gate_proj"), scales(prefix + "mlp.gate_proj"),
                       x_cpu.data(), gate_cpu.data(), tokens, inter, hidden);
        matvecNvfp4Cpu(codes(prefix + "mlp.up_proj"), scales(prefix + "mlp.up_proj"),
                       x_cpu.data(), up_cpu.data(), tokens, inter, hidden);
        for (size_t i = 0; i < inter_count; ++i) gate_cpu[i] = geluTanh(gate_cpu[i]) * up_cpu[i];
        matvecNvfp4Cpu(codes(prefix + "mlp.down_proj"), scales(prefix + "mlp.down_proj"),
                       gate_cpu.data(), y_cpu.data(), tokens, hidden, inter);

        const float* gpu = static_cast<const float*>(backend.contents(y_buf));
        const Diff gpu_ref = diff(gpu, expected, hidden_count);
        const Diff cpu_ref = diff(y_cpu.data(), expected, hidden_count);
        const Diff gpu_cpu = diff(gpu, y_cpu.data(), hidden_count);
        constexpr double tolerance = 1e-5;
        const bool ok = gpu_ref.rel_l2 < tolerance && cpu_ref.rel_l2 < tolerance &&
                        gpu_cpu.rel_l2 < tolerance;
        failures += !ok;
        std::printf("layer %2lld  gpu-vs-ref rel %.2e max %.2e | cpu-vs-ref rel %.2e | "
                    "gpu-vs-cpu rel %.2e  %s\n",
                    static_cast<long long>(layer), gpu_ref.rel_l2, gpu_ref.max_abs,
                    cpu_ref.rel_l2, gpu_cpu.rel_l2, ok ? "ok" : "FAIL");
    }

    for (apex::Buffer* b : {h_buf, x_buf, gate_buf, up_buf, y_buf}) backend.release(b);
    apex::Profile::get().print(stdout);
    std::printf("%lld layers tested, %d failures\n", static_cast<long long>(layer_count), failures);
    return failures ? 1 : 0;
}
```

`main.cpp`: declare `run_test_mlp` next to `run_test_norm` and add

```cpp
if (argc == 4 && std::strcmp(argv[1], "test-mlp") == 0) return run_test_mlp(argv[2], argv[3]);
```

plus the usage line.

## Expected output

```
./build/apex test-mlp /Users/spotta/Desktop/models/gemma4/packed dumps/paris
layer  0  gpu-vs-ref rel 2e-07 max 3e-04 | cpu-vs-ref rel 2e-07 | gpu-vs-cpu rel 1e-07  ok
...
layer 29  ...  ok
phase                      total ms    calls     avg ms     max ms
gpu_sync                     ...           30
30 layers tested, 0 failures
```

Relative L2 will be around 1e-7, not the 1e-8 of the norm test: a
2816-term dot product accumulated in a different order than mlx's kernel
moves the last bit or two. `max_abs` can be 1e-3 or larger because
`dense_out` has entries in the hundreds; relative L2 is the gate. Run
`math` and `para` too: `para` has 54 tokens, which exercises `tg.y` beyond
the first few threadgroups.

## If it goes wrong

| Symptom | Likely cause |
|---|---|
| gpu-vs-cpu ok, both vs ref off by a constant factor on every layer | scale decode wrong (exponent bias, subnormal branch, or `/512`) |
| whole 16-wide groups zero | subnormal E4M3 scales decoded as 0 |
| errors only on some rows / sign flips | nibble order: element 0 must be the low 4 bits; sign is bit 3 of the code |
| rel ~1e-3 everywhere, pattern-free | GeGLU formula (sigmoid vs tanh form, or wrong constant); or `up` and `gate` swapped |
| first token right, others wrong | `tg.y` not used for `x`/`y` offsets, or `y` indexed `[row, token]` |
| `rows` not a multiple of 8 corrupts the last block | missing `if (row >= out_rows) return` |
| hang or garbage after `geglu` | `n` passed as element count of the wrong buffer |
| GPU wildly off, CPU fine | `.weight` and `.scales` offsets swapped in `Nvfp4Matrix` |

## Commit split

1. Shader header dependency support in `MetalShaders.cmake` (tiny, separate).
2. `quant.h`, `matvec.metal`, `geglu.metal`, `nvfp4.h`, `ops.h` additions.
3. `cpu_ref.h` refactor of `test_norm.cpp` plus `test_mlp.cpp` and the subcommand.

## What stage 3 needs from this stage

Attention's q/k/v/o projections are `matvecNvfp4` calls with the shapes in
the table above. Stage 3 adds only RoPE, the per-head norms, the
softmax-over-keys kernel for both window kinds, and a `test-attn`
subcommand against `attn_out`. Stage 5 reuses `Nvfp4Matrix` by pointing
`codes`/`scales` at an expert slot inside a layer file: offsets 0 and
991232 for gate, 1115136 and 2106368 for up, 2230272 and 3221504 for down.
