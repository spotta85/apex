# B6 stage 1: kernel dispatch, RMSNorm, and the parity harness

B6 is "the kernels". It is too big for one doc, so it is split into stages,
each ending with a kernel verified against the A2 dumps. This doc is stage 1.
It builds the machinery every later kernel uses and proves it on the simplest
kernel in the model.

## B6 roadmap (for orientation)

| Stage | Kernels | Verified against |
|---|---|---|
| **1 (this doc)** | dispatch API, `.npy` loader, RMSNorm | `attn_in` dump, all 30 layers |
| 2 | NVFP4 matvec, GeGLU, dense MLP | `dense_out` dump |
| 3 | attention: q/k/v proj, q_norm/k_norm/v_norm, RoPE (two kinds), sliding + full, o_proj | `attn_out` dump |
| 4 | router: RMSNorm with scale, affine-8-bit (group 64) matvec, top-8, softmax, per_expert_scale | `route_ids` / `route_weights` dumps |
| 5 | expert GeGLU over 8 experts, weighted sum | `moe_out` dump |
| 6 | residual + norms + layer_scalar, final norm, output head, softcap | `layer_out`, `logits` dumps |

After stage 6, B7 wires the layers into a forward pass and generates text.

## Gemma 4 layer math (from mlx_lm `gemma4_text.py`, verified 2026-10-12)

Keep this next to you for every stage. `norm(x, w)` is RMSNorm with eps 1e-6
and **no +1 offset**: `x * rsqrt(mean(x^2) + eps) * w`.

```
embed      = embed_tokens[id] * sqrt(2816)
--- per layer ---
attn_in    = norm(h, input_layernorm)                       <- stage 1 verifies this
a          = attention(attn_in)                              (stage 3)
attn_out   = norm(a, post_attention_layernorm)
h          = h + attn_out
dense_out  = down( gelu_tanh(gate(norm(h, pre_ffn_norm))) * up(norm(h, pre_ffn_norm)) )   (stage 2)
ids, w     = router(h)                                       (stage 4; reads the RAW h)
moe_out    = experts(norm(h, pre_ffn_norm_2), ids, w)        (stage 5)
f          = norm(dense_out, post_ffn_norm_1) + norm(moe_out, post_ffn_norm_2)
h          = (h + norm(f, post_ffn_norm)) * layer_scalar
--- end ---
logits     = tanh(embed_tokens^T . norm(h, norm) / 30) * 30
```

Attention geometry differs by layer type:

| | sliding (25 layers) | full (layers 5, 11, 17, 23, 29) |
|---|---|---|
| head_dim | 256 | 512 |
| q heads / kv heads | 16 / 8 | 16 / 2 |
| v_proj | yes | **none, V = K** |
| window | 1024 | unlimited |
| RoPE | theta 1e4, all dims | theta 1e6, "proportional", rotates 25% of dims |

Not needed until stage 3, but add `global_head_dim`, `num_global_key_value_heads`,
`attention_k_eq_v`, and the two rope parameter sets to `ModelConfig` when you get there.

## Stage 1 goal

1. `Backend` gains a generic "run this kernel with these arguments" call and a
   `sync`. **No per-kernel virtual methods, ever.** The Intel port implements
   `run` once; kernels are data.
2. A `.npy` reader so C++ can load the A2 dumps.
3. `rms_norm_f32`: the first real kernel. Used seven times per layer plus the
   final norm plus q/k/v norms, so getting it right pays off nine ways.
4. `apex test-norm <packed> <dump_dir>`: for every layer l, run the norm on the
   layer's input (embed for l = 0, `layer_out[l-1]` otherwise) with
   `layers.l.input_layernorm.weight`, compare against `attn_in[l]`. Three-way
   check: GPU vs dump, GPU vs a CPU reference, CPU vs dump.

The three-way check separates "I misunderstand the math" (CPU disagrees with
dump) from "my kernel is wrong" (GPU disagrees with CPU).

## Before you start: four cleanups

1. **`MAP_PRIVATE` -> `MAP_SHARED`** in `core_weights.h`. Logged decision, not yet in code.
2. **Delete `run_smoke`, `hello.metal`, `checksum.metal`, and `checkSumU32`.**
   `main.cpp` should include no Metal headers. The norm test supersedes the
   checksum as proof the GPU reads the mapped weights. `bench` keeps
   mmap + touch + memory tables.
3. **`.gitignore`**: your unstaged edit ignores `docs/` again. Decide; the log
   is tracked either way, but new docs won't be.
4. **Dumper: add the `attn_in` stage.** In `tools/dump_reference.py`:

   ```python
   STAGES = ("attn_in", "attn_out", "dense_out", "moe_out", "layer_out")
   ...
           # --- attention block ---
           residual = h
           attn_in = layer.input_layernorm(h)
           a, _, _ = layer.self_attn(attn_in, mask, cache)
   ...
           mx.eval(attn_in, attn_out, dense_out, moe_out, h, idx, w)
           per_stage["attn_in"][l].append(to_np(attn_in))
   ```

   Then re-dump all three prompts (10 s each):
   ```bash
   cd tools && uv run python dump_reference.py --name paris --prompt "What is the capital of France?" && uv run python dump_reference.py --name math --prompt "What is 2+2?"
   ```
   (and `para` with its paragraph). `attn_in.npy` is (30, positions, 2816).

5. **Shader precision.** Metal compiles with fast math by default, which lets
   `rsqrt` be approximate. For parity work, add `-fno-fast-math` to the
   `metal` command in `cmake/MetalShaders.cmake`. Revisit in D14.

## Files

| File | New or edit | Purpose |
|---|---|---|
| `src/backend/backend.h` | edit | `Arg`, `run`, `sync`; remove checksum |
| `src/backend/metal_backend.h/.cpp` | edit | implement `run`/`sync` with one command buffer per sync |
| `src/npy.h` | new | read `.npy` float32 / int64 |
| `src/ops.h` | new | Metal-free kernel wrappers: grid math lives here |
| `shaders/norm.metal` | new | `rms_norm_f32` |
| `src/main.cpp` | edit | `test-norm` subcommand |
| `cmake/MetalShaders.cmake`, `CMakeLists.txt` | edit | `-fno-fast-math`, shader list |

---

## 1. `backend.h` additions

```cpp
// One kernel argument. Either a buffer (+ byte offset) or a small inline
// constant copied at encode time. Matches Metal's setBuffer / setBytes.
struct Arg {
    Buffer*     buffer = nullptr;
    size_t      offset = 0;
    const void* bytes  = nullptr;
    size_t      size   = 0;

    static Arg buf(Buffer* b, size_t off = 0) { return {b, off, nullptr, 0}; }
    template <class T>
    static Arg val(const T& v) { return {nullptr, 0, &v, sizeof(T)}; }   // v must outlive run()
};

// Grid or threadgroup shape. One number for 1D work (Dim3{rows}), three for
// tiled kernels (Dim3{tiles_x, tiles_y, 1}). Implicit from a single uint32_t,
// so 1D call sites stay `rows, 256`.
struct Dim3 {
    uint32_t x = 1, y = 1, z = 1;
    Dim3(uint32_t x_, uint32_t y_ = 1, uint32_t z_ = 1) : x(x_), y(y_), z(z_) {}
    uint32_t total() const { return x * y * z; }
};

class Backend {
public:
    // ... wrapNoCopy / allocate / contents / release as before ...

    // Encode one kernel launch. Arguments bind to buffer(0), buffer(1), ... in
    // order. Work is queued, not executed, until sync(). Both shapes may be
    // 1D, 2D or 3D; the kernel reads threadgroup_position_in_grid.xy(z).
    virtual void run(const char* kernel, std::initializer_list<Arg> args,
                     Dim3 threadgroups, Dim3 threads_per_group) = 0;

    // Submit everything queued since the last sync() and wait for it.
    virtual void sync() = 0;
};
```

Why threadgroups + threads-per-group rather than total threads: most kernels
here are "one threadgroup per row" and want `threadgroup_position_in_grid` to
be the row. That maps to `dispatchThreadgroups` directly. Every kernel in
sections B and C is 1D; the 3D shape exists for the tiled matmul in D15
(prefill), where the grid is output tiles in two axes.

## 2. `metal_backend.cpp`: `run` and `sync`

Members to add in the header: `NS::AutoreleasePool* pool_ = nullptr;
MTL::CommandBuffer* cmd_ = nullptr; MTL::ComputeCommandEncoder* enc_ = nullptr;`

```cpp
void MetalBackend::run(const char* kernel, std::initializer_list<Arg> args,
                       Dim3 threadgroups, Dim3 threads_per_group) {
    if (!cmd_) {
        // One pool per batch: commandBuffer() and computeCommandEncoder() are
        // autoreleased; the pool drains them in sync().
        pool_ = NS::AutoreleasePool::alloc()->init();
        cmd_ = queue_->commandBuffer();
        enc_ = cmd_->computeCommandEncoder();
    }
    MTL::ComputePipelineState* pso = pipeline(kernel);
    enc_->setComputePipelineState(pso);
    uint32_t index = 0;
    for (const Arg& a : args) {
        if (a.buffer) enc_->setBuffer(a.buffer->mtl, a.offset, index);
        else          enc_->setBytes(a.bytes, a.size, index);
        ++index;
    }
    // Metal's limit is on the threadgroup's total thread count, not per axis.
    if (threads_per_group.total() > pso->maxTotalThreadsPerThreadgroup()) {
        throw std::runtime_error(std::string(kernel) + ": threadgroup too large");
    }
    enc_->dispatchThreadgroups(MTL::Size::Make(threadgroups.x, threadgroups.y, threadgroups.z),
                               MTL::Size::Make(threads_per_group.x, threads_per_group.y, threads_per_group.z));
}

void MetalBackend::sync() {
    if (!cmd_) return;
    Timer t("gpu_sync");
    enc_->endEncoding();
    cmd_->commit();
    cmd_->waitUntilCompleted();
    pool_->release();            // releases cmd_ and enc_
    pool_ = nullptr; cmd_ = nullptr; enc_ = nullptr;
}
```

The destructor should call `sync()` first. Every `run` between two `sync`s
lands in one command buffer: that is the "as few submissions per token as
possible" rule from D11, free from day one.

## 3. `src/npy.h`

The `.npy` format: 6-byte magic `\x93NUMPY`, major, minor, a little-endian
header length (2 bytes for v1, 4 for v2), then an ASCII Python dict, then raw
data. The dumps are v1, C order, `<f4` or `<i8`.

```cpp
// Purpose: read the A2 dumps. Only what dump_reference.py writes:
// version 1.0, C order, dtype <f4 or <i8.
#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace apex {

struct Npy {
    std::vector<int64_t> shape;
    std::string dtype;               // "<f4" or "<i8"
    std::vector<char> raw;

    const float*   f32() const { return reinterpret_cast<const float*>(raw.data()); }
    const int64_t* i64() const { return reinterpret_cast<const int64_t*>(raw.data()); }
    size_t count() const { size_t n = 1; for (auto d : shape) n *= d; return n; }
};

inline Npy loadNpy(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("npy: cannot open " + path);

    char magic[6]; uint8_t major, minor;
    in.read(magic, 6); in.read(reinterpret_cast<char*>(&major), 1); in.read(reinterpret_cast<char*>(&minor), 1);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) throw std::runtime_error("npy: bad magic " + path);

    uint32_t header_len = 0;
    if (major == 1) { uint16_t h; in.read(reinterpret_cast<char*>(&h), 2); header_len = h; }
    else            { in.read(reinterpret_cast<char*>(&header_len), 4); }
    std::string header(header_len, '\0');
    in.read(header.data(), header_len);

    Npy out;
    // dtype
    size_t d = header.find("'descr':");
    size_t q1 = header.find('\'', d + 8), q2 = header.find('\'', q1 + 1);
    out.dtype = header.substr(q1 + 1, q2 - q1 - 1);
    if (out.dtype != "<f4" && out.dtype != "<i8") throw std::runtime_error("npy: unsupported dtype " + out.dtype);
    if (header.find("'fortran_order': True") != std::string::npos) throw std::runtime_error("npy: fortran order");
    // shape
    size_t s = header.find("'shape':");
    size_t p1 = header.find('(', s), p2 = header.find(')', p1);
    std::string dims = header.substr(p1 + 1, p2 - p1 - 1);
    size_t pos = 0;
    while (pos < dims.size()) {
        size_t comma = dims.find(',', pos);
        std::string tok = dims.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        if (!tok.empty() && tok.find_first_not_of(" ") != std::string::npos) out.shape.push_back(std::stoll(tok));
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    size_t item = out.dtype == "<f4" ? 4 : 8;
    out.raw.resize(out.count() * item);
    in.read(out.raw.data(), out.raw.size());
    if (!in) throw std::runtime_error("npy: truncated " + path);
    return out;
}

}  // namespace apex
```

## 4. `shaders/norm.metal`

One threadgroup per row. 256 threads each own a strided slice of the row
(2816 / 256 = 11 elements). Sum of squares is reduced in two steps: within
each 32-lane SIMD group by `simd_sum`, then across the 8 SIMD groups through
threadgroup memory. Weight is BF16: the top 16 bits of a float32, so
converting is a shift.

```metal
#include <metal_stdlib>
using namespace metal;

// y[row] = x[row] * rsqrt(mean(x[row]^2) + eps) * w     (no +1 on w)
// x, y: fp32 [rows, n] contiguous.  w: bf16 bits [n].
kernel void rms_norm_f32(device const float*  x   [[buffer(0)]],
                         device const ushort* w   [[buffer(1)]],
                         device float*        y   [[buffer(2)]],
                         constant uint&       n   [[buffer(3)]],
                         constant float&      eps [[buffer(4)]],
                         uint row       [[threadgroup_position_in_grid]],
                         uint tid       [[thread_index_in_threadgroup]],
                         uint tg_size   [[threads_per_threadgroup]],
                         uint lane      [[thread_index_in_simdgroup]],
                         uint simd_id   [[simdgroup_index_in_threadgroup]]) {
    x += row * n;
    y += row * n;

    // 1. partial sum of squares per thread
    float ss = 0.0f;
    for (uint i = tid; i < n; i += tg_size) ss += x[i] * x[i];

    // 2. reduce within SIMD group, then across SIMD groups
    ss = simd_sum(ss);
    threadgroup float partial[32];
    if (lane == 0) partial[simd_id] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
        uint nsimd = (tg_size + 31) / 32;
        float v = (lane < nsimd) ? partial[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) partial[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. normalize and scale
    float inv = rsqrt(partial[0] / float(n) + eps);
    for (uint i = tid; i < n; i += tg_size) {
        float wf = as_type<float>(uint(w[i]) << 16);     // bf16 -> f32
        y[i] = x[i] * inv * wf;
    }
}
```

The same kernel does q_norm and k_norm later: rows = heads, n = head_dim.

## 5. `src/ops.h`

The only place that knows kernel names and grid sizes. The model file (B7)
calls these and never sees a kernel name.

```cpp
// Purpose: Metal-free wrappers. One function per kernel; grid math lives here.
#pragma once

#include <cstdint>
#include "backend/backend.h"

namespace apex {

// x, y: fp32 [rows, n]. w: bf16 [n] at (w_buf, w_off), typically a core tensor.
inline void rmsNorm(Backend& b, Buffer* x, Buffer* w_buf, size_t w_off, Buffer* y,
                    uint32_t rows, uint32_t n, float eps) {
    b.run("rms_norm_f32",
          {Arg::buf(x), Arg::buf(w_buf, w_off), Arg::buf(y), Arg::val(n), Arg::val(eps)},
          rows, 256);
}

}  // namespace apex
```

## 6. `test-norm` in `main.cpp`

```cpp
#include <cmath>
#include "npy.h"
#include "ops.h"

// CPU reference for one row, same formula.
static void rmsNormCpu(const float* x, const uint16_t* w_bf16, float* y, size_t n, float eps) {
    double ss = 0;
    for (size_t i = 0; i < n; ++i) ss += double(x[i]) * x[i];
    float inv = 1.0f / std::sqrt(float(ss / n) + eps);
    for (size_t i = 0; i < n; ++i) {
        uint32_t bits = uint32_t(w_bf16[i]) << 16;
        float wf; std::memcpy(&wf, &bits, 4);
        y[i] = x[i] * inv * wf;
    }
}

// max |a-b| and relative L2 ||a-b|| / ||b||
struct Diff { double max_abs = 0, rel_l2 = 0; };
static Diff diff(const float* a, const float* b, size_t n) {
    double num = 0, den = 0; Diff d;
    for (size_t i = 0; i < n; ++i) {
        double e = double(a[i]) - b[i];
        num += e * e; den += double(b[i]) * b[i];
        d.max_abs = std::max(d.max_abs, std::fabs(e));
    }
    d.rel_l2 = std::sqrt(num / den);
    return d;
}

static int run_test_norm(const std::string& packed_dir, const std::string& dump_dir) {
    apex::Manifest m = apex::Manifest::load(packed_dir);
    apex::MetalBackend backend(APEX_METALLIB);
    apex::CoreWeights core(m, backend);

    apex::Npy embed     = apex::loadNpy(dump_dir + "/embed.npy");       // (P, H)
    apex::Npy layer_out = apex::loadNpy(dump_dir + "/layer_out.npy");   // (L, P, H)
    apex::Npy attn_in   = apex::loadNpy(dump_dir + "/attn_in.npy");     // (L, P, H)
    const uint32_t L = attn_in.shape[0], P = attn_in.shape[1], H = attn_in.shape[2];
    const float eps = float(m.config.rms_norm_eps);

    apex::Buffer* x = backend.allocate(size_t(P) * H * 4, "activations");
    apex::Buffer* y = backend.allocate(size_t(P) * H * 4, "activations");
    std::vector<float> cpu(size_t(P) * H);

    int failures = 0;
    for (uint32_t l = 0; l < L; ++l) {
        // input to layer l's norm: embed for l = 0, previous layer's output otherwise
        const float* in = (l == 0) ? embed.f32() : layer_out.f32() + size_t(l - 1) * P * H;
        const float* expect = attn_in.f32() + size_t(l) * P * H;
        std::memcpy(backend.contents(x), in, size_t(P) * H * 4);

        apex::TensorView w = core.tensor("layers." + std::to_string(l) + ".input_layernorm.weight");
        apex::rmsNorm(backend, x, w.buffer, w.offset, y, P, H, eps);
        backend.sync();

        const uint16_t* w_cpu = static_cast<const uint16_t*>(core.cpu("layers." + std::to_string(l) + ".input_layernorm.weight"));
        for (uint32_t p = 0; p < P; ++p) rmsNormCpu(in + size_t(p) * H, w_cpu, cpu.data() + size_t(p) * H, H, eps);

        const float* gpu = static_cast<const float*>(backend.contents(y));
        Diff gd = diff(gpu, expect, size_t(P) * H);        // GPU vs reference dump
        Diff cd = diff(cpu.data(), expect, size_t(P) * H); // CPU vs reference dump
        Diff gc = diff(gpu, cpu.data(), size_t(P) * H);    // GPU vs CPU
        bool ok = gd.rel_l2 < 1e-5;
        failures += !ok;
        std::printf("layer %2u  gpu-vs-ref rel %.2e max %.2e | cpu-vs-ref rel %.2e | gpu-vs-cpu rel %.2e  %s\n",
                    l, gd.rel_l2, gd.max_abs, cd.rel_l2, gc.rel_l2, ok ? "ok" : "FAIL");
    }
    backend.release(x); backend.release(y);
    apex::Profile::get().print(stdout);
    std::printf("%s\n", failures ? "FAILED" : "ALL 30 LAYERS MATCH");
    return failures ? 1 : 0;
}
```

Wire it: `apex test-norm <packed> <dump_dir>`.

Note the one CPU read of core weights (`core.cpu(...)` for the reference).
After B5's findings that's fine with MAP_SHARED and is test-only.

## Expected output

```
layer  0  gpu-vs-ref rel 2.1e-07 max 3.8e-06 | cpu-vs-ref rel 1.9e-07 | gpu-vs-cpu rel 1.5e-07  ok
...
layer 29  ...  ok
phase                      total ms    calls     avg ms     max ms
gpu_sync                        ~15       30
mmap_core                        ~1        1
ALL 30 LAYERS MATCH
```

Relative L2 around 1e-7 to 1e-6 is fp32 reorder noise. 1e-3 or worse means
a real bug. The three columns tell you which:

| gpu-vs-ref | cpu-vs-ref | gpu-vs-cpu | meaning |
|---|---|---|---|
| bad | bad | good | the formula is wrong (eps? the +1? wrong weight?) |
| bad | good | bad | the kernel is wrong (reduction, indexing, bf16 convert) |
| good | good | good | done |

## If it goes wrong

- **All layers off by a near-constant factor** -> missing `* w` or `+1` applied. Gemma 4 has no +1.
- **Only some rows wrong** -> `row * n` offset or a `tg_size` mismatch (ops.h says 256, kernel assumes 32-lane SIMD groups, fine; but if you change 256 to something not a multiple of 32 the reduction breaks).
- **Everything ~1e-3 off** -> fast math. Check `-fno-fast-math` reached the `metal` command (`ninja -v`).
- **`npy: unsupported dtype`** -> you loaded `ids.npy` with the float path; it is `<i8`.
- **Garbage from layer 1 on** -> `layer_out` indexing: layer l's input is `layer_out[l-1]`, not `[l]`.

## Commit split

1. Cleanups: MAP_SHARED, delete smoke/checksum, `-fno-fast-math`.
2. Dumper `attn_in` stage (tools) and re-dumped fixtures (not committed; `dumps/` is ignored).
3. `Backend::run/sync` + `Arg`, metal implementation.
4. `npy.h`, `ops.h`, `norm.metal`, `test-norm`.

## What stage 2 needs from this stage

The NVFP4 matvec reuses everything: `run`, `Arg`, `ops.h`, the three-way
test pattern, `loadNpy`. Stage 2's doc will add only the kernel, its CPU
reference (the NVFP4 decoder), and a `test-mlp` subcommand against `dense_out`.
