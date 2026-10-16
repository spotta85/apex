# B6 stage 3: attention

Stage 3 builds the attention block: projections, per-head norms, RoPE, the
softmax over keys, and the output projection. It also fixes the KV cache
layout, because the attention kernel reads keys and values from the cache
in both prefill (this test) and decode (B7).

Verified against the `attn_out` dump for all 30 layers. Every formula in
this doc was first run in numpy against the dump on layers 0, 5 and 29
(rel L2 1.4e-6, 1.7e-6, 2.3e-6), so if the C++ disagrees, the C++ is wrong.

## What the block computes

From `gemma4_text.py` `Attention.__call__`, per layer:

```
x        = attn_in                                    [T, 2816]   (input_layernorm output, dumped)
q        = W_q x       → [T, 16, hd]
k_raw    = W_k x       → [T, n_kv, hd]
v_raw    = W_v x       → [T, n_kv, hd]      sliding layers
v_raw    = k_raw                             full layers (no v_proj in the checkpoint)

q        = rope( rmsnorm(q,     q_norm.weight) , pos )      per head, over hd
K[pos]   = rope( rmsnorm(k_raw, k_norm.weight) , pos )      → KV cache
V[pos]   =       rmsnorm(v_raw, no weight)                  → KV cache

o[t,h]   = softmax_j( q[t,h] · K[j, kv(h)] ) · V[j, kv(h)]      scale = 1.0 (!)
           j ranges over  max(0, t-1023) .. t   (sliding)   or   0 .. t   (full)
out      = W_o concat_h(o[t,h])                → [T, 2816]
attn_out = rmsnorm(out, post_attention_layernorm.weight)    ← compared with the dump
```

Things that are easy to get wrong, all verified:

| Detail | Value |
|---|---|
| attention scale | **1.0**, not 1/sqrt(hd) (`self.scale = 1.0`) |
| full layers' V | `v_raw = k_raw` is taken **before** k_norm. V gets v_norm (no weight), K gets k_norm + RoPE. So the V cache is different from the K cache even on full layers |
| q_norm / k_norm | plain RMSNorm over hd with a BF16 weight of length hd (256 or 512), no +1 |
| v_norm | RMSNorm over hd, no weight |
| GQA mapping | q head h reads kv head `h / (16 / n_kv)` |
| sliding window | key j is visible to query t iff `j <= t` and `j > t - 1024`: 1024 keys including itself |
| full layers | plain causal |
| eps | 1e-6 everywhere |

Shapes per layer type:

| | sliding (25 layers) | full (5, 11, 17, 23, 29) |
|---|---|---|
| hd | 256 | 512 |
| q heads | 16 | 16 |
| kv heads | 8 | 2 |
| `q_proj.weight` | `[4096, 352]` | `[8192, 352]` |
| `k_proj.weight` | `[2048, 352]` | `[1024, 352]` |
| `v_proj.weight` | `[2048, 352]` | none |
| `o_proj.weight` (rows, K) | `[2816, 4096]` | `[2816, 8192]` |
| `q_norm`, `k_norm` | `[256]` | `[512]` |
| RoPE base | 1e4 | 1e6 |
| RoPE pairs rotated | 128 (all) | 64 (first quarter) |
| window | 1024 | none |

## RoPE, exactly

mlx `traditional=False` pairs element `i` with element `i + hd/2`:

```
half  = hd / 2
for i in 0 .. pairs-1:
    angle = pos * base^(-2i / hd)           <- denominator is the FULL hd, also on full layers
    a, b  = x[i], x[half + i]
    x[i]        = a cos - b sin
    x[half + i] = a sin + b cos
elements i >= pairs and half+i >= half+pairs are untouched
```

`pairs = hd/2` on sliding layers (every element rotates). On full layers
`partial_rotary_factor = 0.25` with rope type `proportional` means
`pairs = 64`: only elements `0..63` and `256..319` of the 512 rotate, and
the frequencies are still computed with `hd = 512` in the exponent. This
was verified against `initialize_rope` at positions 0, 37 and 1500.

Compute `angle` in float, as mlx does. Position `pos` is the absolute
position in the sequence (`pos0 + t`), which is `t` in this test.

## KV cache layout

One buffer each for K and V per layer, head-major so one head's keys are
contiguous for the scan:

```
K: [n_kv][capacity][hd] float32        index = (kv_head * capacity + pos) * hd + d
V: same
```

The attention kernel reads `K[kv(h)][j]` for `j` in the visible range.
Prefill (this test) writes positions `0..T-1` then attends; decode (B7)
writes position `pos0` and attends with `T = 1`. Same kernel, same layout.

What this layout costs at float32 is large (a 4096-token context is 64 MB
per sliding layer). That is section C's problem, not this stage's: C
will store the cache as fp16 and make sliding layers a 1024-entry ring,
which changes `pos` to `pos % 1024` in the index above and nothing else.

## Kernel design

Three kernels plus the matmul from stage 2:

| Kernel | Grid | Does |
|---|---|---|
| `matmul_nvfp4` (stage 2) | | q, k, v, o projections |
| `head_norm_rope_f32` | `Dim3{T, heads}` × 256 threads | per (token, head): RMSNorm over hd, optional weight, optional RoPE, write with caller-chosen strides (so it writes q in place and K/V into the cache) |
| `attention_f32` | `Dim3{T, 16}` × 256 threads | per (query token, q head): online softmax over the visible keys in chunks of 256, output `[T, 16, hd]` |

`attention_f32` is the flash-attention idea in its simplest form. Keys are
processed 256 at a time: each thread scores one key of the chunk, the
threadgroup finds the chunk max and sum, the running max/sum/accumulator
are rescaled, and each thread accumulates its own output dims. Nothing is
stored per key beyond the current chunk, so context length is not limited
by threadgroup memory.

## Files

| File | Change |
|---|---|
| `src/model_info.h` | add `global_head_dim`, `num_global_key_value_heads`, `attention_k_eq_v`, `rope_theta_sliding`, `rope_theta_full`, `partial_rotary_factor_full` to `ModelConfig`; `attentionShape(layer)` helper |
| `shaders/head_norm_rope.metal` | new |
| `shaders/attention.metal` | new |
| `CMakeLists.txt` | list both shaders; add `src/tests/attention.cpp` |
| `src/ops.h` | add `AttentionShape`, `AttentionWeights`, `KvCache`, `headNormRope`, `attention`, `attentionBlock` |
| `src/tests/cpu_ref.h` | `rmsNormCpu` accepts `nullptr` weight; add `ropeCpu`, `attentionCpu` |
| `src/tests/attention.cpp` | new: `run_test_attn` |
| `src/main.cpp` | `test-attn <packed_dir> <dump_dir>` |

## 1. `model_info.h` additions

In `ModelConfig`:

```cpp
    int global_head_dim = 0;             // 512  (full-attention layers)
    int num_global_key_value_heads = 0;  // 2    (full-attention layers)
    bool attention_k_eq_v = false;       // true: full layers have no v_proj
    double rope_theta_sliding = 0;       // 1e4
    double rope_theta_full = 0;          // 1e6
    double partial_rotary_factor_full = 0;  // 0.25
```

In `load`:

```cpp
    c.global_head_dim            = tc.at("global_head_dim").get<int>();
    c.num_global_key_value_heads = tc.at("num_global_key_value_heads").get<int>();
    c.attention_k_eq_v           = tc.at("attention_k_eq_v").get<bool>();
    const auto& rp = tc.at("rope_parameters");
    c.rope_theta_sliding         = rp.at("sliding_attention").at("rope_theta").get<double>();
    c.rope_theta_full            = rp.at("full_attention").at("rope_theta").get<double>();
    c.partial_rotary_factor_full = rp.at("full_attention").at("partial_rotary_factor").get<double>();
```

## 2. `shaders/head_norm_rope.metal`

```metal
#include <metal_stdlib>
using namespace metal;

// Per (token, head): RMSNorm over head_dim with an optional BF16 weight,
// then optional RoPE, written with caller-chosen strides.
//
// input : [tokens, heads, head_dim]  (the matmul output, viewed per head)
// output: output[t * out_token_stride + head * out_head_stride + d]
//         q  : token_stride = heads*head_dim, head_stride = head_dim   (same layout as input)
//         K/V: token_stride = head_dim,       head_stride = capacity*head_dim, output pre-offset to pos0
//
// Threadgroup (t, head): 256 threads stride over head_dim (256 or 512).
kernel void head_norm_rope_f32(device const float*  input      [[buffer(0)]],
                               device const ushort* weight     [[buffer(1)]],   // [head_dim] BF16, used if has_weight
                               device float*        output     [[buffer(2)]],
                               constant uint&  heads           [[buffer(3)]],
                               constant uint&  head_dim        [[buffer(4)]],
                               constant uint&  has_weight      [[buffer(5)]],
                               constant uint&  rope_pairs      [[buffer(6)]],   // 0 = no RoPE
                               constant float& rope_base       [[buffer(7)]],
                               constant uint&  pos0            [[buffer(8)]],   // position of token 0
                               constant uint&  out_token_stride [[buffer(9)]],
                               constant uint&  out_head_stride [[buffer(10)]],
                               constant float& eps             [[buffer(11)]],
                               uint2 tg       [[threadgroup_position_in_grid]],
                               uint  tid      [[thread_index_in_threadgroup]],
                               uint  lane     [[thread_index_in_simdgroup]],
                               uint  simd_id  [[simdgroup_index_in_threadgroup]]) {
    const uint token = tg.x, head = tg.y;
    input  += (token * heads + head) * head_dim;
    output += token * out_token_stride + head * out_head_stride;

    // --- RMSNorm (same reduction as rms_norm_f32) ---
    float ss = 0.0f;
    for (uint i = tid; i < head_dim; i += 256) ss += input[i] * input[i];
    ss = simd_sum(ss);
    threadgroup float partial[8];
    if (lane == 0) partial[simd_id] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
        float v = (lane < 8) ? partial[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) partial[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = rsqrt(partial[0] / float(head_dim) + eps);

    // Normalized row staged in threadgroup memory: RoPE pairs (i, i + half)
    // are owned by different threads when head_dim == 256.
    threadgroup float row[512];
    for (uint i = tid; i < head_dim; i += 256) {
        const float w = has_weight ? as_type<float>(uint(weight[i]) << 16) : 1.0f;
        row[i] = input[i] * inv * w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- RoPE on the first rope_pairs pairs ---
    const uint half = head_dim / 2;
    if (tid < rope_pairs) {
        const float pos   = float(pos0 + token);
        const float angle = pos * pow(rope_base, -2.0f * float(tid) / float(head_dim));
        const float c = cos(angle), s = sin(angle);
        const float a = row[tid], b = row[half + tid];
        row[tid]        = a * c - b * s;
        row[half + tid] = a * s + b * c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < head_dim; i += 256) output[i] = row[i];
}
```

`rope_pairs` is at most 128, so `tid < rope_pairs` never exceeds the 256
threads.

## 3. `shaders/attention.metal`

```metal
#include <metal_stdlib>
using namespace metal;

// One threadgroup per (query token t, q head h). Online softmax over the
// visible keys in chunks of 256: each thread scores one key per chunk and
// accumulates output dims d = tid, tid + 256.
//
// q  : [tokens, q_heads, head_dim]
// K,V: [kv_heads][capacity][head_dim]   (the cache; positions 0..pos0+tokens-1 are valid)
// out: [tokens, q_heads, head_dim]
constant constexpr uint CHUNK = 256;

// Max and sum over the 256 threads; result broadcast through scratch[0].
static float groupMax(float v, uint lane, uint simd_id, threadgroup float* scratch) {
    v = simd_max(v);
    if (lane == 0) scratch[simd_id] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
        float m = (lane < 8) ? scratch[lane] : -INFINITY;
        m = simd_max(m);
        if (lane == 0) scratch[0] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float r = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);   // scratch is reused by the next call
    return r;
}
static float groupSum(float v, uint lane, uint simd_id, threadgroup float* scratch) {
    v = simd_sum(v);
    if (lane == 0) scratch[simd_id] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
        float s = (lane < 8) ? scratch[lane] : 0.0f;
        s = simd_sum(s);
        if (lane == 0) scratch[0] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float r = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

kernel void attention_f32(device const float* q        [[buffer(0)]],
                          device const float* K        [[buffer(1)]],
                          device const float* V        [[buffer(2)]],
                          device float*       out      [[buffer(3)]],
                          constant uint& q_heads       [[buffer(4)]],
                          constant uint& kv_heads      [[buffer(5)]],
                          constant uint& head_dim      [[buffer(6)]],
                          constant uint& capacity      [[buffer(7)]],
                          constant uint& pos0          [[buffer(8)]],
                          constant uint& window        [[buffer(9)]],   // 0 = full attention
                          uint2 tg      [[threadgroup_position_in_grid]],
                          uint  tid     [[thread_index_in_threadgroup]],
                          uint  lane    [[thread_index_in_simdgroup]],
                          uint  simd_id [[simdgroup_index_in_threadgroup]]) {
    const uint token = tg.x, head = tg.y;
    const uint kv_head = head / (q_heads / kv_heads);
    const uint pos = pos0 + token;                       // this query's position
    const uint first = (window && pos + 1 > window) ? pos + 1 - window : 0;   // first visible key

    q   += (token * q_heads + head) * head_dim;
    out += (token * q_heads + head) * head_dim;
    K   += kv_head * capacity * head_dim;
    V   += kv_head * capacity * head_dim;

    threadgroup float q_s[512];
    threadgroup float p[CHUNK];       // this chunk's scores, then probabilities
    threadgroup float scratch[8];
    for (uint i = tid; i < head_dim; i += 256) q_s[i] = q[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m = -INFINITY, l = 0.0f;    // running max and sum
    float acc0 = 0.0f, acc1 = 0.0f;   // output dims tid and tid + 256
    const bool has1 = tid + 256 < head_dim;

    for (uint start = first; start <= pos; start += CHUNK) {
        // 1. score one key per thread (scale 1.0)
        const uint j = start + tid;
        float s = -INFINITY;
        if (j <= pos) {
            device const float* k = K + j * head_dim;
            s = 0.0f;
            for (uint d = 0; d < head_dim; ++d) s += q_s[d] * k[d];
        }
        // 2. new running max, rescale old state
        const float m_new = max(m, groupMax(s, lane, simd_id, scratch));
        const float alpha = exp(m - m_new);            // 0 on the first chunk (m = -inf)
        const float pj = exp(s - m_new);               // 0 for j > pos
        p[tid] = pj;
        l = l * alpha + groupSum(pj, lane, simd_id, scratch);   // barriers inside publish p[]
        // 3. accumulate this chunk's values into my output dims
        const uint n = min(CHUNK, pos + 1 - start);
        acc0 *= alpha; acc1 *= alpha;
        for (uint i = 0; i < n; ++i) {
            device const float* v = V + (start + i) * head_dim;
            acc0 += p[i] * v[tid];
            if (has1) acc1 += p[i] * v[tid + 256];
        }
        m = m_new;
        threadgroup_barrier(mem_flags::mem_threadgroup);   // p[] is rewritten next chunk
    }
    out[tid] = acc0 / l;
    if (has1) out[tid + 256] = acc1 / l;
}
```

Where the barriers are and why: `groupMax` and `groupSum` each end with
two barriers, so by the time `l` is updated every thread has written
`p[tid]`, and the value loop can read all of `p[]`. The barrier at the end
of the loop body keeps a fast thread from overwriting `p[]` for the next
chunk while a slow one is still reading it. `scratch` is reused by both
reductions, which is why each reduction ends with a barrier after reading
the result.

The score loop reads `head_dim` floats of `k` per thread with stride
`head_dim` between threads. That is the simple version; D13 tiles it.

## 4. `src/ops.h` additions

```cpp
#include <cmath>
#include "model_info.h"   // ModelConfig, for attentionShape

// Per-layer attention geometry, derived from ModelConfig and the layer type.
struct AttentionShape {
    uint32_t q_heads;      // 16
    uint32_t kv_heads;     // 8 sliding, 2 full
    uint32_t head_dim;     // 256 sliding, 512 full
    uint32_t rope_pairs;   // 128 sliding, 64 full
    float    rope_base;    // 1e4 sliding, 1e6 full
    uint32_t window;       // 1024 sliding, 0 full
};

inline AttentionShape attentionShape(const ModelConfig& c, int layer) {
    const bool full = c.layer_types[layer] == "full_attention";
    AttentionShape s;
    s.q_heads   = uint32_t(c.num_attention_heads);
    s.kv_heads  = uint32_t(full ? c.num_global_key_value_heads : c.num_key_value_heads);
    s.head_dim  = uint32_t(full ? c.global_head_dim : c.head_dim);
    s.rope_pairs = uint32_t(full ? (c.global_head_dim * c.partial_rotary_factor_full) / 2 : c.head_dim / 2);
    s.rope_base = float(full ? c.rope_theta_full : c.rope_theta_sliding);
    s.window    = uint32_t(full ? 0 : c.sliding_window);
    return s;
}

// One layer's attention weights. v is unused when has_v is false (full layers: v_raw = k_raw).
struct AttentionWeights {
    Nvfp4Matrix q, k, v, o;
    bool has_v;
    Buffer* norms; size_t q_norm_offset, k_norm_offset;   // BF16 [head_dim] each, in the core buffer
};

// One layer's KV cache: K and V are [kv_heads][capacity][head_dim] f32.
struct KvCache {
    Buffer* k;
    Buffer* v;
    uint32_t capacity;
};

// Scratch for one attention call, sized for the largest layer (head_dim 512).
struct AttentionScratch {
    Buffer* q;     // [tokens, 16, hd]  projection, then normed+roped in place
    Buffer* k;     // [tokens, kv, hd]  projection
    Buffer* v;     // [tokens, kv, hd]  projection (sliding layers only)
    Buffer* o;     // [tokens, 16, hd]  attention output
};

// Queue RMSNorm (+ optional RoPE) per (token, head) with explicit output strides.
inline void headNormRope(Backend& backend, Buffer* input, Buffer* weights, size_t weight_offset,
                         bool has_weight, Buffer* output, size_t output_offset, uint32_t tokens,
                         uint32_t heads, uint32_t head_dim, uint32_t rope_pairs, float rope_base,
                         uint32_t pos0, uint32_t out_token_stride, uint32_t out_head_stride,
                         float eps) {
    const uint32_t weight_flag = has_weight ? 1 : 0;
    backend.run("head_norm_rope_f32",
                {Arg::buf(input), Arg::buf(weights, weight_offset), Arg::buf(output, output_offset),
                 Arg::val(heads), Arg::val(head_dim), Arg::val(weight_flag), Arg::val(rope_pairs),
                 Arg::val(rope_base), Arg::val(pos0), Arg::val(out_token_stride),
                 Arg::val(out_head_stride), Arg::val(eps)},
                Dim3{tokens, heads}, Dim3{256});
}

// Queue the softmax-over-keys for every (token, q head).
inline void attention(Backend& backend, Buffer* q, const KvCache& cache, Buffer* output,
                      const AttentionShape& s, uint32_t tokens, uint32_t pos0) {
    backend.run("attention_f32",
                {Arg::buf(q), Arg::buf(cache.k), Arg::buf(cache.v), Arg::buf(output),
                 Arg::val(s.q_heads), Arg::val(s.kv_heads), Arg::val(s.head_dim),
                 Arg::val(cache.capacity), Arg::val(pos0), Arg::val(s.window)},
                Dim3{tokens, s.q_heads}, Dim3{256});
}

// Whole attention block for `tokens` tokens starting at position pos0:
//   x → q/k/v proj → norms + RoPE (K,V into the cache) → attention → o_proj → output [tokens, hidden]
inline void attentionBlock(Backend& backend, Buffer* x, const AttentionWeights& w,
                           const AttentionShape& s, KvCache& cache, AttentionScratch& tmp,
                           Buffer* output, uint32_t tokens, uint32_t pos0, float eps) {
    const uint32_t hd = s.head_dim;
    matmulNvfp4(backend, x, w.q, tmp.q, tokens);
    matmulNvfp4(backend, x, w.k, tmp.k, tokens);
    Buffer* v_raw = tmp.k;
    if (w.has_v) { matmulNvfp4(backend, x, w.v, tmp.v, tokens); v_raw = tmp.v; }

    // q: norm + rope in place.
    headNormRope(backend, tmp.q, w.norms, w.q_norm_offset, true, tmp.q, 0, tokens, s.q_heads, hd,
                 s.rope_pairs, s.rope_base, pos0, s.q_heads * hd, hd, eps);
    // K: norm + rope into the cache at position pos0. V: norm only (no weight, no rope).
    const size_t cache_offset = size_t(pos0) * hd * sizeof(float);
    headNormRope(backend, tmp.k, w.norms, w.k_norm_offset, true, cache.k, cache_offset, tokens,
                 s.kv_heads, hd, s.rope_pairs, s.rope_base, pos0, hd, cache.capacity * hd, eps);
    headNormRope(backend, v_raw, w.norms, w.k_norm_offset, false, cache.v, cache_offset, tokens,
                 s.kv_heads, hd, 0, s.rope_base, pos0, hd, cache.capacity * hd, eps);

    attention(backend, tmp.q, cache, tmp.o, s, tokens, pos0);
    matmulNvfp4(backend, tmp.o, w.o, output, tokens);
}
```

Reading `tmp.q` and writing `tmp.q` in the same `headNormRope` call is
safe: each threadgroup reads its own (token, head) slice into threadgroup
memory, barriers, then writes only that slice.

`Arg::val` takes a reference and `run` copies the bytes right away, so the
temporaries (`weight_flag`, `s.q_heads * hd`) are fine.

## 5. `cpu_ref.h` additions

Change `rmsNormCpu` so `w_bf16 == nullptr` means weight 1.0. Then:

```cpp
// RoPE on one head vector, mlx traditional=False pairing (i, i + half).
inline void ropeCpu(float* x, size_t head_dim, size_t pairs, float base, uint32_t pos) {
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
inline void attentionRowCpu(const float* q, const float* K, const float* V, float* out,
                            size_t head_dim, size_t first, size_t last) {
    std::vector<double> score(last - first + 1);
    double m = -INFINITY;
    for (size_t j = first; j <= last; ++j) {
        double s = 0;
        for (size_t d = 0; d < head_dim; ++d) s += double(q[d]) * K[j * head_dim + d];
        score[j - first] = s;
        m = std::max(m, s);
    }
    double l = 0;
    for (double& s : score) { s = std::exp(s - m); l += s; }
    for (size_t d = 0; d < head_dim; ++d) {
        double acc = 0;
        for (size_t j = first; j <= last; ++j) acc += score[j - first] * V[j * head_dim + d];
        out[d] = float(acc / l);
    }
}
```

## 6. `src/tests/attention.cpp`

```cpp
// Attention parity test: GPU attention block + post norm vs the CPU reference and the Python dump.
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include "../backend/metal_backend.h"
#include "../file.h"
#include "../model_info.h"
#include "../npy.h"
#include "../nvfp4.h"
#include "../ops.h"
#include "cpu_ref.h"

static apex::Nvfp4Matrix quantMatrix(const apex::ModelInfo& info, apex::Buffer* core, const std::string& name);
static apex::AttentionWeights attentionWeights(const apex::ModelInfo& info, apex::Buffer* core, int layer);
static void attentionCpu(const apex::ModelInfo& info, const apex::MappedFile& core, int layer,
                         const apex::AttentionShape& s, const float* x, float* output, size_t tokens);

// Compare each layer's attention output against the CPU reference and the Python dump.
int run_test_attn(const std::string& packed_dir, const std::string& dump_dir) {
    apex::ModelInfo info = apex::ModelInfo::load(packed_dir);
    apex::MetalBackend backend(APEX_METALLIB);
    apex::MappedFile core = apex::mapFile(info.dir + "/core.bin", "core_weights");
    if (core.size != info.core_size) throw std::runtime_error("core.bin size != manifest");
    apex::Buffer* core_buffer = backend.wrapNoCopy(core.data, core.size);

    apex::Npy attn_in  = apex::loadNpy(dump_dir + "/attn_in.npy");    // (L, T, H)
    apex::Npy attn_out = apex::loadNpy(dump_dir + "/attn_out.npy");   // (L, T, H)
    const int64_t layer_count = info.config.num_hidden_layers;
    const int64_t hidden_dim  = info.config.hidden_size;
    if (attn_in.dtype != "<f4" || attn_in.shape.size() != 3 || attn_in.shape != attn_out.shape ||
        attn_in.shape[0] != layer_count || attn_in.shape[2] != hidden_dim) {
        throw std::runtime_error("test-attn: dump shapes do not match model");
    }
    const uint32_t tokens = uint32_t(attn_in.shape[1]);
    const size_t hidden_count = size_t(tokens) * size_t(hidden_dim);
    const float eps = float(info.config.rms_norm_eps);

    // Scratch sized for the largest layer kind (head_dim 512, 16 q heads, 8 kv heads).
    const uint32_t max_hd = uint32_t(std::max(info.config.head_dim, info.config.global_head_dim));
    const uint32_t max_kv = uint32_t(std::max(info.config.num_key_value_heads, info.config.num_global_key_value_heads));
    const size_t q_bytes  = size_t(tokens) * info.config.num_attention_heads * max_hd * sizeof(float);
    const size_t kv_bytes = size_t(tokens) * max_kv * max_hd * sizeof(float);
    apex::Buffer* x_buf = backend.allocate(hidden_count * sizeof(float), "activations");
    apex::AttentionScratch tmp{backend.allocate(q_bytes, "activations"), backend.allocate(kv_bytes, "activations"),
                               backend.allocate(kv_bytes, "activations"), backend.allocate(q_bytes, "activations")};
    apex::KvCache cache{backend.allocate(kv_bytes, "kv_cache"), backend.allocate(kv_bytes, "kv_cache"), tokens};
    apex::Buffer* o_buf   = backend.allocate(hidden_count * sizeof(float), "activations");
    apex::Buffer* out_buf = backend.allocate(hidden_count * sizeof(float), "activations");
    std::vector<float> cpu_out(hidden_count), cpu_normed(hidden_count);

    int failures = 0;
    for (int64_t layer = 0; layer < layer_count; ++layer) {
        const float* x = attn_in.f32() + size_t(layer) * hidden_count;
        const float* expected = attn_out.f32() + size_t(layer) * hidden_count;
        const apex::AttentionShape shape = apex::attentionShape(info.config, int(layer));
        const apex::AttentionWeights w = attentionWeights(info, core_buffer, int(layer));
        const std::string post_norm = "layers." + std::to_string(layer) + ".post_attention_layernorm.weight";

        // GPU: block, then post norm; prefill from position 0 into an empty cache.
        std::memcpy(backend.contents(x_buf), x, hidden_count * sizeof(float));
        apex::attentionBlock(backend, x_buf, w, shape, cache, tmp, o_buf, tokens, 0, eps);
        apex::rmsNorm(backend, o_buf, core_buffer, info.getTensor(post_norm).offset, out_buf, tokens,
                      uint32_t(hidden_dim), eps);
        backend.sync();

        // CPU reference.
        attentionCpu(info, core, int(layer), shape, x, cpu_out.data(), tokens);
        const uint16_t* post_w = reinterpret_cast<const uint16_t*>(
            static_cast<const char*>(core.data) + info.getTensor(post_norm).offset);
        for (uint32_t t = 0; t < tokens; ++t)
            rmsNormCpu(cpu_out.data() + t * hidden_dim, post_w, cpu_normed.data() + t * hidden_dim, hidden_dim, eps);

        const float* gpu = static_cast<const float*>(backend.contents(out_buf));
        const Diff gpu_ref = diff(gpu, expected, hidden_count);
        const Diff cpu_ref = diff(cpu_normed.data(), expected, hidden_count);
        const Diff gpu_cpu = diff(gpu, cpu_normed.data(), hidden_count);
        constexpr double tolerance = 1e-5;
        const bool ok = gpu_ref.rel_l2 < tolerance && cpu_ref.rel_l2 < tolerance && gpu_cpu.rel_l2 < tolerance;
        failures += !ok;
        std::printf("layer %2lld %-7s gpu-vs-ref rel %.2e max %.2e | cpu-vs-ref rel %.2e | gpu-vs-cpu rel %.2e  %s\n",
                    static_cast<long long>(layer), shape.window ? "sliding" : "full", gpu_ref.rel_l2,
                    gpu_ref.max_abs, cpu_ref.rel_l2, gpu_cpu.rel_l2, ok ? "ok" : "FAIL");
    }

    for (apex::Buffer* b : {x_buf, tmp.q, tmp.k, tmp.v, tmp.o, cache.k, cache.v, o_buf, out_buf}) backend.release(b);
    backend.release(core_buffer);
    apex::unmapFile(core);
    apex::Profile::get().print(stdout);
    std::printf("%lld layers tested, %d failures\n", static_cast<long long>(layer_count), failures);
    return failures ? 1 : 0;
}

/// GPU description of an NVFP4 matrix from its .weight and .scales tensors in core.bin.
static apex::Nvfp4Matrix quantMatrix(const apex::ModelInfo& info, apex::Buffer* core, const std::string& name) {
    const apex::TensorInfo& w = info.getTensor(name + ".weight");
    const apex::TensorInfo& s = info.getTensor(name + ".scales");
    return apex::Nvfp4Matrix{core, w.offset, core, s.offset, uint32_t(w.shape[0]), uint32_t(w.shape[1] * 8)};
}

/// One layer's attention weights. Full layers have no v_proj.
static apex::AttentionWeights attentionWeights(const apex::ModelInfo& info, apex::Buffer* core, int layer) {
    const std::string p = "layers." + std::to_string(layer) + ".self_attn.";
    apex::AttentionWeights w;
    w.q = quantMatrix(info, core, p + "q_proj");
    w.k = quantMatrix(info, core, p + "k_proj");
    w.has_v = info.core.count(p + "v_proj.weight") > 0;
    w.v = w.has_v ? quantMatrix(info, core, p + "v_proj") : w.k;
    w.o = quantMatrix(info, core, p + "o_proj");
    w.norms = core;
    w.q_norm_offset = info.getTensor(p + "q_norm.weight").offset;
    w.k_norm_offset = info.getTensor(p + "k_norm.weight").offset;
    return w;
}

/// One layer's attention block on the CPU: x → q/k/v → norms + RoPE → softmax over keys → o_proj.
static void attentionCpu(const apex::ModelInfo& info, const apex::MappedFile& core, int layer,
                         const apex::AttentionShape& s, const float* x, float* output, size_t tokens) {
    const size_t H = size_t(info.config.hidden_size), hd = s.head_dim, nq = s.q_heads, nkv = s.kv_heads;
    const float eps = float(info.config.rms_norm_eps);
    const std::string p = "layers." + std::to_string(layer) + ".self_attn.";
    const char* base = static_cast<const char*>(core.data);
    auto tensor = [&](const std::string& n) { return base + info.getTensor(p + n).offset; };
    auto matmul = [&](const std::string& n, const float* in, float* out, size_t rows, size_t k) {
        matvecNvfp4Cpu(reinterpret_cast<const uint32_t*>(tensor(n + ".weight")),
                       reinterpret_cast<const uint8_t*>(tensor(n + ".scales")), in, out, tokens, rows, k);
    };
    const bool has_v = info.core.count(p + "v_proj.weight") > 0;

    // Projections: q [T, nq, hd], k_raw/v_raw [T, nkv, hd].
    std::vector<float> q(tokens * nq * hd), k_raw(tokens * nkv * hd), v_raw(tokens * nkv * hd);
    matmul("q_proj", x, q.data(), nq * hd, H);
    matmul("k_proj", x, k_raw.data(), nkv * hd, H);
    if (has_v) matmul("v_proj", x, v_raw.data(), nkv * hd, H); else v_raw = k_raw;

    // Norms + RoPE. K, V stored [nkv][T][hd] like the cache.
    const uint16_t* q_norm = reinterpret_cast<const uint16_t*>(tensor("q_norm.weight"));
    const uint16_t* k_norm = reinterpret_cast<const uint16_t*>(tensor("k_norm.weight"));
    std::vector<float> K(nkv * tokens * hd), V(nkv * tokens * hd);
    for (size_t t = 0; t < tokens; ++t) {
        for (size_t h = 0; h < nq; ++h) {
            float* row = q.data() + (t * nq + h) * hd;
            rmsNormCpu(row, q_norm, row, hd, eps);
            ropeCpu(row, hd, s.rope_pairs, s.rope_base, uint32_t(t));
        }
        for (size_t h = 0; h < nkv; ++h) {
            float* k = K.data() + (h * tokens + t) * hd;
            rmsNormCpu(k_raw.data() + (t * nkv + h) * hd, k_norm, k, hd, eps);
            ropeCpu(k, hd, s.rope_pairs, s.rope_base, uint32_t(t));
            rmsNormCpu(v_raw.data() + (t * nkv + h) * hd, nullptr, V.data() + (h * tokens + t) * hd, hd, eps);
        }
    }

    // Softmax over the visible keys, then o_proj.
    std::vector<float> o(tokens * nq * hd);
    for (size_t t = 0; t < tokens; ++t) {
        const size_t first = (s.window && t + 1 > s.window) ? t + 1 - s.window : 0;
        for (size_t h = 0; h < nq; ++h) {
            const size_t kvh = h / (nq / nkv);
            attentionRowCpu(q.data() + (t * nq + h) * hd, K.data() + kvh * tokens * hd,
                            V.data() + kvh * tokens * hd, o.data() + (t * nq + h) * hd, hd, first, t);
        }
    }
    matmul("o_proj", o.data(), output, H, nq * hd);
}
```

`rmsNormCpu(row, q_norm, row, ...)` normalizes in place; the current
implementation reads all of `x` before writing `y`, which makes that fine.
If you change it to a one-pass form, give it a temporary.

`main.cpp`: declare `run_test_attn` and add the `test-attn` subcommand.

## Expected output

```
./build/apex test-attn /Users/spotta/Desktop/models/gemma4/packed dumps/paris
layer  0 sliding gpu-vs-ref rel 1.4e-06 max ... | cpu-vs-ref rel 1.4e-06 | gpu-vs-cpu rel ...  ok
...
layer  5 full    gpu-vs-ref rel 1.7e-06 ...                                                      ok
...
30 layers tested, 0 failures
```

Relative L2 lands around 1e-6 to 3e-6, higher than the MLP's 1e-7,
because the reference ran its attention kernel in a different
accumulation order and the softmax amplifies it. Still 5x under the gate.
Run `para` (54 tokens) too.

The CPU side is heavier here: q_proj on full layers is 8192 × 2816 per
token. Expect ~1 s per layer in Release.

## If it goes wrong

| Symptom | Likely cause |
|---|---|
| sliding layers ok, full layers off | `rope_pairs` 64 with `head_dim` 512 in the exponent; or V taken after k_norm instead of from `k_raw`; or kv_heads 2 not read from `num_global_key_value_heads` |
| all layers off by a lot, CPU matches GPU | scale applied (must be 1.0); or `attn_out` compared before the post norm |
| token 0 ok, later tokens off | RoPE position (`pos0 + token`), or pairing `(i, i+hd/2)` done as `(2i, 2i+1)` |
| full layers ok, sliding off | window test: key `j` visible iff `j > t - 1024` (only matters above 1024 tokens, so not on these dumps) |
| GPU off, CPU ok, errors grow with token index | `attention_f32` missing a barrier before `p[]` is rewritten, or `alpha` not applied to `acc` |
| NaN | first chunk's `m = -inf` reached `exp(m - m_new)` with `m_new = -inf`: `first > pos` (bad window math) |
| some heads off | GQA mapping `h / (q_heads / kv_heads)`; or cache head stride not `capacity * hd` |
| GPU q wrong, K right | in-place `headNormRope` on `tmp.q` with strides not matching the input layout |

## Commit split

1. `ModelConfig` attention fields.
2. `head_norm_rope.metal`, `attention.metal`, `ops.h` additions.
3. `cpu_ref.h` additions, `attention.cpp`, subcommand.

## What stage 4 needs from this stage

Nothing new in kernels. The router is `rms_norm_f32` with the `router.scale`
weight, an affine 8-bit matmul (new small kernel, group 64, scales and
biases in BF16), top-8 and softmax on 128 values per token (one tiny
kernel or the CPU), times `per_expert_scale`. Verified against
`route_ids` and `route_weights`.
