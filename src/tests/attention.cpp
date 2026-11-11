// Attention parity tests.
//
// test-attn:       GPU attention block + post norm vs the CPU reference and the Python dump,
//                  per layer. Tokens go through in two ways at once: a prefill chunk of
//                  T - DECODE_STEPS tokens, then DECODE_STEPS single-token decode steps, so
//                  both the multi-token path and pos0 > 0 are exercised against real weights.
// test-attn-synth: random q/k/v through the norm+RoPE+attention kernels with a tiny window,
//                  chunked writes and rolled-back speculative writes, against the CPU
//                  reference. The dumps are too short to reach the sliding window.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include "../kv_cache.h"
#include "test_util.h"

namespace {

constexpr uint32_t DECODE_STEPS = 3; // single-token steps after the prefill chunk
// fp16 keys/values round each element to 11 significant bits (~2.4e-4 relative), so the
// attention output cannot match the fp32 reference to 1e-5; this gate catches real bugs only.
constexpr double F16_TOLERANCE = 2e-3;

/// One layer's attention block on the CPU: x → q/k/v → norms + RoPE → softmax → o_proj.
void attentionBlockCpu(const apex::TestContext &ctx, const apex::LayerWeights &w,
                       const float *x, float *output, size_t tokens) {
    const apex::AttentionShape &s = w.shape;
    const size_t H = size_t(ctx.info.config.hidden_size), hd = s.head_dim;
    const size_t nq = s.q_heads, nkv = s.kv_heads;
    auto matmul = [&](const apex::Nvfp4Matrix &m, const float *in, float *out) {
        matvecNvfp4Cpu(ctx.cpu<uint32_t>(m.weights_offset), ctx.cpu<uint8_t>(m.scales_offset),
                       in, out, tokens, m.w_rows, m.hidden_dim);
    };
    std::vector<float> q(tokens * nq * hd), k_raw(tokens * nkv * hd), v_raw(tokens * nkv * hd);
    matmul(w.attn.q, x, q.data());
    matmul(w.attn.k, x, k_raw.data());
    if (w.attn.has_v) {
        matmul(w.attn.v, x, v_raw.data());
    } else {
        v_raw = k_raw; // full layers: V is the raw k projection (before k_norm)
    }
    std::vector<float> o(tokens * nq * hd);
    attentionCoreCpu(q.data(), k_raw.data(), v_raw.data(),
                     ctx.cpu<uint16_t>(w.attn.q_norm_offset),
                     ctx.cpu<uint16_t>(w.attn.k_norm_offset), o.data(), tokens, nq, nkv, hd,
                     s.rope_pairs, s.rope_base, s.window, ctx.eps);
    matmul(w.attn.o, o.data(), output);
}

} // namespace

// Compare each layer's attention output against the CPU reference and the Python dump.
int run_test_attn(const std::string &packed_dir, const std::string &dump_dir) {
    apex::TestContext ctx(packed_dir);
    const apex::ModelConfig &c = ctx.info.config;
    const int64_t L = c.num_hidden_layers, H = c.hidden_size;
    apex::Npy probe = apex::loadNpy(dump_dir + "/attn_in.npy");
    if (probe.shape.size() != 3) throw std::runtime_error("test-attn: bad attn_in shape");
    const uint32_t T = uint32_t(probe.shape[1]);
    const apex::Npy attn_in = apex::loadDump(dump_dir, "attn_in", {L, T, H});
    const apex::Npy attn_out = apex::loadDump(dump_dir, "attn_out", {L, T, H});
    const size_t hidden_count = size_t(T) * size_t(H);
    const size_t row_bytes = size_t(H) * sizeof(float);

    // Scratch sized for the largest layer kind (head_dim 512, 16 q heads, 8 kv heads).
    const size_t max_hd = size_t(std::max(c.head_dim, c.global_head_dim));
    const size_t max_kv_rows =
        size_t(std::max(c.num_key_value_heads * c.head_dim,
                        c.num_global_key_value_heads * c.global_head_dim));
    const size_t q_bytes = size_t(T) * c.num_attention_heads * max_hd * sizeof(float);
    const size_t kv_bytes = size_t(T) * max_kv_rows * sizeof(float);
    apex::Buffer *x_buf = ctx.alloc(hidden_count * sizeof(float));
    const apex::AttentionScratch tmp{ctx.alloc(q_bytes), ctx.alloc(kv_bytes), ctx.alloc(kv_bytes),
                                     ctx.alloc(q_bytes)};
    // One linear cache per element type (the dumps are shorter than the window, so a ring
    // would never wrap here; test-attn-synth covers the ring).
    const apex::KvCache caches[2] = {
        {ctx.alloc(kv_bytes, "kv_cache"), ctx.alloc(kv_bytes, "kv_cache"), T, apex::KvDtype::F32},
        {ctx.alloc(kv_bytes, "kv_cache"), ctx.alloc(kv_bytes, "kv_cache"), T, apex::KvDtype::F16}};
    apex::Buffer *o_chunk = ctx.alloc(hidden_count * sizeof(float));
    apex::Buffer *o_all = ctx.alloc(hidden_count * sizeof(float));
    apex::Buffer *out_buf[2] = {ctx.alloc(hidden_count * sizeof(float)),
                                ctx.alloc(hidden_count * sizeof(float))};
    std::vector<float> cpu_out(hidden_count), cpu_normed(hidden_count);
    double worst_f16 = 0;

    // Chunks: [0, prefill) then one token at a time.
    const uint32_t prefill = T > DECODE_STEPS ? T - DECODE_STEPS : T;
    std::vector<std::pair<uint32_t, uint32_t>> chunks{{0, prefill}};
    for (uint32_t t = prefill; t < T; ++t) chunks.push_back({t, 1});

    int failures = 0;
    for (int layer = 0; layer < L; ++layer) {
        const apex::LayerWeights &w = ctx.weights.layers[layer];
        const float *x = attn_in.f32() + size_t(layer) * hidden_count;
        const float *expected = attn_out.f32() + size_t(layer) * hidden_count;

        // GPU: chunked attention into the same cache, then the post norm over all tokens.
        // Once with an fp32 cache, once with an fp16 one.
        for (int d = 0; d < 2; ++d) {
            for (auto [t0, n] : chunks) {
                std::memcpy(ctx.f32(x_buf), x + size_t(t0) * H, n * row_bytes);
                ctx.ops.attentionBlock(x_buf, w.attn, w.shape, caches[d], tmp, o_chunk, n, t0,
                                       ctx.eps);
                ctx.backend.sync();
                std::memcpy(ctx.f32(o_all) + size_t(t0) * H, ctx.f32(o_chunk), n * row_bytes);
            }
            ctx.ops.rmsNorm(o_all, ctx.core_buffer, w.post_attn_norm, out_buf[d], T, uint32_t(H),
                            ctx.eps);
            ctx.backend.sync();
        }

        // CPU reference over the whole sequence at once.
        attentionBlockCpu(ctx, w, x, cpu_out.data(), T);
        const uint16_t *post_w = ctx.cpu<uint16_t>(w.post_attn_norm);
        for (uint32_t t = 0; t < T; ++t)
            rmsNormCpu(cpu_out.data() + t * H, post_w, cpu_normed.data() + t * H, H, ctx.eps);

        const float *gpu = ctx.f32(out_buf[0]);
        failures += !apex::report(w.shape.window ? "sliding" : "full", layer,
                                  diff(gpu, expected, hidden_count),
                                  diff(cpu_normed.data(), expected, hidden_count),
                                  diff(gpu, cpu_normed.data(), hidden_count), 1e-5);
        const Diff f16 = diff(ctx.f32(out_buf[1]), expected, hidden_count);
        worst_f16 = std::max(worst_f16, f16.rel_l2);
        const bool f16_ok = f16.rel_l2 < F16_TOLERANCE;
        failures += !f16_ok;
        std::printf("         fp16 KV  gpu-vs-ref rel %.2e max %.2e  %s\n", f16.rel_l2,
                    f16.max_abs, f16_ok ? "ok" : "FAIL");
    }
    apex::Profile::get().print(stdout);
    std::printf("%lld layers tested (prefill %u + %u decode steps; fp32 KV gate 1e-5, fp16 KV "
                "gate %.0e, worst fp16 %.2e), %d failures\n",
                static_cast<long long>(L), prefill, T - prefill, F16_TOLERANCE, worst_f16,
                failures);
    return failures ? 1 : 0;
}

namespace {

// Round a float to BF16 bits (truncation is fine for synthetic weights).
uint16_t toBf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return uint16_t(u >> 16);
}

// Expect `fn` to throw; returns 1 (a failure) if it does not.
template <class F> int expectThrow(const char *what, F fn) {
    try {
        fn();
    } catch (const std::exception &) {
        std::printf("  rejects %-52s ok\n", what);
        return 0;
    }
    std::printf("  rejects %-52s FAIL (accepted)\n", what);
    return 1;
}

// A two-layer toy model: layer 0 sliding (4 q heads, 2 kv heads, hd 256, window 8), layer 1
// full attention (4 q heads, 1 kv head, hd 512, RoPE on a quarter of the dims). Only the
// fields attentionShape() reads are set.
apex::ModelConfig synthConfig(uint32_t window) {
    apex::ModelConfig c;
    c.num_hidden_layers = 2;
    c.layer_types = {"sliding_attention", "full_attention"};
    c.num_attention_heads = 4;
    c.num_key_value_heads = 2;
    c.head_dim = 256;
    c.num_global_key_value_heads = 1;
    c.global_head_dim = 512;
    c.partial_rotary_factor_full = 0.25;
    c.rope_theta_sliding = 1e4;
    c.rope_theta_full = 1e6;
    c.sliding_window = int(window);
    return c;
}

// One synthetic run over both toy layers with the given cache dtype. Returns failures.
int runSynth(apex::TestContext &ctx, const apex::ModelConfig &cfg, apex::KvDtype dtype,
             uint32_t tokens, uint32_t slack) {
    apex::KvCacheOptions kv;
    kv.max_context = tokens;
    kv.slack = slack;
    kv.dtype = dtype;
    apex::KvCacheSet caches(ctx.backend, cfg, kv);
    const uint32_t max_chunk = caches.maxRollback(); // slack + 1: the largest legal ring write
    const int layers = cfg.num_hidden_layers;

    struct LayerData {
        apex::AttentionShape shape;
        apex::AttentionWeights w{};
        size_t qn, kvn;
        std::vector<float> q_raw, k_raw, v_raw, junk_q, junk_k, junk_v, o_ref, o_gpu;
        apex::Buffer *q_buf, *k_buf, *v_buf, *o_buf;
    };
    std::vector<LayerData> data(static_cast<size_t>(layers));
    for (int l = 0; l < layers; ++l) {
        LayerData &d = data[size_t(l)];
        d.shape = apex::attentionShape(cfg, l);
        const uint32_t hd = d.shape.head_dim;
        d.qn = size_t(d.shape.q_heads) * hd;
        d.kvn = size_t(d.shape.kv_heads) * hd;
        Lcg rng(uint64_t(hd) * 31 + uint64_t(l));
        auto fill = [&](std::vector<float> &v, size_t n, float scale) {
            v.resize(n);
            for (float &x : v) x = scale * rng.next();
        };
        fill(d.q_raw, tokens * d.qn, 3.0f);
        fill(d.k_raw, tokens * d.kvn, 3.0f);
        fill(d.v_raw, tokens * d.kvn, 1.0f);
        // Junk for speculative writes that get rolled back.
        fill(d.junk_q, max_chunk * d.qn, 5.0f);
        fill(d.junk_k, max_chunk * d.kvn, 5.0f);
        fill(d.junk_v, max_chunk * d.kvn, 5.0f);
        std::vector<uint16_t> norms(2 * hd);
        for (uint16_t &v : norms) v = toBf16(1.0f + 0.5f * rng.next());

        d.o_ref.resize(tokens * d.qn);
        d.o_gpu.resize(tokens * d.qn);
        attentionCoreCpu(d.q_raw.data(), d.k_raw.data(), d.v_raw.data(), norms.data(),
                         norms.data() + hd, d.o_ref.data(), tokens, d.shape.q_heads,
                         d.shape.kv_heads, hd, d.shape.rope_pairs, d.shape.rope_base,
                         d.shape.window, ctx.eps);

        apex::Buffer *norm_buf = ctx.alloc(norms.size() * sizeof(uint16_t));
        std::memcpy(ctx.backend.contents(norm_buf), norms.data(), norms.size() * sizeof(uint16_t));
        d.w.norms = norm_buf;
        d.w.q_norm_offset = 0;
        d.w.k_norm_offset = hd * sizeof(uint16_t);
        d.q_buf = ctx.alloc(max_chunk * d.qn * sizeof(float));
        d.k_buf = ctx.alloc(max_chunk * d.kvn * sizeof(float));
        d.v_buf = ctx.alloc(max_chunk * d.kvn * sizeof(float));
        d.o_buf = ctx.alloc(max_chunk * d.qn * sizeof(float));
    }

    // Write n tokens at pos into every layer (real rows, or junk) and attend, like forward().
    auto step = [&](uint32_t n, uint32_t pos, bool junk) {
        caches.checkWrite(pos, n);
        for (int l = 0; l < layers; ++l) {
            LayerData &d = data[size_t(l)];
            const float *q = junk ? d.junk_q.data() : d.q_raw.data() + pos * d.qn;
            const float *k = junk ? d.junk_k.data() : d.k_raw.data() + pos * d.kvn;
            const float *v = junk ? d.junk_v.data() : d.v_raw.data() + pos * d.kvn;
            std::memcpy(ctx.f32(d.q_buf), q, n * d.qn * sizeof(float));
            std::memcpy(ctx.f32(d.k_buf), k, n * d.kvn * sizeof(float));
            std::memcpy(ctx.f32(d.v_buf), v, n * d.kvn * sizeof(float));
            const apex::KvCache &cache = caches.layer(l);
            ctx.ops.qkvNormRope(d.q_buf, d.k_buf, d.v_buf, d.w, d.shape, cache, n, pos, ctx.eps);
            ctx.ops.attention(d.q_buf, cache, d.o_buf, d.shape, n, pos);
        }
        ctx.backend.sync();
        caches.commitWrite(pos, n);
        if (!junk) {
            for (LayerData &d : data)
                std::memcpy(d.o_gpu.data() + pos * d.qn, ctx.f32(d.o_buf),
                            n * d.qn * sizeof(float));
        }
    };

    // Chunk schedule: varying sizes up to max_chunk; every third chunk is preceded by a
    // speculative junk write of max_chunk tokens that is truncated away again, so the ring's
    // rollback limit is exercised at its edge.
    const uint32_t pattern[] = {max_chunk, 1, 1, 3, 1, 2, 1, max_chunk, 1, 1};
    uint32_t pos = 0, chunk_index = 0;
    while (pos < tokens) {
        const uint32_t n = std::min({pattern[chunk_index % 10], max_chunk, tokens - pos});
        if (chunk_index % 3 == 2) {
            const uint32_t junk = std::min(max_chunk, tokens - pos); // full layer is linear
            step(junk, pos, true);
            caches.truncate(pos);
        }
        step(n, pos, false);
        pos += n;
        ++chunk_index;
    }

    int failures = 0;
    const double tolerance = dtype == apex::KvDtype::F16 ? F16_TOLERANCE : 1e-5;
    for (int l = 0; l < layers; ++l) {
        const LayerData &d = data[size_t(l)];
        const apex::KvCache &cache = caches.layer(l);
        const Diff diff_l = diff(d.o_gpu.data(), d.o_ref.data(), d.o_gpu.size());
        const bool ok = diff_l.rel_l2 < tolerance;
        failures += !ok;
        std::printf("%s KV %-7s hd %3u q %u kv %u window %2u capacity %2u writes<=%u tokens %u  "
                    "gpu-vs-cpu rel %.2e max %.2e  %s\n",
                    dtype == apex::KvDtype::F16 ? "fp16" : "fp32",
                    d.shape.window ? "sliding" : "full", d.shape.head_dim, d.shape.q_heads,
                    d.shape.kv_heads, d.shape.window, cache.capacity, max_chunk, tokens,
                    diff_l.rel_l2, diff_l.max_abs, ok ? "ok" : "FAIL");
    }

    // The cache must refuse what would silently corrupt the ring.
    const uint32_t len = caches.length();
    failures += expectThrow("a rollback of slack + 2 positions", [&] {
        caches.truncate(len - (max_chunk + 1));
    });
    failures += expectThrow("a ring write of slack + 2 tokens", [&] {
        caches.truncate(len - max_chunk); // legal rollback, then an oversized write
        caches.checkWrite(caches.length(), max_chunk + 1);
    });
    failures += expectThrow("a write that leaves a gap", [&] {
        caches.checkWrite(caches.length() + 1, 1);
    });
    return failures;
}

} // namespace

// Synthetic attention: tiny window and ring, chunked and rolled-back writes, vs the CPU
// reference, for both KV cache element types.
int run_test_attn_synth(const std::string &packed_dir) {
    apex::TestContext ctx(packed_dir); // only for the backend and eps; no weights are read
    constexpr uint32_t TOKENS = 40;
    constexpr uint32_t WINDOW = 8;
    constexpr uint32_t SLACK = 4; // ring capacity 12: wraps three times over 40 tokens
    const apex::ModelConfig cfg = synthConfig(WINDOW);
    int failures = 0;
    for (apex::KvDtype dtype : {apex::KvDtype::F32, apex::KvDtype::F16}) {
        failures += runSynth(ctx, cfg, dtype, TOKENS, SLACK);
    }
    std::printf("synthetic attention: %d failures\n", failures);
    return failures ? 1 : 0;
}
