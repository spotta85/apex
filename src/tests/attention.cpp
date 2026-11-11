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
#include "test_util.h"

namespace {

constexpr uint32_t DECODE_STEPS = 3; // single-token steps after the prefill chunk

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
    apex::KvCache cache{ctx.alloc(kv_bytes, "kv_cache"), ctx.alloc(kv_bytes, "kv_cache"), T};
    apex::Buffer *o_chunk = ctx.alloc(hidden_count * sizeof(float));
    apex::Buffer *o_all = ctx.alloc(hidden_count * sizeof(float));
    apex::Buffer *out_buf = ctx.alloc(hidden_count * sizeof(float));
    std::vector<float> cpu_out(hidden_count), cpu_normed(hidden_count);

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
        for (auto [t0, n] : chunks) {
            std::memcpy(ctx.f32(x_buf), x + size_t(t0) * H, n * row_bytes);
            ctx.ops.attentionBlock(x_buf, w.attn, w.shape, cache, tmp, o_chunk, n, t0, ctx.eps);
            ctx.backend.sync();
            std::memcpy(ctx.f32(o_all) + size_t(t0) * H, ctx.f32(o_chunk), n * row_bytes);
        }
        ctx.ops.rmsNorm(o_all, ctx.core_buffer, w.post_attn_norm, out_buf, T, uint32_t(H), ctx.eps);
        ctx.backend.sync();

        // CPU reference over the whole sequence at once.
        attentionBlockCpu(ctx, w, x, cpu_out.data(), T);
        const uint16_t *post_w = ctx.cpu<uint16_t>(w.post_attn_norm);
        for (uint32_t t = 0; t < T; ++t)
            rmsNormCpu(cpu_out.data() + t * H, post_w, cpu_normed.data() + t * H, H, ctx.eps);

        const float *gpu = ctx.f32(out_buf);
        failures += !apex::report(w.shape.window ? "sliding" : "full", layer,
                                  diff(gpu, expected, hidden_count),
                                  diff(cpu_normed.data(), expected, hidden_count),
                                  diff(gpu, cpu_normed.data(), hidden_count), 1e-5);
    }
    apex::Profile::get().print(stdout);
    std::printf("%lld layers tested (prefill %u + %u decode steps), %d failures\n",
                static_cast<long long>(L), prefill, T - prefill, failures);
    return failures ? 1 : 0;
}

namespace {

// Round a float to BF16 bits (truncation is fine for synthetic weights).
uint16_t toBf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return uint16_t(u >> 16);
}

struct SynthConfig {
    const char *name;
    uint32_t q_heads, kv_heads, head_dim, rope_pairs;
    float rope_base;
    uint32_t window;
};

// One synthetic run. Returns the rel L2 of GPU vs CPU over all real tokens.
double runSynth(apex::TestContext &ctx, const SynthConfig &cfg, uint32_t tokens,
                uint32_t capacity, uint32_t max_chunk) {
    const uint32_t hd = cfg.head_dim, nq = cfg.q_heads, nkv = cfg.kv_heads;
    const apex::AttentionShape shape{nq, nkv, hd, cfg.rope_pairs, cfg.rope_base, cfg.window};
    Lcg rng(hd * 31 + cfg.window);

    // Real inputs for the whole sequence, plus junk for speculative writes that get rolled back.
    const size_t qn = size_t(nq) * hd, kvn = size_t(nkv) * hd;
    std::vector<float> q_raw(tokens * qn), k_raw(tokens * kvn), v_raw(tokens * kvn);
    for (float &v : q_raw) v = 3.0f * rng.next();
    for (float &v : k_raw) v = 3.0f * rng.next();
    for (float &v : v_raw) v = rng.next();
    std::vector<uint16_t> norms(2 * hd);
    for (uint16_t &v : norms) v = toBf16(1.0f + 0.5f * rng.next());

    // CPU reference.
    std::vector<float> o_ref(tokens * qn);
    attentionCoreCpu(q_raw.data(), k_raw.data(), v_raw.data(), norms.data(), norms.data() + hd,
                     o_ref.data(), tokens, nq, nkv, hd, cfg.rope_pairs, cfg.rope_base, cfg.window,
                     ctx.eps);

    // GPU buffers.
    apex::Buffer *norm_buf = ctx.alloc(norms.size() * sizeof(uint16_t));
    std::memcpy(ctx.backend.contents(norm_buf), norms.data(), norms.size() * sizeof(uint16_t));
    apex::AttentionWeights w{};
    w.norms = norm_buf;
    w.q_norm_offset = 0;
    w.k_norm_offset = hd * sizeof(uint16_t);
    apex::Buffer *q_buf = ctx.alloc(max_chunk * qn * sizeof(float));
    apex::Buffer *k_buf = ctx.alloc(max_chunk * kvn * sizeof(float));
    apex::Buffer *v_buf = ctx.alloc(max_chunk * kvn * sizeof(float));
    apex::Buffer *o_buf = ctx.alloc(max_chunk * qn * sizeof(float));
    const size_t cache_bytes = size_t(nkv) * capacity * hd * sizeof(float);
    apex::KvCache cache{ctx.alloc(cache_bytes, "kv_cache"), ctx.alloc(cache_bytes, "kv_cache"),
                        capacity};
    std::vector<float> o_gpu(tokens * qn);

    // Write n tokens at position pos from the given rows and attend; returns nothing, the
    // caller copies o_buf if it wants the result.
    auto step = [&](const float *q, const float *k, const float *v, uint32_t n, uint32_t pos) {
        std::memcpy(ctx.f32(q_buf), q, n * qn * sizeof(float));
        std::memcpy(ctx.f32(k_buf), k, n * kvn * sizeof(float));
        std::memcpy(ctx.f32(v_buf), v, n * kvn * sizeof(float));
        ctx.ops.qkvNormRope(q_buf, k_buf, v_buf, w, shape, cache, n, pos, ctx.eps);
        ctx.ops.attention(q_buf, cache, o_buf, shape, n, pos);
        ctx.backend.sync();
    };

    // Chunk schedule: varying sizes up to max_chunk; every third chunk is preceded by a
    // speculative junk write of up to max_chunk tokens that is then rolled back (the next real
    // write starts at the same position and overwrites it).
    std::vector<float> junk_q(max_chunk * qn), junk_k(max_chunk * kvn), junk_v(max_chunk * kvn);
    for (float &v : junk_q) v = 5.0f * rng.next();
    for (float &v : junk_k) v = 5.0f * rng.next();
    for (float &v : junk_v) v = 5.0f * rng.next();
    const uint32_t pattern[] = {max_chunk, 1, 1, 3, 1, 2, 1, max_chunk, 1, 1};
    uint32_t pos = 0, chunk_index = 0;
    while (pos < tokens) {
        const uint32_t n = std::min({pattern[chunk_index % 10], max_chunk, tokens - pos});
        if (chunk_index % 3 == 2) {
            // A ring cache (capacity < tokens) takes any write; a linear one only up to its end.
            const bool ring = capacity < tokens;
            const uint32_t junk = ring ? max_chunk : std::min(max_chunk, capacity - pos);
            if (junk > 0) {
                step(junk_q.data(), junk_k.data(), junk_v.data(), junk, pos);
            }
        }
        step(q_raw.data() + pos * qn, k_raw.data() + pos * kvn, v_raw.data() + pos * kvn, n,
             pos);
        std::memcpy(o_gpu.data() + pos * qn, ctx.f32(o_buf), n * qn * sizeof(float));
        pos += n;
        ++chunk_index;
    }
    const Diff d = diff(o_gpu.data(), o_ref.data(), o_gpu.size());
    std::printf("%-8s hd %3u q %u kv %u window %2u capacity %2u chunks<=%u tokens %u  "
                "gpu-vs-cpu rel %.2e max %.2e\n",
                cfg.name, hd, nq, nkv, cfg.window, capacity, max_chunk, tokens, d.rel_l2,
                d.max_abs);
    return d.rel_l2;
}

} // namespace

// Synthetic attention: tiny window, chunked and rolled-back writes, vs the CPU reference.
int run_test_attn_synth(const std::string &packed_dir) {
    apex::TestContext ctx(packed_dir); // only for the backend and eps; no weights are read
    constexpr uint32_t TOKENS = 40;
    constexpr uint32_t WINDOW = 8;
    const SynthConfig sliding{"sliding", 4, 2, 256, 128, 1e4f, WINDOW};
    const SynthConfig full{"full", 4, 1, 512, 64, 1e6f, 0};
    // Linear cache: one slot per position.
    const uint32_t capacity = TOKENS;
    const uint32_t max_chunk = 5;
    int failures = 0;
    for (const SynthConfig &cfg : {sliding, full}) {
        failures += runSynth(ctx, cfg, TOKENS, capacity, max_chunk) > 1e-5;
    }
    std::printf("synthetic attention: %d failures\n", failures);
    return failures ? 1 : 0;
}
