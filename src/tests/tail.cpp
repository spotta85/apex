// Row kernels and the output head against the dumps. Core weights only.
//
// test-tail checks, in model order:
//   embed      embed_norm_nvfp4: h vs embed.npy, x vs attn_in[0]
//   residual   attn_residual_norms_f32 per layer, fed the GPU attention block's o_proj output
//              (test-attn verifies that): h vs layer_in + attn_out, x_mlp / x_moe vs CPU norms
//   tail       layer_tail_f32 per layer on the dumped dense_out / moe_out:
//              h vs layer_out, x vs attn_in[l + 1] (or the CPU final norm after layer 29)
//   head       output head with softcap on the last tail's x: logits vs logits.npy, a CPU
//              check on a sample of vocabulary rows, and argmax at every position
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include "test_util.h"

int run_test_tail(const std::string &packed_dir, const std::string &dump_dir) {
    apex::TestContext ctx(packed_dir);
    const apex::ModelConfig &c = ctx.info.config;
    const int64_t L = c.num_hidden_layers, H = c.hidden_size, V = c.vocab_size;
    const apex::Npy ids = apex::loadNpy(dump_dir + "/ids.npy");
    const int64_t T = ids.shape.at(0);
    const apex::Npy embed = apex::loadDump(dump_dir, "embed", {T, H});
    const apex::Npy attn_in = apex::loadDump(dump_dir, "attn_in", {L, T, H});
    const apex::Npy attn_out = apex::loadDump(dump_dir, "attn_out", {L, T, H});
    const apex::Npy dense_out = apex::loadDump(dump_dir, "dense_out", {L, T, H});
    const apex::Npy moe_out = apex::loadDump(dump_dir, "moe_out", {L, T, H});
    const apex::Npy layer_out = apex::loadDump(dump_dir, "layer_out", {L, T, H});
    const apex::Npy logits = apex::loadDump(dump_dir, "logits", {T, V});
    const size_t count = size_t(T * H);
    const size_t bytes = count * sizeof(float);
    const uint32_t tokens = uint32_t(T), hidden = uint32_t(H);
    constexpr double tolerance = 1e-5;
    int failures = 0;

    apex::Buffer *ids_buf = ctx.alloc(size_t(T) * sizeof(uint32_t));
    apex::Buffer *h_buf = ctx.alloc(bytes);
    apex::Buffer *x_buf = ctx.alloc(bytes);
    apex::Buffer *a_buf = ctx.alloc(bytes); // dense input / o_proj output
    apex::Buffer *b_buf = ctx.alloc(bytes); // moe input
    std::vector<float> ref(count), ref2(count);

    // --- embed ---
    auto *id32 = static_cast<uint32_t *>(ctx.backend.contents(ids_buf));
    for (int64_t t = 0; t < T; ++t) id32[t] = uint32_t(ids.i64()[t]);
    const apex::LayerWeights &w0 = ctx.weights.layers[0];
    ctx.ops.embedNorm(ids_buf, ctx.weights.embed, ctx.core_buffer, w0.input_norm, h_buf, x_buf,
                      tokens, ctx.eps);
    ctx.backend.sync();
    {
        const Diff dh = diff(ctx.f32(h_buf), embed.f32(), count);
        const Diff dx = diff(ctx.f32(x_buf), attn_in.f32(), count);
        const bool ok = dh.rel_l2 < tolerance && dx.rel_l2 < tolerance;
        failures += !ok;
        std::printf("embed      h vs embed rel %.2e max %.2e | x vs attn_in[0] rel %.2e  %s\n",
                    dh.rel_l2, dh.max_abs, dx.rel_l2, ok ? "ok" : "FAIL");
    }

    // Attention scratch (for the residual check), sized for the largest layer.
    const size_t max_hd = size_t(std::max(c.head_dim, c.global_head_dim));
    const size_t q_bytes = size_t(T) * c.num_attention_heads * max_hd * sizeof(float);
    const size_t kv_bytes = size_t(T) * c.num_key_value_heads * max_hd * sizeof(float);
    const apex::AttentionScratch tmp{ctx.alloc(q_bytes), ctx.alloc(kv_bytes), ctx.alloc(kv_bytes),
                                     ctx.alloc(q_bytes)};
    const apex::KvCache cache{ctx.alloc(kv_bytes, "kv_cache"), ctx.alloc(kv_bytes, "kv_cache"),
                              tokens};

    for (int layer = 0; layer < L; ++layer) {
        const apex::LayerWeights &w = ctx.weights.layers[layer];
        const float *layer_in = layer == 0 ? embed.f32() : layer_out.f32() + (layer - 1) * count;
        const size_t lo = size_t(layer) * count;

        // --- residual: h = layer_in + norm(o); x_mlp, x_moe ---
        std::memcpy(ctx.f32(x_buf), attn_in.f32() + lo, bytes);
        std::memcpy(ctx.f32(h_buf), layer_in, bytes);
        ctx.ops.attentionBlock(x_buf, w.attn, w.shape, cache, tmp, a_buf, tokens, 0, ctx.eps);
        ctx.ops.attnResidualNorms(a_buf, h_buf, ctx.core_buffer, w.post_attn_norm, w.pre_ffn_norm,
                                  w.pre_ffn_norm_2, x_buf, b_buf, tokens, hidden, ctx.eps);
        ctx.backend.sync();
        for (size_t i = 0; i < count; ++i) ref[i] = layer_in[i] + attn_out.f32()[lo + i];
        std::vector<float> mlp_ref(count), moe_ref(count);
        for (int64_t t = 0; t < T; ++t) {
            rmsNormCpu(ref.data() + t * H, ctx.cpu<uint16_t>(w.pre_ffn_norm), mlp_ref.data() + t * H,
                       H, ctx.eps);
            rmsNormCpu(ref.data() + t * H, ctx.cpu<uint16_t>(w.pre_ffn_norm_2),
                       moe_ref.data() + t * H, H, ctx.eps);
        }
        {
            const Diff dh = diff(ctx.f32(h_buf), ref.data(), count);
            const Diff dm = diff(ctx.f32(x_buf), mlp_ref.data(), count);
            const Diff de = diff(ctx.f32(b_buf), moe_ref.data(), count);
            const bool ok = dh.rel_l2 < tolerance && dm.rel_l2 < tolerance && de.rel_l2 < tolerance;
            failures += !ok;
            std::printf("layer %2d residual h rel %.2e max %.2e | x_mlp rel %.2e | x_moe rel %.2e  %s\n",
                        layer, dh.rel_l2, dh.max_abs, dm.rel_l2, de.rel_l2, ok ? "ok" : "FAIL");
        }

        // --- tail on the dumped dense_out / moe_out ---
        const bool last = layer == L - 1;
        const size_t next_norm = last ? ctx.weights.final_norm
                                      : ctx.weights.layers[layer + 1].input_norm;
        std::memcpy(ctx.f32(h_buf), ref.data(), bytes); // h = layer_in + attn_out
        std::memcpy(ctx.f32(a_buf), dense_out.f32() + lo, bytes);
        std::memcpy(ctx.f32(b_buf), moe_out.f32() + lo, bytes);
        ctx.ops.layerTail(a_buf, b_buf, h_buf, ctx.core_buffer, w.post_ffn_norm_1,
                          w.post_ffn_norm_2, w.post_ffn_norm, w.layer_scalar, next_norm, x_buf,
                          tokens, hidden, ctx.eps);
        ctx.backend.sync();
        const float *x_expected = nullptr;
        if (last) {
            for (int64_t t = 0; t < T; ++t)
                rmsNormCpu(layer_out.f32() + lo + t * H, ctx.cpu<uint16_t>(next_norm),
                           ref2.data() + t * H, H, ctx.eps);
            x_expected = ref2.data();
        } else {
            x_expected = attn_in.f32() + lo + count;
        }
        const Diff dh = diff(ctx.f32(h_buf), layer_out.f32() + lo, count);
        const Diff dx = diff(ctx.f32(x_buf), x_expected, count);
        const bool ok = dh.rel_l2 < tolerance && dx.rel_l2 < tolerance;
        failures += !ok;
        std::printf("layer %2d tail     h vs layer_out rel %.2e max %.2e | x vs %s rel %.2e  %s\n",
                    layer, dh.rel_l2, dh.max_abs, last ? "cpu final norm" : "next attn_in",
                    dx.rel_l2, ok ? "ok" : "FAIL");
    }

    // --- head: logits = softcap(embed_tokens . x), argmax ---
    // x_buf holds norm(h, final_norm) computed by the last tail from the dumped layer_out.
    apex::Buffer *logit_buf = ctx.alloc(size_t(T) * size_t(V) * sizeof(float));
    apex::Buffer *argmax_buf = ctx.alloc(size_t(T) * sizeof(uint32_t));
    const float softcap = float(c.final_logit_softcapping);
    ctx.ops.matmulNvfp4(x_buf, ctx.weights.embed, logit_buf, tokens, softcap);
    ctx.ops.argmax(logit_buf, argmax_buf, tokens, uint32_t(V));
    ctx.backend.sync();
    const float *gpu_logits = ctx.f32(logit_buf);
    const Diff dl = diff(gpu_logits, logits.f32(), size_t(T) * size_t(V));

    // CPU check on every 64th vocabulary row (the full head is 15 G multiply-adds).
    constexpr size_t SAMPLE = 64;
    const size_t sampled = size_t(V) / SAMPLE;
    std::vector<float> cpu_rows(sampled * size_t(T)), gpu_rows(sampled * size_t(T)), row(H);
    const apex::Nvfp4Matrix &E = ctx.weights.embed;
    const float *x = ctx.f32(x_buf);
    for (size_t s = 0; s < sampled; ++s) {
        const size_t r = s * SAMPLE;
        apex::decodeRowNvfp4(ctx.cpu<uint32_t>(E.weights_offset) + r * (H / 8),
                             ctx.cpu<uint8_t>(E.scales_offset) + r * (H / 16), row.data(), H);
        for (int64_t t = 0; t < T; ++t) {
            double acc = 0;
            for (int64_t i = 0; i < H; ++i) acc += double(row[i]) * x[t * H + i];
            cpu_rows[s * T + t] = float(std::tanh(acc / softcap) * softcap);
            gpu_rows[s * T + t] = gpu_logits[t * V + r];
        }
    }
    const Diff dc = diff(gpu_rows.data(), cpu_rows.data(), cpu_rows.size());

    const auto *am = static_cast<const uint32_t *>(ctx.backend.contents(argmax_buf));
    int argmax_match = 0;
    std::printf("argmax gpu/ref:");
    for (int64_t t = 0; t < T; ++t) {
        const float *rl = logits.f32() + t * V;
        const int64_t ref_id = std::max_element(rl, rl + V) - rl;
        argmax_match += int64_t(am[t]) == ref_id;
        std::printf(" %u/%lld", am[t], static_cast<long long>(ref_id));
    }
    std::printf("\n");
    const bool ok = dl.rel_l2 < tolerance && dc.rel_l2 < tolerance && argmax_match == T;
    failures += !ok;
    std::printf("head       logits rel %.2e max %.2e | gpu-vs-cpu (1/%zu rows) rel %.2e | "
                "argmax %d/%lld  %s\n",
                dl.rel_l2, dl.max_abs, SAMPLE, dc.rel_l2, argmax_match,
                static_cast<long long>(T), ok ? "ok" : "FAIL");

    apex::Profile::get().print(stdout);
    std::printf("test-tail: %d failures\n", failures);
    return failures ? 1 : 0;
}
