// Router and expert parity tests, per layer, against the dumps.
//
// test-router: GPU router on h = layer_in + attn_out (the raw post-attention hidden the
//              reference router reads). Expert ids must match the dump exactly (as sets,
//              sorted by id); weights are compared by rel L2. Core weights only.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include "test_util.h"

namespace {

/// h = residual + attn_out for one layer, from the dumps: what the router and the experts read.
void postAttentionHidden(const apex::Npy &embed, const apex::Npy &layer_out,
                         const apex::Npy &attn_out, int layer, size_t count, float *h) {
    const float *residual =
        layer == 0 ? embed.f32() : layer_out.f32() + size_t(layer - 1) * count;
    const float *attention = attn_out.f32() + size_t(layer) * count;
    for (size_t i = 0; i < count; ++i) h[i] = residual[i] + attention[i];
}

/// Dump route ids/weights as (L, T, K) arrays.
struct RouteDump {
    apex::Npy ids, weights;
};
RouteDump loadRoutes(const std::string &dump_dir, int64_t L, int64_t T, int64_t K) {
    RouteDump r{apex::loadNpy(dump_dir + "/route_ids.npy"),
                apex::loadNpy(dump_dir + "/route_weights.npy")};
    const std::vector<int64_t> shape{L, T, K};
    if (r.ids.dtype != "<i8" || r.ids.shape != shape || r.weights.dtype != "<f4" ||
        r.weights.shape != shape)
        throw std::runtime_error("route dumps: unexpected dtype or shape");
    return r;
}

} // namespace

int run_test_router(const std::string &packed_dir, const std::string &dump_dir) {
    apex::TestContext ctx(packed_dir);
    const apex::ModelConfig &c = ctx.info.config;
    const int64_t L = c.num_hidden_layers, H = c.hidden_size, K = c.top_k_experts;
    const apex::Npy embed = apex::loadNpy(dump_dir + "/embed.npy");
    const int64_t T = embed.shape[0];
    const apex::Npy layer_out = apex::loadDump(dump_dir, "layer_out", {L, T, H});
    const apex::Npy attn_out = apex::loadDump(dump_dir, "attn_out", {L, T, H});
    const RouteDump dump = loadRoutes(dump_dir, L, T, K);
    const size_t count = size_t(T * H);

    apex::Buffer *h_buf = ctx.alloc(count * sizeof(float));
    apex::Buffer *routes = ctx.alloc(size_t(T * K) * sizeof(ExpertRoute));
    apex::Buffer *counts = ctx.alloc(size_t(T) * sizeof(uint32_t));
    std::vector<float> h(count);

    int failures = 0;
    double worst_margin = INFINITY;
    for (int layer = 0; layer < L; ++layer) {
        const apex::LayerWeights &w = ctx.weights.layers[layer];
        const apex::RouterWeights &r = w.router;
        postAttentionHidden(embed, layer_out, attn_out, layer, count, h.data());
        std::memcpy(ctx.f32(h_buf), h.data(), count * sizeof(float));
        const RouterParams p = apex::routerParams(c, r.group_size);
        ctx.ops.router(h_buf, ctx.core_buffer, r.scale, r.proj_weight, r.proj_scales,
                       r.proj_biases, r.per_expert_scale, routes, counts, p, uint32_t(T));
        ctx.backend.sync();

        const auto *gpu_routes = static_cast<const ExpertRoute *>(ctx.backend.contents(routes));
        const auto *gpu_counts = static_cast<const uint32_t *>(ctx.backend.contents(counts));
        std::vector<float> gpu_w(T * K), cpu_w(T * K);
        int gpu_mismatch = 0, cpu_mismatch = 0;
        double layer_margin = INFINITY;
        for (int64_t t = 0; t < T; ++t) {
            const int64_t *ref_ids = dump.ids.i64() + (layer * T + t) * K;
            std::vector<int64_t> gi(K), ci(K);
            for (int64_t k = 0; k < K; ++k) {
                gi[k] = gpu_routes[t * K + k].slot;
                gpu_w[t * K + k] = gpu_routes[t * K + k].weight;
            }
            sortRoute(gi.data(), gpu_w.data() + t * K, size_t(K));
            double margin = 0;
            routerCpu(h.data() + t * H, ctx.cpu<uint16_t>(r.scale),
                      ctx.cpu<uint32_t>(r.proj_weight), ctx.cpu<uint16_t>(r.proj_scales),
                      ctx.cpu<uint16_t>(r.proj_biases), ctx.cpu<uint16_t>(r.per_expert_scale),
                      size_t(H), size_t(c.num_experts), r.group_size, size_t(K), ctx.eps,
                      p.root_scale, ci.data(), cpu_w.data() + t * K, &margin);
            sortRoute(ci.data(), cpu_w.data() + t * K, size_t(K));
            layer_margin = std::min(layer_margin, margin);
            gpu_mismatch += gpu_counts[t] != uint32_t(K) ||
                            !std::equal(gi.begin(), gi.end(), ref_ids);
            cpu_mismatch += !std::equal(ci.begin(), ci.end(), ref_ids);
        }
        worst_margin = std::min(worst_margin, layer_margin);
        const float *ref_w = dump.weights.f32() + size_t(layer * T * K);
        const size_t n = size_t(T * K);
        const Diff gr = diff(gpu_w.data(), ref_w, n), cr = diff(cpu_w.data(), ref_w, n);
        const Diff gc = diff(gpu_w.data(), cpu_w.data(), n);
        const bool ok = gpu_mismatch == 0 && cpu_mismatch == 0 && gr.rel_l2 < 1e-5 &&
                        cr.rel_l2 < 1e-5 && gc.rel_l2 < 1e-5;
        failures += !ok;
        std::printf("layer %2d ids gpu %lld/%lld cpu %lld/%lld match | weights gpu-vs-ref rel "
                    "%.2e cpu-vs-ref %.2e gpu-vs-cpu %.2e | min top-k margin %.2e  %s\n",
                    layer, static_cast<long long>(T - gpu_mismatch), static_cast<long long>(T),
                    static_cast<long long>(T - cpu_mismatch), static_cast<long long>(T),
                    gr.rel_l2, cr.rel_l2, gc.rel_l2, layer_margin, ok ? "ok" : "FAIL");
    }
    std::printf("%lld layers tested, %d failures (smallest 8th-vs-9th score gap %.2e)\n",
                static_cast<long long>(L), failures, worst_margin);
    return failures ? 1 : 0;
}
