// Router and expert parity tests, per layer, against the dumps.
//
// test-router: GPU router on h = layer_in + attn_out (the raw post-attention hidden the
//              reference router reads). Expert ids must match the dump exactly (as sets,
//              sorted by id); weights are compared by rel L2. Core weights only.
// test-moe:    GPU router + pre_feedforward_layernorm_2 + both expert kernels on the same h,
//              compared with the moe_out dump and a CPU reference. The routed experts of
//              each layer are pread into a small slot buffer (slot != expert id), which
//              exercises the slot interface a disk cache will use and never wraps a whole
//              layer file.
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <map>
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

namespace {

/// Read `bytes` at `offset` of `path` into dst, or throw.
void preadAll(int fd, void *dst, size_t bytes, off_t offset) {
    char *p = static_cast<char *>(dst);
    while (bytes > 0) {
        const ssize_t n = pread(fd, p, bytes, offset);
        if (n <= 0) throw std::runtime_error("pread failed");
        p += n;
        bytes -= size_t(n);
        offset += n;
    }
}

/// One expert of one token on the CPU: y = down(gelu(gate x) * up x). slot points at the
/// expert's bytes, laid out like the packed slot.
void expertCpu(const char *slot, const ExpertLayout &l, const float *x, float *y, size_t hidden,
               size_t inter) {
    std::vector<float> g(inter), u(inter);
    matvecNvfp4Cpu(reinterpret_cast<const uint32_t *>(slot + l.gate_w),
                   reinterpret_cast<const uint8_t *>(slot + l.gate_s), x, g.data(), 1, inter,
                   hidden);
    matvecNvfp4Cpu(reinterpret_cast<const uint32_t *>(slot + l.up_w),
                   reinterpret_cast<const uint8_t *>(slot + l.up_s), x, u.data(), 1, inter,
                   hidden);
    for (size_t i = 0; i < inter; ++i) g[i] = geluTanh(g[i]) * u[i];
    matvecNvfp4Cpu(reinterpret_cast<const uint32_t *>(slot + l.down_w),
                   reinterpret_cast<const uint8_t *>(slot + l.down_s), g.data(), y, 1, hidden,
                   inter);
}

} // namespace

int run_test_moe(const std::string &packed_dir, const std::string &dump_dir) {
    apex::TestContext ctx(packed_dir);
    const apex::ModelConfig &c = ctx.info.config;
    const int64_t L = c.num_hidden_layers, H = c.hidden_size, K = c.top_k_experts;
    const size_t inter = size_t(c.moe_intermediate_size);
    const apex::Npy embed = apex::loadNpy(dump_dir + "/embed.npy");
    const int64_t T = embed.shape[0];
    const apex::Npy layer_out = apex::loadDump(dump_dir, "layer_out", {L, T, H});
    const apex::Npy attn_out = apex::loadDump(dump_dir, "attn_out", {L, T, H});
    const apex::Npy moe_out = apex::loadDump(dump_dir, "moe_out", {L, T, H});
    const RouteDump dump = loadRoutes(dump_dir, L, T, K);
    const size_t count = size_t(T * H);
    const ExpertLayout &layout = ctx.weights.expert_layout;

    apex::Buffer *h_buf = ctx.alloc(count * sizeof(float));
    apex::Buffer *x_buf = ctx.alloc(count * sizeof(float));
    apex::Buffer *routes = ctx.alloc(size_t(T * K) * sizeof(ExpertRoute));
    apex::Buffer *counts = ctx.alloc(size_t(T) * sizeof(uint32_t));
    apex::Buffer *act = ctx.alloc(size_t(T * K) * inter * sizeof(float));
    apex::Buffer *out_buf = ctx.alloc(count * sizeof(float));
    std::vector<float> h(count), x(count), cpu_out(count), y(H);

    int failures = 0;
    for (int layer = 0; layer < L; ++layer) {
        const apex::LayerWeights &w = ctx.weights.layers[layer];
        const apex::RouterWeights &r = w.router;
        postAttentionHidden(embed, layer_out, attn_out, layer, count, h.data());
        std::memcpy(ctx.f32(h_buf), h.data(), count * sizeof(float));

        // 1. Router and the experts' input norm on the GPU.
        ctx.ops.router(h_buf, ctx.core_buffer, r.scale, r.proj_weight, r.proj_scales,
                       r.proj_biases, r.per_expert_scale, routes, counts,
                       apex::routerParams(c, r.group_size), uint32_t(T));
        ctx.ops.rmsNorm(h_buf, ctx.core_buffer, w.pre_ffn_norm_2, x_buf, uint32_t(T),
                        uint32_t(H), ctx.eps);
        ctx.backend.sync();

        // 2. Load each routed expert once into a compact slot buffer and remap the routes.
        auto *route = static_cast<ExpertRoute *>(ctx.backend.contents(routes));
        std::map<uint32_t, uint32_t> slot_of; // expert id -> slot
        for (int64_t i = 0; i < T * K; ++i) slot_of.emplace(route[i].slot, 0);
        uint32_t next = 0;
        for (auto &[id, slot] : slot_of) slot = next++;
        const size_t slot_bytes = size_t(layout.stride);
        apex::Buffer *slots = ctx.backend.allocate(slot_of.size() * slot_bytes, "expert_slots");
        char *slot_cpu = static_cast<char *>(ctx.backend.contents(slots));
        const int fd = open(ctx.info.layer_file(layer).c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open " + ctx.info.layer_file(layer));
        for (auto &[id, slot] : slot_of)
            preadAll(fd, slot_cpu + slot * slot_bytes, slot_bytes, off_t(id) * off_t(slot_bytes));
        close(fd);
        for (int64_t i = 0; i < T * K; ++i) route[i].slot = slot_of[route[i].slot];

        // 3. Experts on the GPU.
        ctx.ops.experts(x_buf, slots, routes, counts, act, out_buf,
                        apex::expertParams(c, layout, uint32_t(T)));
        ctx.backend.sync();

        // CPU reference with the DUMP's routing (independent of the GPU router).
        const uint16_t *norm_w = ctx.cpu<uint16_t>(w.pre_ffn_norm_2);
        for (int64_t t = 0; t < T; ++t) {
            rmsNormCpu(h.data() + t * H, norm_w, x.data() + t * H, H, ctx.eps);
            std::vector<double> acc(H, 0.0);
            for (int64_t k = 0; k < K; ++k) {
                const int64_t id = dump.ids.i64()[(layer * T + t) * K + k];
                const float wk = dump.weights.f32()[(layer * T + t) * K + k];
                const auto it = slot_of.find(uint32_t(id));
                if (it == slot_of.end()) throw std::runtime_error("dump expert not loaded");
                expertCpu(slot_cpu + it->second * slot_bytes, layout, x.data() + t * H, y.data(),
                          H, inter);
                for (int64_t i = 0; i < H; ++i) acc[i] += double(wk) * y[i];
            }
            for (int64_t i = 0; i < H; ++i) cpu_out[t * H + i] = float(acc[i]);
        }
        ctx.backend.release(slots);

        const float *expected = moe_out.f32() + size_t(layer) * count;
        const float *gpu = ctx.f32(out_buf);
        char label[32];
        std::snprintf(label, sizeof label, "%zu slots", slot_of.size());
        failures += !apex::report(label, layer, diff(gpu, expected, count),
                                  diff(cpu_out.data(), expected, count),
                                  diff(gpu, cpu_out.data(), count), 1e-5);
    }
    apex::Profile::get().print(stdout);
    std::printf("%lld layers tested, %d failures\n", static_cast<long long>(L), failures);
    return failures ? 1 : 0;
}
