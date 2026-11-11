// Full forward pass in layer-streamed mode against the dumps.
//
// Runs the prompt through Model::forward in two calls: a prefill of the first `prefill` tokens
// (default T - 1) and then the remaining tokens one per call, so both the multi-token path and
// decode (pos0 > 0) go through the real engine. Every layer is fed by the engine's own
// previous layer, so this measures accumulated error, unlike the per-stage tests.
//
// Checked: h after every layer vs layer_out (rel L2 over all positions), routed expert ids per
// layer and position (exact, as sets), final logits (rel L2) and argmax at every position.
//
// Streamed mode wraps one layer file at a time (at most core.bin + one layer, ~1.8 GB).
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include "../backend/metal_backend.h"
#include "../model.h"
#include "../npy.h"
#include "../profiling/memory.h"
#include "cpu_ref.h"

int run_test_forward(const std::string &packed_dir, const std::string &dump_dir,
                     int prefill_arg, bool fp16_kv) {
    const apex::ModelInfo info = apex::ModelInfo::load(packed_dir);
    const apex::ModelConfig &c = info.config;
    const int64_t L = c.num_hidden_layers, H = c.hidden_size, V = c.vocab_size;
    const int64_t K = c.top_k_experts;
    const apex::Npy ids = apex::loadNpy(dump_dir + "/ids.npy");
    const int64_t T = ids.shape.at(0);
    const apex::Npy layer_out = apex::loadNpy(dump_dir + "/layer_out.npy");
    const apex::Npy logits = apex::loadNpy(dump_dir + "/logits.npy");
    const apex::Npy route_ids = apex::loadNpy(dump_dir + "/route_ids.npy");
    if (layer_out.shape != std::vector<int64_t>{L, T, H} ||
        logits.shape != std::vector<int64_t>{T, V} ||
        route_ids.shape != std::vector<int64_t>{L, T, K})
        throw std::runtime_error("test-forward: dump shapes do not match the model");
    const uint32_t prefill =
        prefill_arg > 0 ? uint32_t(std::min<int64_t>(prefill_arg, T)) : uint32_t(T - 1);

    apex::MetalBackend backend(APEX_METALLIB);
    apex::ModelOptions options;
    options.residency = apex::ExpertResidency::LayerStreamed;
    options.max_context = uint32_t(T);
    options.max_tokens = uint32_t(T);
    options.kv_dtype = fp16_kv ? apex::KvDtype::F16 : apex::KvDtype::F32;
    apex::Model model(backend, info, options);
    // fp32 KV: every stage is held to 1e-5 and the routing must match exactly. fp16 KV rounds
    // the cached keys/values (~3e-4 per attention call, see test-attn), which can flip
    // near-tied router picks, so ids are reported but not gated and the hidden-state gate is
    // loose; argmax must still match everywhere.
    const double layer_gate = fp16_kv ? 1e-2 : 1e-5;
    const double logit_gate = fp16_kv ? 1e-2 : 1e-4;

    // Captured per layer by the hook (streamed mode syncs after every layer).
    std::vector<float> gpu_h(size_t(L * T * H));
    std::vector<int64_t> gpu_ids(size_t(L * T * K));
    uint32_t t0 = 0; // first position of the current call
    model.layer_hook = [&](int layer, uint32_t n) {
        const float *h = static_cast<const float *>(backend.contents(model.finalHidden()));
        std::memcpy(gpu_h.data() + (size_t(layer) * T + t0) * H, h, size_t(n) * H * sizeof(float));
        const auto *r = static_cast<const ExpertRoute *>(backend.contents(model.routes()));
        for (uint32_t t = 0; t < n; ++t) {
            int64_t *dst = gpu_ids.data() + (size_t(layer) * T + t0 + t) * K;
            std::vector<float> unused(K);
            for (int64_t k = 0; k < K; ++k) dst[k] = r[t * K + k].slot;
            sortRoute(dst, unused.data(), size_t(K));
        }
    };

    std::vector<uint32_t> tokens(static_cast<size_t>(T));
    for (int64_t t = 0; t < T; ++t) tokens[t] = uint32_t(ids.i64()[t]);
    std::vector<float> gpu_logits(static_cast<size_t>(T * V));
    std::vector<uint32_t> gpu_argmax(static_cast<size_t>(T));
    auto run = [&](uint32_t start, uint32_t n) {
        t0 = start;
        const uint32_t *am = model.forward(tokens.data() + start, n, start, true);
        std::memcpy(gpu_argmax.data() + start, am, n * sizeof(uint32_t));
        std::memcpy(gpu_logits.data() + size_t(start) * V, backend.contents(model.logits()),
                    size_t(n) * V * sizeof(float));
    };
    run(0, prefill);
    for (uint32_t t = prefill; t < uint32_t(T); ++t) run(t, 1);

    int failures = 0;
    const size_t count = size_t(T * H);
    for (int64_t l = 0; l < L; ++l) {
        const Diff d = diff(gpu_h.data() + l * count, layer_out.f32() + l * count, count);
        int match = 0;
        for (int64_t t = 0; t < T; ++t) {
            const int64_t *g = gpu_ids.data() + (l * T + t) * K;
            match += std::equal(g, g + K, route_ids.i64() + (l * T + t) * K);
        }
        const bool ok = d.rel_l2 < layer_gate && (fp16_kv || match == T);
        failures += !ok;
        std::printf("layer %2lld layer_out rel %.2e max %.2e | expert ids %d/%lld positions  %s\n",
                    static_cast<long long>(l), d.rel_l2, d.max_abs, match,
                    static_cast<long long>(T), ok ? "ok" : "FAIL");
    }
    const Diff dl = diff(gpu_logits.data(), logits.f32(), size_t(T * V));
    int argmax_match = 0;
    std::printf("argmax engine/ref:");
    for (int64_t t = 0; t < T; ++t) {
        const float *rl = logits.f32() + t * V;
        const int64_t ref = std::max_element(rl, rl + V) - rl;
        argmax_match += int64_t(gpu_argmax[t]) == ref;
        std::printf(" %u/%lld", gpu_argmax[t], static_cast<long long>(ref));
    }
    std::printf("\n");
    const bool ok = argmax_match == T && dl.rel_l2 < logit_gate;
    failures += !ok;
    std::printf("logits rel %.2e max %.2e | argmax %d/%lld  %s\n", dl.rel_l2, dl.max_abs,
                argmax_match, static_cast<long long>(T), ok ? "ok" : "FAIL");
    apex::Profile::get().print(stdout);
    apex::MemoryTracker::get().print(stdout);
    std::printf("test-forward (streamed, %s KV, prefill %u + %lld decode): %d failures\n",
                fp16_kv ? "fp16" : "fp32", prefill, static_cast<long long>(T - prefill),
                failures);
    return failures ? 1 : 0;
}
