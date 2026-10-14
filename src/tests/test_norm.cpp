
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <cstdio>
#include <limits>
#include "../backend/metal_backend.h"
#include "../file.h"
#include "../model_info.h"
#include "../npy.h"
#include "../ops.h"

// Normalize one token vector on the CPU using BF16 weights.
static void rmsNormCpu(const float *x, const uint16_t *w_bf16, float *y, size_t n, float eps) {
    double ss = 0;
    for (size_t i = 0; i < n; ++i)
        ss += double(x[i]) * x[i];
    float inv = 1.0f / std::sqrt(float(ss / n) + eps);
    for (size_t i = 0; i < n; ++i) {
        uint32_t bits = uint32_t(w_bf16[i]) << 16;
        float wf;
        std::memcpy(&wf, &bits, 4);
        y[i] = x[i] * inv * wf;
    }
}

// max |a-b| and relative L2 ||a-b|| / ||b||
struct Diff {
    double max_abs = 0, rel_l2 = 0;
};
static Diff diff(const float *a, const float *b, size_t n) {
    double num = 0, den = 0;
    Diff d;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            return {std::numeric_limits<double>::infinity(),
                    std::numeric_limits<double>::infinity()};
        }
        double e = double(a[i]) - b[i];
        num += e * e;
        den += double(b[i]) * b[i];
        d.max_abs = std::max(d.max_abs, std::fabs(e));
    }
    d.rel_l2 = den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
    return d;
}

// Compare each layer's RMSNorm output against CPU and Python references.
int run_test_norm(const std::string &packed_dir, const std::string &dump_dir) {
    apex::ModelInfo model_info = apex::ModelInfo::load(packed_dir);
    apex::MetalBackend backend(APEX_METALLIB);

    // Map core.bin (CPU), then let the GPU read the same bytes without a copy.
    apex::MappedFile core = apex::mapFile(model_info.dir + "/core.bin", "core_weights");
    if (core.size != model_info.core_size) {
        throw std::runtime_error("core.bin size != manifest");
    }
    apex::Buffer *core_buffer = backend.wrapNoCopy(core.data, core.size);

    apex::Npy embed = apex::loadNpy(dump_dir + "/embed.npy");         // (P, H)
    apex::Npy layer_out = apex::loadNpy(dump_dir + "/layer_out.npy"); // (L, P, H)
    apex::Npy attn_in = apex::loadNpy(dump_dir + "/attn_in.npy");     // (L, P, H)

    const int64_t layer_count = model_info.config.num_hidden_layers;
    const int64_t hidden_dim = model_info.config.hidden_size;

    if (embed.dtype != "<f4" || layer_out.dtype != "<f4" || attn_in.dtype != "<f4" ||
        embed.shape.size() != 2) {
        throw std::runtime_error("test-norm: expected float32 dumps and 2D embed");
    }

    const int64_t token_count = embed.shape[0];
    if (token_count <= 0 || embed.shape[1] != hidden_dim ||
        layer_out.shape != std::vector<int64_t>{layer_count, token_count, hidden_dim} ||
        attn_in.shape != layer_out.shape) {
        throw std::runtime_error("test-norm: dump shapes do not match model");
    }

    // allocate buffesr
    const size_t value_count = static_cast<size_t>(token_count) * static_cast<size_t>(hidden_dim);
    const size_t buffer_bytes = value_count * sizeof(float);
    const float eps = static_cast<float>(model_info.config.rms_norm_eps);

    apex::Buffer *input = backend.allocate(buffer_bytes, "activations");
    apex::Buffer *result = backend.allocate(buffer_bytes, "activations");
    std::vector<float> cpu(value_count);

    // layer loop

    int failures = 0;
    for (int64_t layer = 0; layer < layer_count; ++layer) {
        // layer 0 gets input and then rest get the ouput of prev layers. 
        const float *saved_input =
            layer == 0 ? embed.f32()
                       : layer_out.f32() + static_cast<size_t>(layer - 1) * value_count;

        const float *expected = attn_in.f32() + static_cast<size_t>(layer) * value_count;

        std::memcpy(backend.contents(input), saved_input, buffer_bytes);

        // Kernel launch
        const std::string weight_name =
            "layers." + std::to_string(layer) + ".input_layernorm.weight";
        const apex::TensorInfo &weights = model_info.getTensor(weight_name);

        apex::rmsNorm(backend, input, core_buffer, weights.offset, result,
                      static_cast<uint32_t>(token_count), static_cast<uint32_t>(hidden_dim), eps);
        backend.sync();

        //calculate cpu answer
        const uint16_t *cpu_weights = reinterpret_cast<const uint16_t *>(
            static_cast<const char *>(core.data) + weights.offset);

        const size_t width = static_cast<size_t>(hidden_dim);
        for (int64_t token = 0; token < token_count; ++token) {
            const size_t offset = static_cast<size_t>(token) * width;
            rmsNormCpu(saved_input + offset, cpu_weights, cpu.data() + offset, width, eps);
        }

        const float *gpu = static_cast<const float *>(backend.contents(result));
        const Diff gpu_reference = diff(gpu, expected, value_count);
        const Diff cpu_reference = diff(cpu.data(), expected, value_count);
        const Diff gpu_cpu = diff(gpu, cpu.data(), value_count);
        constexpr double tolerance = 1e-5;
        const bool ok = gpu_reference.rel_l2 < tolerance &&
                        cpu_reference.rel_l2 < tolerance && gpu_cpu.rel_l2 < tolerance;
        failures += !ok;
        std::printf("layer %2lld  gpu-vs-ref rel %.2e max %.2e | "
                    "cpu-vs-ref rel %.2e | gpu-vs-cpu rel %.2e  %s\n",
                    static_cast<long long>(layer), gpu_reference.rel_l2,
                    gpu_reference.max_abs, cpu_reference.rel_l2, gpu_cpu.rel_l2,
                    ok ? "ok" : "FAIL");
    }
    backend.release(input);
    backend.release(result);

    // GPU lets go of core.bin before it is unmapped.
    backend.release(core_buffer);
    apex::unmapFile(core);
    apex::Profile::get().print(stdout);
    std::printf("%lld layers tested, %d failures\n",
                static_cast<long long>(layer_count), failures);
    return failures ? 1 : 0;
}
