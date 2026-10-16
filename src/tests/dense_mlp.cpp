// Dense MLP parity test: GPU norm + MLP vs the CPU reference and the Python dump, per layer.
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include "../backend/metal_backend.h"
#include "../file.h"
#include "../model_info.h"
#include "../npy.h"
#include "../ops.h"
#include "cpu_ref.h"

// Helpers, defined below run_test_mlp.
static apex::Nvfp4Matrix quantMatrix(const apex::ModelInfo &model_info, apex::Buffer *core_buffer,
                                     const std::string &name);
static void denseMlpCpu(const apex::ModelInfo &model_info, const apex::MappedFile &core,
                        const std::string &prefix, const float *h, float *output, size_t tokens);

// Compare each layer's dense MLP output against the CPU reference and the Python dump.
int run_test_mlp(const std::string &packed_dir, const std::string &dump_dir) {
    apex::ModelInfo model_info = apex::ModelInfo::load(packed_dir);
    apex::MetalBackend backend(APEX_METALLIB);

    // Map core.bin - core weights (CPU), then let the GPU read the same bytes without a copy.
    apex::MappedFile core = apex::mapFile(model_info.dir + "/core.bin", "core_weights");
    if (core.size != model_info.core_size) {
        throw std::runtime_error("core.bin size != manifest");
    }
    apex::Buffer *core_buffer = backend.wrapNoCopy(core.data, core.size);

    // Python reference activations.
    apex::Npy embed = apex::loadNpy(dump_dir + "/embed.npy");         // (T, H)
    apex::Npy attn_out = apex::loadNpy(dump_dir + "/attn_out.npy");   // (L, T, H)
    apex::Npy layer_out = apex::loadNpy(dump_dir + "/layer_out.npy"); // (L, T, H)
    apex::Npy dense_out = apex::loadNpy(dump_dir + "/dense_out.npy"); // (L, T, H)

    if (embed.dtype != "<f4" || attn_out.dtype != "<f4" || layer_out.dtype != "<f4" ||
        dense_out.dtype != "<f4" || embed.shape.size() != 2) {
        throw std::runtime_error("test-mlp: expected float32 dumps and 2D embed");
    }

    // model metadata. 
    const int64_t layer_count = model_info.config.num_hidden_layers;
    const int64_t hidden_dim = model_info.config.hidden_size;             // 2816
    const int64_t intermediate_dim = model_info.config.intermediate_size; // 2112
    const int64_t token_count = embed.shape[0];

    // Every per-layer dump must be (layers, tokens, hidden).
    const std::vector<int64_t> layer_shape{layer_count, token_count, hidden_dim};
    if (embed.shape[1] != hidden_dim || attn_out.shape != layer_shape ||
        layer_out.shape != layer_shape || dense_out.shape != layer_shape) {
        throw std::runtime_error("test-mlp: dump shapes do not match model");
    }

    // Element counts for one layer's activations (tok * hidden_dim)
    const size_t hidden_count = static_cast<size_t>(token_count) * static_cast<size_t>(hidden_dim);
    const size_t intermediate_count =
        static_cast<size_t>(token_count) * static_cast<size_t>(intermediate_dim);
    const float eps = static_cast<float>(model_info.config.rms_norm_eps);

    // Allocate activation buffers for intermediate values. 
    apex::Buffer *input = backend.allocate(hidden_count * sizeof(float), "activations");
    apex::Buffer *norm_output = backend.allocate(hidden_count * sizeof(float), "activations");
    apex::Buffer *gate_buffer = backend.allocate(intermediate_count * sizeof(float), "activations");
    apex::Buffer *up_buffer = backend.allocate(intermediate_count * sizeof(float), "activations");
    apex::Buffer *output_buffer = backend.allocate(hidden_count * sizeof(float), "activations");

    // CPU side: the MLP input (built per layer) and the CPU reference answer.
    std::vector<float> h(hidden_count);
    std::vector<float> cpu_output(hidden_count);

    int failures = 0;
    for (int64_t layer = 0; layer < layer_count; ++layer) {
        // h = residual + attn_out. The residual is this layer's input.
        const float *residual =
            layer == 0 ? embed.f32()
                       : layer_out.f32() + static_cast<size_t>(layer - 1) * hidden_count;
        const float *attention = attn_out.f32() + static_cast<size_t>(layer) * hidden_count;
        for (size_t i = 0; i < hidden_count; ++i) {
            h[i] = residual[i] + attention[i];
        }
        const float *expected = dense_out.f32() + static_cast<size_t>(layer) * hidden_count;

        // This layer's weights: one BF16 norm vector and three NVFP4 matrices.
        const std::string prefix = "layers." + std::to_string(layer) + ".";
        const apex::TensorInfo &norm =
            model_info.getTensor(prefix + "pre_feedforward_layernorm.weight");
        const apex::Nvfp4Matrix gate_weights =
            quantMatrix(model_info, core_buffer, prefix + "mlp.gate_proj");
        const apex::Nvfp4Matrix up_weights =
            quantMatrix(model_info, core_buffer, prefix + "mlp.up_proj");
        const apex::Nvfp4Matrix down_weights =
            quantMatrix(model_info, core_buffer, prefix + "mlp.down_proj");

        // GPU: norm, then the MLP. Kernels run in queue order; sync waits for all of them.
        std::memcpy(backend.contents(input), h.data(), hidden_count * sizeof(float));
        apex::rmsNorm(backend, input, core_buffer, norm.offset, norm_output,
                      static_cast<uint32_t>(token_count), static_cast<uint32_t>(hidden_dim), eps);
        apex::denseMlp(backend, norm_output, gate_weights, up_weights, down_weights, gate_buffer,
                       up_buffer, output_buffer, static_cast<uint32_t>(token_count));
        backend.sync();

        // CPU reference from the same h, independent of the GPU path.
        denseMlpCpu(model_info, core, prefix, h.data(), cpu_output.data(),
                    static_cast<size_t>(token_count));

        // Compare GPU and CPU against the dump, and against each other.
        const float *gpu = static_cast<const float *>(backend.contents(output_buffer));
        const Diff gpu_reference = diff(gpu, expected, hidden_count);
        const Diff cpu_reference = diff(cpu_output.data(), expected, hidden_count);
        const Diff gpu_cpu = diff(gpu, cpu_output.data(), hidden_count);
        constexpr double tolerance = 1e-5;
        const bool ok = gpu_reference.rel_l2 < tolerance && cpu_reference.rel_l2 < tolerance &&
                        gpu_cpu.rel_l2 < tolerance;
        failures += !ok;
        std::printf("layer %2lld  gpu-vs-ref rel %.2e max %.2e | "
                    "cpu-vs-ref rel %.2e | gpu-vs-cpu rel %.2e  %s\n",
                    static_cast<long long>(layer), gpu_reference.rel_l2, gpu_reference.max_abs,
                    cpu_reference.rel_l2, gpu_cpu.rel_l2, ok ? "ok" : "FAIL");
    }

    // Free activations, then let the GPU go of core.bin before unmapping it.
    for (apex::Buffer *buffer : {input, norm_output, gate_buffer, up_buffer, output_buffer}) {
        backend.release(buffer);
    }
    backend.release(core_buffer);
    apex::unmapFile(core);

    apex::Profile::get().print(stdout);
    std::printf("%lld layers tested, %d failures\n", static_cast<long long>(layer_count), failures);
    return failures ? 1 : 0;
}

/// Builds the GPU description of an NVFP4 matrix from its .weight and .scales tensors in core.bin.
static apex::Nvfp4Matrix quantMatrix(const apex::ModelInfo &model_info, apex::Buffer *core_buffer,
                                     const std::string &name) {
    const apex::TensorInfo &weights = model_info.getTensor(name + ".weight"); // uint32 [rows, K/8]
    const apex::TensorInfo &scales = model_info.getTensor(name + ".scales");  // uint8  [rows, K/16]
    return apex::Nvfp4Matrix{core_buffer,
                             weights.offset,
                             core_buffer,
                             scales.offset,
                             static_cast<uint32_t>(weights.shape[0]),
                             static_cast<uint32_t>(weights.shape[1] * 8)}; // 8 fp4 packed into one int.
}

/// One layer's dense MLP on the CPU: h → norm → gate/up → GELU(gate) * up → down → output.
static void denseMlpCpu(const apex::ModelInfo &model_info, const apex::MappedFile &core,
                        const std::string &prefix, const float *h, float *output, size_t tokens) {
    const size_t hidden_dim = static_cast<size_t>(model_info.config.hidden_size);
    const size_t intermediate_dim = static_cast<size_t>(model_info.config.intermediate_size);
    const float eps = static_cast<float>(model_info.config.rms_norm_eps);
    const char *base = static_cast<const char *>(core.data);

    // CPU address of a tensor inside the mapped core.bin.
    auto tensor = [&](const std::string &name) {
        return base + model_info.getTensor(prefix + name).offset;
    };

    std::vector<float> x(tokens * hidden_dim);
    std::vector<float> gate(tokens * intermediate_dim);
    std::vector<float> up(tokens * intermediate_dim);

    const uint16_t *norm = reinterpret_cast<const uint16_t *>(tensor("pre_feedforward_layernorm.weight"));
    for (size_t t = 0; t < tokens; ++t) {
        rmsNormCpu(h + t * hidden_dim, norm, x.data() + t * hidden_dim, hidden_dim, eps);
    }

    matvecNvfp4Cpu(reinterpret_cast<const uint32_t *>(tensor("mlp.gate_proj.weight")),
                   reinterpret_cast<const uint8_t *>(tensor("mlp.gate_proj.scales")), x.data(),
                   gate.data(), tokens, intermediate_dim, hidden_dim);
    matvecNvfp4Cpu(reinterpret_cast<const uint32_t *>(tensor("mlp.up_proj.weight")),
                   reinterpret_cast<const uint8_t *>(tensor("mlp.up_proj.scales")), x.data(),
                   up.data(), tokens, intermediate_dim, hidden_dim);
    for (size_t i = 0; i < gate.size(); ++i) {
        gate[i] = geluTanh(gate[i]) * up[i];
    }
    matvecNvfp4Cpu(reinterpret_cast<const uint32_t *>(tensor("mlp.down_proj.weight")),
                   reinterpret_cast<const uint8_t *>(tensor("mlp.down_proj.scales")), gate.data(),
                   output, tokens, hidden_dim, intermediate_dim);
}