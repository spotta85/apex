// Shared setup for the parity tests: backend, mapped core.bin, resolved weights, dump checks.
#pragma once
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
#include "../weights.h"
#include "cpu_ref.h"

namespace apex {

/// Owns the backend, the no-copy core.bin buffer and the resolved weights for one test run.
/// Core is the only weight file mapped; tests that need experts map one layer at a time.
struct TestContext {
    ModelInfo info;
    MetalBackend backend;
    Ops ops;
    MappedFile core;
    Buffer *core_buffer = nullptr;
    ModelWeights weights;
    float eps;

    explicit TestContext(const std::string &packed_dir)
        : info(ModelInfo::load(packed_dir)), backend(APEX_METALLIB), ops(backend) {
        core = mapFile(info.dir + "/core.bin", "core_weights");
        if (core.size != info.core_size) throw std::runtime_error("core.bin size != manifest");
        core_buffer = backend.wrapNoCopy(core.data, core.size);
        weights = resolveWeights(info, core_buffer);
        eps = float(info.config.rms_norm_eps);
    }
    ~TestContext() {
        for (Buffer *b : owned) backend.release(b);
        backend.release(core_buffer); // GPU lets go of core.bin before it is unmapped
        unmapFile(core);
    }
    TestContext(const TestContext &) = delete;
    TestContext &operator=(const TestContext &) = delete;

    /// CPU address of a core tensor (for the CPU reference only).
    template <class T = char> const T *cpu(size_t offset) const {
        return reinterpret_cast<const T *>(static_cast<const char *>(core.data) + offset);
    }

    /// Activation buffer released with the context.
    Buffer *alloc(size_t bytes, const char *category = "activations") {
        Buffer *b = backend.allocate(bytes, category);
        owned.push_back(b);
        return b;
    }
    float *f32(Buffer *b) { return static_cast<float *>(backend.contents(b)); }

  private:
    std::vector<Buffer *> owned;
};

/// Load a float32 dump with an expected shape, or throw.
inline Npy loadDump(const std::string &dump_dir, const std::string &name,
                    const std::vector<int64_t> &shape) {
    Npy n = loadNpy(dump_dir + "/" + name + ".npy");
    if (n.dtype != "<f4" || n.shape != shape) {
        throw std::runtime_error("dump " + name + ": unexpected dtype or shape");
    }
    return n;
}

/// Prints one comparison line and returns whether it passed.
inline bool report(const char *label, int layer, const Diff &gpu_ref, const Diff &cpu_ref,
                   const Diff &gpu_cpu, double tolerance) {
    const bool ok = gpu_ref.rel_l2 < tolerance && cpu_ref.rel_l2 < tolerance &&
                    gpu_cpu.rel_l2 < tolerance;
    std::printf("layer %2d %-8s gpu-vs-ref rel %.2e max %.2e | cpu-vs-ref rel %.2e | "
                "gpu-vs-cpu rel %.2e  %s\n",
                layer, label, gpu_ref.rel_l2, gpu_ref.max_abs, cpu_ref.rel_l2, gpu_cpu.rel_l2,
                ok ? "ok" : "FAIL");
    return ok;
}

} // namespace apex
