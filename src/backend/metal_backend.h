#pragma once

#include <string>
#include <Metal/Metal.hpp>
#include <map>
#include "backend.h"

namespace apex {

/// @brief Metal buffer
struct Buffer {
    MTL::Buffer *mtl = nullptr; // Metal buffer object.
    std::size_t length = 0;     // length
    std::string category;       // tracker category
};

/// @brief A Metal compute pipeline plus what run() checks before every dispatch.
struct Kernel {
    MTL::ComputePipelineState *pso = nullptr;
    std::string name;            // for error messages only
    uint32_t max_threads = 0;    // maxTotalThreadsPerThreadgroup, read once
};

class MetalBackend : public Backend {
  public:
    explicit MetalBackend(const std::string &metallib_path);

    ~MetalBackend() override;

    Buffer *wrapNoCopy(void *cpu_ptr, std::size_t bytes) override;
    Buffer *allocate(std::size_t bytes, const char *category) override;
    void *contents(Buffer *buffer) override;
    void release(Buffer *buffer) override;
    /// Returns the GPU's name for benchmark output.
    const char *deviceName() const;

    // High level functions. ----
    // Build (or fetch the cached) pipeline for a kernel name.
    Kernel *kernel(const char *name) override;

    // Queue a Metal kernel launch with ordered arguments.
    void run(Kernel *kernel, std::initializer_list<Arg> args, Dim3 threadgroups,
             Dim3 threads_per_group) override;
    using Backend::run; // keep the by-name overload visible on MetalBackend objects

    // Submit queued kernels and wait for completion. Release when done.
    void sync() override;

    // Prevent copying ownership of Metal objects.
    MetalBackend(const MetalBackend &) = delete;
    MetalBackend &operator=(const MetalBackend &) = delete;

  private:
    MTL::Device *device_ = nullptr;      // our device
    MTL::CommandQueue *queue_ = nullptr; // command queue for GPU work
    MTL::Library *library_ = nullptr;    // shaders from metallib
    NS::AutoreleasePool *pool_ = nullptr;
    MTL::CommandBuffer *cmd_ = nullptr;
    MTL::ComputeCommandEncoder *enc_ = nullptr;
    Kernel *bound_ = nullptr; // pipeline currently set on enc_, to skip redundant sets

    std::map<std::string, Kernel *> kernels_; // cached pipelines, owned
};

} // namespace apex
