/// @brief Defines what the engine can ask the GPU to do
/// Implemented by MetalBackend.cpp
// This file is the wrapper around the GPU backend so an Intel GPU backend can be added later.
// Nothing in here may mention Metal: kernels are data (a name resolved once into a handle),
// arguments are buffers or small values, and work is queued then synced.

#pragma once
#include <cstddef>
#include <cstdint>
#include <initializer_list>

namespace apex {
struct Buffer; /// GPU buffer, defined by each backend.
struct Kernel; /// A compiled GPU program, defined by each backend. Resolve once, dispatch often.

/// @brief Describes one input into a GPU kernel.
struct Arg {
    Buffer *buffer = nullptr; // GPU buffer
    size_t offset = 0;
    const void *bytes = nullptr; // address of cpu value to copy
    size_t size = 0;             // how many bytes to copy.

    // Each arg uses either buffer with offset or val
    static Arg buf(Buffer *b, size_t offset = 0) { return {b, offset, nullptr, 0}; }
    template <class T> static Arg val(const T &v) { return {nullptr, 0, &v, sizeof(T)}; }
};

/// Grid or threadgroup size. Construct from one number for 1D work
/// (Dim3{rows}), or three for tiled kernels (Dim3{tiles_x, tiles_y, 1}).
struct Dim3 {
    uint32_t x = 1, y = 1, z = 1;
    Dim3(uint32_t x_, uint32_t y_ = 1, uint32_t z_ = 1) : x(x_), y(y_), z(z_) {}
    uint32_t total() const { return x * y * z; }
};

class Backend {
  public:
    virtual ~Backend() = default;

    // Lets the gpu read memory (for core.bin and the expert layer files)
    virtual Buffer *wrapNoCopy(void *cpu_ptr, size_t bytes) = 0;

    /// Create fresh memory that can be seen by both CPU and GPU
    virtual Buffer *allocate(size_t bytes, const char *category) = 0;

    /// Returns CPU pointer to buffer's bytes
    virtual void *contents(Buffer *buffer) = 0;

    // free a buffer and update memory tracker
    virtual void release(Buffer *buffer) = 0;

    /// Look up a kernel by name and build whatever the backend needs to run it.
    /// Do this once at load time; the returned handle stays valid for the backend's
    /// lifetime and makes run() free of string lookups.
    virtual Kernel *kernel(const char *name) = 0;

    // Queue one kernel launch. threadgroups is the grid in threadgroups,
    // threads_per_group the threadgroup shape; both may be 1D, 2D or 3D.
    // Launches run in queue order, and each one sees the writes of the ones before it.
    virtual void run(Kernel *kernel, std::initializer_list<Arg> args, Dim3 threadgroups,
                     Dim3 threads_per_group) = 0;

    // Convenience for tests and one-off launches: resolves the name every call.
    void run(const char *name, std::initializer_list<Arg> args, Dim3 threadgroups,
             Dim3 threads_per_group) {
        run(kernel(name), args, threadgroups, threads_per_group);
    }

    // submit queued kernels and wait for completion.
    virtual void sync() = 0;
};
} // namespace apex
