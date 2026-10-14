/// @brief Defines what the engine can ask the GPU to do 
/// Implemented by MetalBackend.cpp
// We this file serves as a wrapper for the GPU backend so we can also supprot intel gpu going forward. 


#pragma once
#include <cstddef>
#include <cstdint>
#include <initializer_list>

namespace apex {
    struct Buffer; /// GPU buffer 

    /// @brief Describes one input into a GPU kernel.
    struct Arg {
        Buffer *buffer = nullptr; // GPU buffer
        size_t offset = 0;
        const void *bytes = nullptr; // adress of cpu value to copy 
        size_t size = 0; // how many bytes to copy. 
        
        // Each arg uses either buffer with offset or val
        static Arg buf(Buffer * b, size_t offset = 0) {
            return {b, offset, nullptr, 0};
        }
        template <class T>
        static Arg val(const T& v) {
            return {nullptr, 0, &v, sizeof(T)};
        }
    };

    /// Grid or threadgroup size. Construct from one number for 1D work
    /// (Dim3{rows}), or three for tiled kernels (Dim3{tiles_x, tiles_y, 1}).
    struct Dim3 {
        uint32_t x = 1, y = 1, z = 1;
        Dim3(uint32_t x_, uint32_t y_ = 1, uint32_t z_ = 1) : x(x_), y(y_), z(z_) {}
        uint32_t total() const { return x * y * z; }
    };

    class Backend
    {
    public:
        virtual ~Backend() = default;

        // Lets the gpu read memory (for core.bin)
        virtual Buffer *wrapNoCopy(void *cpu_ptr, size_t bytes) = 0;

        /// Create fresh memory that can be seen by bth CPU and GPU
        virtual Buffer *allocate(size_t bytes, const char *category) = 0;

        /// Returns CPU pointer to buffer's bytes
        virtual void *contents(Buffer *buffer) = 0;

        // free a buffer and update memory tracker
        virtual void release(Buffer *buffer) = 0;

        // Queue one kernel launch. threadgroups is the grid in threadgroups,
        // threads_per_group the threadgroup shape; both may be 1D, 2D or 3D.
        virtual void run(const char *kernel, std::initializer_list<Arg> args,
                         Dim3 threadgroups, Dim3 threads_per_group) = 0;

        // submit queued kernels and wait for completion. 
        virtual void sync() = 0;
    };
}