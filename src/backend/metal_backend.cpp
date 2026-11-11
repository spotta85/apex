#include "metal_backend.h"  // Class declarations we will implement.
#include <stdexcept>        // std::runtime_error for initialization failures.
#include <algorithm>
#include "../profiling/memory.h"
#include "../profiling/timer.h"
namespace apex {

    static NS::String* str(const char* text) {
        return NS::String::string(text, NS::UTF8StringEncoding);
    }

    /// @brief Constructs a MetalBackend with command queue, loading the given metallib shader.
    /// @param metallib_path 
    MetalBackend::MetalBackend(const std::string &metallib_path) {
        auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

        device_ = MTL::CreateSystemDefaultDevice();
        if (!device_) {
            throw std::runtime_error("MetalBackend: no Metal device found");
        }

        // create queue

        queue_ = device_->newCommandQueue();
        if (!queue_) {
            throw std::runtime_error("MetalBackend: failed to create command queue");
        }
        NS::Error* err = nullptr;
        library_ = device_->newLibrary(str(metallib_path.c_str()), &err);
        if (!library_) {
            queue_->release();
            device_->release();
            throw std::runtime_error("cannot load metallib: " + metallib_path);
        }
    };

    /// @brief Just a boring old destructor. Releases all Metal objects we own.
    MetalBackend::~MetalBackend() {
        for (const auto& [name, k] : kernels_) {
            k->pso->release();
            delete k;
        }
        library_->release();
        queue_->release();
        device_->release();
    }

    const char* MetalBackend::deviceName() const {
        return device_->name()->utf8String();
    }

    /// @brief Creates (once) the compute pipeline state object for a function.
    /// Callers keep the returned handle, so the map lookup happens at load time only.
    /// @param name name of the metal kernel.
    /// @return handle owning the pipeline for that kernel.
    Kernel* MetalBackend::kernel(const char* name) {
        auto it = kernels_.find(name);
        if (it != kernels_.end()) {
            return it->second;
        }

        // find shared function inside metallib
        MTL::Function* fn = library_->newFunction(str(name));
        if (!fn) {
            throw std::runtime_error("cannot find function in metallib: " + std::string(name));
        }

        // Prepare that function for execution on this GPU.
        NS::Error* error = nullptr;
        MTL::ComputePipelineState* pso =
            device_->newComputePipelineState(fn, &error);
        fn->release();

        if (!pso) {
            const char* detail = error ? error->localizedDescription()->utf8String()
                                       : "unknown Metal error";
            throw std::runtime_error(std::string("pipeline ") + name + ": " + detail);
        }

        // The cache owns this pipeline until the backend destructor releases it.
        Kernel* k = new Kernel{pso, name, static_cast<uint32_t>(pso->maxTotalThreadsPerThreadgroup())};
        kernels_[name] = k;
        return k;
    }

    /// @brief Creates a metal buffer which is bsically a wrapper around cpu pointer. You create a buffer and then pass the pointer here. 
    /// @param ptr Pointer to CPU memory (page aligned)
    /// @param bytes size of buffer
    /// @return Buffer pointer. 
    Buffer* MetalBackend::wrapNoCopy(void* ptr, size_t bytes) {
        const size_t page = 16384;

        if (reinterpret_cast<uintptr_t>(ptr) % page || bytes % page) {
            throw std::runtime_error("wrapNoCopy: pointer and length must be page multiples");
        }

        // Metal buffer with shared storage mode, no copy.
        MTL::Buffer* b = device_->newBuffer(
            ptr,
            bytes,
            MTL::ResourceStorageModeShared,
            nullptr
        );

        if (!b)
            throw std::runtime_error("wrapNoCopy: newBuffer failed");

        return new Buffer{b, bytes, ""};
    }

    /// @brief Creates a buffer that is accessible by both cpu and gpu
    /// @param bytes size of buffer
    /// @param category category for memory tacker
    /// @return 
    Buffer* MetalBackend::allocate(std::size_t bytes, const char* category) {
    // Allocate fresh memory that both the CPU and GPU can access.
        MTL::Buffer* b = device_->newBuffer( bytes, MTL::ResourceStorageModeShared);

        if (!b) {
            throw std::runtime_error("allocate: newBuffer failed");
        }

        // Record the allocation under a category such as "scratch".
        MemoryTracker::get().add(category, bytes);

        // Store the Metal object, length, and category for later cleanup.
        return new Buffer{b, bytes, category};
    }

    /// @brief Givues us a CPU acessible pointer to the buffer
    /// @param buf 
    /// @return 
    void* MetalBackend::contents(Buffer* buf) { return buf->mtl->contents(); }


    /// @brief Release MTL buffer and update memory tracker.
    /// @param buffer MTL buffer we created before.
    void MetalBackend::release(Buffer* buffer) {
        if (!buffer) return;

        // only allocated buffers are tracked not wrapped buffers
        if (!buffer->category.empty()) {
            MemoryTracker::get().remove(buffer->category, buffer->length);
        }

        // Give up our ownership of the actual Metal buffer.
        buffer->mtl->release();

        // Delte c++ warpper
        delete buffer;
    }
  
    void MetalBackend::run(Kernel* kernel, std::initializer_list<Arg> args,
                           Dim3 threadgroups, Dim3 threads_per_group) {
        if (!cmd_) {
            // One pool per batch: commandBuffer() and computeCommandEncoder() are
            // autoreleased; the pool drains them in sync().
            pool_ = NS::AutoreleasePool::alloc()->init();
            // Unretained references: the engine keeps every buffer alive until sync()
            // returns, so Metal does not need to retain/release each one per dispatch.
            cmd_ = queue_->commandBufferWithUnretainedReferences();
            // Serial dispatch (the default): each dispatch sees the previous one's writes.
            enc_ = cmd_->computeCommandEncoder();
            bound_ = nullptr;
        }
        // Metal's limit is on the threadgroup's total thread count, not per axis.
        if (threads_per_group.total() > kernel->max_threads) {
            throw std::runtime_error(kernel->name + ": threadgroup too large (" +
                                     std::to_string(threads_per_group.total()) + " > " +
                                     std::to_string(kernel->max_threads) + ")");
        }
        if (bound_ != kernel) {
            enc_->setComputePipelineState(kernel->pso);
            bound_ = kernel;
        }
        uint32_t index = 0;
        for (const Arg& a : args) {
            if (a.buffer) enc_->setBuffer(a.buffer->mtl, a.offset, index);
            else          enc_->setBytes(a.bytes, a.size, index);
            ++index;
        }
        enc_->dispatchThreadgroups(MTL::Size::Make(threadgroups.x, threadgroups.y, threadgroups.z),
                                   MTL::Size::Make(threads_per_group.x, threads_per_group.y, threads_per_group.z));
    }

    void MetalBackend::sync() {
        if (!cmd_) return;
        Timer t("gpu_sync");
        enc_->endEncoding();
        cmd_->commit();
        cmd_->waitUntilCompleted();
        if (cmd_->status() == MTL::CommandBufferStatusError) {
            const char* detail = cmd_->error() ? cmd_->error()->localizedDescription()->utf8String()
                                               : "unknown";
            std::string message = std::string("GPU command buffer failed: ") + detail;
            pool_->release();
            pool_ = nullptr; cmd_ = nullptr; enc_ = nullptr; bound_ = nullptr;
            throw std::runtime_error(message);
        }
        pool_->release();            // releases cmd_ and enc_
        pool_ = nullptr; cmd_ = nullptr; enc_ = nullptr; bound_ = nullptr;
    }
} // namespace apex
