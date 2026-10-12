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
        for (const auto& [name, pso] : pipelines_) {
            pso->release();
        }
        library_->release();
        queue_->release();
        device_->release();
    }

    const char* MetalBackend::deviceName() const {
        return device_->name()->utf8String();
    }

    /// @brief Helper to create a compute pipeline state object for a function
    /// @param name name of the metal kernel. 
    /// @return cps object for that kernel. 
    MTL::ComputePipelineState* MetalBackend::pipeline(const char* name) {
        auto it = pipelines_.find(name);
        if (it != pipelines_.end()) {
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
        pipelines_[name] = pso;

        return pso;
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

    /// @brief Checksum a buffer on the GPU. Sums 32 bit words and returns the sum.
    /// @param buffer MTL Buffer
    /// @param offset Offset into buffer to start summing
    /// @param bytes Size of buffer to sum.
    /// @return 
    std::uint32_t MetalBackend::checkSumU32(Buffer* input, std::size_t offset, std::size_t bytes) {
        Timer timer ("checkSumU32");
        // setup
        const std::uint32_t num_32 = static_cast<std::uint32_t>(bytes / 4);
        const std::uint32_t per_thread = 1024; 
        const std::uint32_t num_threads =(num_32 + per_thread - 1) / per_thread; // round up

        Buffer *result = allocate(4, "scratch"); // allocate 4 bytes
        *static_cast<std::uint32_t*>(contents(result)) = 0; // Clear buffer

        // GPU stuff

        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();        MTL::ComputePipelineState* pso = pipeline("checksum_u32");
        MTL::CommandBuffer *cmd = queue_->commandBuffer();
        MTL::ComputeCommandEncoder *enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        enc->setBuffer(input->mtl, offset, 0);
        enc->setBytes(&num_32, sizeof num_32, 1);
        enc->setBytes(&per_thread, sizeof per_thread, 2);
        enc->setBuffer(result->mtl, 0, 3);

        const auto group_size = std::min<std::size_t>(num_threads, pso->maxTotalThreadsPerThreadgroup());

        enc->dispatchThreads(MTL::Size::Make(num_threads, 1, 1), MTL::Size::Make(group_size, 1, 1));
        enc->endEncoding(); // finish up updating the queueu
        cmd->commit(); // send work to gpu
        cmd->waitUntilCompleted(); // cpu waits until this is done.
        pool->release();
        const std::uint32_t sum = *static_cast<std::uint32_t*>(contents(result));
        release(result);

        return sum;
    }

} // namespace apex
