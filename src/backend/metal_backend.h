#pragma once

#include <string> 
#include <Metal/Metal.hpp>
#include <map>
#include "backend.h"

namespace apex {

    /// @brief Metal buffer
    struct Buffer {
        MTL::Buffer *mtl = nullptr; // Metal buffer object.
        std::size_t length = 0; // length
        std::string category; // tracker category
    };


    class MetalBackend : public Backend {
        public: 
            explicit MetalBackend(const std::string& metallib_path);

            ~MetalBackend() override;

            Buffer* wrapNoCopy(void* cpu_ptr, std::size_t bytes) override;
            Buffer* allocate(std::size_t bytes, const char* category) override;
            void* contents(Buffer* buffer) override;
            void release(Buffer* buffer) override;

            /// Returns the GPU's name for benchmark output.
            const char* deviceName() const;

            // Prevent copying ownership of Metal objects.
            MetalBackend(const MetalBackend&) = delete;
            MetalBackend& operator=(const MetalBackend&) = delete;
        
        private:
            MTL::ComputePipelineState* pipeline(const char* name); // creates a gpu prog. w this name.

            MTL::Device* device_ = nullptr; // our device
            MTL::CommandQueue* queue_ = nullptr; // command queue for GPU work
            MTL::Library* library_ = nullptr; // shaders from metallib

            std::map<std::string, MTL::ComputePipelineState*> pipelines_; // cached pipelines
    };

}