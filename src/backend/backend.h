/// @brief Defines what the engine can ask the GPU to do 
/// Implemented by MetalBackend.cpp
// We this file serves as a wrapper for the GPU backend so we can also supprot intel gpu going forward. 


#pragma once
#include <cstddef>
#include <cstdint>

namespace apex {
    struct Buffer; /// GPU buffer 

    class Backend {
        public: 
            virtual ~Backend() = default;
        
            // Lets the gpu read memory (for core.bin)
            virtual Buffer* wrapNoCopy(void* cpu_ptr, size_t bytes) = 0;

            /// Create fresh memory that can be seen by bth CPU and GPU
            virtual Buffer* allocate(size_t bytes, const char* category) = 0;

            /// Returns CPU pointer to buffer's bytes
            virtual void* contents(Buffer* buffer) = 0;

            // free a buffer and update memory tracker
            virtual void release(Buffer* buffer) = 0;

            /// FOR TEST: Sums unsigned 32 bit words on gpu to compare against cpu sum. 
            virtual uint32_t checkSumU32(Buffer* buffer,std::size_t offset, std::size_t bytes) = 0;

    };

}