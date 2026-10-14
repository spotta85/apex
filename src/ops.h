#pragma once
#include "backend/backend.h"

namespace apex {
    // BTW weights has an offset all weights use one buffer. 

// Queue RMSNorm for float32 rows using BF16 weights.
inline void rmsNorm(Backend &backend, Buffer *input, Buffer *weights, size_t weight_offset,
                    Buffer *result, uint32_t rows, uint32_t hidden_dim, float eps) {
    
    constexpr uint32_t THREADS_PER_GROUP = 256;
    backend.run("rms_norm_f32",
                {Arg::buf(input), Arg::buf(weights, weight_offset), Arg::buf(result),
                 Arg::val(hidden_dim), Arg::val(eps)},
                Dim3{rows}, Dim3{THREADS_PER_GROUP});
}

} // namespace apex