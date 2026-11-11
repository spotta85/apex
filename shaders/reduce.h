// Threadgroup-wide reductions shared by the row kernels.
#pragma once
#include <metal_stdlib>
using namespace metal;

// Sum v over every thread of the threadgroup; every thread gets the total.
// scratch needs one float per simdgroup (32 is always enough).
//
// Each simdgroup sums its 32 lanes, lane 0 publishes the partial, and then every simdgroup
// sums the partials itself, so no third broadcast step is needed. The trailing barrier lets
// the caller reuse scratch for the next reduction straight away.
inline float threadgroupSum(float v, threadgroup float *scratch, uint lane, uint simd_id,
                            uint simd_count) {
    v = simd_sum(v);
    if (lane == 0) {
        scratch[simd_id] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = (lane < simd_count) ? scratch[lane] : 0.0f;
    total = simd_sum(total);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}
