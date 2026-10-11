#include <metal_stdlib>
using namespace metal;

kernel void add(device const float* a   [[buffer(0)]],
                device const float* b   [[buffer(1)]],
                device float*       out [[buffer(2)]],
                uint i [[thread_position_in_grid]]) {
  out[i] = a[i] + b[i];
}
