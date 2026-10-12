#include <metal_stdlib>
using namespace metal;

#include <metal_stdlib>  // Metal's types and GPU operations.
using namespace metal;

kernel void checksum_u32(
    device const uint *words [[buffer(0)]], // Input tensor bytes.
    constant uint &count [[buffer(1)]],     // Total number of words.
    constant uint &per_thread [[buffer(2)]], // Words assigned to each thread.
    device atomic_uint *result [[buffer(3)]], // Shared output sum.
    uint tid [[thread_position_in_grid]]) {   // This thread's index.

  uint begin = tid * per_thread;
  uint end = min(begin + per_thread, count);

  uint local = 0;
  for (uint i = begin; i < end; ++i) {
    local += words[i];
  }
  // each thread updtse the reult buffer. 
  atomic_fetch_add_explicit(result, local, memory_order_relaxed);
  
    
}
