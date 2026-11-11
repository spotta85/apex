// Kernel parameter structs and tunables shared by the Metal shaders and the C++ host code.
//
// Both sides include this file, so a struct is laid out identically in both languages and
// argument order cannot drift between a kernel and its launcher. Keep it to plain 4-byte
// fields (uint32 / float) so there are no padding surprises.
//
// The tunables (threadgroup sizes, token tiles, chunk sizes) are named here so they can be
// changed in one place and later tuned against measurements. Every kernel that depends on
// one reads it from here; the host derives its grid sizes from the same macros.
#pragma once

#ifdef __METAL_VERSION__
typedef uint apex_u32;
#else
#include <cstdint>
typedef uint32_t apex_u32;
#endif

// ---------------------------------------------------------------------------------------
// Tunables
// ---------------------------------------------------------------------------------------

// NVFP4 matmuls: one simdgroup per output row, this many simdgroups (rows) per threadgroup.
#define APEX_MATMUL_ROWS_PER_GROUP 8
// Multi-token matmuls decode each weight group once and apply it to up to this many tokens.
// Grid y is ceil(tokens / tile). 8 keeps the per-lane accumulators in registers.
#define APEX_MATMUL_TOKEN_TILE 8

// Row-wise kernels (norms, embedding, layer tail): threads per token row.
#define APEX_ROW_THREADS 256

// Router: one threadgroup per token. More threads means more router rows in flight.
#define APEX_ROUTER_THREADS 512
// Router top-k and the expert lists are sized for at most this many experts per token.
#define APEX_MAX_TOP_K 8
// Router scores live in threadgroup memory; the model has 128 experts.
#define APEX_MAX_EXPERTS 128

// Attention: threads per (token, q head) threadgroup, and keys scored per online-softmax step.
#define APEX_ATTN_THREADS 256
#define APEX_ATTN_KEY_CHUNK 64
// Largest head_dim any layer uses (full-attention layers: 512).
#define APEX_MAX_HEAD_DIM 512

// Head norm + RoPE: threads per (token, head).
#define APEX_HEAD_NORM_THREADS 256

// Argmax over the vocabulary: one threadgroup per token.
#define APEX_ARGMAX_THREADS 1024

// Largest hidden size a row kernel stages in threadgroup memory (Gemma 4 26B: 2816).
#define APEX_MAX_HIDDEN 2816

// ---------------------------------------------------------------------------------------
// Parameter structs (passed with Arg::val, one struct per kernel)
// ---------------------------------------------------------------------------------------

// matmul_nvfp4: up to three matrices that share one input (q/k/v), concatenated by rows.
// A single matrix sets rows1 = rows2 = 0. softcap > 0 applies tanh(y / c) * c to the output.
struct MatmulParams {
    apex_u32 in_dim;        // K, multiple of 16
    apex_u32 rows0;         // rows of matrix 0 (multiple of APEX_MATMUL_ROWS_PER_GROUP)
    apex_u32 rows1;         // rows of matrix 1, or 0
    apex_u32 rows2;         // rows of matrix 2, or 0
    apex_u32 tokens;        // number of input rows
    float softcap;          // 0 = off
};

// gate_up_geglu_nvfp4: act[t, r] = gelu_tanh(gate[r] . x[t]) * (up[r] . x[t]).
struct GateUpParams {
    apex_u32 in_dim;  // 2816
    apex_u32 rows;    // 2112 dense
    apex_u32 tokens;
};

// Byte offsets of the six pieces inside one expert slot, plus the slot stride.
// Read from ModelInfo::expert_pieces at load; never hard-coded.
struct ExpertLayout {
    apex_u32 stride;
    apex_u32 gate_w, gate_s, up_w, up_s, down_w, down_s;
};

// Expert kernels: every token has a list of up to max_k (slot, weight) pairs and a count.
struct ExpertParams {
    apex_u32 in_dim;   // hidden size, 2816
    apex_u32 inter;    // per-expert intermediate size, 704
    apex_u32 tokens;
    apex_u32 max_k;    // stride of the per-token route list (APEX_MAX_TOP_K)
    ExpertLayout layout;
};

// One routed expert for one token. In resident mode slot == expert id; a disk cache maps
// expert ids to the slot that currently holds the expert's bytes.
struct ExpertRoute {
    apex_u32 slot;
    float weight;
};

// Router: norm(h, scale * root) -> 8-bit affine matvec -> top-k -> softmax -> per_expert_scale.
struct RouterParams {
    apex_u32 hidden;       // 2816
    apex_u32 experts;      // 128
    apex_u32 top_k;        // 8
    apex_u32 group_size;   // 64 (affine quantization group)
    float eps;
    float root_scale;      // hidden ** -0.5, rounded to float like mlx does
};

// qkv_norm_rope: per (token, head) RMSNorm + RoPE for q, K and V in one dispatch.
struct QkvNormRopeParams {
    apex_u32 q_heads;
    apex_u32 kv_heads;
    apex_u32 head_dim;
    apex_u32 rope_pairs;   // pairs (i, i + head_dim/2) rotated; 128 sliding, 64 full
    float rope_base;
    apex_u32 pos0;         // absolute position of token 0
    apex_u32 capacity;     // KV cache slots per kv head
    float eps;
};

// attention: online softmax over the visible cached keys for every (token, q head).
struct AttentionParams {
    apex_u32 q_heads;
    apex_u32 kv_heads;
    apex_u32 head_dim;
    apex_u32 capacity;     // KV cache slots per kv head
    apex_u32 pos0;
    apex_u32 window;       // 0 = full attention
    apex_u32 tokens;
};

// Row kernels: norms, embedding, residual/tail.
struct RowParams {
    apex_u32 hidden;       // 2816
    float eps;
    float scale;           // embedding: sqrt(hidden); unused elsewhere
    apex_u32 tokens;
};

// argmax over each row of logits.
struct ArgmaxParams {
    apex_u32 cols;         // vocabulary size
};
