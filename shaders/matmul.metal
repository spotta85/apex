#include "quant.h"
#include "shared.h"

// NVFP4 matrix times a batch of token vectors, for the weights every token shares
// (q/k/v/o projections, dense MLP, output head).
//
// Layout of one dispatch:
//   grid x : row groups of APEX_MATMUL_ROWS_PER_GROUP rows, one simdgroup per output row
//   grid y : token tiles of APEX_MATMUL_TOKEN_TILE tokens
//
// Inside a simdgroup the 32 lanes take the row's 16-weight groups round-robin
// (g = lane, lane + 32, ...), so consecutive lanes read consecutive 8-byte code pairs and
// consecutive scale bytes (coalesced). Each lane decodes its group ONCE and applies it to
// every token in the tile; with N tokens the weight bytes are read once per tile instead of
// once per token. At decode (one token) this is the plain matvec.

constant constexpr uint ROWS_PER_GROUP = APEX_MATMUL_ROWS_PER_GROUP;
constant constexpr uint TOKEN_TILE = APEX_MATMUL_TOKEN_TILE;
constant constexpr uint LANES = 32;

// Up to three matrices that read the same input, stacked by rows: rows [0, rows0) come from
// matrix 0 and are written to out0, the next rows1 rows from matrix 1 into out1, and so on.
// This is how q, k and v are projected in one dispatch. Each out_m is [tokens, rows_m].
kernel void matmul_nvfp4(device const float *input [[buffer(0)]], // [tokens, in_dim]
                         device const uint *w0 [[buffer(1)]],      // [rows0, in_dim/8]
                         device const uchar *s0 [[buffer(2)]],     // [rows0, in_dim/16]
                         device float *out0 [[buffer(3)]],         // [tokens, rows0]
                         device const uint *w1 [[buffer(4)]],
                         device const uchar *s1 [[buffer(5)]],
                         device float *out1 [[buffer(6)]],
                         device const uint *w2 [[buffer(7)]],
                         device const uchar *s2 [[buffer(8)]],
                         device float *out2 [[buffer(9)]],
                         constant MatmulParams &p [[buffer(10)]],
                         uint2 tg [[threadgroup_position_in_grid]],
                         uint simd_id [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    // Which matrix and which of its rows this simdgroup owns. Row counts are multiples of
    // ROWS_PER_GROUP, so a threadgroup never straddles two matrices.
    uint row = tg.x * ROWS_PER_GROUP + simd_id;
    device const uint *weights;
    device const uchar *scales;
    device float *output;
    uint rows;
    if (row < p.rows0) {
        weights = w0; scales = s0; output = out0; rows = p.rows0;
    } else if (row - p.rows0 < p.rows1) {
        row -= p.rows0;
        weights = w1; scales = s1; output = out1; rows = p.rows1;
    } else if (row - p.rows0 - p.rows1 < p.rows2) {
        row -= p.rows0 + p.rows1;
        weights = w2; scales = s2; output = out2; rows = p.rows2;
    } else {
        return; // whole simdgroup leaves together; no barriers in this kernel
    }

    const uint t0 = tg.y * TOKEN_TILE;
    const uint nt = min(TOKEN_TILE, p.tokens - t0); // tokens in this tile
    const uint groups = p.in_dim / NVFP4_GROUP;

    weights += row * (p.in_dim / NVFP4_PER_WORD);
    scales += row * groups;
    input += t0 * p.in_dim;

    float acc[TOKEN_TILE];
    for (uint t = 0; t < TOKEN_TILE; ++t) acc[t] = 0.0f;

    for (uint g = lane; g < groups; g += LANES) {
        float w[NVFP4_GROUP];
        decodeGroup(weights[2 * g], weights[2 * g + 1], w); // once per tile, not per token
        const float scale = e4m3ToFloat(scales[g]);
        for (uint t = 0; t < TOKEN_TILE; ++t) {
            if (t < nt) {
                acc[t] += scale * dotGroup(w, input + t * p.in_dim + g * NVFP4_GROUP);
            }
        }
    }

    for (uint t = 0; t < TOKEN_TILE; ++t) {
        if (t < nt) { // nt is the same for the whole simdgroup, so simd_sum is safe here
            float y = simd_sum(acc[t]);
            if (p.softcap > 0.0f) {
                y = tanh(y / p.softcap) * p.softcap; // Gemma's final logit softcap
            }
            if (lane == 0) {
                output[(t0 + t) * rows + row] = y;
            }
        }
    }
}

// Dense MLP front half, fused: act[t, r] = gelu_tanh(gate_r . x_t) * (up_r . x_t).
// Same grid and lane layout as matmul_nvfp4; each simdgroup computes the gate row and the up
// row with the same index, so the GeGLU happens in registers and gate/up never hit memory.
kernel void gate_up_geglu_nvfp4(device const float *input [[buffer(0)]],  // [tokens, in_dim]
                                device const uint *gate_w [[buffer(1)]],  // [rows, in_dim/8]
                                device const uchar *gate_s [[buffer(2)]], // [rows, in_dim/16]
                                device const uint *up_w [[buffer(3)]],
                                device const uchar *up_s [[buffer(4)]],
                                device float *output [[buffer(5)]], // [tokens, rows]
                                constant GateUpParams &p [[buffer(6)]],
                                uint2 tg [[threadgroup_position_in_grid]],
                                uint simd_id [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tg.x * ROWS_PER_GROUP + simd_id;
    if (row >= p.rows) {
        return;
    }
    const uint t0 = tg.y * TOKEN_TILE;
    const uint nt = min(TOKEN_TILE, p.tokens - t0);
    const uint groups = p.in_dim / NVFP4_GROUP;
    const uint row_words = p.in_dim / NVFP4_PER_WORD;

    gate_w += row * row_words;
    up_w += row * row_words;
    gate_s += row * groups;
    up_s += row * groups;
    input += t0 * p.in_dim;

    float acc_g[TOKEN_TILE], acc_u[TOKEN_TILE];
    for (uint t = 0; t < TOKEN_TILE; ++t) {
        acc_g[t] = 0.0f;
        acc_u[t] = 0.0f;
    }

    for (uint g = lane; g < groups; g += LANES) {
        float wg[NVFP4_GROUP], wu[NVFP4_GROUP];
        decodeGroup(gate_w[2 * g], gate_w[2 * g + 1], wg);
        decodeGroup(up_w[2 * g], up_w[2 * g + 1], wu);
        const float sg = e4m3ToFloat(gate_s[g]);
        const float su = e4m3ToFloat(up_s[g]);
        for (uint t = 0; t < TOKEN_TILE; ++t) {
            if (t < nt) {
                device const float *x = input + t * p.in_dim + g * NVFP4_GROUP;
                acc_g[t] += sg * dotGroup(wg, x);
                acc_u[t] += su * dotGroup(wu, x);
            }
        }
    }

    for (uint t = 0; t < TOKEN_TILE; ++t) {
        if (t < nt) {
            const float g = simd_sum(acc_g[t]);
            const float u = simd_sum(acc_u[t]);
            if (lane == 0) {
                output[(t0 + t) * p.rows + row] = gegluTanh(g, u);
            }
        }
    }
}
