#include <metal_stdlib>
#include "shared.h"
using namespace metal;

// softmax_j(q . K_j) V_j over the visible cached keys, for every (query token, q head).
// Scale is 1.0 (Gemma 4 normalizes q and k instead). GQA: q head h reads kv head
// h / (q_heads / kv_heads).
//
// One threadgroup per (token, q head), APEX_ATTN_THREADS threads = 8 simdgroups.
// Keys are processed in chunks of APEX_ATTN_KEY_CHUNK with an online softmax (the
// flash-attention recurrence), so context length is not limited by threadgroup memory:
//
//   1. score: each simdgroup takes keys kk = simd_id, simd_id + 8, ... of the chunk. Its 32
//      lanes split head_dim (lane reads d = lane, lane + 32, ...), so one key row is read
//      with fully coalesced 128-byte loads, and simd_sum gives the dot product.
//   2. every thread reads the chunk's scores from threadgroup memory (broadcast reads),
//      computes the chunk max, rescales its running state by exp(m_old - m_new) and adds
//      exp(s - m_new) * V[j][d] for the output dims d it owns (d = tid, tid + 256). V rows
//      are therefore also read coalesced. Every thread computes the same running max and
//      sum, so no reduction is needed for them.
//
// q  : [tokens, q_heads, head_dim]
// K,V: [kv_heads][capacity][head_dim] of KV (float or half); position j is in slot
//      j % capacity. Sliding layers are a ring of window + slack slots: every key a query can
//      see (the last `window` positions) is still there as long as a single write or rollback
//      covers at most `slack` positions (the host checks that).
// out: [tokens, q_heads, head_dim]
template <typename KV>
inline void attentionImpl(device const float *q, device const KV *K, device const KV *V,
                          device float *out, constant AttentionParams &p,
                          threadgroup float *scores, // [APEX_ATTN_KEY_CHUNK]
                          uint2 tg, uint tid, uint lane, uint simd_id) {
    constexpr uint THREADS = APEX_ATTN_THREADS;
    constexpr uint SIMDS = THREADS / 32;
    constexpr uint CHUNK = APEX_ATTN_KEY_CHUNK;
    constexpr uint MAX_Q_PER_LANE = APEX_MAX_HEAD_DIM / 32;      // q values held per lane
    constexpr uint MAX_D_PER_THREAD = APEX_MAX_HEAD_DIM / THREADS; // output dims per thread

    const uint token = tg.x;
    const uint head = tg.y;
    const uint hd = p.head_dim;
    const uint kv_head = head / (p.q_heads / p.kv_heads);
    const uint pos = p.pos0 + token; // this query's absolute position
    // Key j is visible iff j <= pos and (full attention or j > pos - window).
    const uint first = (p.window && pos + 1 > p.window) ? pos + 1 - p.window : 0;

    q += (token * p.q_heads + head) * hd;
    out += (token * p.q_heads + head) * hd;
    K += ulong(kv_head) * p.capacity * hd;
    V += ulong(kv_head) * p.capacity * hd;

    // This lane's slice of q, matching the key elements it will read.
    const uint q_per_lane = hd / 32;
    float qr[MAX_Q_PER_LANE];
    for (uint i = 0; i < MAX_Q_PER_LANE; ++i) {
        qr[i] = i < q_per_lane ? q[lane + 32 * i] : 0.0f;
    }

    float m = -INFINITY; // running max
    float l = 0.0f;      // running sum of exp(s - m)
    float acc[MAX_D_PER_THREAD];
    for (uint i = 0; i < MAX_D_PER_THREAD; ++i) acc[i] = 0.0f;

    for (uint start = first; start <= pos; start += CHUNK) {
        const uint n = min(CHUNK, pos + 1 - start); // keys in this chunk (>= 1)

        // 1. Scores, one simdgroup per key.
        for (uint kk = simd_id; kk < n; kk += SIMDS) {
            device const KV *k = K + ((start + kk) % p.capacity) * hd;
            float s = 0.0f;
            for (uint i = 0; i < q_per_lane; ++i) {
                s += qr[i] * float(k[lane + 32 * i]);
            }
            s = simd_sum(s);
            if (lane == 0) {
                scores[kk] = s;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. Online softmax update and value accumulation.
        float chunk_max = -INFINITY;
        for (uint i = 0; i < n; ++i) chunk_max = max(chunk_max, scores[i]);
        const float m_new = max(m, chunk_max);
        const float alpha = exp(m - m_new); // 0 on the first chunk (m = -inf, m_new finite)
        float chunk_sum = 0.0f;
        for (uint d = 0; d < MAX_D_PER_THREAD; ++d) acc[d] *= alpha;
        for (uint i = 0; i < n; ++i) {
            const float e = exp(scores[i] - m_new);
            chunk_sum += e;
            device const KV *v = V + ((start + i) % p.capacity) * hd;
            for (uint d = 0; d < MAX_D_PER_THREAD; ++d) {
                const uint dim = tid + d * THREADS;
                if (dim < hd) acc[d] += e * float(v[dim]);
            }
        }
        l = l * alpha + chunk_sum;
        m = m_new;
        threadgroup_barrier(mem_flags::mem_threadgroup); // scores[] is rewritten next chunk
    }

    for (uint d = 0; d < MAX_D_PER_THREAD; ++d) {
        const uint dim = tid + d * THREADS;
        if (dim < hd) out[dim] = acc[d] / l;
    }
}

#define ATTENTION_KERNEL(NAME, KV)                                                              \
    kernel void NAME(device const float *q [[buffer(0)]], device const KV *K [[buffer(1)]],    \
                     device const KV *V [[buffer(2)]], device float *out [[buffer(3)]],        \
                     constant AttentionParams &p [[buffer(4)]],                                \
                     uint2 tg [[threadgroup_position_in_grid]],                                \
                     uint tid [[thread_index_in_threadgroup]],                                 \
                     uint lane [[thread_index_in_simdgroup]],                                  \
                     uint simd_id [[simdgroup_index_in_threadgroup]]) {                        \
        threadgroup float scores[APEX_ATTN_KEY_CHUNK]; /* this chunk's scores */               \
        attentionImpl<KV>(q, K, V, out, p, scores, tg, tid, lane, simd_id);                    \
    }

ATTENTION_KERNEL(attention_kv_f32, float)
ATTENTION_KERNEL(attention_kv_f16, half)
