/// @file kv_cache.h
/// @brief The KV caches of every layer, plus the sequence length they hold.
#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include "backend/backend.h"
#include "model_info.h"
#include "ops.h"

namespace apex {

/// One KvCache per layer and the number of positions that are valid in all of them.
///
/// Layout per layer: K and V are [kv_heads][capacity][head_dim] float32, position p in slot p
/// (linear). capacity = max_context for every layer.
class KvCacheSet {
  public:
    KvCacheSet(Backend &backend, const ModelConfig &config, uint32_t max_context)
        : backend_(backend), max_context_(max_context) {
        for (int l = 0; l < config.num_hidden_layers; ++l) {
            const AttentionShape s = attentionShape(config, l);
            const size_t bytes = size_t(s.kv_heads) * max_context * s.head_dim * sizeof(float);
            KvCache c;
            c.k = backend.allocate(bytes, "kv_cache");
            c.v = backend.allocate(bytes, "kv_cache");
            c.capacity = max_context;
            layers_.push_back(c);
        }
    }
    ~KvCacheSet() {
        for (KvCache &c : layers_) {
            backend_.release(c.k);
            backend_.release(c.v);
        }
    }
    KvCacheSet(const KvCacheSet &) = delete;
    KvCacheSet &operator=(const KvCacheSet &) = delete;

    const KvCache &layer(int l) const { return layers_.at(size_t(l)); }
    uint32_t length() const { return length_; }
    uint32_t maxContext() const { return max_context_; }

    /// Throws unless n tokens may be written at positions pos0 .. pos0 + n - 1. Writing may
    /// start at or before the current length (re-writing after a rollback), never after it.
    void checkWrite(uint32_t pos0, uint32_t n) const {
        if (pos0 > length_)
            throw std::runtime_error("KV cache: write at " + std::to_string(pos0) +
                                     " leaves a gap after length " + std::to_string(length_));
        if (pos0 + n > max_context_)
            throw std::runtime_error("KV cache: sequence longer than max_context " +
                                     std::to_string(max_context_));
    }

    /// Record that positions pos0 .. pos0 + n - 1 now hold valid keys and values.
    void commitWrite(uint32_t pos0, uint32_t n) { length_ = pos0 + n; }

    /// Drop every position >= new_length (e.g. rejected speculative tokens). The bytes stay;
    /// the next write at new_length overwrites them.
    void truncate(uint32_t new_length) {
        if (new_length > length_)
            throw std::runtime_error("KV cache: truncate beyond the current length");
        length_ = new_length;
    }

  private:
    Backend &backend_;
    uint32_t max_context_;
    uint32_t length_ = 0;
    std::vector<KvCache> layers_;
};

} // namespace apex
