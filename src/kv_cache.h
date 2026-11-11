/// @file kv_cache.h
/// @brief The KV caches of every layer, plus the sequence length they hold.
#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include "backend/backend.h"
#include "model_info.h"
#include "ops.h"

namespace apex {

/// Fewest extra ring slots a sliding layer keeps beyond its window.
constexpr uint32_t KV_RING_MIN_SLACK = 16;

struct KvCacheOptions {
    uint32_t max_context = 2048;      // longest sequence (full-attention layers hold all of it)
    uint32_t slack = KV_RING_MIN_SLACK; // sliding rings hold window + slack positions
    KvDtype dtype = KvDtype::F16;
};

/// One KvCache per layer and the number of positions that are valid in all of them.
///
/// Full-attention layers: capacity = max_context, position p in slot p.
/// Sliding layers: a ring of capacity = window + slack, position p in slot p % capacity.
///
/// Why the slack: a query at position p reads keys p - window + 1 .. p. Writing n positions
/// overwrites the n oldest ring slots; that is only harmless if none of them is still visible,
/// i.e. n <= slack + 1 (checked per write). The same holds for a rollback: after truncating r
/// speculative positions, the keys those writes overwrote must be older than anything the next
/// query sees, i.e. r <= slack + 1. Both limits are enforced here; slack >= the largest
/// forward() batch keeps prefill chunks and speculative steps legal.
class KvCacheSet {
  public:
    KvCacheSet(Backend &backend, const ModelConfig &config, const KvCacheOptions &options)
        : backend_(backend), options_(options) {
        if (options.slack < 1) throw std::runtime_error("KV cache: slack must be >= 1");
        for (int l = 0; l < config.num_hidden_layers; ++l) {
            const AttentionShape s = attentionShape(config, l);
            KvCache c;
            c.dtype = options.dtype;
            // A ring only pays off if it is smaller than the whole context.
            c.capacity = s.window ? std::min(options.max_context, s.window + options.slack)
                                  : options.max_context;
            const size_t bytes =
                size_t(s.kv_heads) * c.capacity * s.head_dim * kvElementBytes(options.dtype);
            c.k = backend.allocate(bytes, "kv_cache");
            c.v = backend.allocate(bytes, "kv_cache");
            layers_.push_back(c);
            windows_.push_back(s.window);
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
    uint32_t maxContext() const { return options_.max_context; }
    const KvCacheOptions &options() const { return options_; }

    /// Throws unless n tokens may be written at positions pos0 .. pos0 + n - 1. Writing may
    /// start at or before the current length (re-writing after a rollback), never after it.
    void checkWrite(uint32_t pos0, uint32_t n) const {
        if (pos0 > length_)
            throw std::runtime_error("KV cache: write at " + std::to_string(pos0) +
                                     " leaves a gap after length " + std::to_string(length_));
        if (pos0 + n > options_.max_context)
            throw std::runtime_error("KV cache: sequence longer than max_context " +
                                     std::to_string(options_.max_context));
        if (pos0 < length_ && length_ - pos0 > maxRollback())
            throw std::runtime_error("KV cache: re-write from " + std::to_string(pos0) +
                                     " rolls back more than the ring slack allows");
        for (size_t l = 0; l < layers_.size(); ++l) checkKvWrite(layers_[l], windows_[l], pos0, n);
    }

    /// Record that positions pos0 .. pos0 + n - 1 now hold valid keys and values.
    void commitWrite(uint32_t pos0, uint32_t n) { length_ = pos0 + n; }

    /// Drop every position >= new_length (e.g. rejected speculative tokens). The bytes stay;
    /// the next write at new_length overwrites them. Throws if a sliding ring cannot roll back
    /// that far (more than slack + 1 positions).
    void truncate(uint32_t new_length) {
        if (new_length > length_)
            throw std::runtime_error("KV cache: truncate beyond the current length");
        if (length_ - new_length > maxRollback())
            throw std::runtime_error("KV cache: truncating " +
                                     std::to_string(length_ - new_length) +
                                     " positions exceeds the ring's rollback limit " +
                                     std::to_string(maxRollback()));
        length_ = new_length;
    }

    /// Most positions a truncate (or re-write) may roll back: slack + 1 when some sliding
    /// layer actually wraps, otherwise unlimited.
    uint32_t maxRollback() const {
        uint32_t limit = UINT32_MAX;
        for (size_t l = 0; l < layers_.size(); ++l) {
            if (windows_[l] && layers_[l].capacity < options_.max_context)
                limit = std::min(limit, layers_[l].capacity - windows_[l] + 1);
        }
        return limit;
    }

  private:
    Backend &backend_;
    KvCacheOptions options_;
    uint32_t length_ = 0;
    std::vector<KvCache> layers_;
    std::vector<uint32_t> windows_;
};

} // namespace apex
