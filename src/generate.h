/// @file generate.h
/// @brief Greedy generation on top of Model::forward.
#pragma once
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <vector>
#include "model.h"

namespace apex {

struct GenerateStats {
    uint32_t prompt_tokens = 0;
    uint32_t generated = 0;
    double prefill_ms = 0;
    double decode_ms = 0; // all decode steps (each one forward of one token)
};

/// Greedy decoding: prefill the prompt in chunks of max_tokens (the head runs only on the last
/// chunk), then one token per forward call until max_new tokens or a stop id. The only GPU
/// readback per generated token is its argmax id. Returns the generated ids (the stop id
/// included if one was produced).
inline std::vector<uint32_t> generateGreedy(Model &model, const std::vector<uint32_t> &prompt,
                                            uint32_t max_new, const std::vector<uint32_t> &stop,
                                            GenerateStats &stats) {
    using clock = std::chrono::steady_clock;
    auto ms = [](clock::time_point a, clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    std::vector<uint32_t> out;
    if (prompt.empty() || max_new == 0) return out;
    const uint32_t chunk = model.options().max_tokens;
    stats.prompt_tokens = uint32_t(prompt.size());

    const auto t0 = clock::now();
    uint32_t pos = 0;
    uint32_t next = 0;
    while (pos < prompt.size()) {
        const uint32_t n = std::min<uint32_t>(chunk, uint32_t(prompt.size()) - pos);
        const bool last = pos + n == prompt.size();
        const uint32_t *argmax = model.forward(prompt.data() + pos, n, pos, last);
        if (last) next = argmax[n - 1];
        pos += n;
    }
    const auto t1 = clock::now();
    stats.prefill_ms = ms(t0, t1);

    while (true) {
        out.push_back(next);
        if (std::find(stop.begin(), stop.end(), next) != stop.end()) break;
        if (out.size() >= max_new) break;
        next = model.forward(&next, 1, pos, true)[0];
        ++pos;
    }
    stats.decode_ms = ms(t1, clock::now());
    stats.generated = uint32_t(out.size());
    return out;
}

} // namespace apex
