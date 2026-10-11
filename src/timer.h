// Purpose of this file is to help us proifle time for variaous operations. 
#pragma once

#include <chrono>
#include <cstdio> // mainly for printing
#include <map>
#include <mutex>
#include <string>
#include <utility>

namespace apex {
    struct PhaseStats {
        double total_ms = 0.0;
        double max_ms = 0.0; // longest single call. 
        long calls = 0;
    };

    /// Used to profile multiple timers of an operatoin
    class Profile {
        // private goes first -_-
        private:
            std::mutex mutex;
            std::map<std::string, PhaseStats> stats;

        public:
            // make Profile shared
            static Profile& get() {
                static Profile p;
                return p;
            }
            void add(const std::string& phase, double ms) {
                std::lock_guard<std::mutex> lock(mutex);
                PhaseStats& s = stats[phase];
                s.total_ms += ms;
                s.calls++;
                if (ms > s.max_ms) {
                    s.max_ms = ms;
                }
            }
            void reset() {
                std::lock_guard<std::mutex> lock(mutex);
                stats.clear();
            }
            void print(FILE* out = stderr) {
                std::lock_guard<std::mutex> lock(mutex);

                std::fprintf(out, "%-24s %10s %8s %10s %10s\n",
                            "phase", "total ms", "calls", "avg ms", "max ms");

                for (const auto& [name, s] : stats) {
                    std::fprintf(out, "%-24s %10.2f %8ld %10.3f %10.3f\n",
                                name.c_str(), s.total_ms, s.calls,
                                s.total_ms / s.calls, s.max_ms);
                }
            }
    };

    // simple class to time an operation with as few lines of code as possible. 
    class Timer{
        private:
            std::string name;
            std::chrono::steady_clock::time_point start_time;
        
        public:
            explicit Timer(std::string name) : 
                name(std::move(name)), 
                start_time(std::chrono::steady_clock::now()) {}

            // no copy construction. 
            Timer(const Timer&) = delete;
            Timer& operator=(const Timer&) = delete;

            // uses desctructor so we end timer when we get out of scope of an operation
            ~Timer() {
                auto end_time = std::chrono::steady_clock::now();
                double ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
                Profile::get().add(name, ms);   // update profile with teh time. 
            }
        
    };
}  // namespace apex
