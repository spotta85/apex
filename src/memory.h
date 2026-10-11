// Purpose of this file is to track how much memory the engine uses, by
// category, and to compare that against what macOS says the process uses.
#pragma once

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <map>
#include <mutex>
#include <string>

#include <mach/mach.h>   // task_info(TASK_VM_INFO): the OS's view of this process

namespace apex {

// Process memory reported by macOS, in bytes.
struct OsMemory {
    std::uint64_t resident;  // pages in RAM right now, including clean mmap'd file pages
    std::uint64_t footprint; // what the process is charged for (heap, Metal buffers, dirty pages)
};

// Ask the kernel. Returns zeros if the call fails.
inline OsMemory getOsMemory() {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    kern_return_t kr = task_info(mach_task_self(), TASK_VM_INFO,
                                 reinterpret_cast<task_info_t>(&info), &count);
    if (kr != KERN_SUCCESS) {
        return {0, 0};
    }
    return {info.resident_size, info.phys_footprint};
}

/// Every engine allocation reports here by category
/// ("core_weights", "expert_slots", "kv_cache", ...).
class MemoryTracker {
    private:
        std::mutex mutex;
        std::map<std::string, std::uint64_t> memory_map;
        std::uint64_t total_memory = 0;
        std::uint64_t peak_memory = 0;

    public:
        // only one of these.
        static MemoryTracker& get() {
            static MemoryTracker memory_tracker;
            return memory_tracker;
        }

        void add(const std::string& category, std::uint64_t bytes) {
            std::lock_guard<std::mutex> lock(mutex);
            memory_map[category] += bytes;
            total_memory += bytes;
            if (total_memory > peak_memory) {
                peak_memory = total_memory;
            }
        }

        void remove(const std::string& category, std::uint64_t bytes) {
            std::lock_guard<std::mutex> lock(mutex);
            assert(memory_map[category] >= bytes && "remove() exceeds what was added");
            assert(total_memory >= bytes);
            memory_map[category] -= bytes;
            total_memory -= bytes;
        }

        std::uint64_t getTotal() {
            std::lock_guard<std::mutex> lock(mutex);
            return total_memory;
        }

        std::uint64_t getPeak() {
            std::lock_guard<std::mutex> lock(mutex);
            return peak_memory;
        }

        // Our ledger per category, then total and peak, then the OS numbers.
        // If "os footprint" is far above "total", something allocates outside the tracker.
        void print(FILE* out = stderr) {
            std::lock_guard<std::mutex> lock(mutex);
            const double MB = 1024.0 * 1024.0;

            std::fprintf(out, "%-24s %12s\n", "category", "MB");
            for (const auto& [name, bytes] : memory_map) {
                std::fprintf(out, "%-24s %12.1f\n", name.c_str(), bytes / MB);
            }
            std::fprintf(out, "%-24s %12.1f\n", "tracked total", total_memory / MB);
            std::fprintf(out, "%-24s %12.1f\n", "tracked peak", peak_memory / MB);

            OsMemory os = getOsMemory();
            std::fprintf(out, "%-24s %12.1f\n", "os resident", os.resident / MB);
            std::fprintf(out, "%-24s %12.1f\n", "os footprint", os.footprint / MB);
        }
};

}  // namespace apex
