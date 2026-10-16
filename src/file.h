/// @file file.h
/// @brief OS file helpers: map a file into memory and pull its pages into RAM. No GPU code.
#pragma once
#include <fcntl.h>     // open(), O_RDONLY.
#include <sys/mman.h>  // mmap(), munmap().
#include <sys/stat.h>  // fstat(), struct stat.
#include <unistd.h>    // close(), sysconf().
#include <cstddef>
#include <stdexcept>
#include <string>
#include "profiling/memory.h"
#include "profiling/timer.h"

namespace apex {

/// A read-only file mapped into this process's memory.
struct MappedFile {
    void* data = nullptr;
    std::size_t size = 0;
    const char* category = "";  // memory tracker category
};

/// @brief Maps a whole file read-only and records it in the memory tracker.
/// Nothing is read from disk here; pages load on first touch.
/// @category Category
/// @path path
inline MappedFile mapFile(const std::string& path, const char* category) {
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error("cannot open " + path);
    }

    struct stat st {};
    if (fstat(fd, &st) != 0 || st.st_size == 0) {
        close(fd);
        throw std::runtime_error("cannot stat or empty file: " + path);
    }
    const std::size_t size = static_cast<std::size_t>(st.st_size);

    void* data = nullptr;
    {
        Timer timer("mmap");
        data = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);  // SHARED: no copy-on-write bookkeeping (see EXPERIMENTS.md)
    }
    close(fd);  // the mapping stays valid after the fd is closed

    if (data == MAP_FAILED) {
        throw std::runtime_error("mmap failed: " + path);
    }

    MemoryTracker::get().add(category, size);
    return MappedFile{data, size, category};
}

/// Unmaps the file. Release any GPU buffer wrapping it first.
inline void unmapFile(MappedFile& file) {
    munmap(file.data, file.size);
    MemoryTracker::get().remove(file.category, file.size);
    file = MappedFile{};
}

/// Reads one byte per page so the whole file is pulled into RAM.
inline void touchAll(const MappedFile& file) {
    Timer timer("touch_pages");
    const long page = sysconf(_SC_PAGESIZE);
    if (page <= 0) {
        throw std::runtime_error("cannot query page size");
    }
    const volatile unsigned char* bytes = static_cast<const unsigned char*>(file.data);
    for (std::size_t offset = 0; offset < file.size; offset += static_cast<std::size_t>(page)) {
        (void)bytes[offset];
    }
}

}  // namespace apex
