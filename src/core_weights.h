#pragma once
#include "backend/backend.h"
#include "manifest.h"
#include <fcntl.h>     // open(), O_RDONLY.
#include <sys/stat.h>  // fstat(), struct stat.
#include <unistd.h>    // close().
#include <stdexcept>
#include <string>
#include <sys/mman.h>  // mmap(), munmap().
#include "profiling/memory.h"
#include "profiling/timer.h"

namespace apex{

    // Core buffer: [ tensor A ][ padding ][ tensor B ][ ... ]
    //                                ↑
    //                         TensorView.offset
    //                                │← nbytes →│
    ///Describes a tensors section of the shared core buffer. 
    /// Doesnt own buffer or metadat, just wraps it in buffer
    struct TensorView {
        Buffer *buffer;
        std::uint64_t offset;
        std::uint64_t size;
        const TensorInfo *info;
    };

    /// @brief Constructs CoreWeights (does mapping & wraps in metal buffer) and looks up tensor by name an dgives its buffer and offset to kernel. 
    class CoreWeights {
        private:
            const Manifest& manifest_; // metadata to locate tensors
            Backend &backend_; // GPU ops

            void *base_ = nullptr; // CPU adress of the mapped file
            std::size_t size_ = 0;
            Buffer *buffer_ = nullptr;

        public:
            CoreWeights(const Manifest &manfiest, Backend &backend);
            ~CoreWeights();

            /// Select a tensor's section of the shared GPU buffer.
            TensorView tensor(const std::string& name) const {
                const TensorInfo& info = manifest_.tensor(name);
                return TensorView{buffer_, info.offset, info.nbytes, &info};
            }

            /// CPU address of the of tensor byte
            const void* cpu(const std::string& name) const {
                return static_cast<const char*>(base_) + manifest_.tensor(name).offset;
            }

            /// force all mapped weights into ram
            void touchAll() const {
                Timer timer("touch_core");
                const long page = sysconf(_SC_PAGESIZE);
                if (page <= 0) throw std::runtime_error("cannot query page size");
                const volatile auto* bytes = static_cast<const unsigned char*>(base_);
                for (std::size_t off = 0; off < size_; off += static_cast<std::size_t>(page)) {
                    (void)bytes[off];
                }
            }

            // no two ojbects can own and free same mapping.
            CoreWeights(const CoreWeights&) = delete;
            CoreWeights& operator=(const CoreWeights&) = delete;
    };

    /// @brief Mmaps core.bin, creates a gpu accessible buffer with it. 
    /// @param manifest 
    /// @param backend 
    inline CoreWeights::CoreWeights(const Manifest& manifest, Backend& backend) : manifest_(manifest), backend_(backend) {

        const std::string path = manifest_.dir + "/core.bin";
        const int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error("cannot open" + path);
        }

        struct stat st {};
        if (fstat(fd, &st) != 0) {
            close(fd);
            throw std::runtime_error("cannot stat " + path);
        }

        size_ = static_cast<std::size_t>(st.st_size);

        // Confirm the file matches the size recorded by the repacker.
        if (size_ == 0 || size_ != manifest_.core_size) {
            close(fd);
            throw std::runtime_error("core.bin size != manifest");
        }

        {
            Timer timer("mmap_core");
            base_ = mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd, 0);  // SHARED: no copy-on-write bookkeeping (see EXPERIMENTS.md)
        }
        close(fd);

        if (base_ == MAP_FAILED) {
            throw ::std::runtime_error("mmap core.bin failed");
        }

        MemoryTracker::get().add("core_weights", size_);

        // Give the GPU access to these same bytes.
        try {
            buffer_ = backend_.wrapNoCopy(base_, size_);
        } catch (...) {
            munmap(base_, size_);
            MemoryTracker::get().remove("core_weights", size_);
            throw;
        }

    }

    inline CoreWeights::~CoreWeights() {
        // Metal must release access before we unmap its backing memory.
        backend_.release(buffer_);
        munmap(base_, size_);
        MemoryTracker::get().remove("core_weights", size_);
    }
}
