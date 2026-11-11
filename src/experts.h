/// @file experts.h
/// @brief Where each layer's experts come from: the seam between the forward pass and
/// expert residency.
///
/// The forward pass only needs, per layer, one GPU buffer of expert slots in which slot s
/// starts at s * stride. Two sources exist:
///
///   ResidentExperts  every layer file is mapped and wrapped no-copy once at load (slot ==
///                    expert id). Nothing happens per token; the whole forward pass is one
///                    command buffer. Costs all 30 layer files of RAM (wired on first use).
///   StreamedExperts  debug / parity only: before a layer runs, its file is mapped and
///                    wrapped; after the layer's work has completed it is released and
///                    unmapped. Keeps at most one layer file wrapped, at the price of one
///                    GPU sync per layer.
#pragma once
#include <memory>
#include <stdexcept>
#include <vector>
#include "backend/backend.h"
#include "file.h"
#include "model_info.h"

namespace apex {

class ExpertSource {
  public:
    virtual ~ExpertSource() = default;
    /// Slot buffer for `layer`; valid until release(layer) (streamed) or forever (resident).
    virtual Buffer *acquire(int layer) = 0;
    /// True if the caller must sync the GPU after each layer and then call release(layer).
    virtual bool needsLayerSync() const = 0;
    /// The layer's GPU work has completed; the source may drop the buffer.
    virtual void release(int layer) = 0;
};

/// All layer files mapped and wrapped at construction.
class ResidentExperts : public ExpertSource {
  public:
    ResidentExperts(Backend &backend, const ModelInfo &info) : backend_(backend) {
        for (int l = 0; l < info.config.num_hidden_layers; ++l) {
            files_.push_back(mapFile(info.layer_file(l), "expert_layers"));
            buffers_.push_back(backend.wrapNoCopy(files_.back().data, files_.back().size));
        }
    }
    ~ResidentExperts() override {
        for (size_t l = 0; l < buffers_.size(); ++l) {
            backend_.release(buffers_[l]); // GPU lets go before the file is unmapped
            unmapFile(files_[l]);
        }
    }
    Buffer *acquire(int layer) override { return buffers_.at(size_t(layer)); }
    bool needsLayerSync() const override { return false; }
    void release(int) override {}

  private:
    Backend &backend_;
    std::vector<MappedFile> files_;
    std::vector<Buffer *> buffers_;
};

/// One layer file wrapped at a time.
class StreamedExperts : public ExpertSource {
  public:
    StreamedExperts(Backend &backend, const ModelInfo &info) : backend_(backend), info_(info) {}
    ~StreamedExperts() override { drop(); }
    Buffer *acquire(int layer) override {
        drop();
        file_ = mapFile(info_.layer_file(layer), "expert_layers");
        buffer_ = backend_.wrapNoCopy(file_.data, file_.size);
        layer_ = layer;
        return buffer_;
    }
    bool needsLayerSync() const override { return true; }
    void release(int layer) override {
        if (layer == layer_) drop();
    }

  private:
    void drop() {
        if (!buffer_) return;
        backend_.release(buffer_);
        unmapFile(file_);
        buffer_ = nullptr;
        layer_ = -1;
    }
    Backend &backend_;
    const ModelInfo &info_;
    MappedFile file_;
    Buffer *buffer_ = nullptr;
    int layer_ = -1;
};

enum class ExpertResidency { Resident, LayerStreamed };

inline std::unique_ptr<ExpertSource> makeExpertSource(ExpertResidency mode, Backend &backend,
                                                      const ModelInfo &info) {
    if (mode == ExpertResidency::Resident) return std::make_unique<ResidentExperts>(backend, info);
    return std::make_unique<StreamedExperts>(backend, info);
}

} // namespace apex
