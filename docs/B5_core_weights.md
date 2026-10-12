# B5: core weights as Metal buffers, zero copy

Reference for writing step B5. Read top to bottom once, then use the file
sections as you type.

## Goal

The GPU must be able to read the 1.3 GB of core weights (embeddings, norms,
attention, dense MLPs, routers) without a second copy in memory. After B5:

- `core.bin` is mmap'd once (already done in A4).
- The whole mapping is wrapped as **one** Metal buffer with `newBuffer(bytesNoCopy:)`.
- Any tensor is "that buffer, at this byte offset", looked up by name from `manifest.json`.
- A test kernel sums a tensor on the GPU and matches a CPU sum of the same bytes.
- `os footprint` in the memory table stays near zero. Nothing was copied.

Every Metal type stays behind a `Backend` interface. This is the seam the
Windows/Intel port (F24) plugs into later.

## Why one buffer and not one per tensor

`newBuffer(bytesNoCopy:)` requires **both** the pointer and the length to be
page multiples (16 KB). The mapping base is page aligned and the repacker
padded `core.bin`'s length to a page multiple, so wrapping the whole file is
legal. Individual tensors start on page boundaries but their *lengths* are
arbitrary (`norm.weight` is 5,632 bytes), so wrapping each one is not. Hence:
one buffer, 863 offsets. Kernels take `(buffer, offset)` pairs via
`setBuffer(buf, offset, index)`.

## Files

| File | New or edit | Purpose |
|---|---|---|
| `third_party/nlohmann/json.hpp` | new (download) | JSON parser, single header |
| `CMakeLists.txt` | edit | include path, new shader, new source file |
| `src/manifest.h` | new | `manifest.json` -> C++ structs |
| `src/backend.h` | new | abstract GPU interface, no Metal types |
| `src/metal_backend.h` | new | `MetalBackend` declaration |
| `src/metal_backend.cpp` | new | the only engine file that touches `MTL::` |
| `src/core_weights.h` | new | mmap + wrap + tensor lookup |
| `shaders/checksum.metal` | new | test kernel |
| `src/main.cpp` | edit | `bench` gains manifest load, wrap, GPU checksum |

Order to write them: 1 → 2 → 3 (test with `apex manifest`) → 4 → 5/6 → 7 → 8 → 9.

---

## 1. Vendor nlohmann/json

Download the single header from the latest release at
https://github.com/nlohmann/json/releases (the asset named `json.hpp`) and
save it as `third_party/nlohmann/json.hpp`. ~900 KB, header only, MIT.

## 2. CMakeLists.txt edits

```cmake
# after the metal_cpp target:
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE third_party)   # #include <nlohmann/json.hpp>

# shader list grows:
add_metal_library(apex_shaders OUTPUT apex.metallib
  SOURCES shaders/hello.metal shaders/checksum.metal)

# source list grows, and we link json:
add_executable(apex src/main.cpp src/metal_impl.cpp src/metal_backend.cpp)
target_link_libraries(apex PRIVATE metal_cpp nlohmann_json)
```

## 3. `src/manifest.h`

Pure C++. Mirrors `manifest.json` as written by `tools/repack.py`:

```
{ version, alignment, text_config{...}, quantization{group_size,bits,mode},
  core{ file, size, tensors[ {name,dtype,shape,offset,nbytes,sha256} ] },
  layers[ { layer, file, size, num_experts, stride,
            pieces[ {name,dtype,shape,offset,nbytes} ], expert_sha256[] } ] }
```

```cpp
// Purpose: read manifest.json so C++ knows where every tensor lives.
#pragma once

#include <cstdint>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace apex {

// One tensor inside core.bin.
struct TensorInfo {
    std::string dtype;           // "U32", "U8", "BF16"
    std::vector<int64_t> shape;
    uint64_t offset = 0;         // byte offset in core.bin, always a page multiple
    uint64_t nbytes = 0;
};

// One of the six pieces inside an expert slot. offset is within the slot.
struct ExpertPiece {
    std::string name;            // "gate_proj.weight", ...
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t offset = 0;
    uint64_t nbytes = 0;
};

// The handful of config values kernels need. Add fields as kernels need them.
struct ModelConfig {
    int hidden_size = 0;            // 2816
    int intermediate_size = 0;      // 2112  (dense MLP)
    int moe_intermediate_size = 0;  // 704   (per expert)
    int num_attention_heads = 0;    // 16
    int num_key_value_heads = 0;    // 8
    int head_dim = 0;               // 256
    int num_hidden_layers = 0;      // 30
    int num_experts = 0;            // 128
    int top_k_experts = 0;          // 8
    int vocab_size = 0;             // 262144
    int sliding_window = 0;         // 1024
    double rms_norm_eps = 0;        // 1e-6
    double final_logit_softcapping = 0;  // 30.0
    std::vector<std::string> layer_types;  // "sliding_attention" | "full_attention"
};

struct Manifest {
    std::string packed_dir;
    ModelConfig config;
    int quant_group_size = 0;       // 16
    int quant_bits = 0;             // 4
    std::string quant_mode;         // "nvfp4"

    uint64_t core_size = 0;
    std::map<std::string, TensorInfo> core;   // name -> where

    uint64_t expert_stride = 0;               // bytes between experts in layer_NN.bin
    std::vector<ExpertPiece> expert_pieces;   // same for every layer

    // Throws std::runtime_error on a missing file or missing tensor.
    static Manifest load(const std::string& packed_dir);

    const TensorInfo& tensor(const std::string& name) const {
        auto it = core.find(name);
        if (it == core.end()) throw std::runtime_error("manifest: no tensor " + name);
        return it->second;
    }

    std::string layer_file(int layer) const {
        char buf[32];
        std::snprintf(buf, sizeof buf, "/layer_%02d.bin", layer);
        return packed_dir + buf;
    }
};

inline Manifest Manifest::load(const std::string& packed_dir) {
    using json = nlohmann::json;
    std::ifstream in(packed_dir + "/manifest.json");
    if (!in) throw std::runtime_error("manifest: cannot open " + packed_dir + "/manifest.json");
    json j = json::parse(in);

    Manifest m;
    m.packed_dir = packed_dir;

    const json& tc = j["text_config"];
    ModelConfig& c = m.config;
    c.hidden_size            = tc["hidden_size"];
    c.intermediate_size      = tc["intermediate_size"];
    c.moe_intermediate_size  = tc["moe_intermediate_size"];
    c.num_attention_heads    = tc["num_attention_heads"];
    c.num_key_value_heads    = tc["num_key_value_heads"];
    c.head_dim               = tc["head_dim"];
    c.num_hidden_layers      = tc["num_hidden_layers"];
    c.num_experts            = tc["num_experts"];
    c.top_k_experts          = tc["top_k_experts"];
    c.vocab_size             = tc["vocab_size"];
    c.sliding_window         = tc["sliding_window"];
    c.rms_norm_eps           = tc["rms_norm_eps"];
    c.final_logit_softcapping = tc["final_logit_softcapping"];
    c.layer_types            = tc["layer_types"].get<std::vector<std::string>>();

    const json& q = j["quantization"];
    m.quant_group_size = q["group_size"];
    m.quant_bits       = q["bits"];
    m.quant_mode       = q["mode"];

    m.core_size = j["core"]["size"];
    for (const json& t : j["core"]["tensors"]) {
        TensorInfo info;
        info.dtype  = t["dtype"];
        info.shape  = t["shape"].get<std::vector<int64_t>>();
        info.offset = t["offset"];
        info.nbytes = t["nbytes"];
        m.core[t["name"]] = info;
    }

    // Every layer has the same stride and piece layout; read layer 0's.
    const json& l0 = j["layers"][0];
    m.expert_stride = l0["stride"];
    for (const json& p : l0["pieces"]) {
        ExpertPiece piece;
        piece.name   = p["name"];
        piece.dtype  = p["dtype"];
        piece.shape  = p["shape"].get<std::vector<int64_t>>();
        piece.offset = p["offset"];
        piece.nbytes = p["nbytes"];
        m.expert_pieces.push_back(piece);
    }
    return m;
}

}  // namespace apex
```

Test before touching Metal: a `manifest` subcommand in `main.cpp` (see §9)
that prints tensor count, `core_size`, `expert_stride`, and the offset of
`norm.weight`. Expect 863 tensors, 1,359,708,160 bytes, stride 3,358,720.

## 4. `src/backend.h`

No Metal types. Everything the engine needs from a GPU, today:

```cpp
// Purpose: the one interface the engine talks to the GPU through.
// MetalBackend implements it; an Intel/Vulkan backend would too.
#pragma once

#include <cstddef>
#include <cstdint>

namespace apex {

// Opaque GPU buffer handle. The backend owns the real object.
struct Buffer;

class Backend {
public:
    virtual ~Backend() = default;

    // Wrap existing memory with no copy. ptr and bytes must be page multiples.
    // The caller keeps ptr alive for the buffer's lifetime (it's our mmap).
    virtual Buffer* wrapNoCopy(void* ptr, size_t bytes) = 0;

    // Fresh GPU-visible memory. Reported to MemoryTracker under category.
    virtual Buffer* allocate(size_t bytes, const char* category) = 0;

    // Returns the CPU pointer of a buffer (unified memory: always valid).
    virtual void* contents(Buffer* buf) = 0;

    virtual void release(Buffer* buf) = 0;

    // B5 test hook: sum of the uint32 words in [offset, offset + nbytes),
    // modulo 2^32, computed on the GPU. Removed once real kernels exist.
    virtual uint32_t checksumU32(Buffer* buf, size_t offset, size_t nbytes) = 0;
};

}  // namespace apex
```

## 5. `src/metal_backend.h`

```cpp
#pragma once

#include <string>
#include <Metal/Metal.hpp>

#include "backend.h"

namespace apex {

// What a Buffer* really is on Metal. Only this file and metal_backend.cpp know.
struct Buffer {
    MTL::Buffer* mtl = nullptr;
    size_t bytes = 0;
    std::string category;        // for MemoryTracker; empty for wrapped (not ours)
};

class MetalBackend : public Backend {
public:
    explicit MetalBackend(const std::string& metallib_path);
    ~MetalBackend() override;

    Buffer* wrapNoCopy(void* ptr, size_t bytes) override;
    Buffer* allocate(size_t bytes, const char* category) override;
    void* contents(Buffer* buf) override;
    void release(Buffer* buf) override;
    uint32_t checksumU32(Buffer* buf, size_t offset, size_t nbytes) override;

    const char* deviceName() const;

private:
    MTL::ComputePipelineState* pipeline(const char* fn);   // cached by name

    MTL::Device* device_ = nullptr;
    MTL::CommandQueue* queue_ = nullptr;
    MTL::Library* library_ = nullptr;
    std::map<std::string, MTL::ComputePipelineState*> pipelines_;
};

}  // namespace apex
```

(Add `#include <map>`.)

## 6. `src/metal_backend.cpp`

```cpp
#include "metal_backend.h"

#include <cstdio>
#include <stdexcept>

#include "memory.h"
#include "timer.h"

namespace apex {

static NS::String* str(const char* s) { return NS::String::string(s, NS::UTF8StringEncoding); }

MetalBackend::MetalBackend(const std::string& metallib_path) {
    device_ = MTL::CreateSystemDefaultDevice();
    if (!device_) throw std::runtime_error("no Metal device");
    queue_ = device_->newCommandQueue();

    NS::Error* err = nullptr;
    library_ = device_->newLibrary(NS::URL::fileURLWithPath(str(metallib_path.c_str())), &err);
    if (!library_) {
        throw std::runtime_error(std::string("metallib: ") + err->localizedDescription()->utf8String());
    }
}

MetalBackend::~MetalBackend() {
    for (auto& [name, pso] : pipelines_) pso->release();
    library_->release();
    queue_->release();
    device_->release();
}

const char* MetalBackend::deviceName() const { return device_->name()->utf8String(); }

MTL::ComputePipelineState* MetalBackend::pipeline(const char* fn) {
    auto it = pipelines_.find(fn);
    if (it != pipelines_.end()) return it->second;
    NS::Error* err = nullptr;
    MTL::Function* f = library_->newFunction(str(fn));
    if (!f) throw std::runtime_error(std::string("no kernel ") + fn);
    MTL::ComputePipelineState* pso = device_->newComputePipelineState(f, &err);
    f->release();
    if (!pso) throw std::runtime_error(std::string("pipeline ") + fn + ": " + err->localizedDescription()->utf8String());
    pipelines_[fn] = pso;
    return pso;
}

Buffer* MetalBackend::wrapNoCopy(void* ptr, size_t bytes) {
    const size_t page = 16384;
    if (reinterpret_cast<uintptr_t>(ptr) % page || bytes % page) {
        throw std::runtime_error("wrapNoCopy: pointer and length must be page multiples");
    }
    // Shared storage + no deallocator: Metal reads our mmap in place and
    // never frees it. We own the mapping.
    MTL::Buffer* b = device_->newBuffer(ptr, bytes, MTL::ResourceStorageModeShared, nullptr);
    if (!b) throw std::runtime_error("wrapNoCopy: newBuffer failed");
    return new Buffer{b, bytes, ""};        // not tracked: the mmap already is
}

Buffer* MetalBackend::allocate(size_t bytes, const char* category) {
    MTL::Buffer* b = device_->newBuffer(bytes, MTL::ResourceStorageModeShared);
    if (!b) throw std::runtime_error("allocate: newBuffer failed");
    MemoryTracker::get().add(category, bytes);
    return new Buffer{b, bytes, category};
}

void* MetalBackend::contents(Buffer* buf) { return buf->mtl->contents(); }

void MetalBackend::release(Buffer* buf) {
    if (!buf) return;
    if (!buf->category.empty()) MemoryTracker::get().remove(buf->category, buf->bytes);
    buf->mtl->release();
    delete buf;
}

uint32_t MetalBackend::checksumU32(Buffer* buf, size_t offset, size_t nbytes) {
    Timer t("gpu_checksum");
    const uint32_t count = static_cast<uint32_t>(nbytes / 4);
    const uint32_t per_thread = 1024;                       // words each thread sums
    const uint32_t threads = (count + per_thread - 1) / per_thread;

    Buffer* result = allocate(4, "scratch");
    *static_cast<uint32_t*>(contents(result)) = 0;

    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    MTL::ComputePipelineState* pso = pipeline("checksum_u32");
    MTL::CommandBuffer* cmd = queue_->commandBuffer();
    MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(buf->mtl, offset, 0);                    // the tensor, by offset
    enc->setBytes(&count, sizeof count, 1);
    enc->setBytes(&per_thread, sizeof per_thread, 2);
    enc->setBuffer(result->mtl, 0, 3);
    enc->dispatchThreads(MTL::Size::Make(threads, 1, 1),
                         MTL::Size::Make(std::min<uint32_t>(threads, pso->maxTotalThreadsPerThreadgroup()), 1, 1));
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
    pool->release();

    uint32_t sum = *static_cast<uint32_t*>(contents(result));
    release(result);
    return sum;
}

}  // namespace apex
```

(Add `#include <algorithm>` for `std::min`.)

Notes:
- `newBuffer(ptr, bytes, options, nullptr)`: the last argument is the
  deallocator. `nullptr` means "don't free this when the buffer dies".
- `setBuffer(buf, offset, index)` is how a kernel sees a tensor inside the
  one big buffer. The kernel's `buffer(0)` pointer starts at `offset`.
- `allocate` is the only place `newBuffer(length:)` is ever called. Every
  GPU allocation is therefore in the tracker. Keep it that way.

## 7. `src/core_weights.h`

Moves the mmap out of `run_bench` and adds the wrap.

```cpp
// Purpose: own the core.bin mapping and its one Metal buffer; hand out
// (buffer, offset) for any tensor by name.
#pragma once

#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "backend.h"
#include "manifest.h"
#include "memory.h"
#include "timer.h"

namespace apex {

struct TensorView {
    Buffer* buffer;      // always the core buffer
    uint64_t offset;
    uint64_t nbytes;
    const TensorInfo* info;
};

class CoreWeights {
public:
    CoreWeights(const Manifest& m, Backend& backend) : manifest_(m), backend_(backend) {
        std::string path = m.packed_dir + "/core.bin";
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open " + path);
        struct stat st {};
        fstat(fd, &st);
        size_ = static_cast<size_t>(st.st_size);
        if (size_ != m.core_size) throw std::runtime_error("core.bin size != manifest");
        {
            Timer t("mmap_core");
            base_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
        }
        close(fd);
        if (base_ == MAP_FAILED) throw std::runtime_error("mmap core.bin failed");
        MemoryTracker::get().add("core_weights", size_);

        buffer_ = backend_.wrapNoCopy(base_, size_);   // zero copy
    }

    ~CoreWeights() {
        backend_.release(buffer_);
        munmap(base_, size_);
        MemoryTracker::get().remove("core_weights", size_);
    }
    CoreWeights(const CoreWeights&) = delete;
    CoreWeights& operator=(const CoreWeights&) = delete;

    TensorView tensor(const std::string& name) const {
        const TensorInfo& info = manifest_.tensor(name);
        return {buffer_, info.offset, info.nbytes, &info};
    }

    // CPU pointer to a tensor's bytes (for checks and for CPU-side reads).
    const void* cpu(const std::string& name) const {
        return static_cast<const char*>(base_) + manifest_.tensor(name).offset;
    }

    // Force every page resident. Returns ms spent.
    void touchAll() const {
        Timer t("touch_core");
        const size_t page = 16384;
        volatile unsigned long sink = 0;
        const unsigned char* p = static_cast<const unsigned char*>(base_);
        for (size_t off = 0; off < size_; off += page) sink += p[off];
    }

    size_t size() const { return size_; }

private:
    const Manifest& manifest_;
    Backend& backend_;
    void* base_ = nullptr;
    size_t size_ = 0;
    Buffer* buffer_ = nullptr;
};

}  // namespace apex
```

## 8. `shaders/checksum.metal`

Each thread sums `per_thread` consecutive words, then one atomic add.
(One atomic per word would serialize 92 million threads on one address.)

```metal
#include <metal_stdlib>
using namespace metal;

kernel void checksum_u32(device const uint*  words      [[buffer(0)]],
                         constant uint&      count      [[buffer(1)]],
                         constant uint&      per_thread [[buffer(2)]],
                         device atomic_uint* result     [[buffer(3)]],
                         uint tid [[thread_position_in_grid]]) {
    uint begin = tid * per_thread;
    uint end = min(begin + per_thread, count);
    uint local = 0;
    for (uint i = begin; i < end; ++i) local += words[i];
    atomic_fetch_add_explicit(result, local, memory_order_relaxed);
}
```

## 9. `src/main.cpp` edits

- Keep `run_smoke` as is.
- Add `run_manifest(packed_dir)`: load, print counts (§3 test).
- Replace the body of `run_bench` with:

```cpp
static int run_bench(const std::string& packed_dir) {
    apex::Manifest manifest = apex::Manifest::load(packed_dir);
    apex::MetalBackend backend(APEX_METALLIB);
    std::printf("device: %s\n", backend.deviceName());

    apex::CoreWeights core(manifest, backend);
    core.touchAll();

    // GPU reads the mapped file directly; CPU sums the same bytes.
    const char* name = "embed_tokens.weight";          // 352 MB, the biggest tensor
    apex::TensorView v = core.tensor(name);
    uint32_t gpu = backend.checksumU32(v.buffer, v.offset, v.nbytes);

    uint32_t cpu = 0;
    {
        apex::Timer t("cpu_checksum");
        const uint32_t* w = static_cast<const uint32_t*>(core.cpu(name));
        for (size_t i = 0; i < v.nbytes / 4; ++i) cpu += w[i];
    }
    std::printf("%s: gpu %08x cpu %08x %s\n", name, gpu, cpu, gpu == cpu ? "MATCH" : "MISMATCH");

    apex::Profile::get().print(stdout);
    apex::MemoryTracker::get().print(stdout);
    return gpu == cpu ? 0 : 1;
}
```

Includes to add: `"manifest.h"`, `"metal_backend.h"`, `"core_weights.h"`.
Remove the now-unused `<sys/mman.h>` etc. from `main.cpp` if nothing else uses them.

## Expected output

```
device: Apple M5
embed_tokens.weight: gpu 7a3f...  cpu 7a3f...  MATCH
phase                      total ms    calls     avg ms     max ms
cpu_checksum                  ~40-80        1
gpu_checksum                  ~10-30        1
mmap_core                      ~1-5         1
touch_core                    ~36 (warm)    1
category                           MB
core_weights                   1296.7
scratch                           0.0
tracked total                  1296.7
tracked peak                   1296.7
os resident                    ~1303
os footprint                   ~small (tens of MB: Metal's own state, not the weights)
```

The three things that prove B5:
1. **MATCH**: the GPU read the mmap'd bytes through the no-copy buffer.
2. `os footprint` did **not** grow by 1.3 GB. Nothing was copied.
3. `scratch` is back to 0 and `tracked peak` is unchanged: `allocate`/`release` are balanced.

## If it goes wrong

- **`wrapNoCopy: pointer and length must be page multiples`**: core.bin was
  not padded, or the mapping isn't from `mmap`. Check `core_size % 16384 == 0`.
- **MISMATCH**: most likely a `setBuffer` offset or `count` mismatch. Print
  `v.offset` and `v.nbytes` and compare with `manifest.json`.
- **`no kernel checksum_u32`**: the shader isn't in the `add_metal_library`
  SOURCES list, or the metallib wasn't rebuilt.
- **GPU hang / timeout**: `threads` computed as 0 (empty tensor) or
  `per_thread` 0. Guard before dispatch.

## Commit split

1. `third_party/nlohmann` + CMake include (vendoring).
2. `manifest.h` + `apex manifest` subcommand.
3. `backend.h`, `metal_backend.*`, `core_weights.h`, `checksum.metal`, new `run_bench`.
