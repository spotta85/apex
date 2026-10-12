#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "profiling/memory.h"
#include "profiling/timer.h"
#include "backend/metal_backend.h"
#include "core_weights.h"
#include "manifest.h"

/// Convert C string to apple string. 
static NS::String* str(const char* s) { 
    return NS::String::string(s, NS::UTF8StringEncoding); 
}

int run_smoke() {
  // For autoreleased objects (commandBuffer(), name(), ...)
  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

  auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  std::printf("device: %s  unified memory: %d\n",
              device->name()->utf8String(), (int)device->hasUnifiedMemory());

  NS::Error* err = nullptr;
  auto lib = NS::TransferPtr(device->newLibrary(NS::URL::fileURLWithPath(str(APEX_METALLIB)), &err));
  if (!lib) { std::fprintf(stderr, "metallib: %s\n", err->localizedDescription()->utf8String()); return 1; }
  auto fn  = NS::TransferPtr(lib->newFunction(str("add")));
  auto pso = NS::TransferPtr(device->newComputePipelineState(fn.get(), &err));
  if (!pso) { std::fprintf(stderr, "pipeline: %s\n", err->localizedDescription()->utf8String()); return 1; }

  const size_t n = 1 << 20;
  const size_t bytes = n * sizeof(float);
  // Shared: one allocation visible to both CPU and GPU. No copies, ever.
  auto a   = NS::TransferPtr(device->newBuffer(bytes, MTL::ResourceStorageModeShared));
  auto b   = NS::TransferPtr(device->newBuffer(bytes, MTL::ResourceStorageModeShared));
  auto out = NS::TransferPtr(device->newBuffer(bytes, MTL::ResourceStorageModeShared));
  float* pa = static_cast<float*>(a->contents());
  float* pb = static_cast<float*>(b->contents());
  for (size_t i = 0; i < n; ++i) { pa[i] = float(i); pb[i] = 1.0f; }

  auto queue = NS::TransferPtr(device->newCommandQueue());
  MTL::CommandBuffer* cmd = queue->commandBuffer();          // autoreleased
  MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
  enc->setComputePipelineState(pso.get());
  enc->setBuffer(a.get(), 0, 0);
  enc->setBuffer(b.get(), 0, 1);
  enc->setBuffer(out.get(), 0, 2);
  // dispatchThreads lets Metal handle the ragged last threadgroup for us.
  enc->dispatchThreads(MTL::Size::Make(n, 1, 1),
                       MTL::Size::Make(pso->maxTotalThreadsPerThreadgroup(), 1, 1));
  enc->endEncoding();
  cmd->commit();
  cmd->waitUntilCompleted();

  const float* po = static_cast<const float*>(out->contents());
  size_t bad = 0;
  for (size_t i = 0; i < n; ++i) if (po[i] != float(i) + 1.0f) ++bad;
  std::printf("%s (%zu mismatches)\n", bad ? "FAIL" : "ok", bad);

  pool->release();
  return bad ? 1 : 0;
}

static int run_bench(const std::string& packed_dir) {
  const auto manifest = apex::Manifest::load(packed_dir);
  apex::MetalBackend backend(APEX_METALLIB);
  std::printf("device: %s\n", backend.deviceName());

  // One mapping and one GPU buffer for the entire core file.
  apex::CoreWeights core(manifest, backend);
  core.touchAll();

  const char* name = "embed_tokens.weight";
  const auto view = core.tensor(name);
  const std::uint32_t gpu = backend.checkSumU32(view.buffer, view.offset, view.size);

  // Sum the same raw words on the CPU, with the same unsigned wraparound.
  std::uint32_t cpu = 0;
  {
    apex::Timer timer("cpu_checksum");
    const auto* words = static_cast<const std::uint32_t*>(core.cpu(name));
    for (std::size_t i = 0; i < view.size / sizeof(*words); ++i) cpu += words[i];
  }
  std::printf("%s: gpu %08x cpu %08x %s\n", name,
              static_cast<unsigned int>(gpu), static_cast<unsigned int>(cpu),
              gpu == cpu ? "MATCH" : "MISMATCH");

  // Print while core_weights is still alive and counted in the ledger.
  apex::Profile::get().print(stdout);
  apex::MemoryTracker::get().print(stdout);
  return gpu == cpu ? 0 : 1;
}

/// Loads and prints packed model metadata without reading weight files.
static int run_manifest(const std::string& packed_dir) {
    const auto m = apex::Manifest::load(packed_dir);

    std::printf("core tensors: %zu\n", m.core.size());

    // Cast to match %llu, which expects unsigned long long.
    std::printf("core bytes: %llu\n",
                static_cast<unsigned long long>(m.core_size));
    std::printf("expert stride: %llu\n",
                static_cast<unsigned long long>(m.expert_slot_size));
    std::printf("norm.weight offset: %llu\n",
                static_cast<unsigned long long>(m.tensor("norm.weight").offset));

    return 0;  // Successful command.
}

int main(int argc, char** argv) {
  if (argc >= 2 && std::strcmp(argv[1], "smoke") == 0) return run_smoke();
  if (argc == 3 && std::strcmp(argv[1], "bench") == 0) return run_bench(argv[2]);
  if (argc == 3 && std::strcmp(argv[1], "manifest") == 0) {
    return run_manifest(argv[2]);
  }
  std::fprintf(stderr,
             "usage: apex smoke | apex bench <packed_dir> | apex manifest <packed_dir>\n");
  return 2;
}