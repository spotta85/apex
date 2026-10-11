#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <cstdio>
#include <cstdlib>

static NS::String* str(const char* s) { return NS::String::string(s, NS::UTF8StringEncoding); }

int main() {
  // Getters (commandBuffer(), name(), ...) return autoreleased objects; the pool reclaims them.
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
