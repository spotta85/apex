#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string>

#include "memory.h"
#include "timer.h"


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
  std::string path = packed_dir + "/core.bin";
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) { std::perror(path.c_str()); return 1; }

  struct stat st {};  // File metadata structure; {} initializes it to zero.
  auto result = fstat(fd, &st);

  if (result != 0) {  std::perror("fstat"); close(fd); return 1;}

  if (st.st_size <= 0) {  std::fprintf(stderr, "core.bin is empty\n"); close(fd); return 1;}

  const size_t size = static_cast<size_t>(st.st_size);  // Convert to mmap's size type.

  void * base; // adress of the mapped file. 

  {
    // measure time to mmap core.bin
    apex::Timer timer("mmap_core");
    base = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
  }

  if (base == MAP_FAILED) { std::perror("mmap"); close(fd); return 1; }
  close(fd);

  apex::MemoryTracker::get().add("core.bin", size); // track mapped bytes

  const long page_size = sysconf(_SC_PAGESIZE);
  const size_t page = static_cast<size_t>(page_size);
  // measure how long it takes to load all pages of file into ram. 
  {
    apex::Timer timer("touch_core");
    const volatile auto* bytes = static_cast<const uint8_t*>(base);

    // touch every page
    for (size_t offset = 0; offset < size; offset += page) {
      (void)bytes[offset];
    }
  }
  
  const size_t pages = size / page + (size % page != 0);
  std::printf("core.bin: %.1f MiB, %zu pages\n",
              size / (1024.0 * 1024.0), pages);
  apex::Profile::get().print(stdout);
  apex::MemoryTracker::get().print(stdout);

  munmap(base, size);
  apex::MemoryTracker::get().remove("core.bin", size);
  return 0;
}

int main(int argc, char** argv) {
  if (argc >= 2 && std::strcmp(argv[1], "smoke") == 0) return run_smoke();
  if (argc >= 3 && std::strcmp(argv[1], "bench") == 0) return run_bench(argv[2]);
  std::fprintf(stderr, "usage: apex smoke | apex bench <packed_dir>\n");
  return 2;
}