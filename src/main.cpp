#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "profiling/memory.h"
#include "profiling/timer.h"
#include "backend/metal_backend.h"
#include "file.h"
#include "model_info.h"

static int run_bench(const std::string& packed_dir) {
  const auto model_info = apex::ModelInfo::load(packed_dir);
  apex::MetalBackend backend(APEX_METALLIB);
  std::printf("device: %s\n", backend.deviceName());

  // Map core.bin (CPU), let the GPU see the same bytes, then pull every page into RAM.
  apex::MappedFile core = apex::mapFile(model_info.dir + "/core.bin", "core_weights");
  if (core.size != model_info.core_size) {
    throw std::runtime_error("core.bin size != manifest");
  }
  apex::Buffer* core_buffer = backend.wrapNoCopy(core.data, core.size);
  apex::touchAll(core);

  // Print while core_weights is still mapped and counted in the ledger.
  apex::Profile::get().print(stdout);
  apex::MemoryTracker::get().print(stdout);

  // GPU lets go of core.bin before it is unmapped.
  backend.release(core_buffer);
  apex::unmapFile(core);
  return 0;
}

/// Loads and prints packed model metadata without reading weight files.
static int run_model_info(const std::string& packed_dir) {
    const auto model_info = apex::ModelInfo::load(packed_dir);

    std::printf("core tensors: %zu\n", model_info.core.size());

    // Cast to match %llu, which expects unsigned long long.
    std::printf("core bytes: %llu\n",
                static_cast<unsigned long long>(model_info.core_size));
    std::printf("expert stride: %llu\n",
                static_cast<unsigned long long>(model_info.expert_slot_size));
    std::printf("norm.weight offset: %llu\n",
                static_cast<unsigned long long>(model_info.getTensor("norm.weight").offset));

    return 0;  // Successful command.
}

// Run the RMSNorm parity test.
int run_test_norm(const std::string &packed_dir, const std::string &dump_dir); 

// Run the dense MLP parity test.
int run_test_mlp(const std::string &packed_dir, const std::string &dump_dir);

int main(int argc, char** argv) {
  if (argc == 4 && std::strcmp(argv[1], "test-norm") == 0) {
    return run_test_norm(argv[2], argv[3]);
  }
  if (argc == 4 && std::strcmp(argv[1], "test-mlp") == 0) {
    return run_test_mlp(argv[2], argv[3]);
  }
  if (argc == 3 && std::strcmp(argv[1], "bench") == 0) return run_bench(argv[2]);
  if (argc == 3 && std::strcmp(argv[1], "manifest") == 0) {
    return run_model_info(argv[2]);
  }
  std::fprintf(stderr,
             "usage: apex bench <packed_dir> | apex manifest <packed_dir> | "
             "apex test-norm <packed_dir> <dump_dir> | apex test-mlp <packed_dir> <dump_dir>\n");
  return 2;
}
