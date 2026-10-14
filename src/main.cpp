#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "profiling/memory.h"
#include "profiling/timer.h"
#include "backend/metal_backend.h"
#include "core_weights.h"
#include "manifest.h"

static int run_bench(const std::string& packed_dir) {
  const auto manifest = apex::Manifest::load(packed_dir);
  apex::MetalBackend backend(APEX_METALLIB);
  std::printf("device: %s\n", backend.deviceName());

  // One mapping and one GPU buffer for the entire core file.
  apex::CoreWeights core(manifest, backend);
  core.touchAll();

  // Print while core_weights is still alive and counted in the ledger.
  apex::Profile::get().print(stdout);
  apex::MemoryTracker::get().print(stdout);
  return 0;
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

// Run the RMSNorm parity test.
int run_test_norm(const std::string &packed_dir, const std::string &dump_dir); 

int main(int argc, char** argv) {
  if (argc == 4 && std::strcmp(argv[1], "test-norm") == 0) {
    return run_test_norm(argv[2], argv[3]);
  }
  if (argc == 3 && std::strcmp(argv[1], "bench") == 0) return run_bench(argv[2]);
  if (argc == 3 && std::strcmp(argv[1], "manifest") == 0) {
    return run_manifest(argv[2]);
  }
  std::fprintf(stderr,
             "usage: apex bench <packed_dir> | apex manifest <packed_dir> | "
             "apex test-norm <packed_dir> <dump_dir>\n");
  return 2;
}
