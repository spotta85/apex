#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "profiling/memory.h"
#include "profiling/timer.h"
#include "backend/metal_backend.h"
#include "file.h"
#include "model_info.h"
#include "generate.h"
#include "model.h"
#include "npy.h"

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

// Attention parity test (dumps) and the synthetic sliding-window test.
int run_test_attn(const std::string &packed_dir, const std::string &dump_dir);
int run_test_attn_synth(const std::string &packed_dir);

// Router parity test (dumps, core weights only).
int run_test_router(const std::string &packed_dir, const std::string &dump_dir);
// Expert parity test (dumps; routed experts pread from one layer file at a time).
int run_test_moe(const std::string &packed_dir, const std::string &dump_dir);
// Embedding, residual/tail row kernels, output head and argmax (dumps, core only).
int run_test_tail(const std::string &packed_dir, const std::string &dump_dir);

// Full forward pass in layer-streamed mode vs the dumps (core + one layer file at a time).
int run_test_forward(const std::string &packed_dir, const std::string &dump_dir, int prefill);

// Parse "1,2,3" (commas and/or whitespace) into token ids.
static std::vector<uint32_t> parseIdList(const std::string& text) {
  std::vector<uint32_t> ids;
  std::string token;
  auto flush = [&]() {
    if (token.empty()) return;
    size_t used = 0;
    const unsigned long v = std::stoul(token, &used);
    if (used != token.size()) throw std::runtime_error("bad token id: " + token);
    ids.push_back(static_cast<uint32_t>(v));
    token.clear();
  };
  for (char ch : text) {
    if (ch == ',' || std::isspace(static_cast<unsigned char>(ch))) flush();
    else token += ch;
  }
  flush();
  return ids;
}

// --ids takes a file (.npy of int64, or text with comma/space separated ids) or a literal list.
static std::vector<uint32_t> loadIds(const std::string& arg) {
  std::ifstream in(arg, std::ios::binary);
  if (!in) return parseIdList(arg);
  if (arg.size() > 4 && arg.compare(arg.size() - 4, 4, ".npy") == 0) {
    const apex::Npy n = apex::loadNpy(arg);
    if (n.dtype != "<i8") throw std::runtime_error("--ids .npy must be int64");
    std::vector<uint32_t> ids;
    for (size_t i = 0; i < n.count(); ++i) ids.push_back(static_cast<uint32_t>(n.i64()[i]));
    return ids;
  }
  std::stringstream text;
  text << in.rdbuf();
  return parseIdList(text.str());
}

// apex generate <packed_dir> --ids <file|list> [--max-tokens N] [--streamed]
//                [--max-context N] [--stop ids]
// Greedy generation. Prints the generated ids and tokens/second. Decode text with
// `uv run tools/chat_ids.py decode <ids>`.
static int run_generate(int argc, char** argv) {
  if (argc < 3) throw std::runtime_error("generate: missing <packed_dir>");
  const std::string packed_dir = argv[2];
  std::string ids_arg;
  uint32_t max_new = 64;
  apex::ModelOptions options;
  std::vector<uint32_t> stop{1, 106, 50};  // generation_config.json eos_token_id
  for (int i = 3; i < argc; ++i) {
    const std::string a = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
      return argv[++i];
    };
    if (a == "--ids") ids_arg = value();
    else if (a == "--max-tokens") max_new = static_cast<uint32_t>(std::stoul(value()));
    else if (a == "--max-context") options.max_context = static_cast<uint32_t>(std::stoul(value()));
    else if (a == "--streamed") options.residency = apex::ExpertResidency::LayerStreamed;
    else if (a == "--stop") stop = parseIdList(value());
    else throw std::runtime_error("generate: unknown option " + a);
  }
  if (ids_arg.empty()) throw std::runtime_error("generate: --ids is required");
  const std::vector<uint32_t> prompt = loadIds(ids_arg);
  if (prompt.empty()) throw std::runtime_error("generate: no prompt ids");
  options.max_context = std::max<uint32_t>(options.max_context,
                                           static_cast<uint32_t>(prompt.size()) + max_new);

  const auto info = apex::ModelInfo::load(packed_dir);
  apex::MetalBackend backend(APEX_METALLIB);
  std::printf("device: %s  experts: %s  prompt: %zu tokens  max new: %u  context: %u\n",
              backend.deviceName(),
              options.residency == apex::ExpertResidency::Resident ? "resident" : "layer-streamed",
              prompt.size(), max_new, options.max_context);
  apex::Model model(backend, info, options);

  apex::GenerateStats stats;
  const std::vector<uint32_t> out = apex::generateGreedy(model, prompt, max_new, stop, stats);

  std::printf("ids:");
  for (size_t i = 0; i < out.size(); ++i) std::printf("%s%u", i ? "," : " ", out[i]);
  std::printf("\n");
  std::printf("prefill: %u tokens in %.1f ms (%.1f tok/s)\n", stats.prompt_tokens,
              stats.prefill_ms, stats.prompt_tokens * 1000.0 / stats.prefill_ms);
  // The first generated token comes out of the prefill; decode covers the rest.
  if (stats.generated > 1) {
    std::printf("decode: %u tokens in %.1f ms (%.2f tok/s)\n", stats.generated - 1,
                stats.decode_ms, (stats.generated - 1) * 1000.0 / stats.decode_ms);
  }
  apex::Profile::get().print(stdout);
  apex::MemoryTracker::get().print(stdout);
  return 0;
}

static int dispatch(int argc, char** argv) {
  if (argc == 4 && std::strcmp(argv[1], "test-norm") == 0) {
    return run_test_norm(argv[2], argv[3]);
  }
  if (argc == 4 && std::strcmp(argv[1], "test-mlp") == 0) {
    return run_test_mlp(argv[2], argv[3]);
  }
  if (argc == 4 && std::strcmp(argv[1], "test-attn") == 0) {
    return run_test_attn(argv[2], argv[3]);
  }
  if (argc == 3 && std::strcmp(argv[1], "test-attn-synth") == 0) {
    return run_test_attn_synth(argv[2]);
  }
  if (argc == 4 && std::strcmp(argv[1], "test-router") == 0) {
    return run_test_router(argv[2], argv[3]);
  }
  if (argc == 4 && std::strcmp(argv[1], "test-moe") == 0) {
    return run_test_moe(argv[2], argv[3]);
  }
  if (argc == 4 && std::strcmp(argv[1], "test-tail") == 0) {
    return run_test_tail(argv[2], argv[3]);
  }
  if ((argc == 4 || argc == 5) && std::strcmp(argv[1], "test-forward") == 0) {
    return run_test_forward(argv[2], argv[3], argc == 5 ? std::atoi(argv[4]) : 0);
  }
  if (argc >= 3 && std::strcmp(argv[1], "generate") == 0) return run_generate(argc, argv);
  if (argc == 3 && std::strcmp(argv[1], "bench") == 0) return run_bench(argv[2]);
  if (argc == 3 && std::strcmp(argv[1], "manifest") == 0) {
    return run_model_info(argv[2]);
  }
  std::fprintf(stderr,
             "usage:\n"
             "  apex generate <packed_dir> --ids <file|list> [--max-tokens N] [--streamed]\n"
             "                [--max-context N] [--stop id,id]\n"
             "  apex bench <packed_dir> | apex manifest <packed_dir>\n"
             "  apex test-norm|test-mlp|test-attn|test-router|test-moe|test-tail <packed_dir> <dump_dir>\n"
             "  apex test-attn-synth <packed_dir>\n"
             "  apex test-forward <packed_dir> <dump_dir> [prefill_tokens]   (layer-streamed)\n");
  return 2;
}

int main(int argc, char** argv) {
  try {
    return dispatch(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
