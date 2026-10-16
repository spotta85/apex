# APEX

A low-memory Mixture-of-Experts inference engine for Apple Silicon, written in
C++20 and Metal. The target model is Gemma 4 26B-A4B (NVFP4, 30 layers,
128 experts per layer, top-8 routing).

The core idea: the user sets a memory budget and the engine stays under it.
Only the shared ("core") weights live in memory. Experts are streamed from
disk into preallocated slots on demand, cached per layer, and evicted when
the budget is tight. Unified memory means a `pread` lands bytes exactly
where the GPU will read them, with no upload step.

## Status

Currently writing the Attention Kernels. 

## Build

Requirements: Xcode 27 with the Metal toolchain
(`xcodebuild -downloadComponent MetalToolchain`), CMake, Ninja,
and [metal-cpp](https://developer.apple.com/metal/cpp/) unzipped at
`third_party/metal-cpp`.

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/apex
```

Build shortcut: run `python3 tools/run.py`, or open `tools/run.py` and use
the editor's Python Run button. It configures and builds Release; run
`./build/apex` separately to execute the GPU smoke test.



## Layout

```
src/        engine (C++20); metal_impl.cpp is the one TU that instantiates metal-cpp
shaders/    Metal kernels, compiled to one .metallib at build time
cmake/      add_metal_library(): .metal -> .air -> .metallib
tools/      Python (uv): reference dumps from mlx-lm, repacker, plots
docs/       EXPERIMENTS.md, the log of every optimization tried, including failures
```

## Plan

A. Foundations: skeleton, reference dumps, repacker, measurement.
B. Correct engine: mmap'd core weights, kernels, layer-by-layer parity, CLI.
C. Low memory: expert streaming, per-layer cache, hard memory limit.
D. Speed: dispatch overhead, disk/GPU overlap, fusion, prefill, GPU sampling.
E. Proof: benchmarks against TurboFieldfare, llama.cpp, mlx-lm.
F. Expansion: Qwen3.6-35B-A3B, Intel iGPU backend.

Rules for section D: one change at a time, parity before benchmark, log every
experiment, let the profile pick the next step.


