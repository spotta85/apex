# APEX Status Summary

Last updated: 2026-05-03

## Current State

APEX now has one main MoE orchestration path:

```text
Runtime
  -> router.route(layer, hidden) -> (expert_ids, weights_or_None)
  -> run_moe_layer(...)
  -> ExpertManager.ensure_loaded(...)
  -> ExpertCache / ExpertStore
  -> Backend.run_expert(...)
```

Fake demos still use uniform combine through `weights=None`. Real Gemma
runtime mode can use router softmax weights through `MultiLayerRealRouter`.

## Implemented So Far

### Phase 1 — Cleanup

Completed.

- Removed dead `Runtime.verbose_router`.
- Hoisted shared synthetic NVFP4 safetensors test helpers into
  `tests/_synthetic_nvfp4.py`.
- Hoisted real-Gemma script bootstrap into `scripts/_real_moe_helpers.py`.
- Removed duplicated helper code from real-cache, real-two-layer,
  quantized-store, and safetensors-loader tests.

Verification:

- Targeted cleanup tests: `22 passed, 2 skipped`.
- Full suite after Phase 1: `212 passed, 2 skipped`.
- All 7 fake demos were byte-identical to the pre-cleanup baseline.
- Real script CLIs imported cleanly via `--help`.

### Phase 2 — Re-validate Steps 10 and 11

Completed.

Steps 10 and 11 were already implemented before this sprint:

- Step 10: `DequantizingExpertStore` + `scripts/demo_real_moe_cache.py`.
- Step 11: `scripts/smoke_real_moe_two_layer.py`.

Real checkpoint validation was run against:

```text
/Users/spotta/Desktop/models/gemma4/Gemma-4-26B-A4B-NVFP4
```

Verification:

```bash
REAL_MOE_PATH=/Users/spotta/Desktop/models/gemma4/Gemma-4-26B-A4B-NVFP4 \
  uv run pytest \
    tests/test_real_moe_cache.py::test_real_gemma_layer_caches_top_8_then_reuses \
    tests/test_real_moe_two_layer.py::test_real_gemma_two_layer_pass_caches_correctly -v
# Result: 2 passed

uv run python scripts/smoke_real_moe_layer.py \
  --path /Users/spotta/Desktop/models/gemma4/Gemma-4-26B-A4B-NVFP4
# Result: passed; output shape (1, 2816), finite=True.

uv run python scripts/demo_real_moe_cache.py \
  --path /Users/spotta/Desktop/models/gemma4/Gemma-4-26B-A4B-NVFP4
# Result: passed; 8 misses, 8 hits, 0 evictions.

uv run python scripts/smoke_real_moe_two_layer.py \
  --path /Users/spotta/Desktop/models/gemma4/Gemma-4-26B-A4B-NVFP4
# Result: passed; first pass 16 misses, second pass 16 hits, 0 evictions.
```

### Phase 3 — Runtime Drives Real Gemma MoE Layers

Completed in commit `40d40f9`.

Core changes:

- Router contract changed from `ids` to `(ids, weights)`.
- Existing fake/npy/safetensors/GGUF/MLX routers return `(ids, None)`.
- `weights=None` means legacy uniform mean combine.
- `Runtime` now calls `apex.moe_layer.run_moe_layer`.
- `Runtime` accepts optional `initial_hidden`.
- `backend.combine_outputs` was removed from NumPy and MLX backends.
- Added `apex.real_router.MultiLayerRealRouter`.
- Added `scripts/demo_real_runtime.py`.
- Added `tests/test_runtime_real_integration.py`.

Important implementation choice:

- `run_moe_layer` itself handles `weights=None` by using the exact old
  formula: `sum(outputs) / len(outputs)`.
- This keeps fake-demo output byte-identical while avoiding another
  combine path inside `Runtime`.

Verification:

```bash
uv run pytest tests/test_moe_layer.py tests/test_gate_router.py \
  tests/test_safetensors_gate_router.py tests/test_gguf_gate_router.py \
  tests/test_mlx_gate_router.py tests/test_runtime_with_gate_router.py \
  tests/test_mlx_hidden_flow.py tests/test_runtime_real_integration.py -q
# Result: 50 passed, 1 skipped

uv run pytest -q
# Result: 217 passed, 3 skipped
```

Fake-demo compatibility:

- All 7 fake demos are byte-identical to the pre-Phase-3 baseline:
  - `demo.py`
  - `demo_predictor.py`
  - `demo_gate_router.py`
  - `demo_safetensors.py`
  - `demo_gguf.py`
  - `demo_mmap_store.py`
  - `demo_mlx_backend.py`

Real-runtime verification status:

- NVIDIA-style `GemmaMoELayout` remains supported.
- The lowercase local checkpoint is handled by Step 13 below.

### Step 13 — Current Local Gemma Checkpoint Support

Implemented.

The currently available checkpoint is:

```text
/Users/spotta/Desktop/models/gemma4/gemma-4-26b-a4b-it-nvfp4
```

It uses lowercase HF/MLX names and stacked expert tensors, so it is not
compatible with the NVIDIA-style `GemmaMoELayout`.

Core changes:

- Added `GemmaHFMoELayout` under layout key `gemma-hf-moe`.
- Updated `validate_layout.py` to understand stacked packed experts.
- Added U32-packed NVFP4 expert dequantization.
- Added U32-packed affine uint8 router dequantization.
- Added `GemmaHFQuantizedExpertStore`, which slices one expert from the
  stacked layer tensor before dequantization.
- Updated `MultiLayerRealRouter` to dequantize quantized router
  projections when the layout exposes router sidecars.
- Updated `scripts/demo_real_runtime.py` to auto-detect `gemma-moe` vs
  `gemma-hf-moe`.

Verification:

```bash
uv run python scripts/validate_layout.py \
  /Users/spotta/Desktop/models/gemma4/gemma-4-26b-a4b-it-nvfp4 \
  --layout gemma-hf-moe --check-shapes
# Result: layout validation passed; detected 30 layers x 128 experts.

uv run python scripts/demo_real_runtime.py \
  --path /Users/spotta/Desktop/models/gemma4/gemma-4-26b-a4b-it-nvfp4
# Result: first run 16 misses; second run 16 hits; finite output.

uv run python scripts/smoke_real_moe_layer.py \
  --path /Users/spotta/Desktop/models/gemma4/gemma-4-26b-a4b-it-nvfp4
uv run python scripts/demo_real_moe_cache.py \
  --path /Users/spotta/Desktop/models/gemma4/gemma-4-26b-a4b-it-nvfp4
uv run python scripts/smoke_real_moe_two_layer.py \
  --path /Users/spotta/Desktop/models/gemma4/gemma-4-26b-a4b-it-nvfp4
# Result: all passed with auto-detected gemma-hf-moe layout.

REAL_MOE_PATH=/Users/spotta/Desktop/models/gemma4/gemma-4-26b-a4b-it-nvfp4 \
  uv run pytest \
  tests/test_runtime_real_integration.py::test_real_gemma_two_layer_runtime_run_then_run -q
# Result: 1 passed

uv run pytest -q
# Result: 226 passed, 3 skipped

REAL_MOE_PATH=/Users/spotta/Desktop/models/gemma4/gemma-4-26b-a4b-it-nvfp4 \
  uv run pytest -q
# Result: 229 passed
```

## What This Means

APEX has moved from two orchestration paths to one:

- Before Phase 3:
  - Fake demos used `Runtime`.
  - Real Gemma scripts bypassed `Runtime` and manually called
    `run_moe_layer`.

- After Phase 3:
  - Fake demos still use `Runtime`.
  - Real Gemma MoE layers can also go through `Runtime`.
  - `run_moe_layer` is the shared compute path for fake and real MoE.

This gets APEX to:

```text
real Gemma MoE blocks through Runtime
```

It does not yet make APEX a full Gemma text-generation runtime.

## Still Not Implemented

Full Gemma generation remains out of scope. Missing pieces:

- Attention.
- RMSNorm.
- Residual wiring.
- Embeddings.
- LM head.
- KV cache.
- Tokenizer/sampling loop.
- Shared dense experts.
- Gemma router scale / per-expert-scale handling.
- MLX-native dequantized real-weight path.
- Predictor/prefetch traces from real model execution.

## Next Recommended Steps

1. Commit Step 13 after review.

Suggested commit message:

```text
step 13: support lowercase Gemma MoE checkpoint
```

2. Decide the next product direction:

- If the goal is stronger MoE-runtime correctness: add Gemma router
  scale / per-expert-scale support.
- If the goal is performance: add MLX-native dequantized real experts.
- If the goal is full model inference: plan attention, norm, residual,
  embeddings, lm_head, KV cache, and sampling as a separate sprint.

## Current Known Limitation

- The local lowercase checkpoint now runs through `Runtime`, but APEX
  still ignores Gemma's `router.scale` and `router.per_expert_scale`
  tensors.
- The always-on dense/shared MLP path is still not executed.
- This is still MoE-block execution, not full Gemma text generation.
