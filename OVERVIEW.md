# APEX — Overview

APEX is a from-scratch Mixture-of-Experts (MoE) inference runtime focused on **bounded-memory expert loading**: the user sets `max_memory_mb` and APEX loads experts on demand, caches them LRU, and evicts when over budget. Built up in tightly-scoped MVP steps; now runs real Gemma 4 NVFP4 weights end-to-end through a single MoE layer.

## How to test

```bash
# Full unit suite (200 tests, ~1s)
uv run pytest -q

# Seven fake-MoE demos — each exercises a different combination of
# store / router / backend; all share the same runtime + cache.
uv run python demo.py                 # FakeRouter, npy store
uv run python demo_predictor.py       # + Markov predictor + prefetcher
uv run python demo_gate_router.py     # real npy gate router
uv run python demo_safetensors.py     # SafetensorsExpertStore + gate router
uv run python demo_gguf.py            # GGUFExpertStore + gate router
uv run python demo_mmap_store.py      # MMapExpertStore (np.memmap)
uv run python demo_mlx_backend.py     # MLXBackend (Apple Silicon)

# Real-checkpoint smoke tests — need ~17 GB Gemma 4 NVFP4 download
uv run python scripts/smoke_real_moe_layer.py        # one layer
uv run python scripts/smoke_real_moe_two_layer.py    # two layers in sequence
uv run python scripts/demo_real_moe_cache.py         # load-once / reuse-many
```

Diagnostic scripts:

```bash
uv run python scripts/inspect_weight_names.py <path>          # list tensor names
uv run python scripts/validate_layout.py <path> --layout gemma-moe [--check-shapes]
```

---

## Architecture diagram

```
                     ┌──────────────────────────────┐
                     │           Runtime            │
                     │  (per-layer route + execute) │
                     └──────────┬───────────────────┘
                                │
              ┌─────────────────┼─────────────────────┐
              ▼                 ▼                     ▼
      ┌─────────────┐   ┌─────────────────┐   ┌─────────────┐
      │   Router    │   │  ExpertManager  │   │ Predictor + │
      │ (selects    │   │ (load + prepare │   │ Prefetcher  │
      │  experts)   │   │  + cache lookup)│   │ (warm cache)│
      └─────┬───────┘   └────┬────────────┘   └─────────────┘
            │                │
            │                ▼
            │         ┌──────────────┐    ┌──────────────────┐
            │         │ ExpertCache  │◄───│ Backend.prepare_ │
            │         │ (LRU bytes)  │    │     expert       │
            │         └──────┬───────┘    └──────────────────┘
            │                │
            │                ▼
            │         ┌──────────────┐
            │         │ Backend.run_ │ ── SwiGLU FFN
            │         │   expert     │
            │         └──────────────┘
            │
            ▼
   ┌────────────────────┐         ┌─────────────────────┐
   │  *GateRouter       │ reads ──│   *ExpertStore      │
   │  (npy / safe / gguf│         │   (npy / safe / gguf│
   │   / mlx)           │         │    / mmap / quant)  │
   └────────┬───────────┘         └─────────┬───────────┘
            │                               │
            └─────────────┬─────────────────┘
                          ▼
                ┌──────────────────────┐
                │   weight_layout.py   │  ← tensor-name strings
                │ (Fake/Mixtral/Qwen/  │     (the only thing
                │  Gemma)              │      that varies per model)
                └──────────────────────┘
```

For real quantized checkpoints the per-expert path looks like:

```
QuantizedSafetensorsExpertStore.load(L,E)
        │  returns ExpertWeights of QuantizedTensor (NVFP4 packed uint8 + scales)
        ▼
dequantize_bundle(...)
        │  returns ExpertWeights of fp32 ndarrays
        ▼
FakeBackend.run_expert(hidden, bundle)
        │  SwiGLU: (silu(h@gate.T) * (h@up.T)) @ down.T
        ▼
weighted-combine via apex.moe_layer.run_moe_layer
```

---

## Module map

### `apex/` — runtime core

| File | What it does | Key exports |
|------|--------------|-------------|
| `runtime.py` | Per-layer loop: route → ensure_loaded → run_expert → combine | `Runtime` |
| `expert_manager.py` | Cache lookup; on miss → store.load → backend.prepare → cache.put | `ExpertManager.ensure_loaded` |
| `expert_cache.py` | Thread-safe LRU keyed on (layer, expert), bounded by `max_memory_mb` | `ExpertCache` |
| `prefetcher.py` | Background thread that warms the cache from predictor hints | `Prefetcher` |
| `predictor.py` | Markov predictor over `(layer, experts) → next_experts` | `MarkovPredictor` |
| `router.py` | `FakeRouter` returns hand-coded picks (no math) | `FakeRouter` |
| `gate_router.py` | Real top-k routing via `hidden @ gate.npy` | `GateRouter` |
| `metrics.py` | Counters (cache_hit/miss/eviction, prefetch_hit/miss) | `Metrics` |
| `trace_recorder.py` | Append-only JSONL transition log for predictor training | `TraceRecorder` |
| `types.py` | `ExpertWeights(gate_proj, up_proj, down_proj)`, `RouteResult` | dataclasses |

### `apex/` — MoE math (step 9)

| File | What it does | Key exports |
|------|--------------|-------------|
| `moe_layer.py` | `route()` (top-k softmax) + `run_moe_layer()` (per-expert SwiGLU + weighted combine) | `route`, `run_moe_layer` |

### `apex/backends/` — compute

| File | What it does | Notes |
|------|--------------|-------|
| `fake_backend.py` | NumPy SwiGLU: `(silu(h@gate.T)*(h@up.T))@down.T` | Default, passthrough conversions |
| `mlx_backend.py` | Same math via `mlx.core.matmul` | Apple Silicon; converts to `mx.array` once in `prepare_expert` |

### `apex/` — on-disk format adapters

| File | Format | Notes |
|------|--------|-------|
| `expert_store.py` | npy (eager `np.load`) | Default fake-MoE store |
| `mmap_expert_store.py` | npy (mmap) | OS pages in on matmul |
| `safetensors_expert_store.py` | safetensors (single file) | uses `safetensors.numpy` |
| `gguf_expert_store.py` | GGUF (mmap-backed) | fp32 unquantized only |
| `quantized_safetensors_store.py` | NVFP4 sharded safetensors | returns `QuantizedTensor` bundles |
| `safetensors_gate_router.py` / `gguf_gate_router.py` / `mlx_gate_router.py` | Router-weight loaders | Same `route(layer, hidden)` interface |

### `apex/` — naming + format helpers

| File | What it does |
|------|--------------|
| `weight_layout.py` | `FakeMoELayout`, `MixtralLayout`, `QwenMoELayout`, `GemmaMoELayout` + `get_layout(name)` registry |
| `checkpoint_index.py` | Metadata-only queries: `list_tensor_names`, `get_tensor_shape`, `read_safetensors_index` (handles single file / sharded dir / gguf) |
| `safetensors_loader.py` | `load_tensor(path, name)` — manual header parsing so BF16 / F8 work without torch |
| `gguf_tensor_index.py` | Lightweight wrapper around `gguf.GGUFReader` with shape-corrected metadata |
| `tensor_types.py` | `RawTensor`, `QuantizedTensor` dataclasses (sidecars for dequant) |
| `tensor_utils.py` | `tensor_nbytes(x)` — single source of truth for cache budget accounting |
| `dequant.py` | `dequantize(t)` + `dequantize_bundle(b)` — NVFP4 (E2M1 lookup + F8_E4M3 block scale + F32 scalar), BF16. Also `DequantizingExpertStore(base)` — wraps any quantized store so the cache holds fp32 bundles |
| `_fake_weights.py` | `write_npy_world` / `write_safetensors_world` / `write_gguf_world` + `make_expert_bundle` + size constants |
| `_numpy_gate_router_base.py` | `BaseNumpyGateRouter` — shared top-k math for the three numpy routers |
| `demo_modes.py` | `RuntimeBundle` dataclass + factory funcs that wire up each demo |

### `scripts/`

| Script | Purpose |
|--------|---------|
| `make_fake_moe_weights.py` / `make_fake_safetensors_moe.py` / `make_fake_gguf_moe.py` | Generate fake-MoE checkpoints + record traces for demos |
| `inspect_weight_names.py` | Print every tensor name in any supported checkpoint |
| `validate_layout.py` | Assert a layout's strings exist in a checkpoint; `--check-shapes` adds SwiGLU consistency |
| `smoke_real_moe_layer.py` | End-to-end one MoE layer of real Gemma 4 NVFP4 (step 9) |
| `demo_real_moe_cache.py` | First-run misses + second-run hits on real Gemma 4 layer 0 (step 10) — ~167x speedup |
| `smoke_real_moe_two_layer.py` | Drives two real MoE layers in sequence; cache key disambiguates by `(layer, expert)` (step 11) |

### `tests/`

31 test modules, 214 tests total. `tests/_world.py` provides `FakeStore` + re-exports the fake-world writers from `apex._fake_weights`. `tests/test_real_moe_cache.py` and `tests/test_real_moe_two_layer.py` each include 1 real-Gemma test that auto-skips unless `REAL_MOE_PATH` (or the default download path) resolves.

---

## What's done (steps 1–11, all on `main`)

1–3. Bounded LRU cache, prefetcher, Markov predictor, metrics.
4. SwiGLU FFN experts (gate/up/down bundles via `ExpertWeights`).
5. Layout adapters (Mixtral, Qwen-MoE, Gemma-MoE) + registry + `inspect_weight_names`.
6. `validate_layout` CLI + `checkpoint_index` library; Gemma layout verified against real NVFP4 checkpoint.
7. NVFP4 dequant + `RawTensor` / `QuantizedTensor` + `safetensors_loader` with BF16/F8 support; real Gemma expert load + dequant + forward.
8. HF (PyTorch) weight orientation throughout — backends compute `h @ proj.T`; no per-load transposes.
9. `apex/moe_layer.py` (`route` + `run_moe_layer`) + `smoke_real_moe_layer.py` end-to-end on real Gemma 4.
10. Real MoE cache integration: `DequantizingExpertStore` + `demo_real_moe_cache.py`. First pass: 8 misses + 8 dequants. Second pass: 8 hits, ~167x speedup.
11. Two-layer real smoke: `smoke_real_moe_two_layer.py`. Cache key disambiguates by `(layer, expert)`; cross-layer cache state survives intact; ~200x cold→warm speedup.

---

## Future / improvements (suggested, not yet committed)

### Near term

- **Plug `apex.moe_layer.run_moe_layer` into `Runtime`** — the runtime today loops `for expert: ensure_loaded; run_expert` with uniform-mean combine. Switching to the real router-weighted combine via `run_moe_layer` removes a layer of duplication and makes the fake demos use the same math the smoke test uses.
- **Real-checkpoint Runtime mode** — `Runtime` itself doesn't yet know how to drive a real checkpoint; the step 10/11 demos wire `ExpertManager` directly. Adding a `RealRuntime` (or extending `Runtime` to take routers + a list of layer ids) closes that last gap.
- **MLX path for real weights** — `MLXBackend` already does SwiGLU; needs a `dequantize` step that produces `mx.array` directly to avoid the numpy round-trip.
- **Predictor + prefetcher for real layers** — step 11 proved 200x speedup once cached; the pre-cache work could happen in the background while attention runs. Wire `Prefetcher` into the demo to overlap layer N+1's expert loads with layer N's compute.

### Medium term

- **Shared experts** — both Qwen MoE and Gemma 4 have an always-on dense FFN per layer alongside routed experts. Needs runtime + cache work, not just naming.
- **Multi-token batches** — `route()` already supports batch dim; runtime + cache assume batch=1.
- **Gemma 4 router scaling** — the model has `router.scale` and `router.per_expert_scale` tensors that bias scoring; APEX's `route()` ignores them. Stub a layout method `router_extras(L)`.
- **GGUF quantized formats** — Q4_0 / Q4_K_M etc. New entry in `dequant.py` + a `QuantizedGGUFExpertStore`.
- **Recovering from cache pressure** — eviction today is pure LRU; could weigh by routing-frequency hints from the predictor.

### Hygiene

- **`scripts/` import boilerplate** — every script does `sys.path.insert(0, ...)`. Could be replaced by making `apex` an installed editable package (`uv pip install -e .`).
- **Replace `np.errstate` wrap in smoke script** with proper bf16-safe matmul (e.g. cast router to fp32 explicitly before the matmul, which is already what `safetensors_loader` does — so the warnings actually come from NVFP4 expert weights spanning a wide dynamic range; investigate whether dequant is producing correct denormals).
- **Single end-to-end real-checkpoint test** — the smoke script currently has no automated verification beyond "all finite + correct shape." Compare against a known-good HuggingFace forward pass for layer 0 to pin down numerical correctness.
