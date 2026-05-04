# APEX — Overview

APEX is a from-scratch Mixture-of-Experts (MoE) inference runtime focused on **bounded-memory expert loading**: the user sets `max_memory_mb` and APEX loads experts on demand, caches them LRU, and evicts when over budget. Built up in tightly-scoped MVP steps; now the same `Runtime` used by fake demos can drive real Gemma 4 NVFP4 MoE layers, and step 16 begins building the surrounding model (attention, norms, embeddings) on top of the proven MoE block.

## How to test

```bash
# Full unit suite (304 always-on tests; +17 real-Gemma tests run when
# REAL_MOE_PATH points at a checkpoint)
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
uv run python scripts/demo_real_runtime.py           # real layers through Runtime
uv run python scripts/demo_real_memory_budget.py     # tight budget + LRU eviction
uv run python scripts/demo_real_prefetch.py          # AsyncPrefetcher on real layers
uv run python scripts/smoke_real_attention_layer.py  # single-token attention on layer 0 (step 16C)
uv run python scripts/smoke_real_decoder_layer.py    # full decoder layer (norms+attn+MoE) (step 16D)
uv run python scripts/demo_one_token_generate.py     # embed → decoder → lm_head → argmax (step 16E)
uv run python scripts/demo_generate_with_kv.py       # multi-token gen with KV cache (step 17D)
uv run python scripts/demo_prefill.py                # prompt prefill + decode tail (step 18)
uv run python scripts/demo_tokenizer_generate.py     # text in / text out generation (step 19)
```

Diagnostic scripts:

```bash
uv run python scripts/inspect_weight_names.py <path>          # list tensor names
uv run python scripts/validate_layout.py <path> --layout gemma-moe [--check-shapes] [--check-non-moe]
uv run python scripts/validate_layout.py <path> --layout gemma-hf-moe [--check-shapes] [--check-non-moe]
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
            │         │ run_moe_layer│ ── weighted SwiGLU FFN
            │         │ + Backend    │
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
| `runtime.py` | Per-layer loop: route → `run_moe_layer` via `ExpertManager` | `Runtime` |
| `expert_manager.py` | Cache lookup; on miss → store.load → backend.prepare → cache.put | `ExpertManager.ensure_loaded` |
| `expert_cache.py` | Thread-safe LRU keyed on (layer, expert), bounded by `max_memory_mb` | `ExpertCache` |
| `prefetcher.py` | Background thread that warms the cache from predictor hints | `Prefetcher` |
| `predictor.py` | Markov predictor over `(layer, experts) → next_experts` | `MarkovPredictor` |
| `router.py` | `FakeRouter` returns hand-coded picks (no math) | `FakeRouter` |
| `gate_router.py` | Real top-k routing via `hidden @ gate.npy` | `GateRouter` |
| `real_router.py` | Lazy per-layer router loading/dequant from real sharded safetensors | `MultiLayerRealRouter` |
| `metrics.py` | Counters (cache_hit/miss/eviction, prefetch_hit/miss) | `Metrics` |
| `memory.py` | `MemoryReport` (expert cache) + `KVMemoryReport` (per-layer KV bytes) + `RuntimeMemoryReport` (joined) with `expert_cache_bytes` / `kv_cache_bytes` / `total_runtime_bytes` properties. Three `*_snapshot` constructors | `MemoryReport`, `KVMemoryReport`, `RuntimeMemoryReport`, `snapshot`, `kv_snapshot`, `runtime_snapshot` |
| `trace_recorder.py` | Append-only JSONL transition log for predictor training | `TraceRecorder` |
| `types.py` | `ExpertWeights(gate_proj, up_proj, down_proj)` | dataclass |

### `apex/` — MoE math (step 9)

| File | What it does | Key exports |
|------|--------------|-------------|
| `moe_layer.py` | `route()` (top-k softmax) + `run_moe_layer()` (per-expert SwiGLU + weighted combine) | `route`, `run_moe_layer` |

### `apex/` — non-MoE building blocks (steps 16 + 17)

| File | What it does | Key exports |
|------|--------------|-------------|
| `ops.py` | Tiny NumPy primitives: `rms_norm` (with Gemma `+1` convention), `linear` (HF orientation), `softmax` (numerically stable), `apply_rope` (split-half RoPE) | `rms_norm`, `linear`, `softmax`, `apply_rope` |
| `kv_cache.py` | `KVCache` — per-layer accumulated K/V history. `append` concatenates along seq, `total_bytes` tracks non-evictable KV memory separately from the LRU expert cache | `KVCache` |
| `attention.py` | `attention_decode_step` (canonical, KV cache + RoPE) and `single_token_attention` (thin wrapper for position-0, no-cache smokes) | `attention_decode_step`, `single_token_attention` |
| `decoder.py` | `decoder_decode_step` (canonical, sandwich-norm + KV cache) and `single_token_decoder_layer` (thin wrapper) | `decoder_decode_step`, `single_token_decoder_layer` |
| `generation.py` | `embed_token` (row lookup), `final_logits` (tied lm_head + softcap), `sample_argmax` (greedy) | `embed_token`, `final_logits`, `sample_argmax` |
| `prefill.py` | `prefill(prompt_ids, *, step_fn)` — loop step_fn over prompt tokens, return last prediction (= first generated). Stateless; KV cache lives in step_fn's closure | `prefill` |
| `tokenizer.py` | Thin wrapper over HF `tokenizers` lib. `Tokenizer.from_pretrained(path)` loads `tokenizer.json` + reads BOS id from `config.json`. `encode` prepends BOS by default; `decode` skips special tokens by default | `Tokenizer` |
| `checkpoint_loaders.py` | High-level component loaders: `read_text_config`, `load_embedding`, `load_final_norm`, `load_layer_norms`, `load_attention_weights`. Built on `safetensors_loader` + `dequant` | five loader functions |

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
| `gemma_hf_store.py` | Lowercase HF/MLX Gemma stacked safetensors | slices stacked experts before dequant |
| `safetensors_gate_router.py` / `gguf_gate_router.py` / `mlx_gate_router.py` | Router-weight loaders | Return `(ids, None)` for uniform combine |

### `apex/` — naming + format helpers

| File | What it does |
|------|--------------|
| `weight_layout.py` | `FakeMoELayout`, `MixtralLayout`, `QwenMoELayout`, `GemmaMoELayout`, `GemmaHFMoELayout` + `get_layout(name)` registry. Gemma layouts also expose optional non-MoE methods for attention / norms / embeddings / lm_head (step 16A) |
| `checkpoint_index.py` | Metadata-only queries: `list_tensor_names`, `get_tensor_shape`, `read_safetensors_index` (handles single file / sharded dir / gguf) |
| `safetensors_loader.py` | `load_tensor(path, name)` — manual header parsing so BF16 / F8 work without torch |
| `gguf_tensor_index.py` | Lightweight wrapper around `gguf.GGUFReader` with shape-corrected metadata |
| `tensor_types.py` | `RawTensor`, `QuantizedTensor` dataclasses (sidecars for dequant) |
| `tensor_utils.py` | `tensor_nbytes(x)` — single source of truth for cache budget accounting |
| `dequant.py` | `dequantize(t)` + `dequantize_bundle(b)` — NVFP4 uint8, U32-packed NVFP4, U32-packed affine uint8 router weights, BF16. Also `DequantizingExpertStore(base)` — wraps any quantized store so the cache holds fp32 bundles |
| `_fake_weights.py` | `write_npy_world` / `write_safetensors_world` / `write_gguf_world` + `make_expert_bundle` + size constants |
| `_numpy_gate_router_base.py` | `BaseNumpyGateRouter` — shared top-k math for the three numpy routers |
| `demo_modes.py` | `RuntimeBundle` dataclass + factory funcs that wire up each demo |

### `scripts/`

| Script | Purpose |
|--------|---------|
| `make_fake_moe_weights.py` / `make_fake_safetensors_moe.py` / `make_fake_gguf_moe.py` | Generate fake-MoE checkpoints + record traces for demos |
| `inspect_weight_names.py` | Print every tensor name in any supported checkpoint |
| `validate_layout.py` | Assert a layout's strings exist in a checkpoint; `--check-shapes` adds SwiGLU consistency; `--check-non-moe` probes optional attention / norm / embedding / lm_head names at layer 0 |
| `smoke_real_moe_layer.py` | End-to-end one MoE layer of real Gemma 4 NVFP4 (step 9) |
| `demo_real_moe_cache.py` | First-run misses + second-run hits on real Gemma 4 layer 0 (step 10) — ~167x speedup |
| `smoke_real_moe_two_layer.py` | Drives two real MoE layers in sequence; cache key disambiguates by `(layer, expert)` (step 11) |
| `demo_real_runtime.py` | Drives real Gemma 4 MoE layers through `Runtime` with cold/warm cache checks (step 12) |
| `demo_real_memory_budget.py` | Drives real Gemma under a small `--max-memory-mb`; LRU eviction enforces the budget; final `MemoryReport` reports peak vs budget (step 14) |
| `demo_real_prefetch.py` | Drives real Gemma 2 layers through `Runtime` with `AsyncPrefetcher`; layer 1 served from prefetched cache; reports `prefetch_hit_rate` (step 15) |
| `smoke_real_attention_layer.py` | Loads + dequants real Gemma 4 layer 0 q/k/v/o + q_norm/k_norm; runs single-token attention through `apex.attention.single_token_attention`; asserts (1, 2816) finite output (step 16C) |
| `smoke_real_decoder_layer.py` | Drives layer 0 through `apex.decoder.single_token_decoder_layer`: norms + attention + routed MoE + residuals; asserts finite (1, 2816) output (step 16D) |
| `demo_one_token_generate.py` | End-to-end one-token generation on real Gemma: embedding lookup → N decoder layers → final RMSNorm → tied lm_head + final_logit_softcapping → argmax (step 16E) |
| `demo_generate_with_kv.py` | Multi-token generation: prefill prompt token-by-token through `decoder_decode_step` (KVCache grows monotonically), then decode N more tokens (step 17D) |
| `demo_prefill.py` | Real prompt prefill via `apex.prefill.prefill`: 5-token prompt + 3-token decode tail; per-step KV cache length printout matches the step 18 spec format (step 18) |
| `demo_tokenizer_generate.py` | Text-in / text-out generation: `Tokenizer.encode("Hello world")` → BOS + token ids → prefill + decode loop → `Tokenizer.decode` back to text (step 19) |

### `tests/`

44 test modules, 304 always-on tests + 17 real-Gemma skipif tests (321 with `REAL_MOE_PATH` set). `tests/_world.py` provides `FakeStore` + re-exports the fake-world writers from `apex._fake_weights`. Real-Gemma tests live in `tests/test_real_moe_cache.py`, `tests/test_real_moe_two_layer.py`, `tests/test_runtime_real_integration.py`, `tests/test_real_memory_budget.py`, `tests/test_real_prefetch.py`, `tests/test_real_attention.py`, `tests/test_real_decoder.py`, and `tests/test_real_generation.py` — all auto-skip unless `REAL_MOE_PATH` (or a known local default path) resolves.

---

## What's done (steps 1–19)

1–3. Bounded LRU cache, prefetcher, Markov predictor, metrics.
4. SwiGLU FFN experts (gate/up/down bundles via `ExpertWeights`).
5. Layout adapters (Mixtral, Qwen-MoE, Gemma-MoE) + registry + `inspect_weight_names`.
6. `validate_layout` CLI + `checkpoint_index` library; Gemma layout verified against real NVFP4 checkpoint.
7. NVFP4 dequant + `RawTensor` / `QuantizedTensor` + `safetensors_loader` with BF16/F8 support; real Gemma expert load + dequant + forward.
8. HF (PyTorch) weight orientation throughout — backends compute `h @ proj.T`; no per-load transposes.
9. `apex/moe_layer.py` (`route` + `run_moe_layer`) + `smoke_real_moe_layer.py` end-to-end on real Gemma 4.
10. Real MoE cache integration: `DequantizingExpertStore` + `demo_real_moe_cache.py`. First pass: 8 misses + 8 dequants. Second pass: 8 hits, ~167x speedup.
11. Two-layer real smoke: `smoke_real_moe_two_layer.py`. Cache key disambiguates by `(layer, expert)`; cross-layer cache state survives intact; ~200x cold→warm speedup.
12. Runtime integration: routers return `(ids, weights)`, `Runtime` calls `run_moe_layer`, accepts `initial_hidden`, and `MultiLayerRealRouter` lets `scripts/demo_real_runtime.py` drive real Gemma MoE layers through the main orchestrator.
13. Lowercase HF/MLX Gemma checkpoint support: `GemmaHFMoELayout`, stacked expert slicing via `GemmaHFQuantizedExpertStore`, U32-packed expert/router dequantization, layout auto-detection in `demo_real_runtime.py`, and real Runtime validation against the local `gemma-4-26b-a4b-it-nvfp4` checkpoint.
14. Real memory-budget enforcement: `ExpertCache.peak_bytes` high-water tracking + `apex/memory.py` (`MemoryReport`, `snapshot`) joining cache/budget/peak with hits/misses/evictions and a `budget_respected` predicate. `demo_real_memory_budget.py` drives real Gemma under a 256 MB budget, forcing LRU eviction (peak 249.6 MB, 5 evictions, finite output, budget respected).
15. Real prefetch on real Runtime: `demo_real_prefetch.py` wires `AsyncPrefetcher` into the production stack — layer 0 cold-misses 8 experts, the worker thread loads layer 1's 8 experts in the background while we wait at `wait_for_layer(1)`, and layer 1 hits the cache on every lookup (prefetch hit rate 100% in the deterministic case, 0% when a misaligned predictor schedules wrong experts).
16A. Layout extension for non-MoE tensors: `WeightLayout` exposes optional `attn_{q,k,v,o}_proj_tensors`, `attn_{q,k}_norm`, `input_layernorm`, `post_attention_layernorm`, `embedding_tensors`, `final_norm`, `lm_head_tensors` (`None` for tied embeddings). All names verified against real `gemma-4-26b-a4b-it-nvfp4` via `validate_layout.py --check-non-moe`.
16B. Tiny ops module: `apex/ops.py` provides `rms_norm` (with Gemma `+1` convention), `linear` (HF orientation), `softmax` (max-subtract stabilized). 12 unit tests pin behavior so 16C/D/E can compose them without surprises.
16C. Single-token attention smoke: `apex/attention.py:single_token_attention` runs q/k/v/o + scaled-dot-product + GQA repeat + optional q_norm/k_norm. `scripts/smoke_real_attention_layer.py` loads + dequants real Gemma layer 0's NVFP4_U32 attention weights and emits a finite (1, 2816) output. No KV cache, no RoPE — those land in step 17.
16D. Full decoder layer: `apex/decoder.py:single_token_decoder_layer` composes the sandwich-norm Gemma block — input_norm → attn → post_attn_norm → +residual → pre_ffn_norm → moe_call → post_ffn_norm → +residual. Layout adds `pre_feedforward_layernorm` and `post_feedforward_layernorm` methods. `scripts/smoke_real_decoder_layer.py` drives layer 0 end-to-end and emits a finite (1, 2816) output (~±100 magnitude). Dense-MLP shared expert + `_1`/`_2` parallel-norm variants intentionally not yet wired.
16E. One-token generation: `apex/generation.py` (`embed_token`, `final_logits` with Gemma's `final_logit_softcapping=30.0`, `sample_argmax`) plus `load_embedding` / `load_final_norm` helpers. `scripts/demo_one_token_generate.py` runs end-to-end on real Gemma 4 — embedding (NVFP4_U32 dequant of the full ~3 GB table) → decoder layer 0 → final RMSNorm → tied lm_head + softcap → argmax. With token id 2 (BOS), produces `next token id: 477` and finite logits.
17A. KVCache class: `apex/kv_cache.py:KVCache` — per-layer keys/values, append concatenates along seq axis, `total_bytes` tracks non-evictable KV memory separately from the LRU expert cache budget. 9 unit tests.
17B. RoPE primitive: `apex/ops.py:apply_rope` — split-half RoPE matching HF/Gemma convention. Position 0 is identity (cos(0)=1, sin(0)=0), so the step-16 smoke math survives unchanged when wired through this code path. 6 unit tests.
17C. KV-cache attention: `apex/attention.py:attention_decode_step` (canonical implementation reading + writing the cache, applying RoPE at the cache's current length). `single_token_attention` becomes a thin wrapper that constructs a fresh 1-layer KVCache and calls the canonical path — single source of truth for the attention math.
17D. Multi-token generation: `apex/decoder.py:decoder_decode_step` (canonical sandwich-norm decoder threading KVCache through to attention) + `scripts/demo_generate_with_kv.py` running real Gemma 4 layer 0 with prefill + decode loops. KV cache grows monotonically (5 tokens × 1 layer × 8 KV heads × 256 head_dim × 4 bytes × 2 = 81920 bytes).
17E. Separate memory budgets: `apex/memory.py:KVMemoryReport` (per-layer KV bytes) + `RuntimeMemoryReport` (joined view with `expert_cache_bytes`, `kv_cache_bytes`, `total_runtime_bytes`). `demo_generate_with_kv.py` now prints a joined `=== memory ===` block at the end of generation: e.g. `expert cache: 385.7 MB (peak 385.7 MB / budget 512.0 MB), kv cache: 0.1 MB, total: 385.8 MB`. KV is non-evictable during generation; the joined report surfaces it next to the LRU expert budget so operators can size both deliberately.
18. Real prompt prefill: `apex/prefill.py:prefill(prompt_ids, *, step_fn)` extracts the prefill loop into a reusable + testable primitive. KV cache lives in `step_fn`'s closure so prefill itself stays stateless. `scripts/demo_prefill.py` drives a 5-token prompt then 3-token decode tail, printing `kv cache length: N` per step. `scripts/demo_generate_with_kv.py` refactored to use the same `prefill` function — single source of truth.
19. Tokenizer integration: `apex/tokenizer.py:Tokenizer` wraps the pure-Rust `tokenizers` library (`Tokenizer.from_file` on `<path>/tokenizer.json`) and reads `bos_token_id` from `<path>/config.json` to replicate `transformers.GemmaTokenizer.add_bos_token=True` behavior. `scripts/demo_tokenizer_generate.py` accepts `--prompt "Hello world"`, encodes to `[2, 9259, 1902]` (BOS + 2 tokens), runs prefill + decode, and decodes generated ids back to text — completing the text-in / text-out path.

---

## Future / improvements (suggested, not yet committed)

### Near term

- **Step 20 — user-facing `MoEModel.generate()`** — wraps embedding + decoder layers + final norm + lm_head + KVCache + tokenizer behind one class with `model.generate(prompt, max_tokens) → str`. Callers stop having to assemble the per-layer attn / norms / moe-call closure themselves. The 19 demo's `_decode` closure is the seed for the class's `step` method.
- **Batched-prompt prefill (multi-token attention with causal mask)** — current `prefill` calls `step_fn` once per prompt token (O(prompt_len) sequential). A real batched prefill would process the whole prompt in a single multi-token attention call, cutting wall time for long prompts.
- **Layer heterogeneity for full-attention layers** — Gemma 4 layers 5/11/17/23/29 use full_attention with K=V tied (no v_proj on disk), head_dim=512, and partial RoPE (rotary_factor=0.25, theta=1e6). Needed for any decoder pass that goes past 5 layers.
- **Dense-MLP shared expert** — Gemma 4 carries both routed MoE and a dense MLP per layer (with the `_1`/`_2` parallel norm variants); driving both would change residual sums.
- **MLX path for real weights** — `MLXBackend` already does SwiGLU; needs a `dequantize` step that produces `mx.array` directly to avoid the numpy round-trip. Same applies to `apex/ops.py`, `apex/attention.py`, `apex/decoder.py`, `apex/generation.py`, `apex/kv_cache.py` (NumPy-only today).
- **Trained predictor on real-Gemma traces** — step 15 proved the prefetch path works on real layers; an actual trained `MarkovPredictor` (instead of the router-peek fallback) would let prefetch overlap with attention/norm in a real model loop. Needs trace data we don't have yet.

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
