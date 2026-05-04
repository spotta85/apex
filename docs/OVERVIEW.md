# APEX — Overview

APEX is a from-scratch Mixture-of-Experts (MoE) inference runtime focused on **bounded-memory expert loading**: the user sets `max_memory_mb` and APEX loads experts on demand, caches them LRU, and evicts when over budget. Built up in tightly-scoped MVP steps. As of step 20, `from apex import MoEModel` is the headline API; as of step 23, all 30 decoder layers of Gemma 4 26B-A4B-IT-NVFP4 (sliding + full attention, K=V tied, partial RoPE) run end-to-end with bounded memory and LRU eviction; as of step 24, router scaling + dense MLP + sliding-window scope + chat-mode EOT stop are all wired in (output is not yet coherent — coherence still needs more, see step 24 in the changelog); as of step 25, the runtime is **measurable, repeatable, and demo-ready** — greedy-decode regression snapshot, long-context window scope, L=30 memory-stress sweep, per-layer parity-harness instrumentation, and a one-call `quickstart.py` (see step 25).

## How to test

```bash
# Full unit suite (348 always-on tests; +84 real-Gemma tests run when
# REAL_MOE_PATH points at a checkpoint, +3 xfail-tolerant coherence
# probes that document where output quality still needs work)
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
uv run python examples/quickstart.py                 # MoEModel.generate() one-liner API (step 20)
uv run python examples/quickstart_multilayer.py      # 1/2/4-layer scaling sweep with memory reports (step 21)
uv run python scripts/benchmark_layer_scaling.py     # tokens/sec + memory comparison table (step 21)
uv run python scripts/audit_config.py                # config audit vs runtime assumptions (step 22A)
uv run python scripts/structural_logit_sanity.py     # top-K logits + cross-prompt diversity (step 22C)
uv run python scripts/audit_router.py                # scaled vs unscaled router top-k (step 24A)
uv run python scripts/benchmark_quality.py           # per-prompt × per-L quality sweep (step 24F)
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
| `generation.py` | `embed_token` (row lookup), `final_logits` (tied lm_head + softcap). Re-exports `sample_argmax` from `apex.sampling` | `embed_token`, `final_logits`, `sample_argmax` |
| `sampling.py` | `sample_argmax` (greedy) and `sample(logits, *, temperature, rng)` (T=0 ⇒ argmax; T>0 ⇒ scaled categorical) | `sample_argmax`, `sample` |
| `prefill.py` | `prefill(prompt_ids, *, step_fn)` — loop step_fn over prompt tokens, return last prediction (= first generated). Stateless; KV cache lives in step_fn's closure | `prefill` |
| `tokenizer.py` | Thin wrapper over HF `tokenizers` lib. `Tokenizer.from_pretrained(path)` loads `tokenizer.json` + reads BOS id from `config.json` + chat template from `chat_template.jinja` or `tokenizer_config.json`. `encode` prepends BOS by default; `decode` skips special tokens by default; `apply_chat_template`/`encode_chat` render via `jinja2` (step 22B) | `Tokenizer` |
| `checkpoint_loaders.py` | High-level component loaders: `read_text_config`, `load_embedding`, `load_final_norm`, `load_layer_norms`, `load_attention_weights`. Built on `safetensors_loader` + `dequant` | five loader functions |
| `config.py` | `ModelConfig` (frozen dataclass). `from_path(checkpoint_dir)` parses `text_config` once into hidden_size, head counts, head_dim, RoPE theta, EOS ids tuple, layer_types, etc. `num_sliding_layers` property = how many decoder layers the runtime currently supports | `ModelConfig` |
| `model.py` | `MoEModel` — the user-facing API. `from_pretrained(path, max_memory_gb)` eager-loads tokenizer + embedding + per-layer attn/norms + cache + manager + router; `generate(prompt, max_tokens, temperature) → str` runs encode → prefill → decode-loop → decode; `generate_chat(messages, **template_kwargs)` formats via the chat template first; `forward(prompt) → logits` returns raw `(1, vocab)` logits without sampling; `memory_report()` returns the joined `RuntimeMemoryReport` | `MoEModel` |
| `__init__.py` | Public API surface: `MoEModel`, `Tokenizer`, `ModelConfig`, `KVCache`, `MemoryReport`, `KVMemoryReport`, `RuntimeMemoryReport` | `apex.MoEModel`, … |

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
| `benchmark_layer_scaling.py` | Sweeps `num_layers` ∈ {1, 2, 4} on the MoEModel and prints a `L | load(s) | gen(s) | tok/s | peak expert | kv | peak total | evict | finite | budget` comparison table (step 21) |
| `audit_config.py` | Walks every hyperparameter the runtime touches and prints a ✓/✗ row per field against the raw `text_config` value (step 22A) |
| `structural_logit_sanity.py` | Forwards 4 prompts through `MoEModel.forward`, prints top-10 token tables + cross-prompt jaccard, asserts argmax varies (step 22C) |
| `audit_router.py` | Layer-0 router decision audit: scaled vs unscaled top-k + per-expert-scale stats (step 24A) |
| `benchmark_quality.py` | 4-prompt × N-layer generation sweep: text, EOS-hit, longest-run, tok/s, peak expert, kv, evictions (step 24F) |
| `parity_harness.py` | Dumps APEX-side per-layer trace (router top-k, attn/MoE/dense norms, top-K logits) per prompt to a JSON file under `traces/` (step 25A) |
| `quickstart.py` | Two-API-call demo: `MoEModel.from_pretrained` → `model.generate` → memory report. The shortest "load real Gemma → generate" entry point (step 25E) |

### `examples/`

| Script | Purpose |
|--------|---------|
| `quickstart.py` | Three-call MoEModel API: `MoEModel.from_pretrained(path, max_memory_gb=8)` → `model.generate("Explain MoE simply.", max_tokens=32)` → `model.memory_report().print_summary()` (step 20) |
| `quickstart_multilayer.py` | Layer-count sweep: builds a fresh `MoEModel` at `num_layers ∈ [1, 2, 4]` and prints memory + timing per trial (step 21) |

### `tests/`

57 test modules, 348 always-on tests + 84 real-Gemma skipif tests + 3 xfail-tolerant coherence probes (432 pass + 3 xfail with `REAL_MOE_PATH` set). `tests/_world.py` provides `FakeStore` + re-exports the fake-world writers from `apex._fake_weights`. Real-Gemma tests live in `tests/test_real_moe_cache.py`, `tests/test_real_moe_two_layer.py`, `tests/test_runtime_real_integration.py`, `tests/test_real_memory_budget.py`, `tests/test_real_prefetch.py`, `tests/test_real_attention.py`, `tests/test_real_decoder.py`, `tests/test_real_generation.py`, `tests/test_generation_regression_real.py` (step 25B), `tests/test_memory_stress_real.py` (step 25D), and `tests/test_parity_harness_real.py` (step 25A) — all auto-skip unless `REAL_MOE_PATH` (or a known local default path) resolves. Synthetic-only long-context coverage lives in `tests/test_long_context.py` (step 25C).

---

## What's done (steps 1–25)

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
20. User-facing API: `from apex import MoEModel` is now a real surface. `apex/config.py:ModelConfig.from_path` parses `text_config` once into a frozen snapshot (hidden_size, head counts, RoPE theta, EOS ids tuple, sliding-vs-full layer cap). `apex/sampling.py:sample` adds temperature-scaled categorical sampling (T=0 collapses to argmax — `apex.generation.sample_argmax` re-exports it for back-compat). `apex/model.py:MoEModel.from_pretrained(path, max_memory_gb=8)` eager-loads tokenizer + embedding + per-layer attn/norms + cache + manager + router; `model.generate(prompt, max_tokens, temperature) → str` runs encode → prefill → decode loop → decode, stopping on EOS. `model.memory_report()` returns the joined `RuntimeMemoryReport`. `examples/quickstart.py` is the three-call quickstart. Clear errors: missing dir → FileNotFoundError; unsupported backend / non-positive memory / num_layers > sliding cap → ValueError.
21. Layer scaling + benchmarking: multi-layer real generation (up to the sliding-attention cap, 5 on Gemma 4) verified with `tests/test_model_multilayer_real.py` (8 new real-Gemma tests covering 2/4-layer generation, budget respected, KV scales linearly, tight-budget eviction still finite, EOS short-circuit). `examples/quickstart_multilayer.py` runs a `[1, 2, 4]`-layer sweep with per-trial memory reports; `scripts/benchmark_layer_scaling.py` prints a `L | load(s) | gen(s) | tok/s | peak expert | kv | peak total | evict | finite | budget` comparison table. Real run: tokens/sec halves per layer doubling (3.0 → 1.8 → 0.9 tok/s); peak expert grows linearly (318 MB → 590 MB → 1202 MB); kv stays in MB-range; budget respected at every L. **Bug fix included:** `apex/attention.py` was dividing fp32 scores by `np.sqrt(head_dim)` (numpy.float64), which silently promoted the layer's output to fp64 and doubled per-layer KV bytes from layer 1 onward. Cast scale to fp32; KV now uniform fp32 across layers. `MoEModel._last_generated_ids` exposes the authoritative generated-id list (re-encoding decoded text doesn't round-trip when output is non-printable bytes).
22. Correctness audit + chat template + structural sanity. **22A:** `scripts/audit_config.py` walks every hyperparameter the runtime uses (hidden, head counts, head_dim, RoPE theta, RMS eps, softcap, BOS/EOS, layer types) and prints ✓ vs the raw `text_config` value plus an info row for unsupported full_attention extras (`attention_k_eq_v`, `global_head_dim=512`, `partial_rotary_factor=0.25`, `rope_theta_full=1e6`). **22B:** `apex/tokenizer.py:Tokenizer.apply_chat_template` + `encode_chat` render Gemma 4's 347-line `chat_template.jinja` (auto-loaded standalone or from `tokenizer_config.json["chat_template"]`) via `jinja2` with `loopcontrols`. `MoEModel.generate_chat(messages, **template_kwargs)` is the one-call API; KV reflects the templated render (markers + turn headers add tokens vs plain `generate`). **22C:** `MoEModel.forward(prompt) → logits` returns raw `(1, vocab)` logits without sampling — built on a new shared `_compute_logits` helper that `_make_step_fn` now also calls (single source of truth). `tests/test_logit_sanity_real.py` locks structural properties: shape, finite, softcap respected, four different prompts produce different argmax (non-degenerate), KV length matches prompt token count. `scripts/structural_logit_sanity.py` prints per-prompt top-10 + cross-prompt jaccard. **22D explicitly deferred:** num_layers > 5 needs a full_attention path (K=V tied, head_dim=512, partial RoPE factor=0.25, rope_theta=1e6) — substantial enough to be its own sprint.
23. **Full-attention path: real Gemma 4 30-layer generation works.** ModelConfig parses the full_attention extras (`global_head_dim=512`, `num_global_key_value_heads=2`, `partial_rotary_factor=0.25`, `rope_theta=1e6`, `attention_k_eq_v=True`) and exposes per-layer accessors `head_dim_for_layer(L)`, `num_kv_heads_for_layer(L)`, `rope_theta_for_layer(L)`, `rotary_dim_for_layer(L)`, `attention_k_eq_v_for_layer(L)`. `apex/ops.py:apply_rope` gained a `rotary_dim` param for partial RoPE (rotates first slice; tail passes through). `apex/attention.py:attention_decode_step` accepts `v_weight=None` for K=V-tied layers — V is aliased after k_norm but before RoPE, so V picks up the per-head normalization but keeps its un-rotated values. `apex/checkpoint_loaders.py:load_attention_weights(..., k_eq_v=True)` skips the (missing on disk) `v_proj`. `MoEModel.from_pretrained` lifted its cap from `num_sliding_layers` to `num_hidden_layers` and builds per-layer attn_kwargs by layer type. Real benchmark on the local Gemma 4 26B-A4B-IT-NVFP4 with `--max-memory-gb 8`: L=8 → 2.6s/2 tok = 0.76 tok/s @ 2.4 GB; L=16 → 8.0s/2 tok = 0.25 tok/s @ 4.8 GB; **L=30 → 16.9s/2 tok = 0.12 tok/s @ 8.2 GB peak with 51 LRU evictions, finite, budget respected**. Tests: `tests/test_full_attention_real.py` (8 new) + `tests/test_real_attention.py` (4 new K=V/partial-RoPE) + `tests/test_ops.py` (7 partial-RoPE) + `tests/test_model_config.py` (3 accessor) = 22 new tests proving the layer-5 path.
24. **Generation quality fixes (still pre-coherence).** Five focused commits closing math + ergonomics gaps that 30-layer generation was running with:
    - **24A** — router scaling. `router.scale` (per-feature, applied to hidden BEFORE projection) and `router.per_expert_scale` (per-expert, applied to logits BEFORE top-k) were ignored. Now plumbed through `apex.moe_layer.route()` via new `hidden_scale` / `per_expert_scale` kwargs (default None ⇒ pre-step-24 byte-identical for Mixtral/Qwen). On Gemma 4 layer 0 the top-k softmax distribution went from near-uniform (top expert 0.18) to extremely peaky (0.997). `scripts/audit_router.py` prints scaled vs unscaled routing.
    - **24B** — dense MLP shared path. Each Gemma 4 layer ships both routed MoE *and* a parallel dense FFN (`mlp.{gate,up,down}_proj`, intermediate=2112). The dense half was unused. New `apex/checkpoint_loaders.py:load_dense_mlp_weights` (AFFINE_UINT8_U32 dequant — same pipeline as router); `apex/decoder.py:decoder_decode_step` accepts `dense_mlp_call=None` (parallel branch summed before `post_ffn_norm`). `MoEModel.per_layer_dense_mlp` is non-evictable (~71 MB/layer × 30 layers ≈ 2.1 GB on top of the LRU expert cache budget).
    - **24C** — sliding-window attention scope. `attention_decode_step(window_size=int|None)` trims K/V on read for sliding layers (Gemma 4: 1024). No-op for prompts ≤ window. Cache itself is not modified (eviction inside KVCache still out of scope). `ModelConfig.sliding_window_for_layer(L)` returns the per-layer cap; full layers get None and attend globally.
    - **24D** — chat-mode EOT stop. `Tokenizer.eot_token_id` reads `<turn|>` (id 106 on Gemma 4) from `tokenizer_config.json`. `MoEModel.generate_chat` stops on EOS ∪ EOT; plain `generate` keeps EOS-only. Without this, `generate_chat` would generate past the model's intended turn boundary into the next user turn.
    - **24E** — three xfail-tolerant plausible-top-K tests document the coherence goal: `"The capital of France is"` should top-K Paris/France, `"2 + 2 ="` should top-K `4`, etc. All three xfail today (output not coherent yet); they exist as the gating signal for when 24A+B+C+D get us to coherence.
    - **24F** — `scripts/benchmark_quality.py` runs a 4-prompt × N-layer sweep with EOS-hit / longest-run / tok/s / peak / evictions reporting. **Honest result:** at L=8 with everything on, generation produces tokens like `'<unused10>'` and immediately hits EOS — finite, budget-respected, and not coherent. Coherence likely needs more (the `_1`/`_2` parallel norm variants we still skip, or HF/MLX reference parity to find the remaining math drift). The plumbing for every remaining quality fix is now in place.
25. **Trustworthiness — measurable, repeatable, demo-ready.** Five focused commits turning "it runs" into "we can prove it runs the same way every time, and localise drift when it doesn't":
    - **25B** — greedy-decode regression snapshot. `tests/test_generation_regression_real.py` + `tests/fixtures/regression_real_l8.json` pin the deterministic per-prompt output (token ids, text, hit_eos, longest_run) at fixed (num_layers=8, max_tokens=8, max_memory_gb=8, temperature=0). Any future math drift in the loader / norms / attention / MoE combine surfaces as a token-id diff against the fixture. `REGEN_REGRESSION_FIXTURE=1` re-bakes after intentional changes. Today's fixture documents the honest L=8 state from 24F: every prompt emits an unused-token (e.g. `<unused10>`) and immediately hits EOS — locked in until the next coherence fix flips it.
    - **25C** — long-context window scope. `tests/test_long_context.py` adds five synthetic tests proving (1) under windowed attention, corruption of pre-window cache entries is invisible to the output; (2) without a window the same corruption is visible; (3) attention stays finite at cache_len=300; (4) `KVCache.total_bytes` grows linearly through 1500+ entries (locks in the no-rolling-window policy); (5) the cache holds well past Gemma 4's `sliding_window=1024` without an internal cap.
    - **25D** — L=30 memory-budget stress sweep. `tests/test_memory_stress_real.py` runs the full 30-layer model at 8 GB (slack) and 4 GB (tight) and pins three contracts: (a) `budget_respected` at every budget; (b) lower budget ⇒ ≥ as many evictions; (c) **byte-identical greedy output across budgets** — LRU eviction is correctness-preserving because NVFP4 dequant returns the same bytes on the next access. This is the central "trustworthy" promise of bounded-memory inference.
    - **25A** — APEX-side parity-harness instrumentation. `apex/decoder.py:decoder_decode_step` accepts `trace=None`; `apex/model.py` adds `MoEModel.forward_with_trace(prompt) → (logits, trace)` capturing per-layer router top-k decisions, post-attn-norm / MoE / dense intermediate norms, and final top-K logits as a JSON-serialisable dict. `scripts/parity_harness.py` dumps captures for a fixed prompt list to `traces/parity_apex_*.json` (gitignored — machine-variant). `tests/test_parity_harness_real.py` (5 tests) proves schema, byte-identical logits vs `forward()`, JSON round-trip, and that router decisions actually vary across layers. **Reference side deferred:** NVFP4 quant means HF transformers can't load the checkpoint without the `compressed-tensors` plugin; bringing that dep in is its own decision. The instrumentation is useful on its own — comparing two APEX configs at this granularity already localises per-layer changes.
    - **25E** — `scripts/quickstart.py`: the absolute-shortest "load real Gemma → generate" demo. Two API calls (`from_pretrained`, `generate`) plus a memory-report print. Use it as the canonical entry point; the existing `scripts/demo_*.py` files build the runtime by hand at the Runtime/ExpertCache level, useful for development but obscuring how trivial the user-facing surface is.

---

## Future / improvements (suggested, not yet committed)

### Near term

- **Generation coherence (step 24 follow-up)** — step 24 plumbed every quality fix we identified (router scaling, dense MLP parallel branch, sliding-window scope, EOT stop) and the runtime is now finite + budget-respected at 30 layers, but the L=8 benchmark still emits unused-token gibberish that hits EOS immediately. The next investigation paths in priority order:
  - **`_1`/`_2` parallel norm variants** — Gemma 4 ships `pre_feedforward_layernorm_2`, `post_feedforward_layernorm_1`, `post_feedforward_layernorm_2` per layer. Step 24 left these on disk and shared the unsuffixed `pre_ffn` / `post_ffn` between routed and dense paths. The most likely fix is that each branch wants its own pre/post norm pair (routed: `pre_ffn` + `post_ffn`; dense: `pre_ffn_2` + `post_ffn_2` or similar). `apex/checkpoint_loaders.py:load_layer_norms` would need to surface the variants and `apex/decoder.py:decoder_decode_step` would need separate norm slots for the dense branch.
  - **Reference loader for parity comparison** — step 25A built the APEX-side instrumentation (`MoEModel.forward_with_trace` + `scripts/parity_harness.py`) so that diffing per-layer norms localizes math drift to a specific operator. The other half — loading the same NVFP4 checkpoint into a reference runtime — needs `transformers` with the `compressed-tensors` quantization plugin (or `mlx-lm`, which does have NVFP4 support). Either is a substantial dep so still deliberately deferred; with 25A in place the diff machinery is one small script away once the reference is loadable.
  - **Router-extras formula** — step 24A applies `hidden * router.scale` then `(hidden @ proj.T) * router.per_expert_scale` then top-k. The shapes match and the audit shows visible peaking, but the multiplicative-vs-additive choice and pre-vs-post softmax order are an educated guess; reference parity would confirm.
  - **MoE expert combination** — `run_moe_layer` does a per-token `sum(w_i * expert_i(h))`. Gemma may use a different combination (e.g. weight renormalization across the top-k slice with per-expert biases, or a different SwiGLU activation order in the expert).
- **Batched-prompt prefill (multi-token attention with causal mask)** — current `prefill` calls `step_fn` once per prompt token (O(prompt_len) sequential). A real batched prefill would process the whole prompt in a single multi-token attention call, cutting wall time for long prompts.
- **MLX path for real weights** — `MLXBackend` already does SwiGLU; needs a `dequantize` step that produces `mx.array` directly to avoid the numpy round-trip. Same applies to `apex/ops.py`, `apex/attention.py`, `apex/decoder.py`, `apex/generation.py`, `apex/kv_cache.py` (NumPy-only today).
- **Trained predictor on real-Gemma traces** — step 15 proved the prefetch path works on real layers; an actual trained `MarkovPredictor` (instead of the router-peek fallback) would let prefetch overlap with attention/norm in a real model loop. Needs trace data we don't have yet.

### Medium term

- **Qwen shared experts** — Qwen MoE also ships an always-on dense FFN per layer; step 24B wired the equivalent for Gemma 4 only. The Qwen layout would need its own `dense_mlp_*` accessors and the `MoEModel` loader path generalized (currently only loads dense MLP weights when the layout returns non-None).
- **Multi-token batches** — `route()` already supports batch dim; runtime + cache assume batch=1.
- **GGUF quantized formats** — Q4_0 / Q4_K_M etc. New entry in `dequant.py` + a `QuantizedGGUFExpertStore`.
- **Recovering from cache pressure** — eviction today is pure LRU; could weigh by routing-frequency hints from the predictor.

### Hygiene

- **`scripts/` import boilerplate** — every script does `sys.path.insert(0, ...)`. Could be replaced by making `apex` an installed editable package (`uv pip install -e .`).
- **Replace `np.errstate` wrap in smoke script** with proper bf16-safe matmul (e.g. cast router to fp32 explicitly before the matmul, which is already what `safetensors_loader` does — so the warnings actually come from NVFP4 expert weights spanning a wide dynamic range; investigate whether dequant is producing correct denormals).
- **Silence `cache hit` / `cache miss` prints in `apex/expert_manager.py`** — currently unconditional. Operator-facing scripts (`scripts/quickstart.py`, `scripts/parity_harness.py`) are forced to grep them out. Adding a `verbose: bool = True` constructor flag (default-on for back-compat with existing demos that grep for these lines) is the obvious fix.
