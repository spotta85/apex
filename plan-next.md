# APEX — Cleanup pass + Step 12 (true Runtime integration on real checkpoints)

## Context

Why this sprint exists: the user asked to "do steps 10 and 11" and to do a cleanup pass first. After exploring the repo, two things are clear:

1. **Steps 10 and 11 already shipped on `main`** as smoke scripts (commits `f0a42fd`, `662db91`). The exact files the user named — `scripts/demo_real_moe_cache.py` and `tests/test_real_moe_cache.py` (plus `scripts/smoke_real_moe_two_layer.py` and `tests/test_real_moe_two_layer.py`) — are present, with 14 passing tests and the documented ~167x / ~200x speedups. The mechanics work end-to-end on real Gemma 4 NVFP4.

2. **The smoke scripts bypass `apex/runtime.py`.** They drive `apex.moe_layer.run_moe_layer` + `ExpertManager` directly. The existing `Runtime` class — what every fake demo uses — still hardcodes `np.ones((1, 8))` for hidden state and uses uniform-mean combine, so it can't drive a real Gemma layer. "Fully integrate" steps 10/11 therefore means **lifting the smoke pattern into `Runtime` itself** so a real-checkpoint stack and a fake-MoE stack drive the same orchestrator. That becomes Step 12.

User decisions captured during planning:

- **Scope**: Cleanup + Step 12 (Runtime integration). Steps 10/11 stand as-shipped.
- **Router contract**: Routers return `(ids, weights)`. Existing routers return `(ids, None)` (None = uniform; fake demos stay bit-equal).

User constraints (from memory):

- Comments on apex code (per `feedback_comments.md` — global "no comments" default is overridden in this repo).
- Separate commits for orthogonal concerns (per `feedback_separate_commits.md`).
- Careful, efficient implementation. Each step proves itself with demo + tests before commit.

## Progress update — 2026-05-03

- **Phase 1 complete.** Implemented the cleanup pass: removed dead `Runtime.verbose_router`, hoisted shared synthetic NVFP4 test helpers, and hoisted real-Gemma script bootstrap into `scripts/_real_moe_helpers.py`.
- **Phase 1 verification complete.** Targeted cleanup tests: `22 passed, 2 skipped`. Full suite: `212 passed, 2 skipped` (214 collected). All 7 fake demos were byte-identical to the pre-cleanup baseline. Real-script CLI imports passed via `--help`.
- **Phase 2 complete.** Re-validated Steps 10 and 11 against the real Gemma 4 NVFP4 checkpoint at `/Users/spotta/Desktop/models/gemma4/Gemma-4-26B-A4B-NVFP4`.
- **Phase 2 verification complete.** Real pytest cases: `2 passed`. `scripts/smoke_real_moe_layer.py`, `scripts/demo_real_moe_cache.py`, and `scripts/smoke_real_moe_two_layer.py` all passed end-to-end on the real checkpoint.
- **Still pending.** Phase 3 / Step 12 remains unimplemented: `Runtime` still needs router weights, arbitrary initial hidden size, and `run_moe_layer` integration before it can drive real Gemma layers directly.

---

## Phase 1 — Cleanup pass (3 separate commits)

The audit surfaced exactly three high-confidence wins. Everything else (`BaseNumpyGateRouter`, `RuntimeBundle`, `demo_modes.py` factories, `sys.path.insert` boilerplate) is appropriate abstraction or intentional design — leave alone. No naming sweeps.

### Commit C.1 — Drop dead `Runtime.verbose_router` parameter

**Why:** `apex/runtime.py:42` defines `verbose_router: bool = True`. Zero callers across `apex/`, `scripts/`, `tests/`, `demo*.py` ever set it. The default `True` makes the gate-print on lines 94–98 unconditional in practice. Dead surface.

**Exact diff (apex/runtime.py):**

```diff
@@ -34,7 +34,6 @@ class Runtime:
         prefetcher=None,
         predictor=None,
         trace_recorder=None,
-        verbose_router: bool = True,
         num_layers: int = 2,
     ):
@@ -60,10 +59,6 @@ class Runtime:
         # Optional: TraceRecorder that captures real transitions to disk.
         self.trace_recorder = trace_recorder
-        # When False, suppress the per-layer "router selected experts"
-        # print. Other prints (predictor, prefetch, hits/misses) are
-        # unaffected. Default True keeps existing demo output.
-        self.verbose_router = verbose_router
         # Number of layers in the toy model. Default 2 keeps every
         # existing demo unchanged; future real-model layouts can pass
@@ -91,11 +86,10 @@ class Runtime:
             expert_ids = self.router.route(layer_id, hidden)
-            if self.verbose_router:
-                # `router.name` lets demos see which routing path is in
-                # use ("numpy" vs "mlx"). All routers declare a name.
-                print(f"router backend: {self.router.name}")
-                print(f"router selected experts: {list(expert_ids)}")
+            # `router.name` lets demos see which routing path is in
+            # use ("numpy" vs "mlx"). All routers declare a name.
+            print(f"router backend: {self.router.name}")
+            print(f"router selected experts: {list(expert_ids)}")
```

**Tests:** `uv run pytest -q` — should still report 214 passed (no test asserts on the parameter).

**Demo verification:** All 7 fake demos produce **byte-identical** output to before this commit. (Capture with: `for demo in demo*.py; do uv run python $demo > /tmp/before_$demo.out; done` — then again after the change and `diff` each.)

**Commit message:** `cleanup: drop unused Runtime.verbose_router parameter`

---

### Commit C.2 — Hoist NVFP4 synthetic-checkpoint helpers into a shared test module

**Why:** ~270 LOC of near-identical helper code lives in 4 test files:

| Helper | Files containing a copy |
|---|---|
| `_write_safetensors` (writer) | `tests/test_real_moe_cache.py:60-83`, `tests/test_real_moe_two_layer.py:51-69`, `tests/test_quantized_safetensors_store.py:40-62`, `tests/test_safetensors_loader.py:17-40` (renamed `_write_minimal_safetensors`) |
| Constants `_E4M3_ONE = 0x38`, `_FP4_PACKED_ONE = 0x22` | `tests/test_real_moe_cache.py:53-57`, `tests/test_real_moe_two_layer.py:47-48`, `tests/test_quantized_safetensors_store.py:29-37` (latter as fns) |
| Synthetic ones-expert / multi-layer-checkpoint builders | `tests/test_real_moe_cache.py:86-118`, `tests/test_real_moe_two_layer.py:72-102`, `tests/test_quantized_safetensors_store.py:65-97` |
| `_real_checkpoint_path()` + `REAL_PATH_ENV` + skipif decorator | `tests/test_real_moe_cache.py:316-328`, `tests/test_real_moe_two_layer.py:274-288` |

**New file `tests/_synthetic_nvfp4.py`:**

```python
"""Shared helpers for tests that build hand-crafted NVFP4 checkpoints.

Every NVFP4 test wants the same building blocks:
  - write a multi-tensor safetensors file with raw bytes
  - build a "ones-expert" (FP4 code 0x2 + E4M3 scale 1.0 + scalar 1.0)
    so dequant produces all-ones matrices and SwiGLU output is
    deterministic per (hidden, intermediate, hidden_dim)
  - skip real-Gemma tests if the checkpoint isn't on disk

Centralizing here means future NVFP4 tests can `from tests._synthetic_nvfp4
import ...` instead of inventing their own helpers (the failure mode
the previous duplication had).
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from apex.dequant import NVFP4_GROUP_SIZE


# F8_E4M3 byte 0x38 = sign 0, exp 0111 (=7=bias), mant 000
# → (1 + 0/8) × 2^(7-7) = 1.0. Used so block_scale × FP4 × scalar = FP4.
E4M3_ONE = 0x38

# Two FP4 ones (code 0x2 each, magnitude 1.0 in the NVFP4 lookup)
# packed in one uint8 byte: low nibble 0x2, high nibble 0x2 ⇒ 0x22.
FP4_PACKED_ONE = 0x22


def write_safetensors(path: Path, items):
    """Write a multi-tensor safetensors file with hand-built bytes.

    `items` is a list of (name, dtype_str, shape_list, raw_bytes). Used
    by every NVFP4 test to inject controlled byte-level checkpoints
    that bypass safetensors.numpy's dtype gating (which rejects F8/BF16).
    """
    header = {}
    cursor = 0
    payload = bytearray()
    for name, dtype, shape, raw in items:
        size = len(raw)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [cursor, cursor + size],
        }
        payload.extend(raw)
        cursor += size
    header_json = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(len(header_json).to_bytes(8, "little"))
        f.write(header_json)
        f.write(bytes(payload))


def build_expert_items(layout, layer_id, expert_id, hidden, intermediate):
    """Return the list of (name, dtype, shape, bytes) tuples for one expert.

    Per the NVFP4 spec: each projection has a packed-uint8 weight,
    F8_E4M3 per-block scales, F32 weight_scale_2 + F32 input_scale.
    HF orientation: gate/up are (intermediate, hidden); down is reversed.
    """
    items = []
    names = layout.expert_tensors(layer_id, expert_id)
    for proj_key, weight_name in names.items():
        if proj_key == "down_proj":
            rows, cols = hidden, intermediate
        else:
            rows, cols = intermediate, hidden
        prefix = weight_name[: -len(".weight")]
        packed = np.full((rows, cols // 2), FP4_PACKED_ONE, dtype=np.uint8)
        scales = np.full(
            (rows, cols // NVFP4_GROUP_SIZE), E4M3_ONE, dtype=np.uint8,
        )
        items.extend([
            (weight_name, "U8", list(packed.shape), packed.tobytes()),
            (f"{prefix}.weight_scale", "F8_E4M3",
             list(scales.shape), scales.tobytes()),
            (f"{prefix}.weight_scale_2", "F32", [], np.float32(1.0).tobytes()),
            (f"{prefix}.input_scale", "F32", [], np.float32(1.0).tobytes()),
        ])
    return items


def build_one_layer_checkpoint(path, layout, num_experts, hidden, intermediate):
    """Build a single-file NVFP4 checkpoint with one layer × num_experts."""
    items = []
    for expert_id in range(num_experts):
        items.extend(build_expert_items(layout, 0, expert_id, hidden, intermediate))
    write_safetensors(path, items)


def build_multi_layer_checkpoint(path, layout, num_layers, num_experts,
                                 hidden, intermediate):
    """Build a single-file NVFP4 checkpoint with num_layers × num_experts."""
    items = []
    for layer_id in range(num_layers):
        for expert_id in range(num_experts):
            items.extend(build_expert_items(layout, layer_id, expert_id,
                                            hidden, intermediate))
    write_safetensors(path, items)


REAL_PATH_ENV = "REAL_MOE_PATH"
DEFAULT_REAL_PATH = "/Users/spotta/Desktop/models/Gemma-4-26B-A4B-NVFP4"


def real_checkpoint_path():
    """Return Path to the real Gemma 4 NVFP4 checkpoint, or None if absent."""
    candidate = os.environ.get(REAL_PATH_ENV, DEFAULT_REAL_PATH)
    p = Path(candidate)
    return p if p.is_dir() else None


def skip_if_no_real_checkpoint():
    """Decorator: skip a test if the real-Gemma checkpoint isn't on disk."""
    return pytest.mark.skipif(
        real_checkpoint_path() is None,
        reason=(
            f"real Gemma 4 NVFP4 checkpoint not found at "
            f"{DEFAULT_REAL_PATH} (set ${REAL_PATH_ENV} to override)"
        ),
    )
```

**Per-file edits:**

- `tests/test_real_moe_cache.py`: delete lines 53–118 (constants + builders), delete lines 316–340 (real-path helpers + decorator); add `from tests._synthetic_nvfp4 import (build_one_layer_checkpoint, real_checkpoint_path, skip_if_no_real_checkpoint)`. Replace `@pytest.mark.skipif(...)` with `@skip_if_no_real_checkpoint()`. Replace `_make_synthetic_stack()` body's call to `_build_synthetic_checkpoint(...)` with `build_one_layer_checkpoint(...)`.

- `tests/test_real_moe_two_layer.py`: same shape — delete the duplicate constants/helpers, import shared ones, swap `_build_two_layer_checkpoint` for `build_multi_layer_checkpoint`.

- `tests/test_quantized_safetensors_store.py`: replace local `_e4m3_one()` / `_nvfp4_packed_ones()` / `_build_fake_nvfp4_expert` / `_write_safetensors` with `build_expert_items` / `write_safetensors`. The 3 tests in this file should keep their existing assertions; only the helper calls change.

- `tests/test_safetensors_loader.py`: replace local `_write_minimal_safetensors` with `write_safetensors` from the shared module. Loader-specific tests don't use the NVFP4 builders, only the writer.

**Verification:**

```bash
uv run pytest tests/test_real_moe_cache.py tests/test_real_moe_two_layer.py \
              tests/test_quantized_safetensors_store.py \
              tests/test_safetensors_loader.py -v
# Expected: same 14 + 3 + N tests pass that passed before, no skips that
# weren't already there.
uv run pytest -q
# Expected: 214 passed.
```

**Commit message:** `cleanup: hoist NVFP4 synthetic-checkpoint helpers into tests/_synthetic_nvfp4.py`

---

### Commit C.3 — Hoist real-Gemma script bootstrap

**Why:** Three real-Gemma scripts duplicate the same setup:

- The same import block (`apex.checkpoint_index.read_safetensors_index`, `apex.safetensors_loader.load_tensor`, `apex.weight_layout.get_layout`)
- The same `np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore")` filter at every NVFP4 matmul site (~5 uses across the 3 scripts)
- The same router-loading pattern (probe shape via `get_tensor_shape`, look up shard via `read_safetensors_index`, call `load_tensor`)
- The same `--path` default pointing at `/Users/spotta/Desktop/models/Gemma-4-26B-A4B-NVFP4`

**New files:**

`scripts/__init__.py` (empty file — needed so `from scripts._real_moe_helpers import ...` works after `sys.path.insert(project_root)`).

`scripts/_real_moe_helpers.py`:

```python
"""Shared bootstrapping for the three real-Gemma scripts.

Every real-checkpoint smoke / demo script wants the same handful of
utilities; centralizing them keeps the per-script code focused on
the actual experiment, not on plumbing.

Folded in:
  - `nvfp4_matmul_warnings`: context manager that silences the
    transient over/underflow warnings real NVFP4 weights provoke
    during partial-sum reduction. Final accumulated output is finite
    (asserted at every script's end); the filter only hides noise.
  - `load_router(path, layout, layer_id)`: returns the BF16-decoded
    fp32 router weight for one layer. One-liner that hides the
    sharded-index dance from callers.
  - `add_real_moe_args(parser, ...)`: registers the --path/--top-k/
    --seed/--max-memory-mb args every real script needs.
"""

import argparse
import contextlib
from pathlib import Path

import numpy as np

from apex.checkpoint_index import get_tensor_shape, read_safetensors_index
from apex.safetensors_loader import load_tensor


DEFAULT_GEMMA_PATH = "/Users/spotta/Desktop/models/Gemma-4-26B-A4B-NVFP4"


@contextlib.contextmanager
def nvfp4_matmul_warnings():
    """Silence over/underflow warnings during real-NVFP4 matmuls.

    Real NVFP4 dequantized weights span ~e-7 to ~e+1 dynamic range;
    partial sums during a 2816-element matmul accumulation briefly
    over/underflow even though the final result is finite. The
    warnings are deafening without this. Callers always assert
    np.all(np.isfinite(out)) afterward — that's the real check.
    """
    with np.errstate(over="ignore", under="ignore",
                     invalid="ignore", divide="ignore"):
        yield


def load_router(path, layout, layer_id):
    """Return (router_weight_fp32, num_experts, hidden_dim) for one layer."""
    name = layout.gate_weight(layer_id)
    num_experts, hidden_dim = get_tensor_shape(path, name)
    weight_map = read_safetensors_index(path)
    router = load_tensor(path / weight_map[name], name)
    return router, num_experts, hidden_dim


def add_real_moe_args(parser: argparse.ArgumentParser, *,
                      include_layer: bool = True,
                      include_max_memory: bool = True,
                      default_max_memory_mb: float = 256):
    """Register the standard real-Gemma CLI flags."""
    parser.add_argument("--path", default=DEFAULT_GEMMA_PATH,
                        help=f"checkpoint directory (default: {DEFAULT_GEMMA_PATH})")
    if include_layer:
        parser.add_argument("--layer", type=int, default=0,
                            help="layer index to drive (default: 0)")
    parser.add_argument("--top-k", type=int, default=8,
                        help="experts per token (default: 8, matches Gemma 4)")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for the fake hidden state (default: 0)")
    if include_max_memory:
        parser.add_argument("--max-memory-mb", type=float,
                            default=default_max_memory_mb,
                            help=f"cache budget in MB (default: {default_max_memory_mb})")
```

**Per-script edits (smoke_real_moe_layer.py, demo_real_moe_cache.py, smoke_real_moe_two_layer.py):**

- Replace the 3 stale-imports + DEFAULT_PATH constant block with `from scripts._real_moe_helpers import (DEFAULT_GEMMA_PATH, load_router, nvfp4_matmul_warnings, add_real_moe_args)`. Note `sys.path.insert(0, project_root)` stays — the helpers themselves use `from apex.X import Y`, but `from scripts.X import Y` requires both `scripts/__init__.py` AND project root on sys.path.
- Each script's `_parse_args()` shrinks: now it's `p = argparse.ArgumentParser(...); add_real_moe_args(p); return p.parse_args()`.
- Replace inline `weight_map = read_safetensors_index(path); router_path = path / weight_map[router_name]; router = load_tensor(...)` with one call to `load_router(path, layout, layer_id)`.
- Replace `with np.errstate(over="ignore", ...)` with `with nvfp4_matmul_warnings()`.

**Verification:**

```bash
# Run each script before C.3 and capture output:
uv run python scripts/smoke_real_moe_layer.py > /tmp/before_smoke.out
uv run python scripts/demo_real_moe_cache.py > /tmp/before_cache.out
uv run python scripts/smoke_real_moe_two_layer.py > /tmp/before_two_layer.out

# Apply C.3, then re-run and diff:
uv run python scripts/smoke_real_moe_layer.py > /tmp/after_smoke.out && diff /tmp/before_smoke.out /tmp/after_smoke.out
# Expected: empty diff. Same for the other two.

uv run pytest -q
# Expected: 214 passed.
```

**Commit message:** `cleanup: hoist real-Gemma script bootstrap to scripts/_real_moe_helpers.py`

---

## Phase 2 — Re-validate Steps 10 & 11 (no new commits)

These already shipped. The plan's job is to confirm there's nothing left to build, not to re-implement. Quick recap so the plan stands alone:

**Status: complete as of 2026-05-03.** Re-validation used the real checkpoint at `/Users/spotta/Desktop/models/gemma4/Gemma-4-26B-A4B-NVFP4` because the older default path `/Users/spotta/Desktop/models/Gemma-4-26B-A4B-NVFP4` is not present on this machine.

Commands run:

```bash
REAL_MOE_PATH=/Users/spotta/Desktop/models/gemma4/Gemma-4-26B-A4B-NVFP4 \
  uv run pytest \
    tests/test_real_moe_cache.py::test_real_gemma_layer_caches_top_8_then_reuses \
    tests/test_real_moe_two_layer.py::test_real_gemma_two_layer_pass_caches_correctly -v
# Result: 2 passed in 3.87s

uv run python scripts/smoke_real_moe_layer.py \
  --path /Users/spotta/Desktop/models/gemma4/Gemma-4-26B-A4B-NVFP4
# Result: passed; output shape (1, 2816), finite=True.

uv run python scripts/demo_real_moe_cache.py \
  --path /Users/spotta/Desktop/models/gemma4/Gemma-4-26B-A4B-NVFP4
# Result: passed; 8 misses, 8 hits, 0 evictions, finite=True.

uv run python scripts/smoke_real_moe_two_layer.py \
  --path /Users/spotta/Desktop/models/gemma4/Gemma-4-26B-A4B-NVFP4
# Result: passed; first pass 16 misses, second pass 16 hits, 0 evictions.
```

### Step 10 — committed in `f0a42fd`
- `apex/dequant.py:DequantizingExpertStore` — wraps any quantized store; `load(layer, expert)` returns dequantized fp32 ExpertWeights bundle. Cache hits skip the ~60 ms NVFP4 decode.
- `scripts/demo_real_moe_cache.py` — runs Gemma 4 layer 0 twice with the same hidden state. First run: 8 misses + 8 dequants (~1037 ms). Second run: 8 cache hits (~6 ms). 167x speedup, bit-identical outputs, 0 evictions.
- `tests/test_real_moe_cache.py` — 8 tests (7 synthetic always-run + 1 real-checkpoint skipif).

### Step 11 — committed in `662db91`
- `scripts/smoke_real_moe_two_layer.py` — drives Gemma 4 layers 0 → 1 in sequence using one shared `ExpertManager`/`ExpertCache`. Threading: layer 0's output → layer 1's router. 16 cold misses, 16 warm hits, ~200x speedup.
- `tests/test_real_moe_two_layer.py` — 6 tests (5 synthetic + 1 real-checkpoint skipif). Includes a deliberate cache-key-disambiguation test: layer 0 expert 121 and layer 1 expert 121 must be distinct cache entries (real Gemma actually picks expert 121 in both layers for seed=0 hidden — test exercises the invariant on real weights).

### What 10–11 left undone (and why Step 12 exists)

`apex/runtime.py:Runtime` — the orchestrator every fake demo uses — does **not** call `apex.moe_layer.run_moe_layer` and cannot drive a real Gemma layer:

1. **Hidden state hardcoded** — `runtime.py:75-77` always starts with `np.ones((1, 8), dtype="float32")`. Wrong shape for Gemma 4's hidden_dim=2816.
2. **Combine is uniform-mean, not router-weighted** — `runtime.py:117` calls `backend.combine_outputs(outputs)` which does `sum(outputs) / len(outputs)`. Discards the router's softmax weights. Wrong numerical answer for any real model.
3. **Routers don't return weights** — every existing router returns `[int, ...]` only; `apex.moe_layer.route()` returns `(ids, weights)`.

So real Gemma works **only** through one-off smoke scripts. Step 12 closes that gap.

---

## Phase 3 — Step 12 — Runtime drives real Gemma 4 layers (single commit)

**Single capability win:** `scripts/demo_real_runtime.py` runs N layers of Gemma 4 through the existing Runtime + cache + manager machinery. Same orchestrator the fake demos use.

**Why one commit, not sub-step commits:** The atomic unit is "Runtime supports real checkpoints." Splitting the router contract change (12.1) from the Runtime change (12.2) would commit broken intermediate state — the router contract changed but nothing consumes the new shape, OR the Runtime calls a method that doesn't yet exist.

### 12.1 — Extend the router contract

Routers go from returning `[int, ...]` to returning `(ids, weights)` where `weights` is **either** a list of floats summing to ~1.0 **or** `None` (meaning uniform). Default to `None` for every existing router — they're score-derived, not softmax-aware, and uniform combine is what `combine_outputs` already produces.

**Diff in `apex/router.py:FakeRouter.route`:**

```diff
     def route(self, layer_id: int, hidden_state=None):
         # Layer 0: use experts 1 and 3.
         if layer_id == 0:
-            return [1, 3]
+            return [1, 3], None  # None ⇒ uniform combine in Runtime
         # All other layers: use experts 0 and 2.
-        return [0, 2]
+        return [0, 2], None
```

**Diff in `apex/_numpy_gate_router_base.py:BaseNumpyGateRouter.route`:**

```diff
     def route(self, layer_id: int, hidden_state):
         gate = self._load_gate(layer_id)
         # scores: (batch=1, num_experts)
         scores = np.asarray(hidden_state) @ gate
         # Top-k by score, descending.
         expert_ids = scores[0].argsort()[-self.top_k :][::-1]
-        return [int(e) for e in expert_ids]
+        # `weights=None` means "uniform combine" in Runtime — these routers
+        # don't apply softmax to the gate scores, so they can't supply
+        # meaningful per-expert weights. Real router (apex.real_router)
+        # returns proper softmax weights instead.
+        return [int(e) for e in expert_ids], None
```

**Diff in `apex/mlx_gate_router.py:MLXGateRouter.route`:** same shape — return `(expert_ids, None)`.

**Test fallout:** Every test that asserts on the router's return shape needs a tuple-unpack. Grep for `router.route(` and `\.route(` to find callers. Most tests just call route() without inspecting the return — those are unaffected. The few that do inspect: update with `ids, _ = router.route(...)`.

### 12.2 — Rewrite Runtime's per-layer loop using `run_moe_layer`

**Current code (`apex/runtime.py:79-117`):**

```python
for layer_id in range(self.num_layers):
    if self.prefetcher is not None and layer_id > 0:
        self.prefetcher.wait_for_layer(layer_id)

    print(f"running layer {layer_id}")
    expert_ids = self.router.route(layer_id, hidden)
    print(f"router backend: {self.router.name}")
    print(f"router selected experts: {list(expert_ids)}")

    outputs = []
    for expert_id in expert_ids:
        expert = self.expert_manager.ensure_loaded(layer_id, expert_id)
        out = self.backend.run_expert(hidden, expert)
        outputs.append(out)
    hidden = self.backend.combine_outputs(outputs)
    # ... end-of-layer side work (unchanged) ...
```

**New code:**

```python
from apex.moe_layer import run_moe_layer

# inside run() loop:
for layer_id in range(self.num_layers):
    if self.prefetcher is not None and layer_id > 0:
        self.prefetcher.wait_for_layer(layer_id)

    print(f"running layer {layer_id}")
    expert_ids, weights = self.router.route(layer_id, hidden)
    print(f"router backend: {self.router.name}")
    print(f"router selected experts: {list(expert_ids)}")

    # Uniform-combine fallback: when a router returns weights=None
    # (every existing fake router) we mean "treat each expert equally."
    # This is exactly what backend.combine_outputs used to do, so the
    # numerical output is bit-identical to the pre-step-12 demos.
    if weights is None:
        weights = [1.0 / len(expert_ids)] * len(expert_ids)

    expert_loader = lambda eid, lid=layer_id: self.expert_manager.ensure_loaded(lid, eid)
    hidden = run_moe_layer(hidden, expert_ids, weights, expert_loader, self.backend)

    # ... end-of-layer side work (unchanged) — same actual_next_experts,
    # trace_recorder, predictor, prefetcher logic. The only thing changed
    # is the per-expert run + combine.

# Need to unpack actual_next_experts from the (ids, weights) tuple too:
#   actual_next_experts, _ = self.router.route(next_layer, hidden)
```

**Drop `combine_outputs` from both backends:**

```diff
# apex/backends/fake_backend.py
-    def combine_outputs(self, outputs):
-        # Uniform mean across the per-expert outputs. Real MoE uses
-        # router-weighted combination; uniform is enough to demonstrate
-        # the loading/caching machinery.
-        return sum(outputs) / len(outputs)
```

```diff
# apex/backends/mlx_backend.py — same removal.
```

Verify nothing else calls `combine_outputs` (grep for it across the repo). After Runtime stops calling it, it's pure dead code.

### 12.3 — Add optional `initial_hidden` to Runtime

**Diff in `apex/runtime.py:__init__` and `run`:**

```diff
     def __init__(
         self,
         router,
         expert_manager,
         backend,
         prefetcher=None,
         predictor=None,
         trace_recorder=None,
         num_layers: int = 2,
+        initial_hidden=None,
     ):
         ...
+        # Optional override for the starting hidden state. Default
+        # np.ones((1, 8)) keeps every fake demo unchanged; real
+        # checkpoints (hidden_dim ≠ 8) pass an explicit ndarray.
+        self.initial_hidden = initial_hidden

     def run(self):
+        if self.initial_hidden is not None:
+            initial = self.initial_hidden
+        else:
+            initial = np.ones((1, 8), dtype="float32")
-        hidden = self.backend.prepare_hidden(
-            np.ones((1, 8), dtype="float32")
-        )
+        hidden = self.backend.prepare_hidden(initial)
```

### 12.4 — Add `apex/real_router.py`

**Per-layer model**: Gemma 4 has 30 distinct routers, one per layer. Build a `MultiLayerRealRouter` that holds the path/layout and lazy-loads each layer's router weight on first use, caching the fp32 result. Same `route(layer_id, hidden)` interface as every other router.

```python
"""Router that scores against pre-loaded BF16 router weights from a real checkpoint.

For HF / NVIDIA MoE checkpoints (Mixtral / Qwen MoE / Gemma 4 NVFP4).
Wraps apex.moe_layer.route so the Runtime sees the same
(route(layer, hidden) -> (ids, weights)) interface every other router
implements. Router weights are loaded lazily per layer (cheap
relative to expert weights — under 50 MB total even for Gemma 4) and
cached in a dict.

Different from the gate routers (apex.gate_router, apex.safetensors_gate_router):
those expect ad-hoc per-layer .npy / safetensors files; this one
threads the layout's gate_weight() naming through the production
sharded checkpoint structure. And it returns *softmax weights* — the
existing gate routers can't, because they don't softmax their scores.
"""

import numpy as np

from apex.checkpoint_index import read_safetensors_index
from apex.moe_layer import route
from apex.safetensors_loader import load_tensor


class MultiLayerRealRouter:
    name = "numpy"

    def __init__(self, path, layout, top_k):
        self.path = path
        self.layout = layout
        self.top_k = top_k
        # Lazy per-layer router weight cache: {layer_id: ndarray}.
        # Loaded on first route() call for that layer.
        self._weights = {}
        # Memoize the shard index so we don't re-read the index file
        # for every layer's first lookup.
        self._weight_map = read_safetensors_index(path)

    def _load_for_layer(self, layer_id: int):
        if layer_id not in self._weights:
            name = self.layout.gate_weight(layer_id)
            self._weights[layer_id] = load_tensor(
                self.path / self._weight_map[name], name,
            )
        return self._weights[layer_id]

    def route(self, layer_id: int, hidden_state):
        weight = self._load_for_layer(layer_id)
        # Same warning-filter rationale as scripts/_real_moe_helpers:
        # NVFP4 / BF16 dynamic range provokes transient warnings during
        # the matmul reduction even though the final logits are finite.
        with np.errstate(over="ignore", under="ignore",
                         invalid="ignore", divide="ignore"):
            ids, weights = route(np.asarray(hidden_state), weight, self.top_k)
        # `route` returns (1, top_k) shapes (batch dim); Runtime expects
        # 1-D iterables for both. Take row 0 of each.
        return ids[0].tolist(), weights[0].tolist()
```

### 12.5 — Add `scripts/demo_real_runtime.py`

Wires the production stack:

```python
"""Step 12 demo: real Gemma 4 layers driven through apex.runtime.Runtime.

Proves the integration: same orchestrator the fake demos use can drive
a real-checkpoint stack with no special-case code. First run: cold
cache, every expert is a miss. Second run: warm cache, every expert
is a hit. Identical Runtime, identical metrics object — only the
router/store/cache/initial_hidden are wired differently.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apex.backends.fake_backend import FakeBackend
from apex.dequant import DequantizingExpertStore
from apex.expert_cache import ExpertCache
from apex.expert_manager import ExpertManager
from apex.metrics import Metrics
from apex.quantized_safetensors_store import QuantizedSafetensorsExpertStore
from apex.real_router import MultiLayerRealRouter
from apex.runtime import Runtime
from apex.weight_layout import get_layout
from scripts._real_moe_helpers import (DEFAULT_GEMMA_PATH, add_real_moe_args)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    add_real_moe_args(p, include_layer=False, default_max_memory_mb=512)
    p.add_argument("--num-layers", type=int, default=2,
                   help="how many real layers to drive (default: 2)")
    args = p.parse_args()

    path = Path(args.path)
    if not path.is_dir():
        raise SystemExit(f"checkpoint not found at {path}")

    layout = get_layout("gemma-moe")
    router = MultiLayerRealRouter(path, layout, top_k=args.top_k)
    quant_store = QuantizedSafetensorsExpertStore(path, layout=layout)
    store = DequantizingExpertStore(quant_store, verbose=False)
    metrics = Metrics()
    cache = ExpertCache(max_memory_mb=args.max_memory_mb,
                        verbose=False, metrics=metrics)
    manager = ExpertManager(store=store, cache=cache, metrics=metrics)
    backend = FakeBackend()

    rng = np.random.default_rng(args.seed)
    # Probe hidden_dim from layer 0's router so we don't hardcode 2816.
    initial = rng.standard_normal((1, router._load_for_layer(0).shape[1])).astype(np.float32)

    runtime = Runtime(
        router=router,
        expert_manager=manager,
        backend=backend,
        num_layers=args.num_layers,
        initial_hidden=initial,
    )

    print("=== run 1 (cold cache) ===")
    out_first = runtime.run()
    expected_misses = args.top_k * args.num_layers
    assert metrics.cache_misses == expected_misses
    assert metrics.cache_hits == 0
    assert metrics.evictions == 0
    assert np.all(np.isfinite(out_first))
    print(f"shape={out_first.shape} finite=True misses={metrics.cache_misses}")

    print("\n=== run 2 (warm cache) ===")
    # Reset metrics counters so we measure the second run cleanly.
    misses_baseline = metrics.cache_misses
    out_second = runtime.run()
    new_misses = metrics.cache_misses - misses_baseline
    assert new_misses == 0, f"second run re-loaded {new_misses}"
    assert metrics.cache_hits == expected_misses
    np.testing.assert_array_equal(out_first, out_second)
    print(f"shape={out_second.shape} finite=True hits={metrics.cache_hits}")

    print("\n[ok] Runtime drives real Gemma layers end-to-end")


if __name__ == "__main__":
    main()
```

**Expected output:**

```
=== run 1 (cold cache) ===
running layer 0
router backend: numpy
router selected experts: [121, 73, 16, 96, 23, 17, 66, 78]
cache miss: loading layer 0 expert 121
... (8 misses)
running layer 1
router backend: numpy
router selected experts: [...]
cache miss: loading layer 1 expert ...
... (8 more misses)
shape=(1, 2816) finite=True misses=16

=== run 2 (warm cache) ===
running layer 0
router backend: numpy
router selected experts: [121, 73, 16, 96, 23, 17, 66, 78]
cache hit: layer 0 expert 121
... (8 hits)
running layer 1
... (8 more hits)
shape=(1, 2816) finite=True hits=16

[ok] Runtime drives real Gemma layers end-to-end
```

### 12.6 — Tests `tests/test_runtime_real_integration.py`

Three synthetic-NVFP4 tests + one real-Gemma skipif:

```python
"""Tests for step 12: Runtime drives real-checkpoint MoE layers.

Synthetic NVFP4 covers the contracts; real-Gemma test is the
integration check that we'd actually catch at the model level.
"""

import numpy as np
import pytest

from apex.backends.fake_backend import FakeBackend
from apex.dequant import DequantizingExpertStore
from apex.expert_cache import ExpertCache
from apex.expert_manager import ExpertManager
from apex.metrics import Metrics
from apex.quantized_safetensors_store import QuantizedSafetensorsExpertStore
from apex.runtime import Runtime
from apex.weight_layout import FakeMoELayout
from tests._synthetic_nvfp4 import (build_one_layer_checkpoint,
                                    skip_if_no_real_checkpoint)


class _StubRouter:
    """Router whose route() returns hand-fed (ids, weights) per layer."""
    name = "numpy"

    def __init__(self, plan):
        self.plan = plan  # {layer_id: (ids, weights_or_None)}

    def route(self, layer_id, hidden):
        return self.plan[layer_id]


def test_runtime_uses_router_weights_for_combine(tmp_path):
    """Router-supplied weights → weighted combine, not uniform mean.

    Construct a 2-expert layer with experts that produce different
    per-expert outputs (because hidden differs from ones, so the
    SwiGLU output of expert 0 ≠ expert 1 even though the weights
    are identical ones-experts). With weights [0.9, 0.1] the
    combined output should be 0.9*out0 + 0.1*out1 — verify by
    computing each expert's output independently and the weighted
    sum, then comparing to Runtime's output.
    """
    layout = FakeMoELayout()
    path = tmp_path / "ck.safetensors"
    build_one_layer_checkpoint(path, layout, num_experts=2,
                               hidden=16, intermediate=32)
    quant_store = QuantizedSafetensorsExpertStore(str(path), layout)
    store = DequantizingExpertStore(quant_store)
    cache = ExpertCache(max_memory_mb=4, verbose=False)
    manager = ExpertManager(store=store, cache=cache)
    backend = FakeBackend()

    initial = np.full((1, 16), 0.5, dtype=np.float32)  # not ones, so SwiGLU
                                                         # produces non-trivial values
    router = _StubRouter({0: ([0, 1], [0.9, 0.1])})
    runtime = Runtime(router, manager, backend, num_layers=1,
                      initial_hidden=initial)

    runtime_out = runtime.run()

    # Recompute expected output by hand: load each expert + run SwiGLU
    # + weight-sum.
    exp0 = backend.run_expert(initial, manager.ensure_loaded(0, 0))
    exp1 = backend.run_expert(initial, manager.ensure_loaded(0, 1))
    expected = 0.9 * exp0 + 0.1 * exp1
    np.testing.assert_allclose(runtime_out, expected, rtol=1e-5)


def test_runtime_uniform_when_router_returns_none(tmp_path):
    """Router returns weights=None ⇒ Runtime substitutes uniform combine.

    This is the back-compat invariant: every existing fake router
    returns (ids, None), so its output must equal pre-step-12
    `mean(per-expert outputs)`.
    """
    # Same setup as above.
    layout = FakeMoELayout()
    path = tmp_path / "ck.safetensors"
    build_one_layer_checkpoint(path, layout, num_experts=2,
                               hidden=16, intermediate=32)
    store = DequantizingExpertStore(QuantizedSafetensorsExpertStore(str(path), layout))
    cache = ExpertCache(max_memory_mb=4, verbose=False)
    manager = ExpertManager(store=store, cache=cache)
    backend = FakeBackend()
    initial = np.full((1, 16), 0.5, dtype=np.float32)

    router = _StubRouter({0: ([0, 1], None)})  # None → uniform
    out = Runtime(router, manager, backend, num_layers=1,
                  initial_hidden=initial).run()

    exp0 = backend.run_expert(initial, manager.ensure_loaded(0, 0))
    exp1 = backend.run_expert(initial, manager.ensure_loaded(0, 1))
    np.testing.assert_allclose(out, 0.5 * exp0 + 0.5 * exp1, rtol=1e-5)


def test_runtime_accepts_arbitrary_hidden_dim(tmp_path):
    """Runtime no longer hardcodes hidden=8; works with hidden=64."""
    layout = FakeMoELayout()
    path = tmp_path / "ck.safetensors"
    build_one_layer_checkpoint(path, layout, num_experts=2,
                               hidden=64, intermediate=128)
    store = DequantizingExpertStore(QuantizedSafetensorsExpertStore(str(path), layout))
    cache = ExpertCache(max_memory_mb=4, verbose=False)
    manager = ExpertManager(store=store, cache=cache)
    backend = FakeBackend()
    initial = np.ones((1, 64), dtype=np.float32)

    router = _StubRouter({0: ([0, 1], None)})
    out = Runtime(router, manager, backend, num_layers=1,
                  initial_hidden=initial).run()
    assert out.shape == (1, 64)
    assert np.all(np.isfinite(out))


@skip_if_no_real_checkpoint()
def test_real_gemma_two_layer_run_then_run():
    """End-to-end on real Gemma 4 NVFP4 through Runtime, not bare smoke script."""
    from apex.real_router import MultiLayerRealRouter
    from apex.weight_layout import get_layout
    from tests._synthetic_nvfp4 import real_checkpoint_path

    path = real_checkpoint_path()
    layout = get_layout("gemma-moe")
    router = MultiLayerRealRouter(path, layout, top_k=8)
    quant_store = QuantizedSafetensorsExpertStore(path, layout=layout)
    store = DequantizingExpertStore(quant_store)
    metrics = Metrics()
    cache = ExpertCache(max_memory_mb=512, verbose=False, metrics=metrics)
    manager = ExpertManager(store=store, cache=cache, metrics=metrics)
    backend = FakeBackend()

    rng = np.random.default_rng(0)
    initial = rng.standard_normal((1, 2816)).astype(np.float32)

    runtime = Runtime(router, manager, backend, num_layers=2,
                      initial_hidden=initial)
    out_first = runtime.run()
    assert metrics.cache_misses == 16
    assert metrics.cache_hits == 0
    assert np.all(np.isfinite(out_first))
    assert out_first.shape == (1, 2816)

    out_second = runtime.run()
    assert metrics.cache_misses == 16  # no new misses
    assert metrics.cache_hits == 16
    assert metrics.evictions == 0
    np.testing.assert_array_equal(out_first, out_second)
```

### 12.7 — Update `OVERVIEW.md`

- Add `apex/real_router.py:MultiLayerRealRouter` to the module map (under `Routers / format adapters`).
- Add `scripts/demo_real_runtime.py` to the script table.
- Move `Plug apex.moe_layer.run_moe_layer into Runtime` from **Future / Improvements** to **What's done** as step 12.
- Move `Real-checkpoint runtime mode` similarly.
- Update the diagram caption: the per-layer compute box is now "run_moe_layer" rather than "Backend.run_expert" so readers know that's the seam.
- Bump the test count: 218 (214 + 4 new step-12 tests).

**Commit message:** `step 12: Runtime drives real Gemma 4 layers via run_moe_layer (router-weighted combine)`

---

## End-to-end verification

After Phase 1 (3 cleanup commits in):

```bash
uv run pytest -q
# Expected: 214 passed (no count change — the test files only got smaller).

# Every fake demo: byte-identical output to before C.1.
for d in demo.py demo_predictor.py demo_gate_router.py demo_safetensors.py \
         demo_gguf.py demo_mmap_store.py demo_mlx_backend.py; do
    uv run python $d > /tmp/before_$d.out
done
# (apply the cleanup commits)
for d in demo.py demo_predictor.py demo_gate_router.py demo_safetensors.py \
         demo_gguf.py demo_mmap_store.py demo_mlx_backend.py; do
    uv run python $d > /tmp/after_$d.out && diff /tmp/before_$d.out /tmp/after_$d.out
done
# Expected: empty diffs.

# Real-checkpoint smoke + demo + two-layer all still produce identical output:
uv run python scripts/smoke_real_moe_layer.py     # same trace
uv run python scripts/demo_real_moe_cache.py      # same trace
uv run python scripts/smoke_real_moe_two_layer.py # same trace
```

After Phase 3 (step 12 commit in):

```bash
uv run pytest -q
# Expected: 218 passed (4 new tests in test_runtime_real_integration.py).

# Existing fake demos: still bit-identical to pre-step-12 output.
# (Critical: uniform-weight-of-N × N outputs == mean(outputs), so Runtime's
# new run_moe_layer-based loop produces the same numerical answer.)

# New demo:
uv run python scripts/demo_real_runtime.py
# Expected:
#   === run 1 (cold cache) ===
#   running layer 0
#   router backend: numpy
#   router selected experts: [121, 73, 16, 96, 23, 17, 66, 78]
#   cache miss: loading layer 0 expert 121
#   ... (etc)
#   shape=(1, 2816) finite=True misses=16
#
#   === run 2 (warm cache) ===
#   running layer 0
#   router backend: numpy
#   router selected experts: [121, 73, 16, 96, 23, 17, 66, 78]
#   cache hit: layer 0 expert 121
#   ... (etc)
#   shape=(1, 2816) finite=True hits=16
#
#   [ok] Runtime drives real Gemma layers end-to-end

# All 3 existing real-checkpoint scripts also still pass:
uv run python scripts/smoke_real_moe_layer.py
uv run python scripts/demo_real_moe_cache.py
uv run python scripts/smoke_real_moe_two_layer.py
```

---

## Critical files

**Phase 1 (Cleanup) — 3 commits, 12 file touches total:**

C.1:
- `apex/runtime.py` — drop `verbose_router` (lines 42, 64, 94)

C.2:
- `tests/_synthetic_nvfp4.py` — NEW
- `tests/test_real_moe_cache.py` — replace local helpers with shared imports
- `tests/test_real_moe_two_layer.py` — same
- `tests/test_quantized_safetensors_store.py` — same
- `tests/test_safetensors_loader.py` — same

C.3:
- `scripts/__init__.py` — NEW (empty)
- `scripts/_real_moe_helpers.py` — NEW
- `scripts/smoke_real_moe_layer.py` — switch to shared bootstrap
- `scripts/demo_real_moe_cache.py` — same
- `scripts/smoke_real_moe_two_layer.py` — same

**Phase 3 (Step 12) — 1 commit, 8 file touches:**

- `apex/runtime.py` — use `run_moe_layer`, accept `initial_hidden`
- `apex/router.py` — return `(ids, None)` from `FakeRouter.route`
- `apex/_numpy_gate_router_base.py` — return `(ids, None)` from base `route`
- `apex/mlx_gate_router.py` — return `(ids, None)`
- `apex/backends/fake_backend.py` — drop `combine_outputs` (now unused)
- `apex/backends/mlx_backend.py` — same
- `apex/real_router.py` — NEW (MultiLayerRealRouter)
- `scripts/demo_real_runtime.py` — NEW
- `tests/test_runtime_real_integration.py` — NEW
- `OVERVIEW.md` — module map + diagram + status updates

---

## What this plan deliberately does NOT do

- **Implement attention / RMSNorm / embedding / lm_head / KV cache / sampling.** Even with step 12 done, APEX still can't generate text from Gemma 4 — the MoE block is just one piece. A "make Gemma 4 actually generate" sprint is a substantially larger lift and not part of "fully integrate steps 10/11."
- **Add MLX path for real-Gemma weights.** `MLXBackend.run_expert` already does HF-orientation SwiGLU on `mx.array`, but `dequantize_bundle` produces ndarray. Bridging requires a `dequantize_bundle_mlx` variant. Worth doing later, not part of this plan.
- **Train a `MarkovPredictor` on real-Gemma transitions.** The predictor needs trace data we don't have (real Gemma routes from real attention output, not from random hidden). The prefetcher itself is layer-agnostic and would Just Work with real-Gemma traces if they existed — this is upstream/data-collection work, not an APEX code gap.
- **Sweep naming, restructure modules, add type hints.** Out of scope for this plan; the user has been clear about preferring focused commits.
- **Touch `apex/types.py`, `apex/expert_cache.py`, `apex/expert_manager.py`, `apex/metrics.py`** — those are stable load-bearing surfaces.

---

## Glossary (so the plan stands alone)

- **HF orientation:** PyTorch convention. `gate_proj.shape == up_proj.shape == (intermediate, hidden)`; `down_proj.shape == (hidden, intermediate)`. Backends compute `h @ proj.T`. APEX standardized on this in step 8 — every store / fixture / writer uses HF orientation, so real Mixtral / Qwen MoE / Gemma 4 / NVFP4 checkpoints flow through `run_expert` with no per-load transpose.
- **NVFP4:** NVIDIA's 4-bit FP quantization. Each value is 4 bits in E2M1 layout (sign | 2-bit exp | 1-bit mantissa, bias=1). Two values pack into one uint8. Two-level scaling: per-block (group_size=16) F8_E4M3 scale + tensor-wide F32 scale. `apex.dequant._dequantize_nvfp4` does the decode.
- **`run_moe_layer`:** `apex/moe_layer.py:78`. Pure function: `(hidden, expert_ids, weights, expert_loader, backend) → combined_output`. Loader is a callable `expert_id → ExpertWeights` so the same function drives fake-MoE and real-Gemma paths.
- **`DequantizingExpertStore`:** `apex/dequant.py`. Wraps any store-of-quantized-bundles so its `load()` returns dequantized fp32 ExpertWeights. The point: the cache holds compute-ready bundles, so cache hits skip the ~60 ms NVFP4 decode.
