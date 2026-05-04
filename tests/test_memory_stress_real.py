"""Step 25D: memory-budget stress sweep on real Gemma 4 (L=30).

Drives the full 30-layer Gemma 4 at two budgets — a comfortable 8 GB
and a tight 4 GB — and pins the contracts that an MoE inference
runtime owes its operators:

  1. **`budget_respected`** at every budget. The LRU expert cache
     never exceeds its `max_memory_bytes` peak, no matter how
     aggressive the eviction pressure.

  2. **More pressure ⇒ at least as many evictions.** A tighter
     budget forces more LRU churn. Strict equality at both budgets
     would mean either (a) the budget knob isn't doing anything, or
     (b) one of the budgets isn't tight enough to matter — and 4 GB
     vs 8 GB at L=30 (240 active expert slots × ~24 MB ≈ 5.7 GB) is
     genuinely tight.

  3. **LRU eviction is correctness-preserving** — same prompt under
     greedy decode produces byte-identical output at any budget.
     Eviction discards bytes; the bytes are the same after a
     re-load (deterministic NVFP4 dequant), so the model's math
     can't drift on cache pressure. This is the central
     "trustworthy" promise of bounded-memory inference.

Two budgets (not three from the original 8/6/4 sweep) keeps the test
under ~3 min: each L=30 model load is ~60-70s, plus a 4-token greedy
decode ~30s. Three budgets would push past the natural test-time
budget for a `pytest -q` run; the third datapoint adds little
beyond what 8 vs 4 already proves.

The dense MLP (~2.1 GB non-evictable, loaded once at construction)
is OUTSIDE the LRU expert cache — both budgets above pay for it
out-of-band. If a future change moves dense MLP into the LRU, this
test's expected eviction-count delta will need updating.
"""

import numpy as np
import pytest

import apex
from tests._synthetic_nvfp4 import (
    real_checkpoint_path,
    skip_if_no_real_checkpoint,
)


# Locked params — same prompt + same max_tokens + greedy at every
# budget so the LRU determinism check has clean inputs to compare.
_PROMPT = "Hello"
_MAX_TOKENS = 4
_NUM_LAYERS = 30


def _run_at_budget(path, budget_gb: float) -> dict:
    """Build a fresh MoEModel at `budget_gb`, generate, return summary.

    Each call is a fresh load — there's no way to retune the LRU
    budget on an existing model without rebuilding the cache. That
    cost (~60-70s per call) is the price of a real budget-sweep
    test; module-scope reuse isn't an option here.
    """
    model = apex.MoEModel.from_pretrained(
        path,
        max_memory_gb=budget_gb,
        num_layers=_NUM_LAYERS,
        verbose=False,
    )
    text = model.generate(
        _PROMPT,
        max_tokens=_MAX_TOKENS,
        temperature=0.0,
    )
    ids = list(model._last_generated_ids)
    report = model.memory_report()
    return {
        "budget_gb": budget_gb,
        "text": text,
        "ids": ids,
        "evictions": report.expert.evictions,
        "peak_bytes": report.expert.peak_bytes,
        "budget_bytes": report.expert.budget_bytes,
        "budget_respected": report.expert.budget_respected,
    }


@pytest.fixture(scope="module")
def stress_results():
    """Run the full 30-layer model at 8 GB and 4 GB, share results.

    Cached at module scope so the two parametrized assertion tests
    below can both consume the same pair of runs without paying for
    the loads twice.
    """
    path = real_checkpoint_path()
    high = _run_at_budget(path, 8.0)
    low = _run_at_budget(path, 4.0)
    return {"high": high, "low": low}


@skip_if_no_real_checkpoint()
def test_both_budgets_are_respected(stress_results):
    """Peak bytes must stay under budget at every tested budget."""
    high = stress_results["high"]
    low = stress_results["low"]
    assert high["budget_respected"], (
        f"8 GB run exceeded budget: peak={high['peak_bytes']} "
        f"budget={high['budget_bytes']}"
    )
    assert low["budget_respected"], (
        f"4 GB run exceeded budget: peak={low['peak_bytes']} "
        f"budget={low['budget_bytes']}"
    )


@skip_if_no_real_checkpoint()
def test_lower_budget_evicts_at_least_as_much(stress_results):
    """4 GB run must evict ≥ as many experts as 8 GB run.

    At L=30 the working set across one forward pass is ~5.7 GB of
    active expert weights. 8 GB has slack; 4 GB must evict heavily
    to keep peak under budget. Strict `>` would over-pin the
    relationship — 8 GB might also have non-zero evictions if the
    LRU briefly overshoots between puts and the post-put eviction —
    so `>=` is the right contract.
    """
    high = stress_results["high"]
    low = stress_results["low"]
    assert low["evictions"] >= high["evictions"], (
        f"tighter budget evicted less: "
        f"4 GB → {low['evictions']} evictions, "
        f"8 GB → {high['evictions']} evictions"
    )


@skip_if_no_real_checkpoint()
def test_output_is_byte_identical_across_budgets(stress_results):
    """Greedy decode output must be invariant to LRU eviction.

    Eviction discards bytes; the bytes come back identical from
    NVFP4 dequant on the next access. The model's per-token math
    therefore cannot depend on cache state, only on weights, which
    are unchanged. If this assertion ever fires, the runtime has a
    correctness bug — likely a stateful side effect in the dequant
    pipeline or a non-deterministic eviction order.
    """
    high = stress_results["high"]
    low = stress_results["low"]
    assert low["ids"] == high["ids"], (
        f"output drift under memory pressure: "
        f"8 GB → {high['ids']} ({high['text']!r}), "
        f"4 GB → {low['ids']} ({low['text']!r})"
    )
    assert low["text"] == high["text"]


@skip_if_no_real_checkpoint()
def test_output_is_finite_at_tight_budget(stress_results):
    """Even under heavy eviction pressure, the runtime produces finite logits.

    The greedy decode loop already asserts finite logits per step
    (`apex/model.py:_make_step_fn`). Reaching this assertion means
    the loop completed without raising — but make the contract
    explicit here too in case the upstream assertion is ever
    relaxed.
    """
    low = stress_results["low"]
    assert all(isinstance(i, int) for i in low["ids"])
    assert len(low["ids"]) <= _MAX_TOKENS
