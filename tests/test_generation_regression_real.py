"""Step 25B: deterministic generation regression on real Gemma 4.

Locks in the current per-prompt behavior at a fixed
(num_layers, max_tokens, max_memory_gb, temperature) tuple so that any
future change to the math, the loader, or the sampling path that
silently perturbs greedy decode shows up immediately as a token-id
diff against the fixture — instead of a slow drift in "quality" that
no one notices until a benchmark week later.

The fixture (`tests/fixtures/regression_real_l8.json`) records, per
prompt:

  - `token_ids`      — the deterministic greedy-decode output
  - `text`           — the same ids, decoded (human-readable in CI diff)
  - `hit_eos`        — did the run terminate on EOS, or run to max_tokens
  - `longest_run`    — longest contiguous identical-id run (a degeneration
                       proxy: high values usually mean the model got stuck)

Wall time, absolute peak bytes, and tok/s are NOT asserted (machine-
variant); they're separately captured by `scripts/benchmark_quality.py`.

Eviction count at L=8 is expected to be 0 (8 layers × 8 active experts
× ~24 MB ≈ 1.5 GB, well under the 8 GB cap), so the regression here is
purely about per-token determinism, not memory pressure. The L=30
memory-stress regression lives in step 25D.

To regenerate after an intentional math change, run:

    REGEN_REGRESSION_FIXTURE=1 uv run pytest \\
        tests/test_generation_regression_real.py -v

The regen path overwrites the fixture file with whatever the current
runtime produces. Diff the file in your commit so the change is
visible to a reviewer.
"""

import json
import os
from pathlib import Path

import pytest

import apex
from tests._synthetic_nvfp4 import (
    real_checkpoint_path,
    skip_if_no_real_checkpoint,
)


# Where the snapshot lives. New `tests/fixtures/` directory — the first
# golden-file pattern in the repo. Future regression-style tests can
# share this directory.
FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "regression_real_l8.json"
)
REGEN_ENV = "REGEN_REGRESSION_FIXTURE"

# Locked params for the snapshot. Any change to these numbers
# invalidates the existing fixture and forces a regen — the assertion
# below cross-checks the params on disk against these constants.
_NUM_LAYERS = 8
_MAX_TOKENS = 8
_MAX_MEMORY_GB = 8.0
_TEMPERATURE = 0.0  # greedy ⇒ deterministic given same weights + math

# Prompt set: spans factual / arithmetic / narrative / code so a math
# drift in any one operator surface (norm, attention, MoE combine,
# softcap, …) tends to show up in at least one prompt's token
# sequence.
_PROMPTS = (
    "The capital of France is",
    "2 + 2 =",
    "Once upon a time",
    "def fibonacci(n):",
)


@pytest.fixture(scope="module")
def model_l8():
    """Load the real-Gemma model once for all regression assertions.

    Module scope amortizes the ~30-60s checkpoint load (embedding
    dequant + 8 layers' attention + dense MLP) across every prompt
    parameterization.
    """
    return apex.MoEModel.from_pretrained(
        real_checkpoint_path(),
        max_memory_gb=_MAX_MEMORY_GB,
        num_layers=_NUM_LAYERS,
        verbose=False,
    )


def _longest_repeat_run(ids: list[int]) -> int:
    """Length of the longest contiguous run of identical ids in `ids`.

    Mirrors the helper in `scripts/benchmark_quality.py` — kept local
    so the test file stays self-contained (the script imports `apex`,
    not `tests`).
    """
    if not ids:
        return 0
    best = cur = 1
    prev = ids[0]
    for x in ids[1:]:
        if x == prev:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
            prev = x
    return best


def _run_one(model, prompt: str) -> dict:
    """Greedy-decode `prompt` for `_MAX_TOKENS` and return the snapshot dict.

    Pulls token ids from `_last_generated_ids` (authoritative — the
    decoded text round-trip can be lossy when the model emits unusual
    bytes, which is exactly what L=8 does today).
    """
    text = model.generate(
        prompt,
        max_tokens=_MAX_TOKENS,
        temperature=_TEMPERATURE,
    )
    ids = list(model._last_generated_ids)
    eos_set = set(model.config.eos_token_ids)
    return {
        "token_ids": ids,
        "text": text,
        "hit_eos": any(i in eos_set for i in ids),
        "longest_run": _longest_repeat_run(ids),
    }


def _read_fixture() -> dict:
    """Load the JSON snapshot; fail loudly if absent or stale."""
    if not FIXTURE_PATH.exists():
        pytest.fail(
            f"regression fixture missing at {FIXTURE_PATH}. "
            f"Run: {REGEN_ENV}=1 uv run pytest "
            f"tests/test_generation_regression_real.py"
        )
    with FIXTURE_PATH.open() as f:
        return json.load(f)


def _write_fixture(data: dict) -> None:
    """Overwrite the fixture file with `data`. Creates the dir if needed."""
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FIXTURE_PATH.open("w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _assert_params_match(fixture: dict) -> None:
    """Cross-check fixture params against the constants above.

    The snapshot is only meaningful at a fixed (layers, max_tokens,
    budget, temperature) tuple. If anyone bumps a constant without
    regenerating, fail with an actionable message instead of a
    confusing per-prompt id mismatch.
    """
    params = fixture.get("params", {})
    expected = {
        "num_layers": _NUM_LAYERS,
        "max_tokens": _MAX_TOKENS,
        "max_memory_gb": _MAX_MEMORY_GB,
        "temperature": _TEMPERATURE,
    }
    if params != expected:
        pytest.fail(
            f"fixture params {params} do not match test constants {expected}. "
            f"Set {REGEN_ENV}=1 to regenerate after an intentional change."
        )


@skip_if_no_real_checkpoint()
@pytest.mark.parametrize(
    "prompt",
    _PROMPTS,
    ids=lambda p: p.replace(" ", "_").replace("(", "").replace(")", "")[:24],
)
def test_generation_regression(model_l8, prompt):
    """Each prompt's deterministic greedy output must match the fixture.

    Two modes:

      - default: read the fixture, assert per-prompt fields match.
      - REGEN_REGRESSION_FIXTURE=1: regenerate the fixture in place
        (the parametrize loop runs each prompt and the last call
        writes the consolidated dict). Used when an intentional math
        change has perturbed the snapshots and you've reviewed the
        diff.
    """
    actual = _run_one(model_l8, prompt)

    if os.environ.get(REGEN_ENV) == "1":
        # Build / merge into the on-disk fixture. We can't atomically
        # write all four prompts in one shot under parametrize, so we
        # merge against whatever's there and overwrite — the result
        # converges after a full pytest run.
        fixture = (
            json.loads(FIXTURE_PATH.read_text())
            if FIXTURE_PATH.exists()
            else {"params": {}, "prompts": {}}
        )
        fixture["params"] = {
            "num_layers": _NUM_LAYERS,
            "max_tokens": _MAX_TOKENS,
            "max_memory_gb": _MAX_MEMORY_GB,
            "temperature": _TEMPERATURE,
        }
        fixture.setdefault("prompts", {})[prompt] = actual
        _write_fixture(fixture)
        return

    fixture = _read_fixture()
    _assert_params_match(fixture)
    expected = fixture.get("prompts", {}).get(prompt)
    if expected is None:
        pytest.fail(
            f"prompt {prompt!r} missing from fixture. "
            f"Set {REGEN_ENV}=1 to add it."
        )
    # Compare field-by-field for clear diagnostics on failure.
    assert actual["token_ids"] == expected["token_ids"], (
        f"token id drift for {prompt!r}: "
        f"actual={actual['token_ids']} expected={expected['token_ids']}"
    )
    assert actual["text"] == expected["text"], (
        f"decoded-text drift for {prompt!r}: "
        f"actual={actual['text']!r} expected={expected['text']!r}"
    )
    assert actual["hit_eos"] == expected["hit_eos"], (
        f"EOS-hit changed for {prompt!r}: "
        f"actual={actual['hit_eos']} expected={expected['hit_eos']}"
    )
    assert actual["longest_run"] == expected["longest_run"], (
        f"longest_run changed for {prompt!r}: "
        f"actual={actual['longest_run']} expected={expected['longest_run']}"
    )


@skip_if_no_real_checkpoint()
def test_finite_logits_per_prompt(model_l8):
    """Every prompt's forward must produce finite logits.

    Cheap secondary check that runs the same prompts through `forward`
    (no decode loop) and asserts finiteness. If a future math change
    introduces a NaN/inf path, this trips before the slower
    parametrized regression even runs.
    """
    import numpy as np
    for prompt in _PROMPTS:
        logits = model_l8.forward(prompt)
        assert np.all(np.isfinite(logits)), (
            f"non-finite logits for prompt {prompt!r}"
        )
