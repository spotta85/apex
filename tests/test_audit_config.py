"""Tests for scripts/audit_config.py — script smoke + parsing contract.

We mostly test ModelConfig directly (already covered in
`tests/test_model_config.py`); this file just smokes the audit script
end-to-end on the real Gemma checkpoint so a future regression that
breaks the audit (e.g. import path drift, ANSI escape changes) is
caught at suite time rather than the next time someone runs it.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from tests._synthetic_nvfp4 import (
    real_checkpoint_path,
    skip_if_no_real_checkpoint,
)


_REPO_ROOT = Path(__file__).resolve().parent.parent
_AUDIT_SCRIPT = _REPO_ROOT / "scripts" / "audit_config.py"


@skip_if_no_real_checkpoint()
def test_audit_script_runs_on_real_checkpoint():
    """`scripts/audit_config.py --path <real>` exits 0 and prints expected fields."""
    result = subprocess.run(
        [sys.executable, str(_AUDIT_SCRIPT), "--path", str(real_checkpoint_path())],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"audit failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    out = result.stdout
    # Spot-check a few expected lines so the suite catches refactors
    # that quietly break the audit's output shape.
    for needle in [
        "model config audit:",
        "hidden size: 2816",
        "q heads: 16",
        "kv heads: 8",
        "rope theta (sliding): 10000",
        # Step 23: full-attention extras now appear as ✓ rows.
        "full-attn global_head_dim: 512",
        "full-attn num_global_key_value_heads: 2",
        "full-attn rope_theta: 1000000",
        "full-attn partial_rotary_factor: 0.25",
        "[ok] audit complete",
    ]:
        assert needle in out, f"missing {needle!r} in audit output"


def test_audit_script_clear_error_on_missing_path(tmp_path):
    """Pointing at a directory without config.json fails fast."""
    bogus = tmp_path / "no-config-here"
    bogus.mkdir()
    result = subprocess.run(
        [sys.executable, str(_AUDIT_SCRIPT), "--path", str(bogus)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    # ModelConfig.from_path raises FileNotFoundError; missing
    # config.json triggers a different but equally clear error from
    # read_text_config (FileNotFoundError when opening the file).
    assert "config" in (result.stderr + result.stdout).lower()
