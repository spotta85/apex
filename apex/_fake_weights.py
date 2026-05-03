"""Helpers shared by the scripts/make_fake_*_moe.py setup scripts.

The setup scripts all do the same final step: run a fully-wired
runtime once, silently, with a TraceRecorder attached, so the resulting
JSONL file matches what the gate router would actually produce. Lifting
that loop into one place keeps the scripts focused on weight format
and avoids three copies of the same `redirect_stdout` boilerplate.

Underscore-prefixed because this is internal scaffolding, not part of
APEX's API surface.
"""

import contextlib
import io
from pathlib import Path


def record_silent_traces(runtime, output_path, num_runs: int = 10) -> Path:
    """Run `runtime.run()` num_runs times silently and return the trace path.

    Assumes the runtime was constructed with a TraceRecorder writing to
    `output_path`. Wipes the file first so re-running a setup script
    doesn't accumulate stale entries from a previous seed/config.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    # Silence every print the runtime emits — the setup script's own
    # output should be a clean "weights + traces created" summary.
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(num_runs):
            runtime.run()
    return output_path
