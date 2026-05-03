"""Records (current_layer, current_experts) -> (next_layer, next_experts)
transitions to a JSONL file as the runtime executes.

This is the data source for MarkovPredictor.fit(): once you've recorded
enough real router decisions, you can train a predictor on them and
have it prefetch future-layer experts without needing to peek the
router. In the MVP it doubles as a debug aid — printing a one-liner per
recorded transition lets you watch what the runtime is committing to
disk.

Append mode by design: traces accumulate across runs so the predictor
learns from the user's actual usage over time. Truncate the file
manually (`rm traces/runtime_traces.jsonl`) to start fresh.
"""

import json
from pathlib import Path


class TraceRecorder:
    def __init__(self, output_path):
        self.path = Path(output_path)
        # Make sure the parent dir exists; first-time runs shouldn't
        # crash because traces/ doesn't exist yet.
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, current_layer: int, current_experts, next_layer: int, next_experts):
        # Normalize to plain lists so the JSON is stable regardless of
        # whether the caller passed a list, tuple, or numpy array.
        record = {
            "current_layer": current_layer,
            "current_experts": list(current_experts),
            "next_layer": next_layer,
            "next_experts": list(next_experts),
        }
        # Append one JSON object per line. Open/close per record keeps
        # things crash-safe (no buffering loss) at the cost of some I/O
        # — fine for MVP scale.
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
        # Show only the current-layer half in the trace; the next-layer
        # half is what the predictor is about to *guess at*, and we
        # don't want to give the impression we're peeking.
        print(f"trace recorder: layer {current_layer} experts {list(current_experts)}")
