"""Markov-style predictor for next-layer experts.

Trains on a JSONL file of (current_layer, current_experts, next_layer,
next_experts) records. Internally stores a Counter per (layer, sorted
current-expert tuple, next_layer) key, counting how often each next
expert appeared. `predict()` returns the top-k most common next experts
for that key, or [] if the key was never seen — letting the runtime
fall back to the router for unknown contexts.

Why "Markov": prediction is conditioned only on the *current* layer's
expert set, not on history. Cheap to train, cheap to query, good enough
to demonstrate that learned prefetch can beat blind router peeks.

Why sorted tuple as key: the router's expert IDs are an unordered set
("which experts are active"), not a sequence. Sorting normalizes
[1,3] and [3,1] into the same key.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path


class MarkovPredictor:
    def __init__(self):
        # Key: (current_layer, tuple(sorted(current_experts)), next_layer)
        # Value: Counter mapping next-expert-id -> observation count.
        # defaultdict(Counter) means new keys auto-init cleanly.
        self.transitions: dict = defaultdict(Counter)

    def fit(self, trace_path):
        # Read every JSONL record and bump the per-(key, next_expert) count.
        # Multiple calls accumulate (you can fit on additional traces later).
        path = Path(trace_path)
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                key = (
                    record["current_layer"],
                    tuple(sorted(record["current_experts"])),
                    record["next_layer"],
                )
                # Each next expert in the record contributes one count.
                for expert_id in record["next_experts"]:
                    self.transitions[key][expert_id] += 1

    def predict(self, current_layer: int, current_experts, next_layer: int, k: int):
        # Returns a list of the top-k most common next-layer expert IDs
        # observed for this (layer, sorted-experts, next_layer) key.
        # Returns [] if we've never seen this transition — caller
        # should fall back to a different source (e.g. the router).
        key = (current_layer, tuple(sorted(current_experts)), next_layer)
        counter = self.transitions.get(key)
        if counter is None:
            return []
        return [expert_id for expert_id, _count in counter.most_common(k)]
