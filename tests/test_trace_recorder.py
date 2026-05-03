"""Tests for TraceRecorder: writes valid JSONL, appends across calls."""

import json

from apex.trace_recorder import TraceRecorder


def read_jsonl(path):
    # Helper: parse a JSONL file into a list of dicts.
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def test_records_single_transition(tmp_path):
    # One record() call -> one JSON object on disk with exactly the
    # fields expected by MarkovPredictor.fit().
    path = tmp_path / "traces.jsonl"
    rec = TraceRecorder(path)

    rec.record(current_layer=0, current_experts=[1, 3], next_layer=1, next_experts=[0, 2])

    rows = read_jsonl(path)
    assert rows == [
        {
            "current_layer": 0,
            "current_experts": [1, 3],
            "next_layer": 1,
            "next_experts": [0, 2],
        }
    ]


def test_appends_multiple_records(tmp_path):
    # Successive record() calls accumulate, not overwrite.
    path = tmp_path / "traces.jsonl"
    rec = TraceRecorder(path)

    rec.record(0, [1, 3], 1, [0, 2])
    rec.record(0, [2, 3], 1, [1, 2])
    rec.record(0, [1, 3], 1, [0, 2])

    rows = read_jsonl(path)
    assert len(rows) == 3
    assert rows[1]["current_experts"] == [2, 3]
    assert rows[1]["next_experts"] == [1, 2]


def test_creates_parent_directory(tmp_path):
    # If traces/ doesn't exist yet, the recorder should create it
    # rather than crash on first write.
    path = tmp_path / "deep" / "nested" / "dir" / "traces.jsonl"
    rec = TraceRecorder(path)

    rec.record(0, [1], 1, [0])
    assert path.exists()


def test_normalizes_iterables_to_lists(tmp_path):
    # Tuples and other iterables should serialize as JSON lists, not
    # crash json.dumps. (Tuples encode as lists in JSON anyway, but
    # the recorder should normalize defensively for predictability.)
    path = tmp_path / "traces.jsonl"
    rec = TraceRecorder(path)

    rec.record(0, (1, 3), 1, (0, 2))

    rows = read_jsonl(path)
    assert rows[0]["current_experts"] == [1, 3]
    assert rows[0]["next_experts"] == [0, 2]
