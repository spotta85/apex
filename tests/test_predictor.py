"""Tests for MarkovPredictor: training from JSONL and top-k prediction."""

import json
import textwrap

import pytest

from apex.predictor import MarkovPredictor


def write_traces(tmp_path, lines):
    # Helper: write a list of dicts to a JSONL file in a tmp dir.
    path = tmp_path / "traces.jsonl"
    with open(path, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    return path


def test_fit_reads_jsonl(tmp_path):
    # After fit(), the predictor should have one transition entry per
    # unique (current_layer, sorted-experts, next_layer) key.
    path = write_traces(
        tmp_path,
        [
            {"current_layer": 0, "current_experts": [1, 3], "next_layer": 1, "next_experts": [0, 2]},
            {"current_layer": 0, "current_experts": [1, 3], "next_layer": 1, "next_experts": [0, 2]},
        ],
    )
    p = MarkovPredictor()
    p.fit(path)

    key = (0, (1, 3), 1)
    assert key in p.transitions
    # Each line contributes one count per next-expert mentioned.
    assert p.transitions[key][0] == 2
    assert p.transitions[key][2] == 2


def test_predicts_top_k_by_count(tmp_path):
    # If multiple next-experts compete, top-k returns the most-counted ones.
    path = write_traces(
        tmp_path,
        # 3x [0,2] vs 1x [4,5] -> 0 and 2 win.
        [
            {"current_layer": 0, "current_experts": [1, 3], "next_layer": 1, "next_experts": [0, 2]},
            {"current_layer": 0, "current_experts": [1, 3], "next_layer": 1, "next_experts": [0, 2]},
            {"current_layer": 0, "current_experts": [1, 3], "next_layer": 1, "next_experts": [0, 2]},
            {"current_layer": 0, "current_experts": [1, 3], "next_layer": 1, "next_experts": [4, 5]},
        ],
    )
    p = MarkovPredictor()
    p.fit(path)

    top2 = p.predict(current_layer=0, current_experts=[1, 3], next_layer=1, k=2)
    assert top2 == [0, 2]


def test_sorted_current_experts_key(tmp_path):
    # The key should normalize current_experts so that input order
    # doesn't matter when querying.
    path = write_traces(
        tmp_path,
        [
            {"current_layer": 0, "current_experts": [1, 3], "next_layer": 1, "next_experts": [0, 2]},
        ],
    )
    p = MarkovPredictor()
    p.fit(path)

    # Same set of experts but different input order -> same prediction.
    a = p.predict(current_layer=0, current_experts=[1, 3], next_layer=1, k=2)
    b = p.predict(current_layer=0, current_experts=[3, 1], next_layer=1, k=2)
    assert a == b == [0, 2]


def test_unknown_transition_returns_empty(tmp_path):
    # Querying a key the predictor has never seen should return [],
    # signaling the caller to fall back (e.g. to the router).
    path = write_traces(
        tmp_path,
        [
            {"current_layer": 0, "current_experts": [1, 3], "next_layer": 1, "next_experts": [0, 2]},
        ],
    )
    p = MarkovPredictor()
    p.fit(path)

    # Different current_experts -> never seen -> empty.
    assert p.predict(current_layer=0, current_experts=[5, 7], next_layer=1, k=2) == []
    # Different layer -> never seen -> empty.
    assert p.predict(current_layer=2, current_experts=[1, 3], next_layer=3, k=2) == []


def test_fit_skips_blank_lines(tmp_path):
    # Robustness: blank lines in the JSONL file shouldn't crash fit().
    path = tmp_path / "traces.jsonl"
    with open(path, "w") as f:
        f.write('{"current_layer": 0, "current_experts": [1], "next_layer": 1, "next_experts": [0]}\n')
        f.write("\n")
        f.write("\n")
        f.write('{"current_layer": 0, "current_experts": [1], "next_layer": 1, "next_experts": [0]}\n')

    p = MarkovPredictor()
    p.fit(path)
    assert p.transitions[(0, (1,), 1)][0] == 2
