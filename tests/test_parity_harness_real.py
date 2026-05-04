"""Step 25A: tests for `MoEModel.forward_with_trace` on real Gemma 4.

The harness has two contracts:

  1. Schema contract: trace dict has the documented fields, with one
     entry per layer, top-K final logits, and JSON-serializable
     values throughout.

  2. No-mutation contract: calling `forward_with_trace(p)` produces
     the SAME logits as `forward(p)` would. Tracing must be a side
     channel, not a code path that changes the math. If this fails,
     the trace is meaningless because it's reporting a different
     forward pass than production.

A 4-layer fixture is enough to exercise both contracts without paying
for a 30-layer load — the per-layer trace logic is depth-independent.
"""

import json
import math

import numpy as np
import pytest

import apex
from tests._synthetic_nvfp4 import (
    real_checkpoint_path,
    skip_if_no_real_checkpoint,
)


@pytest.fixture(scope="module")
def model_4l():
    """Reuse the same 4-layer setup the logit-sanity tests use."""
    return apex.MoEModel.from_pretrained(
        real_checkpoint_path(),
        max_memory_gb=4.0,
        num_layers=4,
        verbose=False,
    )


@skip_if_no_real_checkpoint()
def test_trace_has_documented_schema(model_4l):
    """Trace dict must carry the keys + per-layer fields the harness
    documents — schema drift means downstream consumers (parity script,
    OVERVIEW examples) silently break."""
    _, trace = model_4l.forward_with_trace("Hello", top_k_logits=10)

    # Top-level fields
    assert set(trace.keys()) >= {
        "prompt", "input_ids", "num_layers", "layers", "top_k_logits",
    }
    assert trace["prompt"] == "Hello"
    assert isinstance(trace["input_ids"], list)
    assert trace["num_layers"] == 4

    # Per-layer fields. Every layer must have all keys; dense_out_norm
    # is None on layouts without a parallel dense MLP, so we don't
    # assert non-None on it.
    layers = trace["layers"]
    assert len(layers) == 4
    for layer in layers:
        assert set(layer.keys()) >= {
            "layer_id",
            "router",
            "attn_out_norm",
            "moe_out_norm",
            "dense_out_norm",
            "post_layer_hidden_norm",
        }
        # Norms are positive finite floats.
        for k in ("attn_out_norm", "moe_out_norm", "post_layer_hidden_norm"):
            v = layer[k]
            assert isinstance(v, float)
            assert math.isfinite(v)
            assert v > 0
        # Router decision shape: top_k experts + weights, lengths match.
        router = layer["router"]
        assert isinstance(router["expert_ids"], list)
        assert isinstance(router["weights"], list)
        assert len(router["expert_ids"]) == len(router["weights"])
        # Gemma 4 ⇒ top-k = 8.
        assert len(router["expert_ids"]) == model_4l.config.top_k_experts

    # Top-K logits: 10 (id, value) pairs, sorted descending by value.
    top = trace["top_k_logits"]
    assert len(top) == 10
    for i in range(len(top) - 1):
        assert top[i][1] >= top[i + 1][1], (
            "top_k_logits must be sorted descending"
        )


@skip_if_no_real_checkpoint()
def test_trace_logits_match_forward(model_4l):
    """`forward_with_trace` must produce the same logits as `forward`.

    Tracing populates a side dict; it must not change the math. If
    this fails, every diff captured by the harness is comparing two
    different forward passes and is therefore meaningless.
    """
    prompt = "The capital of France is"
    plain = model_4l.forward(prompt)
    traced, _ = model_4l.forward_with_trace(prompt)
    np.testing.assert_array_equal(plain, traced)


@skip_if_no_real_checkpoint()
def test_trace_dense_out_norm_present_on_gemma4(model_4l):
    """Gemma 4 ships parallel dense MLP per layer ⇒ dense_out_norm
    populated (non-None) at every layer."""
    _, trace = model_4l.forward_with_trace("Hi")
    for layer in trace["layers"]:
        assert layer["dense_out_norm"] is not None, (
            f"layer {layer['layer_id']} missing dense_out_norm — "
            f"either dense MLP loading regressed or the trace forgot "
            f"to populate the field"
        )
        assert math.isfinite(layer["dense_out_norm"])
        assert layer["dense_out_norm"] > 0


@skip_if_no_real_checkpoint()
def test_trace_is_json_serializable(model_4l, tmp_path):
    """Whole trace must round-trip through JSON without dtype errors.

    Catches the common bug where a numpy scalar (np.float32, np.int64)
    sneaks in instead of a python float/int, which json.dump rejects.
    """
    _, trace = model_4l.forward_with_trace("hello world")
    out = tmp_path / "trace.json"
    with out.open("w") as f:
        json.dump(trace, f)
    # And it must round-trip identically.
    with out.open() as f:
        back = json.load(f)
    assert back["prompt"] == trace["prompt"]
    assert back["num_layers"] == trace["num_layers"]
    assert len(back["layers"]) == len(trace["layers"])


@skip_if_no_real_checkpoint()
def test_trace_router_decision_changes_across_layers(model_4l):
    """Different layers should generally route to different expert sets.

    Catches a degenerate bug where every layer's router collapses to
    the same top-k decision (which would suggest the per-layer
    `proj.weight` isn't being loaded distinctly). Strict "all layers
    differ" can flake for a small prompt; assert at least 2 of the 4
    layers' top-k sets differ from each other.
    """
    _, trace = model_4l.forward_with_trace("Hello")
    sets = [tuple(layer["router"]["expert_ids"]) for layer in trace["layers"]]
    distinct = len(set(sets))
    assert distinct >= 2, (
        f"all 4 layers picked identical top-k experts {sets[0]} — "
        "router weights are likely not loaded per-layer"
    )
