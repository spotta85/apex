"""Tests for FakeMoELayout: tensor-name conventions for FFN-style experts."""

from apex.weight_layout import FakeMoELayout


def test_gate_weight_format():
    layout = FakeMoELayout()
    assert layout.gate_weight(0) == "layers.0.gate.weight"
    assert layout.gate_weight(12) == "layers.12.gate.weight"


def test_expert_projection_formats():
    layout = FakeMoELayout()
    assert layout.expert_gate_proj(0, 1) == "layers.0.experts.1.gate_proj.weight"
    assert layout.expert_up_proj(0, 1) == "layers.0.experts.1.up_proj.weight"
    assert layout.expert_down_proj(0, 1) == "layers.0.experts.1.down_proj.weight"


def test_expert_tensors_returns_all_three_projections():
    # The bundle helper is what stores will call — it must return all
    # three projection names keyed by canonical short names.
    layout = FakeMoELayout()
    names = layout.expert_tensors(0, 1)
    assert names == {
        "gate_proj": "layers.0.experts.1.gate_proj.weight",
        "up_proj": "layers.0.experts.1.up_proj.weight",
        "down_proj": "layers.0.experts.1.down_proj.weight",
    }


def test_expert_and_gate_names_are_distinct():
    # Sanity: gate name shouldn't collide with any of the per-expert
    # projection names. Distinct namespaces inside the same file.
    layout = FakeMoELayout()
    gate = layout.gate_weight(0)
    for expert_id in range(10):
        names = layout.expert_tensors(0, expert_id)
        assert gate not in names.values()


def test_idempotent():
    # Same args -> same name. Stable hashing into the file.
    layout = FakeMoELayout()
    assert layout.expert_gate_proj(2, 3) == layout.expert_gate_proj(2, 3)
    assert layout.gate_weight(5) == layout.gate_weight(5)
    assert layout.expert_tensors(2, 3) == layout.expert_tensors(2, 3)
