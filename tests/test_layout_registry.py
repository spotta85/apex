"""Tests for the layout registry: get_layout(name) -> instance."""

import pytest

from apex.weight_layout import (
    FakeMoELayout,
    GemmaMoELayout,
    MixtralLayout,
    QwenMoELayout,
    get_layout,
)


@pytest.mark.parametrize(
    "name,expected_cls",
    [
        ("fake", FakeMoELayout),
        ("mixtral", MixtralLayout),
        ("qwen-moe", QwenMoELayout),
        ("gemma-moe", GemmaMoELayout),
    ],
)
def test_get_layout_returns_instance_of_expected_class(name, expected_cls):
    layout = get_layout(name)
    assert isinstance(layout, expected_cls)


def test_get_layout_returns_a_fresh_instance_each_call():
    # Two calls should produce two distinct objects so callers can't
    # accidentally share mutable state across components.
    a = get_layout("fake")
    b = get_layout("fake")
    assert a is not b


def test_get_layout_unknown_name_raises_key_error_with_known_names():
    with pytest.raises(KeyError) as excinfo:
        get_layout("not-a-real-layout")
    msg = str(excinfo.value)
    # The message should name what was missing AND what is available.
    assert "not-a-real-layout" in msg
    for known in ("fake", "mixtral", "qwen-moe", "gemma-moe"):
        assert known in msg
