"""Test-side facade over apex._fake_weights.

Re-exports the shared writers/constants so test files can import them
from `tests._world` (their long-standing import path) without caring
that the implementation now lives in apex/_fake_weights.py. FakeStore
remains here because it's only useful in tests.
"""

from apex._fake_weights import (
    HIDDEN,
    INTERMEDIATE,
    NUM_EXPERTS,
    make_expert_bundle,
    write_gguf_world,
    write_npy_world,
    write_safetensors_world,
)
from apex.types import ExpertWeights


# Make the re-exports visible to `from tests._world import *` consumers
# and silence any "imported but unused" checker on this facade module.
__all__ = [
    "HIDDEN",
    "INTERMEDIATE",
    "NUM_EXPERTS",
    "make_expert_bundle",
    "write_gguf_world",
    "write_npy_world",
    "write_safetensors_world",
    "FakeStore",
]


class FakeStore:
    """In-memory stand-in for ExpertStore — returns ExpertWeights bundles.

    Tests that don't care about disk I/O can use this to construct an
    ExpertManager quickly. The fill value is the expert_id, so the same
    key always returns the same content — useful for identity / hit
    checks across calls.
    """

    def __init__(self, intermediate: int = INTERMEDIATE):
        self.intermediate = intermediate
        self.load_calls = []

    def load(self, layer_id: int, expert_id: int) -> ExpertWeights:
        self.load_calls.append((layer_id, expert_id))
        return make_expert_bundle(fill=float(expert_id), intermediate=self.intermediate)
