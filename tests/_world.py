"""Shared helpers for setting up fake-MoE worlds in tests.

After step 4, every test that needs an on-disk MoE has to write *three*
projection tensors per expert. Centralizing the setup here keeps each
test file focused on what it's actually verifying.
"""

import numpy as np
from pathlib import Path

from apex.types import ExpertWeights
from apex.weight_layout import FakeMoELayout


# Sizes used by tests by default. Tiny so writes are cheap; large
# enough that matmul shapes are non-trivial.
HIDDEN = 8
INTERMEDIATE = 16
NUM_EXPERTS = 4


def make_expert_bundle(
    fill: float = 1.0,
    hidden: int = HIDDEN,
    intermediate: int = INTERMEDIATE,
    dtype: str = "float32",
) -> ExpertWeights:
    """Build an ExpertWeights with constant-filled projections.

    Useful when the test only cares about plumbing (cache accounting,
    flow through manager) rather than numerical correctness.
    """
    return ExpertWeights(
        gate_proj=np.full((hidden, intermediate), fill, dtype=dtype),
        up_proj=np.full((hidden, intermediate), fill, dtype=dtype),
        down_proj=np.full((intermediate, hidden), fill, dtype=dtype),
    )


def write_npy_world(
    root: Path,
    num_layers: int = 2,
    num_experts: int = NUM_EXPERTS,
    hidden: int = HIDDEN,
    intermediate: int = INTERMEDIATE,
    seed: int = 42,
    with_gates: bool = True,
) -> Path:
    """Materialize an experts/-style npy directory tree in `root`.

    Layout matches `scripts/make_fake_moe_weights.py`:
      <root>/layer_<L>/gate.npy                       (if with_gates)
      <root>/layer_<L>/expert_<E>/{gate,up,down}_proj.npy

    Returns `root` for convenient chaining.
    """
    rng = np.random.default_rng(seed)
    for layer in range(num_layers):
        layer_dir = root / f"layer_{layer}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        if with_gates:
            np.save(
                layer_dir / "gate.npy",
                rng.standard_normal((hidden, num_experts)).astype("float32"),
            )
        for expert in range(num_experts):
            expert_dir = layer_dir / f"expert_{expert}"
            expert_dir.mkdir(parents=True, exist_ok=True)
            np.save(
                expert_dir / "gate_proj.npy",
                rng.standard_normal((hidden, intermediate)).astype("float32"),
            )
            np.save(
                expert_dir / "up_proj.npy",
                rng.standard_normal((hidden, intermediate)).astype("float32"),
            )
            np.save(
                expert_dir / "down_proj.npy",
                rng.standard_normal((intermediate, hidden)).astype("float32"),
            )
    return root


def write_safetensors_world(
    path: Path,
    num_layers: int = 2,
    num_experts: int = NUM_EXPERTS,
    hidden: int = HIDDEN,
    intermediate: int = INTERMEDIATE,
    seed: int = 42,
    layout=None,
) -> Path:
    """Write a fake-MoE .safetensors file with the canonical tensor names."""
    from safetensors.numpy import save_file

    if layout is None:
        layout = FakeMoELayout()

    rng = np.random.default_rng(seed)
    tensors = {}
    for layer in range(num_layers):
        tensors[layout.gate_weight(layer)] = rng.standard_normal(
            (hidden, num_experts)
        ).astype("float32")
        for expert in range(num_experts):
            names = layout.expert_tensors(layer, expert)
            tensors[names["gate_proj"]] = rng.standard_normal(
                (hidden, intermediate)
            ).astype("float32")
            tensors[names["up_proj"]] = rng.standard_normal(
                (hidden, intermediate)
            ).astype("float32")
            tensors[names["down_proj"]] = rng.standard_normal(
                (intermediate, hidden)
            ).astype("float32")
    save_file(tensors, str(path))
    return path


def write_gguf_world(
    path: Path,
    num_layers: int = 2,
    num_experts: int = NUM_EXPERTS,
    hidden: int = HIDDEN,
    intermediate: int = INTERMEDIATE,
    seed: int = 42,
    layout=None,
) -> Path:
    """Write a fake-MoE .gguf file with the canonical tensor names."""
    from gguf import GGUFWriter

    if layout is None:
        layout = FakeMoELayout()

    rng = np.random.default_rng(seed)
    writer = GGUFWriter(str(path), "probe")
    for layer in range(num_layers):
        writer.add_tensor(
            layout.gate_weight(layer),
            rng.standard_normal((hidden, num_experts)).astype("float32"),
        )
        for expert in range(num_experts):
            names = layout.expert_tensors(layer, expert)
            writer.add_tensor(
                names["gate_proj"],
                rng.standard_normal((hidden, intermediate)).astype("float32"),
            )
            writer.add_tensor(
                names["up_proj"],
                rng.standard_normal((hidden, intermediate)).astype("float32"),
            )
            writer.add_tensor(
                names["down_proj"],
                rng.standard_normal((intermediate, hidden)).astype("float32"),
            )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


class FakeStore:
    """In-memory stand-in for ExpertStore — returns ExpertWeights bundles.

    Tests that don't care about disk I/O can use this to construct an
    ExpertManager quickly. `fill_seed=expert_id` (default) means
    different experts produce visibly different bundles for assertions.
    """

    def __init__(self, intermediate: int = INTERMEDIATE):
        self.intermediate = intermediate
        self.load_calls = []

    def load(self, layer_id: int, expert_id: int) -> ExpertWeights:
        self.load_calls.append((layer_id, expert_id))
        # Use expert_id as the fill so the same key returns the same
        # value across calls — handy for identity / hit checks.
        return make_expert_bundle(fill=float(expert_id), intermediate=self.intermediate)
