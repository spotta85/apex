"""Helpers for generating fake-MoE weights and recording warmup traces.

Single source of truth for the fake-MoE shape so the setup scripts
(scripts/make_fake_*_moe.py) and the test fixtures don't drift apart.
Three writers cover the three on-disk formats APEX supports:

  - write_npy_world(root, ...)            — directory tree of .npy files
  - write_safetensors_world(path, ...)    — single .safetensors file
  - write_gguf_world(path, ...)           — single .gguf file

All writers share the same shape constants (HIDDEN, INTERMEDIATE,
NUM_EXPERTS) and the same RNG (np.random.default_rng(seed)) so a
given seed produces matching weights across formats — useful for
cross-format comparison.

Also houses record_silent_traces, the trace-capture helper used by
all three setup scripts to seed the predictor demos.

Underscore-prefixed because this is internal scaffolding, not part of
APEX's public API surface.
"""

import contextlib
import io
from pathlib import Path

import numpy as np

from apex.types import ExpertWeights
from apex.weight_layout import FakeMoELayout


# Default sizes for fake-MoE weights. Tiny so writes are cheap; large
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

    Shapes follow HF (PyTorch) convention — weights are stored as
    (out_features, in_features) — so they match every real MoE
    checkpoint without a per-load transpose. The backends' SwiGLU
    math compensates by multiplying against `.T`.
    """
    return ExpertWeights(
        gate_proj=np.full((intermediate, hidden), fill, dtype=dtype),
        up_proj=np.full((intermediate, hidden), fill, dtype=dtype),
        down_proj=np.full((hidden, intermediate), fill, dtype=dtype),
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

    Layout:
      <root>/layer_<L>/gate.npy                       (if with_gates)
      <root>/layer_<L>/expert_<E>/{gate,up,down}_proj.npy

    Projection shapes follow HF (PyTorch) convention: gate/up are
    (intermediate, hidden), down is (hidden, intermediate). The gate
    router weight stays (hidden, num_experts) since APEX's gate routers
    do `hidden @ gate` directly (no transpose).

    Returns `root` for convenient chaining.
    """
    root = Path(root)
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
                rng.standard_normal((intermediate, hidden)).astype("float32"),
            )
            np.save(
                expert_dir / "up_proj.npy",
                rng.standard_normal((intermediate, hidden)).astype("float32"),
            )
            np.save(
                expert_dir / "down_proj.npy",
                rng.standard_normal((hidden, intermediate)).astype("float32"),
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
    """Write a fake-MoE .safetensors file with FakeMoELayout tensor names.

    Projection shapes are HF-oriented (see write_npy_world for the
    rationale).
    """
    from safetensors.numpy import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
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
                (intermediate, hidden)
            ).astype("float32")
            tensors[names["up_proj"]] = rng.standard_normal(
                (intermediate, hidden)
            ).astype("float32")
            tensors[names["down_proj"]] = rng.standard_normal(
                (hidden, intermediate)
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
    arch: str = "fake_moe",
) -> Path:
    """Write a fake-MoE .gguf file with FakeMoELayout tensor names.

    Projection shapes are HF-oriented (see write_npy_world for the
    rationale).
    """
    from gguf import GGUFWriter

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if layout is None:
        layout = FakeMoELayout()

    rng = np.random.default_rng(seed)
    writer = GGUFWriter(str(path), arch)
    for layer in range(num_layers):
        writer.add_tensor(
            layout.gate_weight(layer),
            rng.standard_normal((hidden, num_experts)).astype("float32"),
        )
        for expert in range(num_experts):
            names = layout.expert_tensors(layer, expert)
            writer.add_tensor(
                names["gate_proj"],
                rng.standard_normal((intermediate, hidden)).astype("float32"),
            )
            writer.add_tensor(
                names["up_proj"],
                rng.standard_normal((intermediate, hidden)).astype("float32"),
            )
            writer.add_tensor(
                names["down_proj"],
                rng.standard_normal((hidden, intermediate)).astype("float32"),
            )
    # GGUF write happens in three phases (header, KV metadata, tensor data).
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


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
