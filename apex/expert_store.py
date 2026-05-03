"""Eager npy-backed expert store.

After step 4, each expert is FFN-style: three projection matrices
(gate_proj / up_proj / down_proj) grouped in a per-expert subdirectory.
Layout on disk:

  <root>/layer_<L>/expert_<E>/gate_proj.npy
  <root>/layer_<L>/expert_<E>/up_proj.npy
  <root>/layer_<L>/expert_<E>/down_proj.npy

`load(layer, expert)` returns the bundle as `ExpertWeights`. The cache
treats the bundle as one entry; backends destructure it inside
`run_expert`. See `scripts/make_fake_moe_weights.py` for how this
layout is created.
"""

import numpy as np
from pathlib import Path

from apex.types import ExpertWeights


# Names of the projection files inside each expert subdirectory.
# Centralized so future structure changes (e.g., a single .npz per
# expert) only need updating in one place.
_PROJ_FILENAMES = ("gate_proj", "up_proj", "down_proj")


class ExpertStore:
    def __init__(self, root: str):
        self.root = Path(root)

    def load(self, layer_id: int, expert_id: int) -> ExpertWeights:
        # Pay disk I/O for all three projections; cache holds the
        # bundle as a single entry so future hits are free.
        expert_dir = self.root / f"layer_{layer_id}" / f"expert_{expert_id}"
        return ExpertWeights(
            **{name: self._load_projection(expert_dir, name) for name in _PROJ_FILENAMES}
        )

    def _load_projection(self, expert_dir: Path, projection_name: str):
        # Eager load — np.load reads the whole array into RAM.
        # MMapExpertStore overrides this to use mmap_mode="r" instead.
        return np.load(expert_dir / f"{projection_name}.npy")
