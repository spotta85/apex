"""Memory-mapped variant of ExpertStore.

Identical interface and on-disk layout to `ExpertStore` (so the manager
/ runtime don't need to know which one they're using), but each
projection file is opened via `np.load(..., mmap_mode="r")`. The
returned `ExpertWeights` bundle holds three np.memmap views — the OS
pages each one in lazily as the matmul touches it.

Why this matters for APEX: cache budget accounting still uses
`tensor_nbytes`, which sums the three projections' nbytes — so the
budget is a coarse upper bound (counts file size, not RSS). In
practice, mmap means the OS shares pages and discards cold ones under
memory pressure, so resident memory stays well under the bound.

The cache stores the ExpertWeights bundle directly. Backends'
`run_expert` reads each projection from the bundle and matmul
participates in mmap-driven page-in unchanged.
"""

import numpy as np

from apex.expert_store import ExpertStore


class MMapExpertStore(ExpertStore):
    # Reuses ExpertStore.load; only the per-projection load mode differs.

    def _load_projection(self, expert_dir, projection_name: str):
        # mmap_mode="r" returns an np.memmap in read-only mode. Pages
        # are not read from disk until the array is indexed / used in
        # a computation.
        return np.load(expert_dir / f"{projection_name}.npy", mmap_mode="r")
