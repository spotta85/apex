"""Shared types passed between the loader, cache, backend, and runtime.

These exist so the moving pieces don't need to agree on dict shapes or
plain tuples — each type makes its contract explicit.

`ExpertWeights` is what every expert store will return once experts
become FFN-style (gate / up / down projections instead of a single
matmul). The cache treats it as one cached unit; backends destructure
it inside `run_expert`.

`RouteResult` lets routers return optional per-expert weights alongside
the expert IDs. The runtime today uses uniform averaging via
`backend.combine_outputs`, so `weights` is allowed to be None and most
callers ignore it. When we move to router-weighted combination, this
type already carries the data.
"""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ExpertWeights:
    """Bundle of the three projection matrices that make up one expert.

    Each field holds whatever array type the backend prepared:
      - NumPy backend / pre-prepared cache: np.ndarray (or np.memmap)
      - MLX backend: mx.array
    The runtime never touches the contents — it hands the whole bundle
    to `backend.run_expert`, which knows how to compute against it.
    """

    gate_proj: Any
    up_proj: Any
    down_proj: Any


@dataclass
class RouteResult:
    """What a router returns for a single layer.

    `expert_ids` is a list of int IDs, in priority order (highest score
    first). `weights` is an optional parallel list of routing weights
    (sums to ~1 for top-k softmax routing) — used by router-weighted
    combination once we wire that in.
    """

    expert_ids: list
    weights: Optional[list] = None
