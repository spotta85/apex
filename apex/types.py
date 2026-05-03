"""Shared types passed between the loader, cache, backend, and runtime.

These exist so the moving pieces don't need to agree on dict shapes or
plain tuples — each type makes its contract explicit.

`ExpertWeights` is what every expert store will return once experts
become FFN-style (gate / up / down projections instead of a single
matmul). The cache treats it as one cached unit; backends destructure
it inside `run_expert`.

Routers return plain `(expert_ids, weights_or_None)` tuples; the tuple
contract stays lightweight because no runtime code needs a richer
object today.
"""

from dataclasses import dataclass
from typing import Any


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
