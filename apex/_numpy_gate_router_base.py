"""Base class for numpy-side gate routers (npy / safetensors / GGUF).

All three concrete numpy gate routers (`GateRouter`, `SafetensorsGateRouter`,
`GGUFGateRouter`) share the routing math:

    scores = np.asarray(hidden) @ gate
    return top-k expert IDs as plain Python ints

Only the per-layer gate-loading step differs (file path vs safetensors
get_tensor vs GGUF tensor index). This base lifts the common logic into
one place; subclasses only have to implement `_load_gate(layer_id)`.

Underscore-prefixed because it's internal scaffolding — apex's public
surface is the concrete router classes, not this base.
"""

import numpy as np


class BaseNumpyGateRouter:
    # All numpy-side gate routers report `name = "numpy"` so the runtime
    # can print "router backend: <name>" consistently.
    name = "numpy"

    def __init__(self, top_k: int = 2):
        self.top_k = top_k
        # Per-layer cache of loaded gate matrices (numpy arrays).
        # Subclasses populate via _load_gate.
        self.gates: dict = {}

    def route(self, layer_id: int, hidden_state):
        # `np.asarray` accepts ndarray AND mx.array (via __array__),
        # so this router stays usable from the MLX runtime path even
        # though its own scoring is numpy.
        gate = self._load_gate(layer_id)
        # scores: (batch=1, num_experts)
        scores = np.asarray(hidden_state) @ gate
        # Top-k by score, descending: argsort ascends, slice last k,
        # reverse for highest-first. Plain Python ints so cache keys /
        # JSON serialization stay clean across the rest of the system.
        expert_ids = scores[0].argsort()[-self.top_k :][::-1]
        return [int(e) for e in expert_ids]

    def _load_gate(self, layer_id: int):
        # Subclasses must return the per-layer gate as a numpy array.
        # Caching is the subclass's responsibility — usually populating
        # `self.gates[layer_id]` on first access and returning from
        # there on subsequent calls.
        raise NotImplementedError
