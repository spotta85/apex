"""Fake router that decides which experts each layer should run.

In a real MoE model the router is a small learned gate that, given the
hidden state, picks top-k experts per token. For deterministic tests
and the basic demo we hard-code the choice via FakeRouter so we can
focus on loading / caching / eviction behavior. See `apex.gate_router`
for the gate-score-based variant.
"""


class FakeRouter:
    # `name` lets the runtime print "router backend: <name>" so demos
    # show which routing path is in use. FakeRouter is numpy-side
    # (no compute, but consistent with the other numpy routers).
    name = "numpy"

    # Accepts hidden_state to match GateRouter's signature, but ignores
    # it — selections are hardcoded so test traces stay deterministic.
    def route(self, layer_id: int, hidden_state=None):
        # Layer 0: use experts 1 and 3.
        if layer_id == 0:
            return [1, 3], None
        # All other layers: use experts 0 and 2.
        # (Two distinct expert sets per layer is what forces the cache
        # to evict when max_experts=2 — that's the behavior the original
        # MVP was meant to prove.)
        return [0, 2], None
