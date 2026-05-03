"""Generate a single fake-MoE .safetensors file plus seed training traces.

Produces:
  - weights/fake_moe.safetensors with tensors named per FakeMoELayout:
      layers.<L>.gate.weight                       (hidden, num_experts)
      layers.<L>.experts.<E>.gate_proj.weight      (hidden, intermediate)
      layers.<L>.experts.<E>.up_proj.weight        (hidden, intermediate)
      layers.<L>.experts.<E>.down_proj.weight      (intermediate, hidden)
  - traces/safetensors_traces.jsonl: 10 deterministic gate-routed
    transitions captured by running the runtime once silently. Used by
    demo_safetensors.py to train its MarkovPredictor.

Uses safetensors.numpy.save_file (no torch dependency). Tensor dtype
is float32 throughout.
"""

import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

# Make the apex package importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apex.weight_layout import FakeMoELayout

# Fixed seed so re-runs produce identical weights and matching traces.
np.random.seed(42)

num_layers = 2
num_experts = 4
hidden_size = 8
intermediate_size = 16

layout = FakeMoELayout()

# --- weights ----------------------------------------------------------
# Build the full tensor dict via the layout, then write atomically.
# Going through the layout instead of f-string-ing names keeps the
# script's tensor naming aligned with what the stores will look up.
tensors = {}
for layer in range(num_layers):
    # Gate: hidden_size x num_experts. GateRouter does hidden @ gate.
    tensors[layout.gate_weight(layer)] = (
        np.random.randn(hidden_size, num_experts).astype("float32")
    )
    # Each expert is three projections (SwiGLU FFN).
    for expert in range(num_experts):
        names = layout.expert_tensors(layer, expert)
        tensors[names["gate_proj"]] = (
            np.random.randn(hidden_size, intermediate_size).astype("float32")
        )
        tensors[names["up_proj"]] = (
            np.random.randn(hidden_size, intermediate_size).astype("float32")
        )
        tensors[names["down_proj"]] = (
            np.random.randn(intermediate_size, hidden_size).astype("float32")
        )

weights_path = Path("weights/fake_moe.safetensors")
weights_path.parent.mkdir(parents=True, exist_ok=True)
save_file(tensors, str(weights_path))

# --- training traces --------------------------------------------------
# Imported here (after weights exist) so the safetensors store/router
# can read them.
from apex._fake_weights import record_silent_traces
from apex.backends.fake_backend import FakeBackend
from apex.expert_cache import ExpertCache
from apex.expert_manager import ExpertManager
from apex.runtime import Runtime
from apex.safetensors_expert_store import SafetensorsExpertStore
from apex.safetensors_gate_router import SafetensorsGateRouter
from apex.trace_recorder import TraceRecorder

trace_path = Path("traces/safetensors_traces.jsonl")
cache = ExpertCache(max_memory_mb=10, verbose=False)  # plenty of room
manager = ExpertManager(SafetensorsExpertStore(str(weights_path), layout), cache)
recorder = TraceRecorder(trace_path)
runtime = Runtime(
    router=SafetensorsGateRouter(str(weights_path), layout, top_k=2),
    expert_manager=manager,
    backend=FakeBackend(),
    trace_recorder=recorder,
)

NUM_TRACE_RUNS = 10
record_silent_traces(runtime, trace_path, num_runs=NUM_TRACE_RUNS)

print("Fake MoE safetensors and training traces created.")
print(f"  weights: {weights_path}")
print(f"  traces:  {trace_path} ({NUM_TRACE_RUNS} entries)")
