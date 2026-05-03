"""Generate the full fake MoE weights (npy directory) and seed training traces.

After step 4, each expert is FFN-style: three projection matrices
(gate_proj, up_proj, down_proj) instead of one. The on-disk layout
groups them in a per-expert subdirectory:

  experts/
    layer_0/
      gate.npy                 (hidden_size, num_experts)
      expert_0/
        gate_proj.npy          (hidden_size, intermediate_size)
        up_proj.npy            (hidden_size, intermediate_size)
        down_proj.npy          (intermediate_size, hidden_size)
      expert_1/
        ...

Also records 10 deterministic gate-routed transitions to
`traces/gate_traces.jsonl` so the predictor demo can train on them.
Why bake trace generation into setup: the GateRouter's selections are
random-but-deterministic (we fix np.random.seed); easier to run the
gate router once and record what it does than to hand-author traces.
"""

import sys
from pathlib import Path

import numpy as np

# Make the apex package importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Fixed seed so every run produces identical experts + gates and the
# trained predictor's behavior is reproducible.
np.random.seed(42)

experts_root = Path("experts")
num_layers = 2
num_experts = 4
hidden_size = 8
intermediate_size = 16

# --- weights: experts (3 projections each) + gate per layer -------------

for layer in range(num_layers):
    layer_dir = experts_root / f"layer_{layer}"
    layer_dir.mkdir(parents=True, exist_ok=True)

    # Gate weights: hidden_size x num_experts.
    # GateRouter computes scores = hidden @ gate, then takes top-k.
    gate = np.random.randn(hidden_size, num_experts).astype("float32")
    np.save(layer_dir / "gate.npy", gate)

    # Expert weights: three projections per expert in its own subdir.
    for expert in range(num_experts):
        expert_dir = layer_dir / f"expert_{expert}"
        expert_dir.mkdir(parents=True, exist_ok=True)

        gate_proj = np.random.randn(hidden_size, intermediate_size).astype("float32")
        up_proj = np.random.randn(hidden_size, intermediate_size).astype("float32")
        down_proj = np.random.randn(intermediate_size, hidden_size).astype("float32")

        np.save(expert_dir / "gate_proj.npy", gate_proj)
        np.save(expert_dir / "up_proj.npy", up_proj)
        np.save(expert_dir / "down_proj.npy", down_proj)

# --- training traces: silently run the gate-routed runtime once ---------
# Imported here (after weights exist) so GateRouter can load them.
from apex._fake_weights import record_silent_traces
from apex.backends.fake_backend import FakeBackend
from apex.expert_cache import ExpertCache
from apex.expert_manager import ExpertManager
from apex.expert_store import ExpertStore
from apex.gate_router import GateRouter
from apex.runtime import Runtime
from apex.trace_recorder import TraceRecorder

trace_path = Path("traces/gate_traces.jsonl")
cache = ExpertCache(max_memory_mb=10, verbose=False)  # plenty of room
manager = ExpertManager(ExpertStore(experts_root), cache)
recorder = TraceRecorder(trace_path)
runtime = Runtime(
    router=GateRouter(experts_root, top_k=2),
    expert_manager=manager,
    backend=FakeBackend(),
    trace_recorder=recorder,
)

NUM_TRACE_RUNS = 10
record_silent_traces(runtime, trace_path, num_runs=NUM_TRACE_RUNS)

print("Fake MoE weights and training traces created.")
print(f"  weights: {experts_root}/layer_{{0,1}}/{{gate.npy,expert_*/<proj>.npy}}")
print(f"  traces:  {trace_path} ({NUM_TRACE_RUNS} entries)")
