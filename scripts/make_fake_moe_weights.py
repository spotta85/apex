"""Generate the fake MoE weights (npy directory) and seed training traces.

Weight layout (matches FakeMoELayout / what ExpertStore loads):
  experts/layer_<L>/gate.npy                      (hidden, num_experts)
  experts/layer_<L>/expert_<E>/gate_proj.npy      (hidden, intermediate)
  experts/layer_<L>/expert_<E>/up_proj.npy        (hidden, intermediate)
  experts/layer_<L>/expert_<E>/down_proj.npy      (intermediate, hidden)

Also records 10 deterministic gate-routed transitions to
`traces/gate_traces.jsonl` so the predictor demo can train on them.
The shape + RNG live in apex/_fake_weights.py so this script and the
test fixtures share one source of truth.
"""

import sys
from pathlib import Path

# Make the apex package importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apex._fake_weights import record_silent_traces, write_npy_world
from apex.backends.fake_backend import FakeBackend
from apex.expert_cache import ExpertCache
from apex.expert_manager import ExpertManager
from apex.expert_store import ExpertStore
from apex.gate_router import GateRouter
from apex.runtime import Runtime
from apex.trace_recorder import TraceRecorder


# --- weights ----------------------------------------------------------
experts_root = Path("experts")
write_npy_world(experts_root)

# --- training traces: silently run the gate-routed runtime ------------
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
