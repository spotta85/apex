"""Generate a fake-MoE .gguf file plus seed training traces.

Tensors are named per FakeMoELayout:
  layers.<L>.gate.weight                       (hidden, num_experts)
  layers.<L>.experts.<E>.gate_proj.weight      (hidden, intermediate)
  layers.<L>.experts.<E>.up_proj.weight        (hidden, intermediate)
  layers.<L>.experts.<E>.down_proj.weight      (intermediate, hidden)

Tensor dtype is float32, 2D only — no quantization for this sprint.
The architecture string is "fake_moe" (required by the GGUF spec but
not consumed by anything in apex; real model files would put e.g.
"llama" or "qwen2moe" here).

The shape + RNG live in apex/_fake_weights.py so this script and the
test fixtures share one source of truth.
"""

import sys
from pathlib import Path

# Make the apex package importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apex._fake_weights import record_silent_traces, write_gguf_world
from apex.backends.fake_backend import FakeBackend
from apex.expert_cache import ExpertCache
from apex.expert_manager import ExpertManager
from apex.gguf_expert_store import GGUFExpertStore
from apex.gguf_gate_router import GGUFGateRouter
from apex.runtime import Runtime
from apex.trace_recorder import TraceRecorder
from apex.weight_layout import FakeMoELayout


# --- weights ----------------------------------------------------------
weights_path = Path("weights/fake_moe.gguf")
layout = FakeMoELayout()
write_gguf_world(weights_path, layout=layout)

# --- training traces --------------------------------------------------
trace_path = Path("traces/gguf_traces.jsonl")
cache = ExpertCache(max_memory_mb=10, verbose=False)  # plenty of room
manager = ExpertManager(GGUFExpertStore(str(weights_path), layout), cache)
recorder = TraceRecorder(trace_path)
runtime = Runtime(
    router=GGUFGateRouter(str(weights_path), layout, top_k=2),
    expert_manager=manager,
    backend=FakeBackend(),
    trace_recorder=recorder,
)

NUM_TRACE_RUNS = 10
record_silent_traces(runtime, trace_path, num_runs=NUM_TRACE_RUNS)

print("Fake MoE GGUF and training traces created.")
print(f"  weights: {weights_path}")
print(f"  traces:  {trace_path} ({NUM_TRACE_RUNS} entries)")
