"""Generate a fake-MoE .gguf file plus seed training traces.

Produces:
  - weights/fake_moe.gguf with tensors named per FakeMoELayout:
      layers.<L>.gate.weight                      (hidden, num_experts)
      layers.<L>.experts.<E>.gate_proj.weight     (hidden, intermediate)
      layers.<L>.experts.<E>.up_proj.weight       (hidden, intermediate)
      layers.<L>.experts.<E>.down_proj.weight     (intermediate, hidden)
  - traces/gguf_traces.jsonl: 10 deterministic gate-routed transitions
    captured by running the runtime once silently. Used by demo_gguf.py
    to train its MarkovPredictor.

Tensor dtype is float32, 2D only — no quantization for this sprint.
The architecture string is "fake_moe" (required by the GGUF spec but
not consumed by anything in apex; real model files would put e.g.
"llama" or "qwen2moe" here).
"""

import sys
from pathlib import Path

import numpy as np
from gguf import GGUFWriter

# Make the apex package importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apex.weight_layout import FakeMoELayout

# Fixed seed so re-runs produce identical weights (and matching traces).
np.random.seed(42)

num_layers = 2
num_experts = 4
hidden_size = 8
intermediate_size = 16

layout = FakeMoELayout()

# --- weights ----------------------------------------------------------
weights_path = Path("weights/fake_moe.gguf")
weights_path.parent.mkdir(parents=True, exist_ok=True)

# Naming goes through the layout so the script can't drift from what
# the stores look up. Each expert is now three projections (SwiGLU FFN).
writer = GGUFWriter(str(weights_path), "fake_moe")
for layer in range(num_layers):
    gate = np.random.randn(hidden_size, num_experts).astype("float32")
    writer.add_tensor(layout.gate_weight(layer), gate)

    for expert in range(num_experts):
        names = layout.expert_tensors(layer, expert)
        writer.add_tensor(
            names["gate_proj"],
            np.random.randn(hidden_size, intermediate_size).astype("float32"),
        )
        writer.add_tensor(
            names["up_proj"],
            np.random.randn(hidden_size, intermediate_size).astype("float32"),
        )
        writer.add_tensor(
            names["down_proj"],
            np.random.randn(intermediate_size, hidden_size).astype("float32"),
        )
# GGUF write happens in three phases (header, KV metadata, tensor data).
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()

# --- training traces --------------------------------------------------
# Imported here (after weights exist) so the GGUF store/router can read them.
from apex._fake_weights import record_silent_traces
from apex.backends.fake_backend import FakeBackend
from apex.expert_cache import ExpertCache
from apex.expert_manager import ExpertManager
from apex.gguf_expert_store import GGUFExpertStore
from apex.gguf_gate_router import GGUFGateRouter
from apex.runtime import Runtime
from apex.trace_recorder import TraceRecorder

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
