"""Factory functions for the various demo configurations.

Each `make_*_runtime` returns a `RuntimeBundle` so demo files always
unpack the same shape regardless of which factory they used. Optional
fields (metrics, cache) are None when a factory doesn't surface them.

Naming convention:
  - basic / mmap demos: verbose pipeline (cache + prefetcher print
    everything, no metrics object).
  - predictor / gate / safetensors / gguf / mlx demos: quiet pipeline
    (cache + prefetcher silent; a metrics object collects counters
    that the demo prints at the end).
"""

from dataclasses import dataclass
from typing import Optional

from apex.expert_store import ExpertStore
from apex.mmap_expert_store import MMapExpertStore
from apex.safetensors_expert_store import SafetensorsExpertStore
from apex.gguf_expert_store import GGUFExpertStore
from apex.expert_cache import ExpertCache
from apex.expert_manager import ExpertManager
from apex.prefetcher import AsyncPrefetcher
from apex.router import FakeRouter
from apex.gate_router import GateRouter
from apex.safetensors_gate_router import SafetensorsGateRouter
from apex.gguf_gate_router import GGUFGateRouter
# MLXBackend / MLXGateRouter imported lazily inside make_mlx_runtime so
# the rest of demo_modes still imports cleanly on machines without mlx.
from apex.backends.fake_backend import FakeBackend
from apex.runtime import Runtime
from apex.predictor import MarkovPredictor
from apex.trace_recorder import TraceRecorder
from apex.metrics import Metrics
from apex.weight_layout import FakeMoELayout


@dataclass
class RuntimeBundle:
    """Everything a demo needs to run + tear down a configured pipeline.

    `metrics` and `cache` are populated only by factories that build
    them; demos that don't need them just don't read those fields.
    Always call `bundle.prefetcher.shutdown()` at the end so the worker
    thread exits cleanly.
    """

    runtime: Runtime
    prefetcher: AsyncPrefetcher
    metrics: Optional[Metrics] = None
    cache: Optional[ExpertCache] = None


def make_basic_runtime(experts_root: str = "experts", max_memory_mb: float = 0.005) -> RuntimeBundle:
    """Verbose pipeline: cache + prefetcher print every load / memory change."""
    backend = FakeBackend()
    cache = ExpertCache(max_memory_mb=max_memory_mb, verbose=True)
    store = ExpertStore(experts_root)
    manager = ExpertManager(store, cache, backend=backend)
    prefetcher = AsyncPrefetcher(manager, verbose=True)
    runtime = Runtime(
        router=FakeRouter(),
        expert_manager=manager,
        backend=backend,
        prefetcher=prefetcher,
    )
    return RuntimeBundle(runtime=runtime, prefetcher=prefetcher, cache=cache)


def make_predictor_runtime(
    experts_root: str = "experts",
    trace_path: str = "traces/fake_traces.jsonl",
    output_trace_path: str = "traces/runtime_traces.jsonl",
    max_memory_mb: float = 0.005,
) -> RuntimeBundle:
    """Quiet pipeline: trained predictor, recorder, and metrics object.

    `trace_path` is read once at startup to train the MarkovPredictor.
    `output_trace_path` is appended to during runtime so future runs
    can train on the additional data — pass None to disable recording.
    """
    backend = FakeBackend()
    metrics = Metrics()
    cache = ExpertCache(max_memory_mb=max_memory_mb, verbose=False, metrics=metrics)
    store = ExpertStore(experts_root)
    manager = ExpertManager(store, cache, metrics=metrics, backend=backend)
    prefetcher = AsyncPrefetcher(manager, verbose=False, metrics=metrics)

    predictor = MarkovPredictor()
    predictor.fit(trace_path)

    recorder = TraceRecorder(output_trace_path) if output_trace_path else None

    runtime = Runtime(
        router=FakeRouter(),
        expert_manager=manager,
        backend=backend,
        prefetcher=prefetcher,
        predictor=predictor,
        trace_recorder=recorder,
    )
    return RuntimeBundle(runtime=runtime, prefetcher=prefetcher, metrics=metrics, cache=cache)


def make_mmap_runtime(experts_root: str = "experts", max_memory_mb: float = 0.005) -> RuntimeBundle:
    """Verbose pipeline using MMapExpertStore in place of ExpertStore.

    Identical to make_basic_runtime except the store is mmap-backed,
    so cache entries are np.memmap references whose pages the OS pulls
    in lazily.
    """
    backend = FakeBackend()
    cache = ExpertCache(max_memory_mb=max_memory_mb, verbose=True)
    store = MMapExpertStore(experts_root)
    manager = ExpertManager(store, cache, backend=backend)
    prefetcher = AsyncPrefetcher(manager, verbose=True)
    runtime = Runtime(
        router=FakeRouter(),
        expert_manager=manager,
        backend=backend,
        prefetcher=prefetcher,
    )
    return RuntimeBundle(runtime=runtime, prefetcher=prefetcher, cache=cache)


def make_gate_router_runtime(
    experts_root: str = "experts",
    trace_path: str = "traces/gate_traces.jsonl",
    output_trace_path: str = "traces/runtime_gate_traces.jsonl",
    max_memory_mb: float = 0.005,
    top_k: int = 2,
) -> RuntimeBundle:
    """Quiet pipeline driven by a real gate-style router (npy weights)."""
    backend = FakeBackend()
    metrics = Metrics()
    cache = ExpertCache(max_memory_mb=max_memory_mb, verbose=False, metrics=metrics)
    store = ExpertStore(experts_root)
    manager = ExpertManager(store, cache, metrics=metrics, backend=backend)
    prefetcher = AsyncPrefetcher(manager, verbose=False, metrics=metrics)

    predictor = MarkovPredictor()
    predictor.fit(trace_path)

    recorder = TraceRecorder(output_trace_path) if output_trace_path else None

    runtime = Runtime(
        router=GateRouter(experts_root, top_k=top_k),
        expert_manager=manager,
        backend=backend,
        prefetcher=prefetcher,
        predictor=predictor,
        trace_recorder=recorder,
    )
    return RuntimeBundle(runtime=runtime, prefetcher=prefetcher, metrics=metrics, cache=cache)


def make_safetensors_runtime(
    weights_path: str = "weights/fake_moe.safetensors",
    trace_path: str = "traces/safetensors_traces.jsonl",
    output_trace_path: str = "traces/runtime_safetensors_traces.jsonl",
    max_memory_mb: float = 0.005,
    top_k: int = 2,
    layout=None,
) -> RuntimeBundle:
    """Quiet pipeline driven by safetensors-backed store + gate router."""
    if layout is None:
        layout = FakeMoELayout()

    backend = FakeBackend()
    metrics = Metrics()
    cache = ExpertCache(max_memory_mb=max_memory_mb, verbose=False, metrics=metrics)
    store = SafetensorsExpertStore(weights_path, layout)
    manager = ExpertManager(store, cache, metrics=metrics, backend=backend)
    prefetcher = AsyncPrefetcher(manager, verbose=False, metrics=metrics)

    predictor = MarkovPredictor()
    predictor.fit(trace_path)

    recorder = TraceRecorder(output_trace_path) if output_trace_path else None

    runtime = Runtime(
        router=SafetensorsGateRouter(weights_path, layout, top_k=top_k),
        expert_manager=manager,
        backend=backend,
        prefetcher=prefetcher,
        predictor=predictor,
        trace_recorder=recorder,
    )
    return RuntimeBundle(runtime=runtime, prefetcher=prefetcher, metrics=metrics, cache=cache)


def make_gguf_runtime(
    weights_path: str = "weights/fake_moe.gguf",
    trace_path: str = "traces/gguf_traces.jsonl",
    output_trace_path: str = "traces/runtime_gguf_traces.jsonl",
    max_memory_mb: float = 0.005,
    top_k: int = 2,
    layout=None,
) -> RuntimeBundle:
    """Quiet pipeline driven by GGUF-backed store + gate router."""
    if layout is None:
        layout = FakeMoELayout()

    backend = FakeBackend()
    metrics = Metrics()
    cache = ExpertCache(max_memory_mb=max_memory_mb, verbose=False, metrics=metrics)
    store = GGUFExpertStore(weights_path, layout)
    manager = ExpertManager(store, cache, metrics=metrics, backend=backend)
    prefetcher = AsyncPrefetcher(manager, verbose=False, metrics=metrics)

    predictor = MarkovPredictor()
    predictor.fit(trace_path)

    recorder = TraceRecorder(output_trace_path) if output_trace_path else None

    runtime = Runtime(
        router=GGUFGateRouter(weights_path, layout, top_k=top_k),
        expert_manager=manager,
        backend=backend,
        prefetcher=prefetcher,
        predictor=predictor,
        trace_recorder=recorder,
    )
    return RuntimeBundle(runtime=runtime, prefetcher=prefetcher, metrics=metrics, cache=cache)


def make_mlx_runtime(
    weights_path: str = "weights/fake_moe.safetensors",
    trace_path: str = "traces/safetensors_traces.jsonl",
    output_trace_path: str = "traces/runtime_mlx_traces.jsonl",
    max_memory_mb: float = 0.005,
    top_k: int = 2,
    layout=None,
) -> RuntimeBundle:
    """End-to-end MLX pipeline: MLXBackend + MLXGateRouter + safetensors store.

    Backend-prepared cache holds mx.array experts; hidden state stays
    in mx.array form across layers; routing matmul runs on MLX. Only
    the final boundary (runtime.run() return) is converted back to
    ndarray.

    MLX is Apple-Silicon-only — imports lazily so this module still
    loads cleanly on machines without mlx installed.
    """
    from apex.backends.mlx_backend import MLXBackend
    from apex.mlx_gate_router import MLXGateRouter

    if layout is None:
        layout = FakeMoELayout()

    backend = MLXBackend()
    metrics = Metrics()
    cache = ExpertCache(max_memory_mb=max_memory_mb, verbose=False, metrics=metrics)
    store = SafetensorsExpertStore(weights_path, layout)
    manager = ExpertManager(store, cache, metrics=metrics, backend=backend)
    prefetcher = AsyncPrefetcher(manager, verbose=False, metrics=metrics)

    predictor = MarkovPredictor()
    predictor.fit(trace_path)

    recorder = TraceRecorder(output_trace_path) if output_trace_path else None

    runtime = Runtime(
        router=MLXGateRouter(weights_path, layout, top_k=top_k),
        expert_manager=manager,
        backend=backend,
        prefetcher=prefetcher,
        predictor=predictor,
        trace_recorder=recorder,
    )
    return RuntimeBundle(runtime=runtime, prefetcher=prefetcher, metrics=metrics, cache=cache)
