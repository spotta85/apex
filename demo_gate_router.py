"""Gate-router demo (npy weights + predictor + prefetch)."""

from apex.demo_modes import make_gate_router_runtime

bundle = make_gate_router_runtime()
bundle.runtime.run()
print()
bundle.metrics.print_summary()
bundle.prefetcher.shutdown()
