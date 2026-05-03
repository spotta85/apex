"""Trace-trained Markov predictor + async prefetch demo (npy stack)."""

from apex.demo_modes import make_predictor_runtime

bundle = make_predictor_runtime()
bundle.runtime.run()
print()
bundle.metrics.print_summary()
bundle.prefetcher.shutdown()
