"""Demo backed entirely by a single .gguf file.

Setup before first run:
    uv run python scripts/make_fake_gguf_moe.py
"""

from apex.demo_modes import make_gguf_runtime

bundle = make_gguf_runtime()
bundle.runtime.run()
print()
bundle.metrics.print_summary()
bundle.prefetcher.shutdown()
