"""Demo backed entirely by a single .safetensors file.

Setup before first run:
    uv run python scripts/make_fake_safetensors_moe.py
"""

from apex.demo_modes import make_safetensors_runtime

bundle = make_safetensors_runtime()
bundle.runtime.run()
print()
bundle.metrics.print_summary()
bundle.prefetcher.shutdown()
