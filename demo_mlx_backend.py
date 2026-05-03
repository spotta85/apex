"""End-to-end MLX demo: MLXBackend + MLXGateRouter + safetensors store.

Setup before first run:
    uv run python scripts/make_fake_safetensors_moe.py
"""

from apex.demo_modes import make_mlx_runtime

bundle = make_mlx_runtime()

print(f"backend: {bundle.runtime.backend.name}\n")
bundle.runtime.run()
print()
bundle.metrics.print_summary()
bundle.prefetcher.shutdown()
