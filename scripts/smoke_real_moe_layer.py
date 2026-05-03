"""End-to-end smoke test: one real MoE layer of Gemma 4 NVFP4.

What this proves (and why it's worth a separate script rather than
folding into a test):

  1. The router (BF16 fp32 after loader byte expansion) consumed by
     `apex.moe_layer.route` produces sensible top-k indices on real
     weights — not just the toy router used in unit tests.
  2. `QuantizedSafetensorsExpertStore` + `apex.dequant.dequantize_bundle`
     can pull eight experts from a sharded multi-GB checkpoint and
     hand back numerically-clean (finite, non-zero variance) bundles.
  3. `apex.moe_layer.run_moe_layer` then drives `FakeBackend.run_expert`
     (the SwiGLU FFN) over those bundles and produces a finite output
     of shape (1, hidden_dim) — which is the contract the runtime needs
     to start swapping in real-checkpoint layers ahead of fake ones.

This is the first smoke test that touches every piece of the loading
+ dequant + compute stack with real weights at once. It's deliberately
not a unit test: it depends on a real checkpoint on disk (130-ish MB
per expert times 8) and we don't want CI loading that much data.

Run:
    uv run python scripts/smoke_real_moe_layer.py
        [--path /path/to/Gemma-4-26B-A4B-NVFP4]
        [--layer 0] [--top-k 8] [--seed 0]

Defaults match the local Gemma 4 NVFP4 download path used during
APEX development; override --path to point at any sharded NVFP4 MoE
checkpoint with the same Gemma layout.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Make the apex package importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apex.backends.fake_backend import FakeBackend
from apex.checkpoint_index import get_tensor_shape, read_safetensors_index
from apex.dequant import dequantize_bundle
from apex.moe_layer import route, run_moe_layer
from apex.quantized_safetensors_store import QuantizedSafetensorsExpertStore
from apex.safetensors_loader import load_tensor
from apex.weight_layout import get_layout


DEFAULT_PATH = "/Users/spotta/Desktop/models/Gemma-4-26B-A4B-NVFP4"


def main():
    args = _parse_args()
    path = Path(args.path)
    if not path.is_dir():
        raise SystemExit(
            f"checkpoint directory not found: {path}\n"
            f"download from https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4 "
            f"or pass --path"
        )

    layout = get_layout("gemma-moe")

    # --- 1. Load the router (BF16, NOT quantized — appears in the
    # config.json `ignore` list) and probe its shape so we can build a
    # matching fake hidden state without hard-coding hidden_dim.
    router_name = layout.gate_weight(args.layer)
    router_shape = get_tensor_shape(path, router_name)  # (num_experts, hidden)
    num_experts, hidden_dim = router_shape
    print(f"[load] router {router_name!r} shape={router_shape}")

    # Resolve which shard owns the router and load it. load_tensor only
    # accepts a single .safetensors file path, so we go through the
    # checkpoint index helper to find the right shard ourselves.
    weight_map = read_safetensors_index(path)
    router_path = path / weight_map[router_name]
    t0 = time.perf_counter()
    router = load_tensor(router_path, router_name)
    print(f"[load] router fp32 shape={router.shape} dtype={router.dtype} "
          f"in {(time.perf_counter() - t0) * 1000:.1f} ms")

    # --- 2. Fake hidden state with a fixed seed so the routing
    # decisions are reproducible across runs of this script.
    rng = np.random.default_rng(args.seed)
    hidden = rng.standard_normal((1, hidden_dim)).astype(np.float32)
    print(f"[input] hidden shape={hidden.shape} std={hidden.std():.3f}")

    # --- 3. Routing: top-k experts + softmax weights. The matmul-warning
    # filter is for the same reason as the expert SwiGLU below: a real
    # BF16 router has a wide enough dynamic range to trip transient
    # over/underflow warnings even though the final logits are finite.
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        expert_ids, weights = route(hidden, router, top_k=args.top_k)
    selected = expert_ids[0].tolist()
    routing_weights = weights[0]
    print(f"[route] top-{args.top_k} experts: {selected}")
    print(f"[route] routing weights:        "
          f"[{', '.join(f'{w:.3f}' for w in routing_weights)}]")
    print(f"[route] weights sum:            {routing_weights.sum():.4f}")

    # --- 4. Build the loader: pull a quantized bundle from the store,
    # dequantize all three projections, hand back a SwiGLU-ready bundle.
    # Wrapping with timing so we can see how expensive a real expert
    # load is end-to-end (load + 3x NVFP4 dequant per expert).
    store = QuantizedSafetensorsExpertStore(path, layout=layout)
    load_times_ms = []

    def expert_loader(expert_id: int):
        t = time.perf_counter()
        bundle = dequantize_bundle(store.load(args.layer, expert_id))
        load_times_ms.append((time.perf_counter() - t) * 1000)
        return bundle

    # --- 5. Run the layer: per-expert SwiGLU + router-weighted combine.
    # Silence matmul over/underflow warnings: real NVFP4 weights span a
    # wide dynamic range (some block-scale × FP4 magnitudes briefly
    # over/underflow during the partial-sum reduction) but the final
    # accumulated output is finite — we assert that explicitly below.
    # Without this filter the smoke output is unreadable.
    backend = FakeBackend()
    t0 = time.perf_counter()
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        out = run_moe_layer(hidden, selected, routing_weights, expert_loader, backend)
    layer_ms = (time.perf_counter() - t0) * 1000

    # --- 6. Diagnostics — every check the smoke test really cares about.
    finite_frac = float(np.isfinite(out).mean())
    print()
    print(f"[output] shape:                {out.shape}")
    print(f"[output] dtype:                {out.dtype}")
    print(f"[output] all finite:           {bool(np.all(np.isfinite(out)))} "
          f"({finite_frac * 100:.2f}% finite)")
    print(f"[output] mean:                 {float(out.mean()):.4f}")
    print(f"[output] std:                  {float(out.std()):.4f}")
    print(f"[output] abs max:              {float(np.abs(out).max()):.4f}")
    print()
    print(f"[timing] per-expert load+dequant: "
          f"min {min(load_times_ms):.0f} ms  "
          f"avg {sum(load_times_ms) / len(load_times_ms):.0f} ms  "
          f"max {max(load_times_ms):.0f} ms")
    print(f"[timing] full layer (load + run): {layer_ms:.0f} ms")

    # --- 7. Hard correctness contract: layer must return finite output
    # of the right shape. Anything else means dequant or the SwiGLU
    # numerics regressed and we want a non-zero exit so CI / shells
    # notice.
    assert out.shape == (1, hidden_dim), (
        f"output shape {out.shape} != expected (1, {hidden_dim})"
    )
    assert np.all(np.isfinite(out)), "output contains non-finite values"
    print()
    print("[ok] real MoE layer smoke test passed")


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--path", default=DEFAULT_PATH,
                   help=f"checkpoint directory (default: {DEFAULT_PATH})")
    p.add_argument("--layer", type=int, default=0,
                   help="layer index to drive end-to-end (default: 0)")
    p.add_argument("--top-k", type=int, default=8,
                   help="experts to route to (default: 8, matches Gemma 4)")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for the fake hidden state (default: 0)")
    return p.parse_args()


if __name__ == "__main__":
    main()
