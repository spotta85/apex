"""Validate that a layout class's tensor names actually exist in a checkpoint.

Usage:
    uv run python scripts/validate_layout.py <checkpoint> --layout <name>
    uv run python scripts/validate_layout.py <checkpoint> --layout mixtral --check-shapes

Where <checkpoint> is the same shape inspect_weight_names.py accepts
(single .safetensors, single .gguf, or directory with
model.safetensors.index.json), and <name> is one of the layouts
registered in apex.weight_layout (run with --help to see them).

What it checks:
  - the layout's gate_weight at layer 0 exists
  - the layout's expert_tensors(0, 0) all three projections exist
  - the layout's expert_tensors(last_layer, last_expert) all three exist
    (last_* are auto-detected by probing the layout's name format
     against the checkpoint's tensor inventory — no config.json needed)

With --check-shapes:
  - fetches shape metadata for one expert's three projections + the
    layer-0 gate (no tensor data is loaded)
  - asserts up_proj.shape == gate_proj.shape and down_proj.shape is the
    transpose — these are the SwiGLU FFN's load-bearing invariants

Exits 0 on success, 1 on any failure. Failure messages name the exact
missing tensor or shape mismatch so you can map the gap to a layout
fix (or a wrong --layout choice).
"""

import argparse
import sys
from pathlib import Path

# Make the apex package importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apex.checkpoint_index import get_tensor_shape, list_tensor_names
from apex.weight_layout import available_layouts, get_layout


# Caps on the auto-detection probe. 1024 layers / 4096 experts comfortably
# covers every public MoE architecture as of early 2026; anything beyond
# that can override these by editing the script.
_MAX_PROBE_LAYERS = 1024
_MAX_PROBE_EXPERTS = 4096


def find_last_layer(layout, names_set: set) -> int:
    """Largest L such that layout.gate_weight(L) is in `names_set`, else -1."""
    last = -1
    for L in range(_MAX_PROBE_LAYERS):
        if layout.gate_weight(L) in names_set:
            last = L
        elif last >= 0:
            # Found a hit earlier and now hit a gap — assume contiguous range.
            break
    return last


def find_last_expert(layout, names_set: set, layer: int) -> int:
    """Largest E with layout.expert_gate_proj(layer, E) in `names_set`, else -1."""
    if getattr(layout, "stacked_experts", False):
        if layout.expert_gate_proj(layer, 0) not in names_set:
            return -1
        return getattr(layout, "num_experts", 1) - 1

    last = -1
    for E in range(_MAX_PROBE_EXPERTS):
        if layout.expert_gate_proj(layer, E) in names_set:
            last = E
        elif last >= 0:
            break
    return last


def _check_name(label: str, name: str, names_set: set) -> bool:
    """Print one check line; return whether the name was found."""
    found = name in names_set
    mark = "ok" if found else "MISSING"
    print(f"  [{mark}] {label}: {name}")
    return found


def _check_non_moe(layout, names_set: set, last_layer: int) -> bool:
    """Probe optional non-MoE methods and report per-name presence.

    Methods are intentionally optional (Mixtral / Qwen don't implement
    them). For each method the layout *does* expose, report `[ok]` or
    `[MISSING]` against the checkpoint's tensor inventory. Layer-scoped
    methods are checked at layer 0 and last_layer to catch off-by-one
    naming bugs that only show up at the boundary.
    """
    print()
    print("non-moe names:")
    all_ok = True

    # Single-tensor lookups (no sidecar).
    if hasattr(layout, "final_norm"):
        all_ok &= _check_name("final_norm", layout.final_norm(), names_set)

    # Embedding bundle: (weight + optional scales sidecar).
    if hasattr(layout, "embedding_tensors"):
        names = layout.embedding_tensors()
        for key, name in names.items():
            all_ok &= _check_name(f"embedding.{key}", name, names_set)

    # lm_head can be intentionally None when tied to embeddings.
    if hasattr(layout, "lm_head_tensors"):
        names = layout.lm_head_tensors()
        if names is None:
            print("  [ok] lm_head: tied to embeddings (None)")
        else:
            for key, name in names.items():
                all_ok &= _check_name(f"lm_head.{key}", name, names_set)

    layer_methods_dict = (
        "attn_q_proj_tensors",
        "attn_k_proj_tensors",
        "attn_v_proj_tensors",
        "attn_o_proj_tensors",
    )
    layer_methods_str = (
        "attn_q_norm",
        "attn_k_norm",
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
    )

    # Probe layer 0 only. Gemma 4 has heterogeneous attention across
    # layers (sliding_attention vs full_attention; full layers tie K=V
    # so v_proj is absent), so blanket probing the last layer would
    # false-positive even when the layout itself is correct. Layer 0
    # uses the "standard" attention and is what step 16C smokes.
    for layer_id in (0,):
        for method_name in layer_methods_dict:
            if hasattr(layout, method_name):
                bundle = getattr(layout, method_name)(layer_id)
                for key, name in bundle.items():
                    all_ok &= _check_name(
                        f"layer {layer_id} {method_name}.{key}", name, names_set,
                    )
        for method_name in layer_methods_str:
            if hasattr(layout, method_name):
                name = getattr(layout, method_name)(layer_id)
                all_ok &= _check_name(
                    f"layer {layer_id} {method_name}", name, names_set,
                )

    return all_ok


def _check_shapes(path: Path, layout, last_layer: int, last_expert: int) -> bool:
    """Fetch shapes for one expert + gate; assert SwiGLU consistency."""
    print()
    print("shapes (expert 0,0):")
    bundle = layout.expert_tensors(0, 0)
    gate_proj_shape = get_tensor_shape(path, bundle["gate_proj"])
    up_proj_shape = get_tensor_shape(path, bundle["up_proj"])
    down_proj_shape = get_tensor_shape(path, bundle["down_proj"])
    gate_shape = get_tensor_shape(path, layout.gate_weight(0))
    print(f"  gate_proj  {gate_proj_shape}")
    print(f"  up_proj    {up_proj_shape}")
    print(f"  down_proj  {down_proj_shape}")
    print(f"  gate       {gate_shape}")

    # The SwiGLU FFN math (gate*up then *down) requires up_proj to match
    # gate_proj, and down_proj to reverse the logical projection shape.
    # Stacked packed layouts include an expert axis and store multiple
    # values per last-axis element, so compare their unpacked shapes.
    ok = True
    if up_proj_shape != gate_proj_shape:
        print(
            f"  [MISMATCH] up_proj {up_proj_shape} != gate_proj {gate_proj_shape}",
            file=sys.stderr,
        )
        ok = False
    if getattr(layout, "stacked_experts", False):
        pack = getattr(layout, "expert_pack_factor", 1)
        logical_gate = (gate_proj_shape[1], gate_proj_shape[2] * pack)
        logical_down = (down_proj_shape[1], down_proj_shape[2] * pack)
        router_pack = getattr(layout, "router_pack_factor", 1)
        logical_router = (gate_shape[0], gate_shape[1] * router_pack)
        down_ok = logical_down == tuple(reversed(logical_gate))
    else:
        logical_gate = gate_proj_shape
        logical_down = down_proj_shape
        logical_router = gate_shape
        down_ok = down_proj_shape == tuple(reversed(gate_proj_shape))
    if not down_ok:
        print(
            f"  [MISMATCH] down_proj logical {logical_down} is not the transpose of "
            f"gate_proj logical {logical_gate}",
            file=sys.stderr,
        )
        ok = False

    # Gate shape: one of its dims must equal a dim of gate_proj (the
    # hidden_size). Real MoEs differ on which dim is which (HF-Mixtral
    # gate is (num_experts, hidden), APEX-fake gate is (hidden, num_experts)),
    # so we don't enforce orientation — just share-a-dim.
    if not (set(logical_router) & set(logical_gate)):
        print(
            f"  [WARN] gate logical {logical_router} doesn't share a dim with "
            f"gate_proj logical {logical_gate}; check that --layout matches the model",
            file=sys.stderr,
        )

    # Brief inferred-shape summary so the caller knows what they got.
    print(
        f"  inferred: hidden∈{set(logical_gate) & set(logical_router) or '?'}, "
        f"layers={last_layer + 1}, experts={last_expert + 1}"
    )
    return ok


def validate(
    path: Path,
    layout_name: str,
    check_shapes: bool = False,
    check_non_moe: bool = False,
) -> int:
    layout = get_layout(layout_name)
    print(f"layout: {layout_name}")
    print(f"checkpoint: {path}")

    try:
        names = list_tensor_names(path)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    names_set = set(names)

    # Detect the model's range first so the (last_layer, last_expert)
    # check has concrete indices. If the gate format itself never
    # matches, surface that as a "wrong layout" hint immediately.
    last_layer = find_last_layer(layout, names_set)
    if last_layer < 0:
        print(
            f"error: no gate weight matched layout {layout_name!r}",
            file=sys.stderr,
        )
        print(
            f"  expected (layer 0): {layout.gate_weight(0)}",
            file=sys.stderr,
        )
        print(
            "  hint: run scripts/inspect_weight_names.py to see what's actually in the file",
            file=sys.stderr,
        )
        return 1
    last_expert = find_last_expert(layout, names_set, last_layer)
    if last_expert < 0:
        print(
            f"error: no expert tensors matched at layer {last_layer}",
            file=sys.stderr,
        )
        print(
            f"  expected: {layout.expert_gate_proj(last_layer, 0)}",
            file=sys.stderr,
        )
        return 1
    print(f"detected: {last_layer + 1} layers x {last_expert + 1} experts")
    print()

    print("names:")
    all_ok = True
    all_ok &= _check_name("gate (layer 0)", layout.gate_weight(0), names_set)
    for label, (L, E) in [
        ("expert (0, 0)", (0, 0)),
        (f"expert ({last_layer}, {last_expert})", (last_layer, last_expert)),
    ]:
        bundle = layout.expert_tensors(L, E)
        all_ok &= _check_name(f"{label} gate_proj", bundle["gate_proj"], names_set)
        all_ok &= _check_name(f"{label} up_proj", bundle["up_proj"], names_set)
        all_ok &= _check_name(f"{label} down_proj", bundle["down_proj"], names_set)

    if not all_ok:
        print(
            "\nlayout validation FAILED — see [MISSING] entries above",
            file=sys.stderr,
        )
        return 1

    if check_shapes:
        try:
            shapes_ok = _check_shapes(path, layout, last_layer, last_expert)
        except KeyError as e:
            print(f"\nshape lookup error: {e}", file=sys.stderr)
            return 1
        if not shapes_ok:
            print("\nlayout validation FAILED — shape mismatch", file=sys.stderr)
            return 1

    if check_non_moe:
        non_moe_ok = _check_non_moe(layout, names_set, last_layer)
        if not non_moe_ok:
            print(
                "\nlayout validation FAILED — non-moe tensor missing",
                file=sys.stderr,
            )
            return 1

    print("\nlayout validation passed")
    return 0


def main(argv) -> int:
    parser = argparse.ArgumentParser(
        description="validate a layout class against a real checkpoint",
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="path to .safetensors, .gguf, or directory with model.safetensors.index.json",
    )
    parser.add_argument(
        "--layout",
        required=True,
        choices=available_layouts(),
        help="layout short name (registered in apex.weight_layout)",
    )
    parser.add_argument(
        "--check-shapes",
        action="store_true",
        help="also load tensor shape metadata and check SwiGLU consistency",
    )
    parser.add_argument(
        "--check-non-moe",
        action="store_true",
        help=(
            "also probe optional non-MoE methods (attention, norms, embeddings, "
            "lm_head). Reports per-tensor presence; methods the layout doesn't "
            "expose are skipped silently."
        ),
    )
    args = parser.parse_args(argv[1:])

    if not args.checkpoint.exists():
        print(f"error: path does not exist: {args.checkpoint}", file=sys.stderr)
        return 1

    return validate(
        args.checkpoint,
        args.layout,
        check_shapes=args.check_shapes,
        check_non_moe=args.check_non_moe,
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv))
