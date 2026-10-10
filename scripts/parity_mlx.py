"""Layer-by-layer parity vs the mlx_lm Gemma 4 reference.

Runs the same token ids through mlx_lm's `gemma4_text.Model` and through
`apex.MoEModel`, captures every layer's intermediate tensors at every
prompt position, and reports the first layer / stage where the two
diverge. Final logits are compared last.

The two models never share a process. The reference keeps the ~15 GB
quantized checkpoint in unified memory, and APEX keeps ~13 GB of fp32
fixed overhead plus the expert cache — together they overflow a 24 GB
machine. So each side runs as its own phase, dumps an .npz under
`traces/`, and `diff` compares the two files.

Captured per layer, per position (names match `decoder_decode_step`'s
trace keys on the APEX side):

    attn_out    post_attention_layernorm(attention(...))
    dense_out   raw dense-MLP output (before post_feedforward_layernorm_1)
    route_ids / route_weights   router top-k decision
    moe_out     raw routed-experts output (before ..._layernorm_2)
    layer_out   the layer's final output (after layer_scalar)

Usage:
    uv run python scripts/parity_mlx.py                  # ref + apex + diff
    uv run python scripts/parity_mlx.py ref              # reference only
    uv run python scripts/parity_mlx.py apex --backend mlx
    uv run python scripts/parity_mlx.py diff
    uv run python scripts/parity_mlx.py --chat --prompt "What is 2+2?"
    uv run python scripts/parity_mlx.py ref --ref-dtype bfloat16 --tol 0.1
"""

import argparse
import contextlib
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
# Make the apex package importable when this script is run directly.
sys.path.insert(0, str(_REPO_ROOT))

CKPT = Path("/Users/spotta/Desktop/models/gemma4/gemma-4-26b-a4b-it-nvfp4")
_TRACE_DIR = _REPO_ROOT / "traces"
_REF_NPZ = _TRACE_DIR / "parity_ref.npz"
_APEX_NPZ = _TRACE_DIR / "parity_apex.npz"

# Per-layer hidden-state stages, in execution order inside one layer.
# The router sits between dense_out and moe_out (handled separately
# because it is an id set + weights, not a hidden vector).
_STAGES = ("attn_out", "dense_out", "moe_out", "layer_out")

# Relative L2 error above which a stage counts as "diverged". Both
# sides read the same quantized weights, so against an fp32 reference
# healthy drift is ~1e-6 and a real math bug shows up as O(0.1-1).
# Against a bf16 reference rounding alone reaches a few percent by the
# later layers — pass a looser --tol with --ref-dtype bfloat16.
_DEFAULT_TOL = 1e-3


# --------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------

def _encode(ckpt: Path, prompt: str, chat: bool) -> list:
    """Token ids both sides will consume (APEX tokenizer, BOS included)."""
    from apex.tokenizer import Tokenizer
    tok = Tokenizer.from_pretrained(str(ckpt))
    if chat:
        return tok.encode_chat([{"role": "user", "content": prompt}])
    return tok.encode(prompt)


def _sort_route(ids, weights):
    """Canonicalise a top-k decision: sort by expert id.

    The reference picks top-k with `argpartition` (arbitrary order) and
    APEX sorts by descending logit, so position-by-position comparison
    is only meaningful after both are put in the same order.
    """
    ids = np.asarray(ids).reshape(-1).astype(np.int64)
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    order = np.argsort(ids)
    return ids[order], weights[order]


# --------------------------------------------------------------------
# Reference side (mlx_lm)
# --------------------------------------------------------------------

def load_reference(ckpt: Path):
    """Build mlx_lm's Gemma 4 text model and load the checkpoint into it.

    Mirrors `mlx_lm.utils.load_model` (build the fp model, quantize the
    modules the checkpoint stores quantized, then load weights) but
    loads only the text tower: `mlx_lm.load` would also pull in the
    tokenizer stack, and the vision tower is dead weight here. Raises
    if any model parameter is missing from the checkpoint — a silently
    random-initialised tensor would make the "reference" meaningless.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm.models import gemma4_text as g4

    cfg = json.loads((ckpt / "config.json").read_text())
    text_cfg = cfg["text_config"]
    args = g4.ModelArgs(**{
        k: text_cfg[k]
        for k in text_cfg
        if k in g4.ModelArgs.__dataclass_fields__
    })
    print(f"  layers={args.num_hidden_layers}  hidden={args.hidden_size}  "
          f"experts={args.num_experts}  top_k={args.top_k_experts}")
    ref = g4.Model(args)

    weights = {}
    for f in sorted(ckpt.glob("*.safetensors")):
        weights.update(mx.load(str(f)))
    # The checkpoint is multimodal; keep only the text tower and strip
    # the "language_model." prefix so names line up with `g4.Model`.
    weights = {
        k.removeprefix("language_model."): v
        for k, v in weights.items()
        if k.startswith("language_model.")
    }
    weights = ref.sanitize(weights)

    quant = cfg.get("quantization")
    if quant:
        # Per-module overrides (the 8-bit affine routers) are keyed by
        # full path in the config; strip the same prefix as above.
        per_path = {
            k.removeprefix("language_model."): v
            for k, v in quant.items()
            if isinstance(v, dict)
        }

        def class_predicate(path, m):
            if path in per_path:
                return per_path[path]
            if not hasattr(m, "to_quantized"):
                return False
            # Same rule mlx_lm uses: a module is quantized iff the
            # checkpoint carries quantization scales for it.
            return f"{path}.scales" in weights

        nn.quantize(
            ref,
            group_size=quant["group_size"],
            bits=quant["bits"],
            mode=quant.get("mode", "affine"),
            class_predicate=class_predicate,
        )

    model_keys = {k for k, _ in tree_flatten(ref.parameters())}
    missing = sorted(model_keys - set(weights))
    unexpected = sorted(set(weights) - model_keys)
    if unexpected:
        print(f"  note: {len(unexpected)} checkpoint tensors unused by the "
              f"text model (e.g. {unexpected[:3]})")
    if missing:
        raise RuntimeError(
            f"{len(missing)} reference parameters are not in the "
            f"checkpoint, e.g. {missing[:5]} — refusing to run with "
            f"random-initialised weights"
        )
    ref.load_weights(list(weights.items()), strict=False)
    ref.eval()
    print("  weights loaded")
    return ref, args


def capture_reference(ckpt: Path, ids: list, ref_dtype: str) -> dict:
    """Run `ids` through the reference, recording every layer's stages.

    The layer body below is `gemma4_text.DecoderLayer.__call__` unrolled
    (MoE path) using the reference's own sub-modules, so intermediates
    can be read out. To prove the unrolled copy hasn't drifted from the
    real thing, its logits are checked against `ref(...)` at each token.

    Tokens go through ONE AT A TIME with the reference's KV caches, not
    as one batched prefill — the same shape of computation APEX does.
    It also keeps the reference trustworthy on older MLX: 0.31.2's nvfp4
    `quantized_matmul` is wrong (~18% relative error) for batches of
    12-32 rows on the dense MLP's down_proj shape (fixed by 0.32.3), so
    a batched call on a prompt of that length was not a usable ground
    truth there. One row per call is exact on both.

    `ref_dtype="float32"` casts the reference's bf16 parameters and
    activations up to fp32 so the comparison isolates math differences;
    `"bfloat16"` leaves them as mlx_lm ships them, which shows the
    rounding gap a stock mlx_lm run has against fp32 APEX.
    """
    import mlx.core as mx

    ref, args = load_reference(ckpt)
    if not args.enable_moe_block or args.hidden_size_per_layer_input \
            or args.num_kv_shared_layers:
        raise NotImplementedError(
            "capture_reference only unrolls the 26B MoE layer shape "
            "(enable_moe_block, no per-layer inputs, no shared KV)"
        )
    as_fp32 = ref_dtype == "float32"
    if as_fp32:
        # Floating parameters only (norm weights, router scales, layer
        # scalars); the packed uint weights and E4M3 scales are untouched.
        ref.set_dtype(mx.float32)
    model = ref.model
    n_layers = len(model.layers)

    def to_np(x):
        # (1, 1, ...) single-token result -> flat fp32 vector.
        return np.array(x.astype(mx.float32)).reshape(-1)

    caches = ref.make_cache()        # consumed by the unrolled loop
    check_caches = ref.make_cache()  # consumed by the built-in forward
    embed, logits_out = [], []
    per_stage = {s: [[] for _ in range(n_layers)] for s in _STAGES}
    route_ids = [[] for _ in range(n_layers)]
    route_weights = [[] for _ in range(n_layers)]
    worst_drift = 0.0
    act_dtype = None

    for token_id in ids:
        emb = model.embed_tokens(mx.array([[token_id]]))
        if as_fp32:
            emb = emb.astype(mx.float32)
        h = emb * model.embed_scale
        act_dtype = str(h.dtype)
        embed.append(to_np(h))

        masks = model._make_masks(h, caches)
        for l, (layer, mask, cache) in enumerate(
            zip(model.layers, masks, caches)
        ):
            residual = h
            a = layer.input_layernorm(h)
            a, _, _ = layer.self_attn(a, mask, cache)
            attn_out = layer.post_attention_layernorm(a)
            h = residual + attn_out

            residual = h
            dense_out = layer.mlp(layer.pre_feedforward_layernorm(h))
            # Router reads the RAW post-attention hidden, not a normed one.
            idx, w = layer.router(h)
            moe_out = layer.experts(
                layer.pre_feedforward_layernorm_2(h), idx, w,
            )
            f = (layer.post_feedforward_layernorm_1(dense_out)
                 + layer.post_feedforward_layernorm_2(moe_out))
            h = residual + layer.post_feedforward_layernorm(f)
            h = h * layer.layer_scalar

            # Force this layer now so the lazy graph never spans 30 layers.
            mx.eval(attn_out, dense_out, moe_out, h, idx, w)
            per_stage["attn_out"][l].append(to_np(attn_out))
            per_stage["dense_out"][l].append(to_np(dense_out))
            per_stage["moe_out"][l].append(to_np(moe_out))
            per_stage["layer_out"][l].append(to_np(h))
            ids_l, w_l = _sort_route(np.array(idx), to_np(w))
            route_ids[l].append(ids_l)
            route_weights[l].append(w_l)

        logits = model.embed_tokens.as_linear(model.norm(h))
        logits = mx.tanh(logits / ref.final_logit_softcapping) \
            * ref.final_logit_softcapping
        logits_out.append(to_np(logits))

        # Self-check: unrolled loop vs the reference's own forward, fed
        # the same embedding so both run in the same dtype.
        builtin = to_np(ref(None, cache=check_caches, input_embeddings=emb))
        worst_drift = max(
            worst_drift, float(np.max(np.abs(builtin - logits_out[-1]))),
        )

    print(f"  reference activations: {act_dtype}, one token per call")
    print(f"  unrolled-vs-builtin reference logit diff: {worst_drift:.2e}")
    if worst_drift > 1e-2:
        raise RuntimeError(
            "the unrolled reference loop no longer matches "
            "gemma4_text.Model.__call__ — update capture_reference"
        )

    out = {
        "embed": np.stack(embed),
        "logits": np.stack(logits_out),
        "ref_dtype": act_dtype,
    }
    for s in _STAGES:
        # (layers, positions, H)
        out[s] = np.stack([np.stack(rows) for rows in per_stage[s]])
    out["route_ids"] = np.stack([np.stack(rows) for rows in route_ids])
    out["route_weights"] = np.stack([np.stack(rows) for rows in route_weights])
    return out


# --------------------------------------------------------------------
# APEX side
# --------------------------------------------------------------------

def capture_apex(ckpt: Path, ids: list, *, num_layers, max_memory_gb,
                 backend) -> dict:
    """Run `ids` through APEX token by token, recording the same stages.

    Wraps `apex.model.decoder_decode_step` for the duration of the run so
    every (position, layer) call hands back its trace dict and output —
    no change to the library's forward path.
    """
    import apex
    import apex.model as apex_model
    from apex.kv_cache import KVCache

    m = apex.MoEModel.from_pretrained(
        str(ckpt), max_memory_gb=max_memory_gb, num_layers=num_layers,
        backend=backend, verbose=False,
    )
    n_layers = m.num_layers

    records = []  # one dict per call, ordered (pos 0: L0..Ln, pos 1: ...)
    original = apex_model.decoder_decode_step

    def recording_step(hidden, **kwargs):
        trace = {}
        kwargs["trace"] = trace
        result = original(hidden, **kwargs)
        trace["layer_in"] = hidden
        trace["layer_out"] = result
        records.append(trace)
        return result

    apex_model.decoder_decode_step = recording_step
    logits = []
    try:
        kv = KVCache(num_layers=n_layers)
        # ExpertManager prints one "cache miss/hit" line per expert
        # access; swallow them so the phase output stays readable.
        with contextlib.redirect_stdout(io.StringIO()):
            for token_id in ids:
                logits.append(m._compute_logits(token_id, kv))
    finally:
        apex_model.decoder_decode_step = original

    def flat(x):
        return np.asarray(x, dtype=np.float32).reshape(-1)

    n_pos = len(ids)
    rec = lambda layer, pos: records[pos * n_layers + layer]
    out = {
        # Layer 0's input is the scaled embedding.
        "embed": np.stack([flat(rec(0, p)["layer_in"]) for p in range(n_pos)]),
        "logits": np.stack([flat(x) for x in logits]),
    }
    for s in _STAGES:
        out[s] = np.stack([
            np.stack([flat(rec(l, p)[s]) for p in range(n_pos)])
            for l in range(n_layers)
        ])
    routes = [
        [_sort_route(rec(l, p)["route_ids"], rec(l, p)["route_weights"])
         for p in range(n_pos)]
        for l in range(n_layers)
    ]
    out["route_ids"] = np.stack(
        [np.stack([r[0] for r in row]) for row in routes])
    out["route_weights"] = np.stack(
        [np.stack([r[1] for r in row]) for row in routes])
    return out


# --------------------------------------------------------------------
# Diff
# --------------------------------------------------------------------

def _cos(a, b):
    """Per-position cosine similarity along the last axis."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return (a * b).sum(-1) / np.maximum(den, 1e-30)


def _rel(a, ref):
    """Per-position relative L2 error of `a` against `ref`."""
    a = a.astype(np.float64)
    ref = ref.astype(np.float64)
    return (np.linalg.norm(a - ref, axis=-1)
            / np.maximum(np.linalg.norm(ref, axis=-1), 1e-30))


def diff(ckpt: Path, tol: float) -> bool:
    """Print the per-layer comparison. Returns True when parity holds."""
    ref = np.load(_REF_NPZ)
    apx = np.load(_APEX_NPZ)
    ids = ref["input_ids"].tolist()
    if ids != apx["input_ids"].tolist():
        raise SystemExit(
            "ref and apex captures used different token ids — re-run "
            "both phases with the same --prompt / --chat"
        )
    cfg = json.loads((ckpt / "config.json").read_text())["text_config"]
    layer_types = cfg["layer_types"]
    n_ref, n_apx = ref["layer_out"].shape[0], apx["layer_out"].shape[0]
    n_layers = min(n_ref, n_apx)
    top_k = ref["route_ids"].shape[-1]

    print("=" * 78)
    print(f"prompt ids ({len(ids)}): {ids}")
    print(f"reference activation dtype: {ref['ref_dtype']}")
    print(f"layers compared: {n_layers} "
          f"(ref {n_ref}, apex {n_apx})   tolerance: rel err > {tol}")
    print("each cell: worst position's  cosine / relative-L2-error")
    print("-" * 78)

    first_bad = None  # (layer, stage, value)

    def note(layer, stage, bad, value):
        nonlocal first_bad
        if bad and first_bad is None:
            first_bad = (layer, stage, value)

    e_cos, e_rel = _cos(apx["embed"], ref["embed"]), _rel(apx["embed"], ref["embed"])
    print(f"embed          {e_cos.min():.5f} / {e_rel.max():.4f}")
    note(-1, "embed", e_rel.max() > tol, e_rel.max())

    print(f"{'L':>2} {'type':<8}" + "".join(f"{s:>17}" for s in _STAGES)
          + f"{'route':>9}{'d_weight':>10}")
    for l in range(n_layers):
        cells = {}
        for s in _STAGES:
            c, r = _cos(apx[s][l], ref[s][l]), _rel(apx[s][l], ref[s][l])
            cells[s] = (c.min(), r.max())
        # Router: how many of the top-k experts agree at the worst
        # position, and the worst weight gap among positions where the
        # id sets match exactly (weights are only comparable there).
        same = (apx["route_ids"][l] == ref["route_ids"][l])
        overlap = min(
            len(set(a.tolist()) & set(b.tolist()))
            for a, b in zip(apx["route_ids"][l], ref["route_ids"][l])
        )
        rows_equal = same.all(axis=-1)
        d_w = (
            float(np.abs(apx["route_weights"][l][rows_equal]
                         - ref["route_weights"][l][rows_equal]).max())
            if rows_equal.any() else float("nan")
        )
        # Execution order within a layer, so the first flagged stage
        # really is the earliest operator that went wrong.
        note(l, "attn_out", cells["attn_out"][1] > tol, cells["attn_out"][1])
        note(l, "dense_out", cells["dense_out"][1] > tol, cells["dense_out"][1])
        note(l, "router", overlap < top_k, overlap)
        note(l, "moe_out", cells["moe_out"][1] > tol, cells["moe_out"][1])
        note(l, "layer_out", cells["layer_out"][1] > tol, cells["layer_out"][1])

        kind = "full" if layer_types[l] == "full_attention" else "sliding"
        print(f"{l:>2} {kind:<8}"
              + "".join(f"{cells[s][0]:>9.5f}/{cells[s][1]:<7.4f}"
                        for s in _STAGES)
              + f"{overlap:>6}/{top_k}{d_w:>10.4f}")

    print("-" * 78)
    logits_ok = True
    logits_compared = n_ref == n_apx
    if logits_compared:
        from apex.tokenizer import Tokenizer
        tok = Tokenizer.from_pretrained(str(ckpt))
        r_log, a_log = ref["logits"], apx["logits"]
        top1 = (r_log.argmax(-1) == a_log.argmax(-1))
        print(f"next-token argmax agreement: {int(top1.sum())}/{len(top1)} "
              f"positions")
        r_top = np.argsort(-r_log[-1])[:10]
        a_top = np.argsort(-a_log[-1])[:10]
        print(f"last position — max |logit diff| "
              f"{float(np.abs(r_log[-1] - a_log[-1]).max()):.4f}, "
              f"top-10 overlap {len(set(r_top) & set(a_top))}/10")
        print(f"  {'mlx_lm reference':<38}{'APEX'}")
        for i, j in zip(r_top, a_top):
            left = f"{r_log[-1][i]:>7.3f} {tok.decode([int(i)])!r}"
            right = f"{a_log[-1][j]:>7.3f} {tok.decode([int(j)])!r}"
            print(f"  {left:<38}{right}")
        logits_ok = bool(top1.all())
    else:
        print("logits not compared (layer counts differ)")

    print("=" * 78)
    if first_bad is not None:
        layer, stage, value = first_bad
        where = "embedding" if layer < 0 else f"layer {layer} {stage}"
        print(f"FIRST DIVERGENCE: {where} (value {value:.4f})")
        return False
    if not logits_ok:
        print("layers within tolerance but next-token argmax disagrees")
        return False
    if logits_compared:
        print("PARITY OK: every stage within tolerance, argmax matches")
    else:
        # A partial-depth APEX run can only vouch for the layers it ran.
        print(f"PARITY OK for the first {n_layers} layers "
              f"(final logits not compared)")
    return True


# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "phase", nargs="?", default="all",
        choices=("all", "ref", "apex", "diff"),
        help="all = ref, apex (each in its own process), then diff",
    )
    parser.add_argument("--path", default=str(CKPT),
                        help="checkpoint directory")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--chat", action="store_true",
                        help="wrap the prompt in the chat template")
    parser.add_argument("--num-layers", type=int, default=None,
                        help="APEX decoder layers (default: all)")
    parser.add_argument("--max-memory-gb", type=float, default=4.0,
                        help="APEX expert cache budget (default 4)")
    parser.add_argument("--backend", choices=("numpy", "mlx"),
                        default="numpy", help="APEX expert backend")
    parser.add_argument("--ref-dtype", choices=("float32", "bfloat16"),
                        default="float32",
                        help="reference precision: float32 isolates math "
                             "differences; bfloat16 is stock mlx_lm")
    parser.add_argument("--tol", type=float, default=_DEFAULT_TOL,
                        help="relative-L2 error that counts as diverged")
    args = parser.parse_args()
    ckpt = Path(args.path)
    if not ckpt.is_dir():
        parser.error(f"checkpoint directory not found: {ckpt}")
    _TRACE_DIR.mkdir(exist_ok=True)

    if args.phase == "all":
        # Re-invoke this script once per model so each gets the whole
        # machine's memory to itself, then diff the two dumps here.
        passthrough = sys.argv[1:]
        for phase in ("ref", "apex"):
            subprocess.run(
                [sys.executable, __file__, phase, *passthrough], check=True,
            )
        sys.exit(0 if diff(ckpt, args.tol) else 1)

    if args.phase == "diff":
        sys.exit(0 if diff(ckpt, args.tol) else 1)

    ids = _encode(ckpt, args.prompt, args.chat)
    print("=" * 78)
    print(f"[{args.phase}] prompt {args.prompt!r} "
          f"(chat={args.chat}) -> {len(ids)} ids")
    if args.phase == "ref":
        capture = capture_reference(ckpt, ids, args.ref_dtype)
        target = _REF_NPZ
    else:
        capture = capture_apex(
            ckpt, ids, num_layers=args.num_layers,
            max_memory_gb=args.max_memory_gb, backend=args.backend,
        )
        target = _APEX_NPZ
    np.savez(target, input_ids=np.asarray(ids, dtype=np.int64), **capture)
    print(f"  wrote {target}")


if __name__ == "__main__":
    main()
