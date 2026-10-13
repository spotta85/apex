"""Dump per-layer reference activations from mlx_lm's Gemma 4 model.

This is the answer key the C++ engine checks itself against. For one
fixed prompt it runs mlx_lm's `gemma4_text.Model` one token at a time
and records, at every layer and every prompt position:
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CKPT = Path("/Users/spotta/Desktop/models/gemma4/gemma-4-26b-a4b-it-nvfp4")
REPO_ROOT = Path(__file__).resolve().parent.parent
DUMP_ROOT = REPO_ROOT / "dumps"

# Per-layer hidden-state stages, in execution order inside one layer.
# The router sits between dense_out and moe_out (handled separately
# because it is an id set + weights, not a hidden vector).
STAGES = ("attn_in", "attn_out", "dense_out", "moe_out", "layer_out")


# --------------------------------------------------------------------
# Tokenization
# --------------------------------------------------------------------

def encode(ckpt: Path, prompt: str, chat: bool) -> list[int]:
    """Token ids for the prompt, with <bos> exactly once.

    The chat template itself emits `<bos>` (and, with thinking off, an
    empty thought channel), so we render it to text and encode with
    `add_special_tokens=False`. Raw prompts get <bos> prepended the way
    Gemma's tokenizer does by default.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ckpt))
    if chat:
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        return tok.encode(text, add_special_tokens=False)
    return [tok.bos_token_id] + tok.encode(prompt, add_special_tokens=False)


# --------------------------------------------------------------------
# Reference model
# --------------------------------------------------------------------

def load_reference(ckpt: Path):
    """Build mlx_lm's Gemma 4 text model and load the checkpoint into it.

    Mirrors `mlx_lm.utils.load_model` (build the fp model, quantize the
    modules the checkpoint stores quantized, then load weights) but
    loads only the text tower: the vision tower is dead weight here.
    Raises if any model parameter is missing from the checkpoint; a
    silently random-initialised tensor would make the answer key wrong.
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
        # Per-module overrides are keyed by full path in the config;
        # strip the same prefix as above.
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
            f"checkpoint, e.g. {missing[:5]}; refusing to run with "
            f"random-initialised weights"
        )
    ref.load_weights(list(weights.items()), strict=False)
    ref.eval()
    print("  weights loaded")
    return ref, args, text_cfg


def sort_route(ids, weights):
    """Canonicalise a top-k decision: sort by expert id.

    The reference picks top-k with `argpartition` (arbitrary order).
    Sorting by id makes position-by-position comparison well-defined
    regardless of how the C++ router orders its picks.
    """
    ids = np.asarray(ids).reshape(-1).astype(np.int64)
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    order = np.argsort(ids)
    return ids[order], weights[order]


def capture_reference(ckpt: Path, ids: list[int], ref_dtype: str):
    """Run `ids` through the reference, recording every layer's stages.

    Tokens go through ONE AT A TIME with the reference's KV caches, not
    as one batched prefill. That is the same shape of computation the
    engine's decode path does, and it sidesteps the batched nvfp4
    `quantized_matmul` bug in MLX < 0.32.3.

    `ref_dtype="float32"` casts the reference's floating parameters and
    activations up to fp32 so comparisons isolate math differences;
    `"bfloat16"` leaves them as mlx_lm ships them.
    """
    import mlx.core as mx

    ref, args, text_cfg = load_reference(ckpt)
    if not args.enable_moe_block or args.hidden_size_per_layer_input \
            or args.num_kv_shared_layers:
        raise NotImplementedError(
            "capture_reference only unrolls the 26B MoE layer shape "
            "(enable_moe_block, no per-layer inputs, no shared KV)"
        )
    if ref_dtype == "float32":
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
    per_stage = {s: [[] for _ in range(n_layers)] for s in STAGES}
    route_ids = [[] for _ in range(n_layers)]
    route_weights = [[] for _ in range(n_layers)]
    worst_drift = 0.0
    act_dtype = None

    for pos, token_id in enumerate(ids):
        emb = model.embed_tokens(mx.array([[token_id]]))
        if ref_dtype == "float32":
            emb = emb.astype(mx.float32)
        h = emb * model.embed_scale
        act_dtype = str(h.dtype)
        embed.append(to_np(h))

        masks = model._make_masks(h, caches)
        for l, (layer, mask, cache) in enumerate(
            zip(model.layers, masks, caches)
        ):
            # --- attention block ---
            residual = h
            a = layer.input_layernorm(h)
            attn_in = a                       # input_layernorm(h): the input to attention
            a, _, _ = layer.self_attn(attn_in, mask, cache)
            attn_out = layer.post_attention_layernorm(a)
            h = residual + attn_out

            # --- feed-forward block: dense MLP and routed experts run
            #     side by side on the same hidden, each with its own norms ---
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
            mx.eval(attn_in, attn_out, dense_out, moe_out, h, idx, w)
            per_stage["attn_in"][l].append(to_np(attn_in))
            per_stage["attn_out"][l].append(to_np(attn_out))
            per_stage["dense_out"][l].append(to_np(dense_out))
            per_stage["moe_out"][l].append(to_np(moe_out))
            per_stage["layer_out"][l].append(to_np(h))
            ids_l, w_l = sort_route(np.array(idx), to_np(w))
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
        print(f"  pos {pos:3d} token {token_id:7d}  "
              f"argmax {int(np.argmax(logits_out[-1])):7d}", flush=True)

    print(f"  reference activations: {act_dtype}, one token per call")
    print(f"  unrolled-vs-builtin reference logit diff: {worst_drift:.2e}")
    if worst_drift > 1e-2:
        raise RuntimeError(
            "the unrolled reference loop no longer matches "
            "gemma4_text.Model.__call__; update capture_reference"
        )

    out = {
        "embed": np.stack(embed),            # (positions, H)
        "logits": np.stack(logits_out),      # (positions, vocab)
    }
    for s in STAGES:
        # (layers, positions, H)
        out[s] = np.stack([np.stack(rows) for rows in per_stage[s]])
    out["route_ids"] = np.stack([np.stack(rows) for rows in route_ids])
    out["route_weights"] = np.stack([np.stack(rows) for rows in route_weights])
    return out, act_dtype, text_cfg


# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--name", required=True,
                        help="dump directory name under dumps/")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--raw", action="store_true",
                        help="skip the chat template (default: apply it)")
    parser.add_argument("--ckpt", default=str(CKPT))
    parser.add_argument("--ref-dtype", choices=("float32", "bfloat16"),
                        default="float32")
    args = parser.parse_args()
    ckpt = Path(args.ckpt)
    if not ckpt.is_dir():
        parser.error(f"checkpoint directory not found: {ckpt}")

    ids = encode(ckpt, args.prompt, chat=not args.raw)
    print(f"prompt {args.prompt!r} (chat={not args.raw}) -> {len(ids)} ids")

    capture, act_dtype, text_cfg = capture_reference(ckpt, ids, args.ref_dtype)

    out_dir = DUMP_ROOT / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "ids.npy", np.asarray(ids, dtype=np.int64))
    shapes = {}
    for k, v in capture.items():
        np.save(out_dir / f"{k}.npy", v)
        shapes[k] = list(v.shape)
    meta = {
        "prompt": args.prompt,
        "chat_template": not args.raw,
        "checkpoint": str(ckpt),
        "ref_dtype": act_dtype,
        "num_tokens": len(ids),
        "shapes": shapes,
        "layout": {
            "embed": "(positions, hidden)",
            "logits": "(positions, vocab)",
            "attn_in/attn_out/dense_out/moe_out/layer_out": "(layers, positions, hidden)",
            "route_ids/route_weights": "(layers, positions, top_k), sorted by expert id",
        },
        "text_config": text_cfg,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    total = sum((out_dir / f).stat().st_size for f in
                [p.name for p in out_dir.iterdir()])
    print(f"wrote {out_dir} ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
