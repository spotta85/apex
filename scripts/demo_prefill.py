"""Step 18 demo: real prompt prefill on Gemma 4 + a short decode tail.

Builds the same one-decoder-layer stack that the 17D / 17E demos use
(embedding → decoder layer 0 → final RMSNorm → tied lm_head + softcap
→ argmax), then drives it through two phases:

  prefill: apex.prefill.prefill(prompt_ids, step_fn=_decode)
           — loops _decode once per prompt token. KV cache grows
             from 0 → len(prompt_ids).

  decode:  feed the last prediction back through _decode N times.
           KV cache grows further by N entries.

Done criteria from the step 18 spec:
  ✓ multiple prompt tokens fill KV cache
  ✓ decode starts after prompt cache is built
  ✓ generated tokens depend on prompt tokens (run with two
    different --prompt values to see the predicted next-id change)
  ✓ memory report includes prompt KV cache (via runtime_snapshot)

What this script intentionally does NOT do:
  - Tokenizer integration (token ids only — step 19)
  - Batched-prompt-attention prefill (step 18 spec defers this; the
    "loop over prompt tokens" path is the spec's "start simple")

Run:
    uv run python scripts/demo_prefill.py
        [--path /path/to/checkpoint] [--num-layers 1]
        [--prompt 2,235,1234,5,9] [--generate 3]
        [--top-k 8] [--max-memory-mb 512]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Make the apex package importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apex.backends.fake_backend import FakeBackend
from apex.checkpoint_loaders import (
    load_attention_weights,
    load_embedding,
    load_final_norm,
    load_layer_norms,
    read_text_config,
)
from apex.decoder import decoder_decode_step
from apex.dequant import DequantizingExpertStore
from apex.expert_cache import ExpertCache
from apex.expert_manager import ExpertManager
from apex.generation import embed_token, final_logits, sample_argmax
from apex.kv_cache import KVCache
from apex.memory import runtime_snapshot
from apex.metrics import Metrics
from apex.moe_layer import run_moe_layer
from apex.ops import rms_norm
from apex.prefill import prefill
from apex.real_router import MultiLayerRealRouter
from scripts._real_moe_helpers import (
    add_real_moe_args,
    detect_real_moe_layout,
    nvfp4_matmul_warnings,
    quantized_store_for_layout,
)


# Same cap as 16E / 17D demos: layers 5+ use full_attention with K=V
# tied and head_dim=512 — not yet supported.
_MAX_SLIDING_LAYERS = 5


def _parse_prompt(prompt_str: str) -> list:
    return [int(tok) for tok in prompt_str.split(",") if tok.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    add_real_moe_args(parser, default_max_memory_mb=512)
    parser.add_argument(
        "--num-layers",
        type=int,
        default=1,
        help=(
            f"how many decoder layers (default 1; max {_MAX_SLIDING_LAYERS} — "
            "layer 5+ uses full_attention with K=V tied)"
        ),
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="2,235,1234,5,9",
        help=(
            "comma-separated prompt token ids (default '2,235,1234,5,9' — "
            "BOS plus four arbitrary ids to make prefill ≠ decode obvious)"
        ),
    )
    parser.add_argument(
        "--generate",
        type=int,
        default=3,
        help="how many tokens to generate after the prompt (default 3)",
    )
    args = parser.parse_args()

    if args.num_layers > _MAX_SLIDING_LAYERS:
        raise SystemExit(
            f"--num-layers {args.num_layers} > {_MAX_SLIDING_LAYERS}: "
            "Gemma 4 layer 5+ uses full_attention with K=V tied; this "
            "demo only supports the sliding-attention layers (0-4)."
        )

    prompt_ids = _parse_prompt(args.prompt)
    if not prompt_ids:
        raise SystemExit("--prompt must contain at least one token id")

    path = Path(args.path)
    if not path.is_dir():
        raise SystemExit(f"checkpoint not found: {path}")

    layout_name, layout = detect_real_moe_layout(path)
    text_cfg = read_text_config(path)
    print(f"layout: {layout_name}")
    print(f"prompt: {prompt_ids}  ({len(prompt_ids)} tokens)")
    print(f"generate: {args.generate}  decoder_layers: {args.num_layers}")
    print()

    # Big tensors loaded once — embedding dequant is the dominant
    # cost (~3 GB on Gemma 4).
    print("loading + dequantizing embedding table (a few seconds)...")
    embedding = load_embedding(path, layout)
    final_norm_w = load_final_norm(path, layout)

    per_layer_attn = []
    per_layer_norms = []
    for layer_id in range(args.num_layers):
        per_layer_attn.append(load_attention_weights(path, layout, layer_id))
        per_layer_norms.append(load_layer_norms(path, layout, layer_id))

    metrics = Metrics()
    cache = ExpertCache(
        max_memory_mb=args.max_memory_mb, verbose=False, metrics=metrics,
    )
    quant_store = quantized_store_for_layout(path, layout)
    store = DequantizingExpertStore(quant_store, verbose=False)
    manager = ExpertManager(store=store, cache=cache, metrics=metrics)
    backend = FakeBackend()
    router = MultiLayerRealRouter(path, layout, top_k=args.top_k)

    # KV cache spans both prefill and decode — that's the whole point.
    kv = KVCache(num_layers=args.num_layers)

    def _decode(token_id: int) -> int:
        """One forward pass through embed → N layers → lm_head → argmax."""
        hidden = embed_token(embedding, token_id)
        for layer_id in range(args.num_layers):
            attn_weights = per_layer_attn[layer_id]
            norms = per_layer_norms[layer_id]
            attn_kwargs = dict(
                q_weight=attn_weights["q_weight"],
                k_weight=attn_weights["k_weight"],
                v_weight=attn_weights["v_weight"],
                o_weight=attn_weights["o_weight"],
                num_heads=text_cfg["num_attention_heads"],
                num_kv_heads=text_cfg["num_key_value_heads"],
                head_dim=text_cfg["head_dim"],
                q_norm_weight=attn_weights["q_norm"],
                k_norm_weight=attn_weights["k_norm"],
                rms_eps=text_cfg["rms_norm_eps"],
                rope_theta=10000.0,
            )

            def moe_call(h, lid=layer_id):
                with nvfp4_matmul_warnings():
                    ids, weights = router.route(lid, h)
                    return run_moe_layer(
                        h, ids, weights,
                        lambda eid: manager.ensure_loaded(lid, eid),
                        backend,
                    )

            with nvfp4_matmul_warnings():
                hidden = decoder_decode_step(
                    hidden,
                    norms=norms,
                    attn_kwargs=attn_kwargs,
                    moe_call=moe_call,
                    kv_cache=kv,
                    layer_id=layer_id,
                    rms_eps=text_cfg["rms_norm_eps"],
                )

        hidden = rms_norm(
            hidden, final_norm_w,
            eps=text_cfg["rms_norm_eps"], gemma_unit_offset=True,
        )
        with nvfp4_matmul_warnings():
            logits = final_logits(
                hidden, embedding,
                softcap=text_cfg.get("final_logit_softcapping"),
            )
        assert np.all(np.isfinite(logits)), "logits must be finite"
        return sample_argmax(logits)

    # === Prefill: process every prompt token, growing the KV cache ===
    print("=== prefill ===")
    # Wrap _decode to print per-token progress so the user can see the
    # cache growing alongside the prompt. The wrapped fn matches the
    # spec's "prefill token N / kv cache length: N" trace.
    step_idx = 0

    def step_with_log(token_id):
        nonlocal step_idx
        step_idx += 1
        next_id = _decode(token_id)
        print(f"prefill token {step_idx}")
        print(f"kv cache length: {kv.length(0)}")
        return next_id

    first_generated = prefill(prompt_ids, step_fn=step_with_log)
    print()

    # === Decode: continue generating tokens, KV cache keeps growing ===
    print("=== decode ===")
    generated = [first_generated]
    decode_idx = 1
    print(f"decode token {decode_idx}  (= last-prompt-argmax)")
    print(f"kv cache length: {kv.length(0)}")
    print(f"  → id {first_generated}")
    last_pred = first_generated
    for _ in range(args.generate - 1):
        decode_idx += 1
        last_pred = _decode(last_pred)
        generated.append(last_pred)
        print(f"\ndecode token {decode_idx}")
        print(f"kv cache length: {kv.length(0)}")
        print(f"  → id {last_pred}")

    print()
    print(f"prompt:    {prompt_ids}")
    print(f"generated: {generated}")
    print(f"final kv_cache length: {kv.length(0)}  "
          f"(prompt {len(prompt_ids)} + generated {len(generated)})")

    # Joined memory report — includes prompt KV cache, per the step 18
    # done criterion.
    print()
    print("=== memory ===")
    runtime_snapshot(cache, metrics, kv).print_summary()

    print()
    print("[ok] real prompt prefill demo passed")


if __name__ == "__main__":
    main()
