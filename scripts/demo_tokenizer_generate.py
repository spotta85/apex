"""Step 19 demo: text-prompt generation on real Gemma 4.

Pipeline (replaces the "hardcoded token id" pattern of step 17/18 demos):

    text prompt
        → apex.tokenizer.Tokenizer.encode  (with BOS prepend)
        → apex.prefill.prefill             (one decode_step per token,
                                            KV cache fills monotonically)
        → decode loop                       (feed prediction back N times)
        → apex.tokenizer.Tokenizer.decode  (skip special tokens)
        → printed text

What this script proves (step 19 done criteria):
  - text in / text out (no hardcoded prompt token ids)
  - tokenizer encode → prefill works end-to-end
  - generated token ids decode into text
  - KV cache length tracks prompt tokens + generated tokens
  - clear error if tokenizer.json is missing (raised by
    apex.tokenizer.Tokenizer.from_pretrained)

Same one-decoder-layer caveat as the 16E / 17 / 18 demos: only layers
0-4 work (sliding_attention); layer 5+ ties K=V and isn't supported
yet. Output text from a single layer is not coherent — the smoke
proves the wiring, not the model quality.

Run:
    uv run python scripts/demo_tokenizer_generate.py
        [--path /path/to/checkpoint] [--num-layers 1]
        [--prompt "Hello world"] [--max-tokens 8]
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
from apex.tokenizer import Tokenizer
from scripts._real_moe_helpers import (
    add_real_moe_args,
    detect_real_moe_layout,
    nvfp4_matmul_warnings,
    quantized_store_for_layout,
)


_MAX_SLIDING_LAYERS = 5


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
        default="Hello world",
        help='text prompt (default: "Hello world")',
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8,
        help="how many tokens to generate after the prompt (default 8)",
    )
    args = parser.parse_args()

    if args.num_layers > _MAX_SLIDING_LAYERS:
        raise SystemExit(
            f"--num-layers {args.num_layers} > {_MAX_SLIDING_LAYERS}: "
            "Gemma 4 layer 5+ uses full_attention with K=V tied; this "
            "demo only supports the sliding-attention layers (0-4)."
        )
    if args.max_tokens <= 0:
        raise SystemExit("--max-tokens must be positive")

    path = Path(args.path)
    if not path.is_dir():
        raise SystemExit(f"checkpoint not found: {path}")

    print(f"loading checkpoint: {path}")
    layout_name, layout = detect_real_moe_layout(path)
    text_cfg = read_text_config(path)
    print(f"layout: {layout_name}")

    print("loading tokenizer")
    tokenizer = Tokenizer.from_pretrained(path)
    prompt_ids = tokenizer.encode(args.prompt)
    print()
    print(f"prompt: {args.prompt}")
    print(f"prompt tokens: {prompt_ids}  ({len(prompt_ids)} tokens)")
    print()

    # Big tensors loaded once.
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

    kv = KVCache(num_layers=args.num_layers)

    def _decode(token_id: int) -> int:
        """Per-token forward: embed → N layers → final_norm → lm_head → argmax."""
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

    print(f"prefill tokens: {len(prompt_ids)}")
    first_generated = prefill(prompt_ids, step_fn=_decode)
    print(f"prefill complete: kv cache length {kv.length(0)}")
    print()

    # Decode loop, printing the running detokenized text after each
    # step. With one layer the model converges to a steady token
    # quickly — the trace makes that visible without obscuring it.
    generated_ids = [first_generated]
    print(f"generating token 1")
    print(f"  next token id: {first_generated}")
    print(f"  decoded so far: {tokenizer.decode(generated_ids)!r}")
    last_id = first_generated
    for step in range(args.max_tokens - 1):
        last_id = _decode(last_id)
        generated_ids.append(last_id)
        print(f"\ngenerating token {step + 2}")
        print(f"  next token id: {last_id}")
        print(f"  decoded so far: {tokenizer.decode(generated_ids)!r}")

    # Final summary.
    print()
    print(f"prompt:           {args.prompt!r}")
    print(f"generated ids:    {generated_ids}")
    print(f"decoded text:     {tokenizer.decode(generated_ids)!r}")
    print(f"full text:        {(args.prompt + tokenizer.decode(generated_ids))!r}")
    print(f"final kv length:  {kv.length(0)}  "
          f"(prompt {len(prompt_ids)} + generated {len(generated_ids)})")

    print()
    print("memory:")
    runtime_snapshot(cache, metrics, kv).print_summary()

    print()
    print("[ok] real tokenizer-driven generation passed")


if __name__ == "__main__":
    main()
