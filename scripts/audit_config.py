"""Step 22A: audit a checkpoint's config against APEX's runtime assumptions.

Walks the fields APEX uses inside `attention_decode_step`,
`decoder_decode_step`, and `final_logits` (head counts, head_dim,
RoPE theta, RMS eps, sliding/full layer split, EOS ids, tied
embedding) and prints one ✓ / ✗ per field with the value APEX is
parsing alongside the value on disk. Surfaces drift between the
runtime defaults and the model's actual hyperparameters before
generation quality starts to depend on them.

Done criteria from the step 22A spec:
    model config audit:
      ✓ hidden size: 2816
      ✓ layers: 30
      ✓ q heads: ...
      ✓ kv heads: ...
      ✓ rope theta: ...
      ✓ sliding window: ...
      ✓ norm eps: ...

The audit is non-fatal — every check prints its result and we exit 0
unless a field is structurally missing. The point is visibility, not
gating.

Run:
    uv run python scripts/audit_config.py \
        --path /Users/spotta/Desktop/models/gemma4/gemma-4-26b-a4b-it-nvfp4
"""

import argparse
import json
import sys
from pathlib import Path

# Make the apex package importable when run directly from a clone.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apex.config import ModelConfig


_OK = "\033[32m✓\033[0m"
_BAD = "\033[31m✗\033[0m"
_INFO = "\033[33m·\033[0m"


def _row(ok: bool, label: str, apex_value, raw_value=None):
    """Print one audit row. `raw_value` is shown when it differs from apex."""
    marker = _OK if ok else _BAD
    if raw_value is not None and raw_value != apex_value:
        print(f"  {marker} {label}: apex={apex_value} raw={raw_value}")
    else:
        print(f"  {marker} {label}: {apex_value}")


def _info(label, value):
    """Informational row (no pass/fail) for fields APEX exposes but
    doesn't yet act on."""
    print(f"  {_INFO} {label}: {value}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--path",
        required=True,
        help="checkpoint directory (must contain config.json)",
    )
    args = parser.parse_args()

    path = Path(args.path)
    config = ModelConfig.from_path(path)
    with open(path / "config.json") as f:
        raw = json.load(f)
    text_raw = raw.get("text_config", raw)

    print(f"checkpoint: {path}")
    print(f"layout:     {config.layout_name}")
    print()

    print("model config audit:")

    # Things APEX uses every forward pass — must match what's on disk.
    _row(config.hidden_size == text_raw["hidden_size"],
         "hidden size", config.hidden_size, text_raw["hidden_size"])
    _row(config.num_hidden_layers == text_raw["num_hidden_layers"],
         "layers", config.num_hidden_layers, text_raw["num_hidden_layers"])
    _row(config.num_attention_heads == text_raw["num_attention_heads"],
         "q heads", config.num_attention_heads, text_raw["num_attention_heads"])
    _row(config.num_kv_heads == text_raw["num_key_value_heads"],
         "kv heads", config.num_kv_heads, text_raw["num_key_value_heads"])
    _row(config.head_dim == text_raw["head_dim"],
         "head_dim (sliding)", config.head_dim, text_raw["head_dim"])
    _row(config.rms_norm_eps == text_raw["rms_norm_eps"],
         "rms norm eps", config.rms_norm_eps, text_raw["rms_norm_eps"])

    rope = text_raw.get("rope_parameters", {})
    sliding_rope = rope.get("sliding_attention", {})
    raw_theta_sliding = sliding_rope.get("rope_theta")
    _row(config.rope_theta_sliding == raw_theta_sliding,
         "rope theta (sliding)",
         config.rope_theta_sliding, raw_theta_sliding)

    sliding_window = text_raw.get("sliding_window")
    _info("sliding window", sliding_window)

    softcap_raw = text_raw.get("final_logit_softcapping")
    _row(config.final_logit_softcapping == softcap_raw,
         "final logit softcap", config.final_logit_softcapping, softcap_raw)

    bos_raw = text_raw.get("bos_token_id")
    _row(config.bos_token_id == bos_raw,
         "bos token id", config.bos_token_id, bos_raw)

    eos_raw = text_raw.get("eos_token_id")
    if isinstance(eos_raw, (list, tuple)):
        eos_normalized = tuple(int(x) for x in eos_raw)
    elif eos_raw is None:
        eos_normalized = ()
    else:
        eos_normalized = (int(eos_raw),)
    # Show the normalized form as the "raw" comparator so a scalar 1
    # on disk doesn't false-flag against the tuple (1,) we parse to.
    _row(config.eos_token_ids == eos_normalized,
         "eos token ids", list(config.eos_token_ids))

    tie = text_raw.get("tie_word_embeddings", True)
    _info("tie word embeddings", tie)

    _info("vocab_size", config.vocab_size)
    _info("num_experts", config.num_experts)
    _info("top_k_experts", config.top_k_experts)

    # Layer-type audit.
    print()
    print("layer types:")
    sliding = config.num_sliding_layers
    full = sum(1 for kind in config.layer_types if kind == "full_attention")
    _info("total layers", len(config.layer_types))
    _info(f"sliding (supported, 0-{sliding-1})", sliding)
    _info("full (supported as of step 23)", full)
    if full > 0:
        # Step 23 turned these from info-only to ✓ rows; the runtime
        # consumes them per-layer via `config.*_for_layer(L)`.
        _row(config.attention_k_eq_v == bool(text_raw.get("attention_k_eq_v")),
             "  full-attn attention_k_eq_v", config.attention_k_eq_v)
        _row(config.head_dim_full == text_raw.get("global_head_dim"),
             "  full-attn global_head_dim", config.head_dim_full)
        _row(
            config.num_kv_heads_full
                == text_raw.get("num_global_key_value_heads"),
            "  full-attn num_global_key_value_heads",
            config.num_kv_heads_full,
        )
        _row(
            config.rope_theta_full
                == rope.get("full_attention", {}).get("rope_theta"),
            "  full-attn rope_theta",
            config.rope_theta_full,
        )
        _row(
            config.partial_rotary_factor_full
                == rope.get("full_attention", {}).get("partial_rotary_factor"),
            "  full-attn partial_rotary_factor",
            config.partial_rotary_factor_full,
        )
        rotary_dim = int(
            config.head_dim_full * config.partial_rotary_factor_full,
        )
        _info("  full-attn rotary_dim (derived)", rotary_dim)

    print()
    print(f"max generate(num_layers): {config.num_hidden_layers} "
          f"(both sliding + full attention)")
    print()
    print("[ok] audit complete")


if __name__ == "__main__":
    main()
