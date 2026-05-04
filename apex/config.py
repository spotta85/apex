"""ModelConfig — frozen view of one checkpoint's hyperparameters.

`MoEModel.from_pretrained` builds one of these once at load time;
nothing else writes to it. Pulling the parsing into one place keeps
`apex/model.py` focused on wiring runtime objects together rather
than juggling `text_config[...]` lookups everywhere.

Fields are deliberately a flat snapshot (no nested dicts) so the
generation loop can read them as plain attributes without poking back
into JSON. Whatever the underlying file calls a thing, we surface it
under a stable Python name.

Two Gemma 4 quirks worth noting:

  - `eos_token_id` may be a scalar (1) or a list ([1, 106, 50]) on disk;
    `.eos_token_ids` is always a tuple so the decode loop can `if id
    in self.config.eos_token_ids`.
  - Per-layer RoPE: Gemma 4 uses different `rope_theta` for sliding
    vs full attention. We surface only the sliding value here because
    full_attention layers (5+) aren't supported by the runtime yet
    (they tie K=V and use partial RoPE — a separate sprint).
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    """Immutable snapshot of one checkpoint's text-model hyperparameters.

    Built once via `from_path`; consumed by `MoEModel` to drive the
    decoder. Field names mirror the names callers already use in the
    demos so the migration to the new API is mechanical.
    """

    path: Path
    layout_name: str
    layout: object

    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    num_hidden_layers: int
    vocab_size: int

    num_experts: int
    top_k_experts: int

    rms_norm_eps: float
    final_logit_softcapping: float | None

    layer_types: tuple
    bos_token_id: int | None
    eos_token_ids: tuple

    rope_theta_sliding: float

    @classmethod
    def from_path(cls, path, *, layout_name=None) -> "ModelConfig":
        """Detect layout, read text_config, return a fully-populated ModelConfig.

        `layout_name` lets callers force a specific layout (e.g. tests
        that ship a hand-built checkpoint); when omitted we use the
        same auto-detect that the demo scripts use.

        Raises FileNotFoundError / ValueError with messages that name
        the missing piece — the operator-facing API needs clear errors
        more than it needs a single uniform exception type.
        """
        from apex.checkpoint_loaders import read_text_config

        path = Path(path)
        if not path.is_dir():
            raise FileNotFoundError(
                f"checkpoint directory not found: {path}"
            )

        if layout_name is not None:
            from apex.weight_layout import get_layout

            layout = get_layout(layout_name)
            resolved_layout_name = layout_name
        else:
            from scripts._real_moe_helpers import detect_real_moe_layout

            resolved_layout_name, layout = detect_real_moe_layout(path)

        text_cfg = read_text_config(path)

        # eos may be scalar or list — normalize to a tuple.
        eos = text_cfg.get("eos_token_id")
        if eos is None:
            eos_ids = ()
        elif isinstance(eos, (list, tuple)):
            eos_ids = tuple(int(x) for x in eos)
        else:
            eos_ids = (int(eos),)

        # Gemma 4 nests RoPE per layer-type; sliding is the only path
        # the runtime exercises today.
        rope = text_cfg.get("rope_parameters", {})
        sliding_rope = rope.get("sliding_attention", {})
        rope_theta_sliding = float(sliding_rope.get("rope_theta", 10000.0))

        return cls(
            path=path,
            layout_name=resolved_layout_name,
            layout=layout,
            hidden_size=int(text_cfg["hidden_size"]),
            num_attention_heads=int(text_cfg["num_attention_heads"]),
            num_kv_heads=int(text_cfg["num_key_value_heads"]),
            head_dim=int(text_cfg["head_dim"]),
            num_hidden_layers=int(text_cfg["num_hidden_layers"]),
            vocab_size=int(text_cfg["vocab_size"]),
            num_experts=int(text_cfg.get("num_experts", 0)),
            top_k_experts=int(text_cfg.get("top_k_experts", 8)),
            rms_norm_eps=float(text_cfg["rms_norm_eps"]),
            final_logit_softcapping=(
                float(text_cfg["final_logit_softcapping"])
                if text_cfg.get("final_logit_softcapping") is not None
                else None
            ),
            layer_types=tuple(text_cfg.get("layer_types", ())),
            bos_token_id=(
                int(text_cfg["bos_token_id"])
                if text_cfg.get("bos_token_id") is not None
                else None
            ),
            eos_token_ids=eos_ids,
            rope_theta_sliding=rope_theta_sliding,
        )

    @property
    def num_sliding_layers(self) -> int:
        """Count of leading layers using sliding_attention.

        The runtime only supports sliding-attention layers today; this
        is the natural cap for `num_layers` in `from_pretrained`. On
        Gemma 4 this returns 5 (layers 0-4 are sliding, layer 5 is
        full_attention with K=V tied).
        """
        count = 0
        for kind in self.layer_types:
            if kind == "sliding_attention":
                count += 1
            else:
                break
        return count
