"""ModelConfig — frozen view of one checkpoint's hyperparameters.

`MoEModel.from_pretrained` builds one of these once at load time;
nothing else writes to it. Pulling the parsing into one place keeps
`apex/model.py` focused on wiring runtime objects together rather
than juggling `text_config[...]` lookups everywhere.

Fields are deliberately a flat snapshot (no nested dicts) so the
generation loop can read them as plain attributes without poking back
into JSON. Whatever the underlying file calls a thing, we surface it
under a stable Python name.

Three Gemma 4 quirks worth noting:

  - `eos_token_id` may be a scalar (1) or a list ([1, 106, 50]) on disk;
    `.eos_token_ids` is always a tuple so the decode loop can `if id
    in self.config.eos_token_ids`.
  - Per-layer RoPE: Gemma 4 uses different `rope_theta` for sliding
    vs full attention (10000 vs 1e6). Both are exposed; consumers
    pick via `rope_theta_for_layer(L)`.
  - Full-attention extras (step 23): on top of a different head_dim
    (`global_head_dim`, e.g. 512), full layers also have a different
    KV-head count (`num_global_key_value_heads`, e.g. 2 instead of 8),
    tie K=V (`attention_k_eq_v=True`, no `v_proj` on disk), and use
    partial RoPE (`partial_rotary_factor=0.25`). The per-layer
    accessors `head_dim_for_layer`, `num_kv_heads_for_layer`,
    `rotary_dim_for_layer`, `rope_theta_for_layer` hide the branch.
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

    # --- Step 23: full_attention extras ---
    # `global_head_dim`: head_dim used by full_attention layers.
    # On Gemma 4 = 512 (vs 256 for sliding).
    head_dim_full: int
    # `num_global_key_value_heads`: KV-head count for full layers.
    # On Gemma 4 = 2 (vs 8 for sliding).
    num_kv_heads_full: int
    # `attention_k_eq_v`: when True, full_attention layers don't have
    # `v_proj` on disk — V is tied to K (a memory + compute saving
    # the model was trained with).
    attention_k_eq_v: bool
    # `partial_rotary_factor`: fraction of head_dim that gets RoPE.
    # On Gemma 4 = 0.25 → first 128 of 512 dims rotated, last 384 left
    # alone (the unrotated dims absorb position-invariant features).
    partial_rotary_factor_full: float
    # Different RoPE base for full_attention. Bigger base = longer
    # frequency wavelengths = better long-range positional sensitivity.
    rope_theta_full: float

    # --- Step 24C: sliding-window attention scope -----------------------
    # `sliding_window`: max attention scope (in tokens) for sliding
    # layers. Gemma 4 = 1024. Full_attention layers ignore this. None
    # for models that don't differentiate (every layer attends globally).
    sliding_window: int | None

    # --- Step 26A: FFN activation function ------------------------------
    # `hidden_activation` from text_config. Gemma 4 = "gelu_pytorch_tanh"
    # (tanh-approximation GELU); Mixtral / Qwen MoE = "silu"; default
    # falls back to "silu" because that's what every public MoE used
    # before Gemma 4. The backend reads this on construction and picks
    # the matching activation in run_expert / dense MLP. Using SiLU
    # against a GeLU-trained model is a major source of incoherence —
    # SiLU and GeLU have similar shapes but the layer-after-layer error
    # compounds quickly across 30 decoder layers.
    hidden_activation: str

    # --- Step 26C: model_type + embedding scale -------------------------
    # `model_type` from text_config (e.g. "gemma4_text", "mixtral",
    # "qwen2_moe"). Used to pick architecture-specific behaviors that
    # aren't on a single config flag.
    model_type: str
    # `embed_scale`: multiplier applied to the embedding output at
    # model entry. Gemma family multiplies by `sqrt(hidden_size)` (a
    # large factor — ~53 on Gemma 4 26B). Mixtral / Qwen don't, so
    # default 1.0 for non-Gemma layouts. Without this scale the
    # residual stream enters the first decoder layer at ~1/53 of the
    # magnitude the model expects.
    embed_scale: float

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

        # Gemma 4 nests RoPE per layer-type. Step 23 wires the
        # full_attention values too; pre-step-23 only the sliding
        # value was consumed.
        rope = text_cfg.get("rope_parameters", {})
        sliding_rope = rope.get("sliding_attention", {})
        rope_theta_sliding = float(sliding_rope.get("rope_theta", 10000.0))
        full_rope = rope.get("full_attention", {})
        # `partial_rotary_factor` is only present for full_attention
        # in Gemma 4; default to 1.0 (full rotation) for layouts that
        # omit it.
        partial_factor_full = float(
            full_rope.get("partial_rotary_factor", 1.0)
        )
        rope_theta_full = float(full_rope.get("rope_theta", rope_theta_sliding))

        # Full-attention shape extras. Default to the sliding values
        # when the model doesn't differentiate (most non-Gemma models),
        # so checkpoints with a homogeneous attention shape continue
        # to work without touching these knobs.
        head_dim_full = int(
            text_cfg.get("global_head_dim", text_cfg["head_dim"])
        )
        num_kv_heads_full = int(
            text_cfg.get(
                "num_global_key_value_heads",
                text_cfg["num_key_value_heads"],
            )
        )
        attention_k_eq_v = bool(text_cfg.get("attention_k_eq_v", False))

        # Sliding-window scope for sliding_attention layers (step 24C).
        # Cast to int when present; missing → None ⇒ no window enforced.
        sliding_window = text_cfg.get("sliding_window")
        sliding_window = int(sliding_window) if sliding_window is not None else None

        # Step 26A: FFN activation choice. Gemma 4 ships
        # `hidden_activation: gelu_pytorch_tanh`; older configs use
        # `hidden_act` (also a string). Try both keys, default to silu
        # so synthetic / Mixtral / Qwen layouts that omit the field
        # keep the SiLU activation they were using before step 26.
        hidden_activation = text_cfg.get(
            "hidden_activation", text_cfg.get("hidden_act", "silu"),
        )

        # Step 26C: model type drives architecture-specific behaviors
        # not captured by a single flag. `gemma` family scales
        # embeddings by sqrt(hidden_size) at model entry; others
        # don't. Default to "" so synthetic / Mixtral configs that
        # omit the field stay scale-1.0.
        model_type = str(text_cfg.get("model_type", ""))
        embed_scale = (
            float(text_cfg["hidden_size"]) ** 0.5
            if model_type.startswith("gemma")
            else 1.0
        )

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
            head_dim_full=head_dim_full,
            num_kv_heads_full=num_kv_heads_full,
            attention_k_eq_v=attention_k_eq_v,
            partial_rotary_factor_full=partial_factor_full,
            rope_theta_full=rope_theta_full,
            sliding_window=sliding_window,
            hidden_activation=str(hidden_activation),
            model_type=model_type,
            embed_scale=embed_scale,
        )

    @property
    def num_sliding_layers(self) -> int:
        """Count of leading layers using sliding_attention.

        Used historically (pre-step-23) as the cap for `num_layers`
        in `from_pretrained` because the runtime only supported
        sliding-attention layers. As of step 23 the runtime also
        supports `full_attention` layers, so the relevant cap is
        `num_hidden_layers`. This property still describes the
        contiguous prefix of sliding layers; on Gemma 4 it returns 5.
        """
        count = 0
        for kind in self.layer_types:
            if kind == "sliding_attention":
                count += 1
            else:
                break
        return count

    def is_full_attention(self, layer_id: int) -> bool:
        """Whether `layer_id` uses full_attention (vs sliding_attention).

        Out-of-range layer ids return False — defensive default for
        configs that don't include `layer_types` (every layer is
        treated as sliding-style). Real Gemma 4 always provides the
        list, so the fallback is a safety net for synthetic configs.
        """
        if 0 <= layer_id < len(self.layer_types):
            return self.layer_types[layer_id] == "full_attention"
        return False

    def head_dim_for_layer(self, layer_id: int) -> int:
        """Per-layer head_dim. Full layers use `head_dim_full`."""
        return self.head_dim_full if self.is_full_attention(layer_id) else self.head_dim

    def num_kv_heads_for_layer(self, layer_id: int) -> int:
        """Per-layer KV head count. Full layers may shrink to `num_kv_heads_full`."""
        return (
            self.num_kv_heads_full
            if self.is_full_attention(layer_id)
            else self.num_kv_heads
        )

    def rope_theta_for_layer(self, layer_id: int) -> float:
        """Per-layer RoPE base. Full layers use `rope_theta_full`."""
        return (
            self.rope_theta_full
            if self.is_full_attention(layer_id)
            else self.rope_theta_sliding
        )

    def rope_type_for_layer(self, layer_id: int) -> str:
        """Per-layer RoPE flavor.

        Step 26F: Gemma 4 full_attention layers use ProportionalRoPE
        (`rope_type: "proportional"` in `text_config.rope_parameters
        ["full_attention"]`); sliding layers use standard
        (`rope_type: "default"`). The choice is keyed purely on
        `is_full_attention(layer_id)` — the config's own `rope_type`
        field is not read — so a model with no `full_attention` layers
        (or an empty `layer_types`) gets "default" everywhere, while any
        `full_attention` layer gets "proportional".
        """
        return "proportional" if self.is_full_attention(layer_id) else "default"

    def rotary_dim_for_layer(self, layer_id: int) -> int:
        """How many of head_dim's dims are rotated by RoPE for `layer_id`.

        Sliding layers always rotate the full head_dim (factor=1.0).
        Full layers use `partial_rotary_factor_full` (Gemma 4: 0.25
        of 512 = 128). Result is always even — RoPE rotates pairs.
        """
        head_dim = self.head_dim_for_layer(layer_id)
        if self.is_full_attention(layer_id):
            rotary_dim = int(head_dim * self.partial_rotary_factor_full)
            # RoPE pairs adjacent halves; force even.
            if rotary_dim % 2 != 0:
                rotary_dim -= 1
            return rotary_dim
        return head_dim

    def attention_k_eq_v_for_layer(self, layer_id: int) -> bool:
        """Whether `layer_id` ties V to K (no v_proj on disk)."""
        return self.attention_k_eq_v and self.is_full_attention(layer_id)

    def sliding_window_for_layer(self, layer_id: int) -> int | None:
        """Attention scope for `layer_id`, or None for no limit.

        Sliding-attention layers cap attention to `sliding_window`
        tokens (1024 on Gemma 4); full_attention layers attend
        globally and return None. None is also returned for any
        layer when `sliding_window` is missing from text_config —
        layouts without the field get the pre-step-24C "attend to
        the whole cache" behavior.
        """
        if self.sliding_window is None:
            return None
        if self.is_full_attention(layer_id):
            return None
        return self.sliding_window
