"""MoEModel — drag-and-drop user-facing inference API.

Two methods is the whole surface:

    model = MoEModel.from_pretrained(path, max_memory_gb=8)
    text  = model.generate("Explain MoE simply.", max_tokens=64)

`from_pretrained` does all the heavy work: detect layout, read text
config, load tokenizer + embedding + final norm + per-layer attn +
norms, set up the expert cache + manager + router + backend. Eager
loading on purpose — generation should be ready as soon as the
constructor returns, not lazily-faulted on the first call.

`generate` is the canonical pattern from the step 18/19 demos:

    encode → prefill (one decode_step per prompt token) →
    decode loop (feed predicted token back, stop on EOS or max_tokens)
    → decode

`memory_report()` returns the joined `RuntimeMemoryReport` so callers
can print expert + KV memory after generation. The KV cache lives on
the model instance (kept across `generate` calls if you want to inspect
it) but is rebuilt at the start of each `generate` so independent
prompts don't share history — cleanest semantics for the MVP.

Caveats today:
  - num_layers is capped at `config.num_hidden_layers` (30 on Gemma 4)
    as of step 23. Both sliding_attention and full_attention layers
    are supported. Per-layer head_dim / num_kv_heads / RoPE / K=V-tied
    attention are wired via `config.head_dim_for_layer(L)` etc.
  - backend="numpy" is the only choice — the FakeBackend NumPy MoE.
  - Sampling is greedy by default (`temperature=0.0`); temperature > 0
    delegates to `apex.sampling.sample`.
"""

import sys
from pathlib import Path

import numpy as np

# Allow tests / users to import `MoEModel` even when `scripts/` is not on
# sys.path (the layout-detection helper lives there for now).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from apex.backends.fake_backend import FakeBackend
from apex.checkpoint_loaders import (
    load_attention_weights,
    load_dense_mlp_weights,
    load_embedding,
    load_final_norm,
    load_layer_norms,
)
from apex.config import ModelConfig
from apex.decoder import decoder_decode_step
from apex.dequant import DequantizingExpertStore
from apex.expert_cache import ExpertCache
from apex.expert_manager import ExpertManager
from apex.generation import embed_token, final_logits
from apex.kv_cache import KVCache
from apex.memory import runtime_snapshot
from apex.metrics import Metrics
from apex.moe_layer import run_moe_layer
from apex.ops import rms_norm
from apex.prefill import prefill
from apex.real_router import MultiLayerRealRouter
from apex.sampling import sample
from apex.tokenizer import Tokenizer
from apex.types import ExpertWeights

from scripts._real_moe_helpers import (
    nvfp4_matmul_warnings,
    quantized_store_for_layout,
)


_SUPPORTED_BACKENDS = ("numpy",)


class MoEModel:
    """User-facing MoE inference handle.

    Construct via `from_pretrained`; never `__init__` directly. The
    constructor takes pre-built runtime objects so tests can swap
    components if needed, but the public path is `from_pretrained`.
    """

    def __init__(
        self,
        *,
        config: ModelConfig,
        tokenizer: Tokenizer,
        embedding,
        final_norm_w,
        per_layer_attn,
        per_layer_norms,
        cache: ExpertCache,
        manager: ExpertManager,
        router: MultiLayerRealRouter,
        backend,
        metrics: Metrics,
        num_layers: int,
        per_layer_dense_mlp=None,
        verbose: bool = False,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.embedding = embedding
        self.final_norm_w = final_norm_w
        self.per_layer_attn = per_layer_attn
        self.per_layer_norms = per_layer_norms
        # Step 24B: per-layer dense MLP weights when the layout exposes
        # them (Gemma 4 yes, Mixtral / Qwen / synthetic no). Each entry
        # is `{gate_weight, up_weight, down_weight}` (fp32) or None.
        # Lives outside the LRU expert cache — these are non-evictable.
        self.per_layer_dense_mlp = (
            per_layer_dense_mlp
            if per_layer_dense_mlp is not None
            else [None] * num_layers
        )
        self.cache = cache
        self.manager = manager
        self.router = router
        self.backend = backend
        self.metrics = metrics
        self.num_layers = num_layers
        self.verbose = verbose
        # Most-recent KVCache from `generate` — surfaced via
        # `memory_report` so callers can see KV usage. None until first
        # `generate` call.
        self._last_kv: KVCache | None = None
        # Most-recent generated token id list (in order). Useful for
        # tests that need an authoritative count rather than re-encoding
        # the decoded text (BPE doesn't round-trip cleanly when the
        # model emits unusual byte sequences — e.g. one-layer output).
        self._last_generated_ids: list[int] | None = None

    @classmethod
    def from_pretrained(
        cls,
        path,
        *,
        max_memory_gb: float = 8.0,
        layout: str | None = None,
        num_layers: int | None = None,
        backend: str = "numpy",
        verbose: bool = True,
    ) -> "MoEModel":
        """Load a checkpoint into a ready-to-generate MoEModel.

        Args:
            path: directory containing config.json, tokenizer.json,
                and the sharded safetensors weights.
            max_memory_gb: expert cache budget in GB. Real Gemma 4
                weights at NVFP4_U32 dequant to ~30 MB per expert,
                so 8 GB holds ~250 experts (roughly 2 layers' worth).
            layout: optional layout-name override (e.g. "gemma-hf-moe").
                When None, auto-detects via the same logic as the demos.
            num_layers: optional cap on how many decoder layers to
                load. Defaults to `config.num_sliding_layers` (5 on
                Gemma 4); larger values are rejected since layer 5+
                isn't supported yet.
            backend: "numpy" only today; passing anything else raises
                ValueError so misconfiguration fails fast.
            verbose: print one-line load-progress traces.

        Raises:
            FileNotFoundError: missing checkpoint dir or tokenizer.json.
            ValueError: unsupported backend, num_layers > sliding cap,
                or non-positive max_memory_gb.
        """
        if backend not in _SUPPORTED_BACKENDS:
            raise ValueError(
                f"backend {backend!r} not supported; "
                f"available: {_SUPPORTED_BACKENDS}"
            )
        if max_memory_gb <= 0:
            raise ValueError(
                f"max_memory_gb must be positive, got {max_memory_gb}"
            )

        config = ModelConfig.from_path(path, layout_name=layout)

        # Step 23 lifted the cap from `num_sliding_layers` to
        # `num_hidden_layers` — both attention kinds are supported now.
        max_supported = config.num_hidden_layers
        if max_supported == 0:
            raise ValueError(
                f"checkpoint at {config.path} has no decoder layers"
            )
        if num_layers is None:
            num_layers = max_supported
        elif num_layers <= 0:
            raise ValueError(
                f"num_layers must be positive, got {num_layers}"
            )
        elif num_layers > max_supported:
            raise ValueError(
                f"num_layers={num_layers} exceeds the "
                f"{max_supported} decoder layers in this checkpoint"
            )

        if verbose:
            num_full = sum(
                1 for L in range(num_layers) if config.is_full_attention(L)
            )
            num_sliding = num_layers - num_full
            print(f"loading checkpoint: {config.path}")
            print(f"  layout: {config.layout_name}")
            print(f"  layers: {num_layers}/{config.num_hidden_layers} "
                  f"({num_sliding} sliding + {num_full} full)")
            print(f"  experts: {config.num_experts} "
                  f"top_k={config.top_k_experts}")

        tokenizer = Tokenizer.from_pretrained(config.path)

        if verbose:
            print("  loading + dequantizing embedding (a few seconds)...")
        embedding = load_embedding(config.path, config.layout)
        final_norm_w = load_final_norm(config.path, config.layout)

        per_layer_attn = []
        per_layer_norms = []
        per_layer_dense_mlp = []
        for layer_id in range(num_layers):
            kind = "full" if config.is_full_attention(layer_id) else "sliding"
            if verbose:
                print(f"  loading layer {layer_id} attention + norms ({kind})")
            per_layer_attn.append(
                load_attention_weights(
                    config.path, config.layout, layer_id,
                    # Full layers don't ship a v_proj — the tied flag
                    # tells the loader to skip it and pass v_weight=None
                    # downstream to attention_decode_step.
                    k_eq_v=config.attention_k_eq_v_for_layer(layer_id),
                )
            )
            per_layer_norms.append(
                load_layer_norms(config.path, config.layout, layer_id)
            )
            # Step 24B: load the parallel dense MLP if the layout
            # exposes one. None for layouts without (e.g. Mixtral),
            # which leaves `decoder_decode_step` in routed-only mode.
            dense = load_dense_mlp_weights(
                config.path, config.layout, layer_id,
            )
            per_layer_dense_mlp.append(dense)
            if verbose and dense is not None:
                # Heads up that this is non-evictable RAM — operators
                # need to budget for it on top of the LRU expert cache.
                bytes_dense = sum(w.nbytes for w in dense.values())
                print(f"    dense MLP loaded: "
                      f"{bytes_dense / (1024 * 1024):.1f} MB")

        metrics = Metrics()
        max_memory_mb = max_memory_gb * 1024.0
        cache = ExpertCache(
            max_memory_mb=max_memory_mb, verbose=False, metrics=metrics,
        )
        quant_store = quantized_store_for_layout(config.path, config.layout)
        store = DequantizingExpertStore(quant_store, verbose=False)
        manager = ExpertManager(store=store, cache=cache, metrics=metrics)
        backend_obj = FakeBackend()
        router = MultiLayerRealRouter(
            config.path, config.layout, top_k=config.top_k_experts,
        )

        if verbose:
            print(f"  expert cache budget: {max_memory_gb:.1f} GB")
            print("ready to generate")

        return cls(
            config=config,
            tokenizer=tokenizer,
            embedding=embedding,
            final_norm_w=final_norm_w,
            per_layer_attn=per_layer_attn,
            per_layer_norms=per_layer_norms,
            per_layer_dense_mlp=per_layer_dense_mlp,
            cache=cache,
            manager=manager,
            router=router,
            backend=backend_obj,
            metrics=metrics,
            num_layers=num_layers,
            verbose=verbose,
        )

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 64,
        temperature: float = 0.0,
        verbose: bool = False,
    ) -> str:
        """Generate text continuation from `prompt`.

        Pipeline matches the step 18/19 demos: tokenize → prefill
        (one `decode_step` per prompt token) → decode loop (feed
        predicted token back; stop on EOS or `max_tokens`) → detokenize.

        A fresh KVCache is built per call so independent prompts don't
        share history; the cache is retained on `self._last_kv` for
        `memory_report()` to inspect afterwards.

        Returns the decoded continuation only (no prompt prefix).
        """
        prompt_ids = self.tokenizer.encode(prompt)
        if not prompt_ids:
            raise ValueError("prompt encoded to zero tokens")
        return self._generate_from_ids(
            prompt_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            verbose=verbose,
            prompt_label=prompt,
        )

    def generate_chat(
        self,
        messages: list,
        *,
        max_tokens: int = 64,
        temperature: float = 0.0,
        verbose: bool = False,
        **template_kwargs,
    ) -> str:
        """Format `messages` via the chat template, then generate.

        `messages` is the standard HF list-of-dicts format
        (`{"role": ..., "content": ...}`). The chat template renders
        with `add_generation_prompt=True`, so generation continues
        directly into the model's reply.

        `template_kwargs` are forwarded to the chat template (e.g.
        Gemma 4's `enable_thinking=True`). Returns the decoded model
        reply (no prompt prefix).

        Raises ValueError if the checkpoint has no chat template.
        """
        prompt_ids = self.tokenizer.encode_chat(
            messages,
            add_generation_prompt=True,
            **template_kwargs,
        )
        if not prompt_ids:
            raise ValueError("rendered chat encoded to zero tokens")
        return self._generate_from_ids(
            prompt_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            verbose=verbose,
            prompt_label=f"<chat with {len(messages)} message(s)>",
        )

    def _generate_from_ids(
        self,
        prompt_ids: list,
        *,
        max_tokens: int,
        temperature: float,
        verbose: bool,
        prompt_label: str,
    ) -> str:
        """Shared prefill + decode-loop body for `generate` / `generate_chat`.

        `prompt_label` is just for the verbose trace — both entry
        points hand it a human-readable rendering so the trace is
        legible without the caller having to peek at the ids.
        """
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")

        kv = KVCache(num_layers=self.num_layers)
        self._last_kv = kv
        step_fn = self._make_step_fn(kv, temperature=temperature)

        if verbose:
            print(f"prompt ({len(prompt_ids)} tokens): {prompt_label!r}")
            print(f"prefilling {len(prompt_ids)} prompt tokens...")

        first_id = prefill(prompt_ids, step_fn=step_fn)
        generated = [first_id]
        self._last_generated_ids = generated

        if verbose:
            print(f"  → token 1: id {first_id} "
                  f"= {self.tokenizer.decode([first_id])!r}")

        eos = set(self.config.eos_token_ids)

        if first_id in eos:
            return self.tokenizer.decode(generated)

        last_id = first_id
        for step in range(max_tokens - 1):
            last_id = step_fn(last_id)
            generated.append(last_id)
            if verbose:
                print(f"  → token {step + 2}: id {last_id} "
                      f"= {self.tokenizer.decode([last_id])!r}")
            if last_id in eos:
                break

        return self.tokenizer.decode(generated)

    def forward(self, prompt: str) -> "np.ndarray":
        """Run prefill on `prompt` and return the next-token logits.

        Returns shape `(1, vocab_size)`. The KV cache built during
        this forward pass is left on `self._last_kv` so subsequent
        `memory_report()` reflects the prompt's KV. No sampling,
        no decode loop, no softcap removal — exactly what the model
        sees right before sampling.

        Useful for step 22C structural sanity (top-k diversity, finite
        check) and for any future reference-logits comparison against
        an HF/MLX baseline. The `temperature` knob lives on `generate`,
        not here — `forward` returns raw logits so callers compose.
        """
        prompt_ids = self.tokenizer.encode(prompt)
        if not prompt_ids:
            raise ValueError("prompt encoded to zero tokens")
        return self._forward_ids(prompt_ids)

    def _forward_ids(self, prompt_ids: list) -> "np.ndarray":
        """Implementation of forward(); shared with future variants.

        Builds a fresh KVCache, runs prefill via a logits-returning
        step closure, returns the logits at the *last* prefill step
        (the one that would feed the decode loop). Uses the same
        per-layer math as `_make_step_fn` but stops before sampling.
        """
        if not prompt_ids:
            raise ValueError("prompt_ids must not be empty (zero tokens)")
        kv = KVCache(num_layers=self.num_layers)
        self._last_kv = kv
        logits = None
        for token_id in prompt_ids:
            logits = self._compute_logits(token_id, kv)
        return logits

    def _compute_logits(self, token_id: int, kv: KVCache):
        """Per-token forward through embed → N layers → final norm → lm_head.

        Returns `(1, vocab_size)` logits with softcap applied. Internal
        helper shared by `_forward_ids` and `_make_step_fn`.

        Step 23: each layer's attention shape (head_dim, num_kv_heads,
        rope_theta, rotary_dim, K=V-tied) is per-layer, picked via
        `config.*_for_layer(L)` so sliding and full layers can be
        intermixed without a separate code path.
        """
        config = self.config
        hidden = embed_token(self.embedding, token_id)
        for layer_id in range(self.num_layers):
            attn_weights = self.per_layer_attn[layer_id]
            norms = self.per_layer_norms[layer_id]
            attn_kwargs = dict(
                q_weight=attn_weights["q_weight"],
                k_weight=attn_weights["k_weight"],
                v_weight=attn_weights["v_weight"],  # None for k_eq_v layers
                o_weight=attn_weights["o_weight"],
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_kv_heads_for_layer(layer_id),
                head_dim=config.head_dim_for_layer(layer_id),
                q_norm_weight=attn_weights["q_norm"],
                k_norm_weight=attn_weights["k_norm"],
                rms_eps=config.rms_norm_eps,
                rope_theta=config.rope_theta_for_layer(layer_id),
                rotary_dim=config.rotary_dim_for_layer(layer_id),
            )

            def moe_call(h, lid=layer_id):
                with nvfp4_matmul_warnings():
                    ids, weights = self.router.route(lid, h)
                    return run_moe_layer(
                        h, ids, weights,
                        lambda eid: self.manager.ensure_loaded(lid, eid),
                        self.backend,
                    )

            # Step 24B: parallel dense MLP. ExpertWeights bundle is the
            # natural container — same shape contract as one routed
            # expert, just with a wider intermediate dim. Reusing
            # `backend.run_expert` keeps the SwiGLU math in one place
            # so a future MLX port lights up the dense path for free.
            dense_weights = self.per_layer_dense_mlp[layer_id]
            dense_mlp_call = None
            if dense_weights is not None:
                bundle = ExpertWeights(
                    gate_proj=dense_weights["gate_weight"],
                    up_proj=dense_weights["up_weight"],
                    down_proj=dense_weights["down_weight"],
                )
                def dense_mlp_call(h, b=bundle):
                    with nvfp4_matmul_warnings():
                        return self.backend.run_expert(h, b)

            with nvfp4_matmul_warnings():
                hidden = decoder_decode_step(
                    hidden,
                    norms=norms,
                    attn_kwargs=attn_kwargs,
                    moe_call=moe_call,
                    dense_mlp_call=dense_mlp_call,
                    kv_cache=kv,
                    layer_id=layer_id,
                    rms_eps=config.rms_norm_eps,
                )
        hidden = rms_norm(
            hidden, self.final_norm_w,
            eps=config.rms_norm_eps, gemma_unit_offset=True,
        )
        with nvfp4_matmul_warnings():
            logits = final_logits(
                hidden, self.embedding,
                softcap=config.final_logit_softcapping,
            )
        return logits

    def memory_report(self):
        """Joined expert + KV memory snapshot.

        Returns a `RuntimeMemoryReport`. Before the first `generate`
        call the KV side reports zero (an empty KVCache is built so
        the snapshot still has the right shape).
        """
        kv = self._last_kv if self._last_kv is not None else KVCache(
            num_layers=self.num_layers,
        )
        return runtime_snapshot(self.cache, self.metrics, kv)

    def _make_step_fn(self, kv: KVCache, *, temperature: float):
        """Build the `step_fn(token_id) -> next_token_id` closure for prefill/decode.

        Body is `_compute_logits` + `sample` — keeps the per-layer math
        in one place so any future correctness fix lands in both
        `generate*` and `forward`.
        """
        def step(token_id: int) -> int:
            logits = self._compute_logits(token_id, kv)
            assert np.all(np.isfinite(logits)), "logits must be finite"
            return sample(logits, temperature=temperature)

        return step
