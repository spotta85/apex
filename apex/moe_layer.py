"""One MoE layer's forward pass: route → load+run selected experts → combine.

Two pure functions, deliberately decoupled so they can be tested
without a real checkpoint and reused by both the fake-MoE pipeline
and the real-checkpoint smoke test:

  - `route(hidden, router_weight, top_k, *, hidden_scale=None,
           per_expert_scale=None, rms_eps=1e-6)`     returns (top_k_expert_ids,
                                                    softmax-and-scaled
                                                    top-k weights)
  - `run_moe_layer(hidden, expert_ids, weights,
                   expert_loader, backend)`         returns combined output

The expert_loader argument is a callable `expert_id -> ExpertWeights`
so the same `run_moe_layer` works whether you're loading from disk
or producing experts inline. For real quantized checkpoints, the
typical loader is

    lambda eid: dequantize_bundle(store.load(layer_id, eid))

Step 26D rewrote `route()`'s formula to match the Gemma 4 reference
(`mlx_lm.models.gemma4_text:Router.__call__`):

  - `hidden_scale` is no longer a multiplicative pre-scale. It is now
    the LEARNED WEIGHT of an internal RMSNorm (no `+1` Gemma-unit
    offset, weight pre-multiplied by `1/sqrt(hidden_size)`). The
    pre-step-26D semantics were a guess that produced visibly peaky
    distributions but wrong-architecture logits.
  - `per_expert_scale` is no longer applied to all 128 logits before
    top-k. It is applied to the SELECTED top-k softmax weights as a
    multiplicative scale at the gathered indices. Top-k selection
    runs on the raw post-projection logits; per-expert-scale only
    reweights the chosen winners.

Both still default to None for synthetic / Mixtral / Qwen routers
(byte-identical to pre-step-24A behavior in those layouts).
"""

import numpy as np

from apex.ops import rms_norm


def route(
    hidden_state: np.ndarray,
    router_weight: np.ndarray,
    top_k: int,
    *,
    hidden_scale: np.ndarray | None = None,
    per_expert_scale: np.ndarray | None = None,
    rms_eps: float = 1e-6,
):
    """Compute top-k expert IDs and their softmax routing weights.

    Args:
      hidden_state:    shape (batch, hidden_dim)
      router_weight:   shape (num_experts, hidden_dim)  -- HF orientation
      top_k:           number of experts to keep per token
      hidden_scale:    optional shape (hidden_dim,). When provided,
                       `hidden_state` is RMS-normalized along the last
                       axis with the effective weight
                       `hidden_scale * (1/sqrt(hidden_dim))` (no +1
                       Gemma offset; matches the reference's
                       `mx.fast.rms_norm(x, scale * root_size, eps)`).
                       Gemma 4 stores this as `router.scale`.
      per_expert_scale: optional shape (num_experts,). When provided,
                       the top-k softmax weights are multiplied by
                       `per_expert_scale[expert_ids]` element-wise
                       AFTER softmax. Gemma 4 stores this as
                       `router.per_expert_scale`.
      rms_eps:         epsilon for the internal RMSNorm. Defaults to
                       1e-6 (Gemma default); should track the model's
                       `rms_norm_eps`.

    Returns:
      expert_ids:    int array, shape (batch, top_k), per-token IDs
                     ordered from highest to lowest logit
      weights:       float32 array, shape (batch, top_k). Softmax
                     normalized over the top_k logits, then optionally
                     multiplied by per_expert_scale at the selected
                     ids. Note: when per_expert_scale is non-uniform,
                     the weights NO LONGER sum to 1 along the last
                     axis — that's the reference's intentional
                     behavior (per-expert scale acts as a learned
                     output gain).

    Synthetic routers (Mixtral / Qwen / fake) leave both scale
    arguments as None; the RMSNorm and per-expert reweight are
    skipped, behavior stays byte-identical to the pre-step-24 path.
    """
    if hidden_state.ndim != 2:
        raise ValueError(
            f"hidden_state must be 2-D (batch, hidden), got shape {hidden_state.shape}"
        )
    num_experts = router_weight.shape[0]
    hidden_dim = router_weight.shape[1]
    if top_k > num_experts:
        raise ValueError(
            f"top_k={top_k} exceeds num_experts={num_experts}"
        )
    if hidden_scale is not None:
        if hidden_scale.shape != (hidden_dim,):
            raise ValueError(
                f"hidden_scale shape {hidden_scale.shape} does not match "
                f"hidden_dim ({hidden_dim},)"
            )
        # Step 26D: router applies its own RMSNorm on hidden BEFORE
        # projection. Effective scale weight is `hidden_scale *
        # (1/sqrt(hidden_dim))` per the reference; no +1 Gemma offset
        # (the router's RMSNorm uses a plain scale, not the unit-offset
        # convention the rest of the model uses on its norms).
        effective_weight = hidden_scale.astype(np.float32) * (
            np.float32(hidden_dim) ** -0.5
        )
        hidden_state = rms_norm(
            hidden_state, effective_weight, eps=rms_eps,
            gemma_unit_offset=False,
        )
    if per_expert_scale is not None:
        if per_expert_scale.shape != (num_experts,):
            raise ValueError(
                f"per_expert_scale shape {per_expert_scale.shape} does not "
                f"match num_experts ({num_experts},)"
            )

    # Project hidden to per-expert logits. router_weight is HF-oriented
    # ((num_experts, hidden_dim)) so we multiply against its transpose.
    logits = hidden_state @ router_weight.T  # (batch, num_experts)
    # Top-k selection runs on the RAW post-projection logits — no
    # per-expert-scale tweak before top-k anymore (that was the
    # pre-step-26D formula; reference applies per_expert_scale to the
    # softmax weights at the SELECTED ids, not to all 128 logits).

    # Top-k along the experts axis, sorted descending by logit.
    # argpartition is faster than full argsort for k << num_experts;
    # we then sort just those k indices to keep the contract stable.
    top_k_unsorted = np.argpartition(-logits, top_k - 1, axis=-1)[..., :top_k]
    top_k_logits_unsorted = np.take_along_axis(logits, top_k_unsorted, axis=-1)
    sort_within = np.argsort(-top_k_logits_unsorted, axis=-1)
    expert_ids = np.take_along_axis(top_k_unsorted, sort_within, axis=-1).astype(np.int32)
    top_k_logits = np.take_along_axis(top_k_logits_unsorted, sort_within, axis=-1)

    # Softmax over the kept logits. Subtract max for numerical stability.
    shifted = top_k_logits - top_k_logits.max(axis=-1, keepdims=True)
    exps = np.exp(shifted)
    weights = (exps / exps.sum(axis=-1, keepdims=True)).astype(np.float32)

    if per_expert_scale is not None:
        # Step 26D: gather scales at selected expert ids and multiply
        # element-wise into the softmax weights. This is the reference
        # behavior — per-expert-scale is a per-expert output gain, not
        # a top-k selection bias.
        per_expert_f32 = per_expert_scale.astype(np.float32)
        weights = weights * per_expert_f32[expert_ids]

    return expert_ids, weights


def run_moe_layer(hidden_state, expert_ids, weights, expert_loader, backend):
    """Load each selected expert, run SwiGLU, combine via weighted sum.

    Args:
      hidden_state:   shape (1, hidden_dim) -- batch=1 for now
      expert_ids:     iterable of int, length top_k
      weights:        iterable of float, length top_k, or None. None
                      preserves the legacy runtime behavior: uniform
                      mean across selected experts.
      expert_loader:  callable expert_id -> ExpertWeights bundle ready
                      for backend.run_expert (already dequantized)
      backend:        anything with run_expert(hidden, bundle) -> array

    Returns:
      combined output of the same shape as backend.run_expert's return
      (typically (1, hidden_dim)). Weighted mode sums as:
        sum_i weight_i * backend.run_expert(hidden, expert_loader(id_i))

    The None/uniform branch uses `sum(outputs) / len(outputs)` exactly
    like the pre-step-12 Runtime did, keeping fake demos byte-identical.
    """
    expert_ids = list(expert_ids)
    if not expert_ids:
        raise ValueError("expert_ids must not be empty")
    if weights is None:
        weights_list = None
    else:
        weights_list = [float(w) for w in weights]
        if len(expert_ids) != len(weights_list):
            raise ValueError(
                f"expert_ids ({len(expert_ids)}) and weights ({len(weights_list)}) "
                f"must have the same length"
            )

    outputs = []
    for expert_id in expert_ids:
        bundle = expert_loader(expert_id)
        outputs.append(backend.run_expert(hidden_state, bundle))

    if weights_list is None:
        return sum(outputs) / len(outputs)

    return sum(w * out for w, out in zip(weights_list, outputs))
