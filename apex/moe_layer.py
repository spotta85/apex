"""One MoE layer's forward pass: route → load+run selected experts → combine.

Two pure functions, deliberately decoupled so they can be tested
without a real checkpoint and reused by both the fake-MoE pipeline
and the real-checkpoint smoke test:

  - `route(hidden, router_weight, top_k)`           returns (top_k_expert_ids,
                                                    softmax-of-top-k weights)
  - `run_moe_layer(hidden, expert_ids, weights,
                   expert_loader, backend)`         returns combined output

The expert_loader argument is a callable `expert_id -> ExpertWeights`
so the same `run_moe_layer` works whether you're loading from disk
or producing experts inline. For real quantized checkpoints, the
typical loader is

    lambda eid: dequantize_bundle(store.load(layer_id, eid))

Out of scope here: caching across experts (use ExpertManager for
that), parallelizing the expert loop, model-specific router scaling
(e.g. Gemma 4's `router.scale` / `router.per_expert_scale` tensors).
This module owns the pure math + the per-expert loop.
"""

import numpy as np


def route(hidden_state: np.ndarray, router_weight: np.ndarray, top_k: int):
    """Compute top-k expert IDs and their softmax routing weights.

    Args:
      hidden_state:  shape (batch, hidden_dim)
      router_weight: shape (num_experts, hidden_dim)  -- HF orientation
      top_k:         number of experts to keep per token

    Returns:
      expert_ids:    int array, shape (batch, top_k), per-token IDs
                     ordered from highest to lowest logit
      weights:       float32 array, shape (batch, top_k), softmax
                     normalized over the top_k logits (sum to 1
                     along the last axis)

    Note: this implements the canonical "top-k softmax" routing used
    by Mixtral and most public MoE checkpoints. Some models (Gemma 4)
    layer additional scale tensors on top — those are out of scope
    for this primitive; callers can apply them before/after.
    """
    if hidden_state.ndim != 2:
        raise ValueError(
            f"hidden_state must be 2-D (batch, hidden), got shape {hidden_state.shape}"
        )
    num_experts = router_weight.shape[0]
    if top_k > num_experts:
        raise ValueError(
            f"top_k={top_k} exceeds num_experts={num_experts}"
        )

    # Project hidden to per-expert logits. router_weight is HF-oriented
    # ((num_experts, hidden_dim)) so we multiply against its transpose.
    logits = hidden_state @ router_weight.T  # (batch, num_experts)

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
    return expert_ids, weights


def run_moe_layer(hidden_state, expert_ids, weights, expert_loader, backend):
    """Load each selected expert, run SwiGLU, combine via weighted sum.

    Args:
      hidden_state:   shape (1, hidden_dim) -- batch=1 for now
      expert_ids:     iterable of int, length top_k
      weights:        iterable of float, length top_k (must sum to ~1
                      for combination to behave like a "router-weighted
                      average"; this function does not re-normalize)
      expert_loader:  callable expert_id -> ExpertWeights bundle ready
                      for backend.run_expert (already dequantized)
      backend:        anything with run_expert(hidden, bundle) -> array

    Returns:
      combined output of the same shape as backend.run_expert's return
      (typically (1, hidden_dim)), summed as
        sum_i  weight_i * backend.run_expert(hidden, expert_loader(id_i))

    The accumulator type follows whatever backend.run_expert returns,
    so this works for both numpy and MLX backends (`float * mx.array`
    is well-defined; Python's sum() folds left starting from int 0,
    which mx.array's __radd__ handles).
    """
    expert_ids = list(expert_ids)
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

    return sum(w * out for w, out in zip(weights_list, outputs))
