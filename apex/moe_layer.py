"""One MoE layer's forward pass: route → load+run selected experts → combine.

Two pure functions, deliberately decoupled so they can be tested
without a real checkpoint and reused by both the fake-MoE pipeline
and the real-checkpoint smoke test:

  - `route(hidden, router_weight, top_k, *, hidden_scale=None,
           per_expert_scale=None)`                  returns (top_k_expert_ids,
                                                    softmax-of-top-k weights)
  - `run_moe_layer(hidden, expert_ids, weights,
                   expert_loader, backend)`         returns combined output

The expert_loader argument is a callable `expert_id -> ExpertWeights`
so the same `run_moe_layer` works whether you're loading from disk
or producing experts inline. For real quantized checkpoints, the
typical loader is

    lambda eid: dequantize_bundle(store.load(layer_id, eid))

Step 24A added `hidden_scale` (per-feature scale on hidden, matches
Gemma 4's `router.scale` shape (hidden_size,)) and `per_expert_scale`
(per-expert scale on logits, matches Gemma 4's `router.per_expert_scale`
shape (num_experts,)) — both optional, both multiplicative, both default
to None for back-compat with Mixtral / Qwen / synthetic routers.
"""

import numpy as np


def route(
    hidden_state: np.ndarray,
    router_weight: np.ndarray,
    top_k: int,
    *,
    hidden_scale: np.ndarray | None = None,
    per_expert_scale: np.ndarray | None = None,
):
    """Compute top-k expert IDs and their softmax routing weights.

    Args:
      hidden_state:    shape (batch, hidden_dim)
      router_weight:   shape (num_experts, hidden_dim)  -- HF orientation
      top_k:           number of experts to keep per token
      hidden_scale:    optional shape (hidden_dim,). Multiplied
                       element-wise into hidden_state BEFORE the
                       projection. Gemma 4 stores this as
                       `router.scale` and the runtime loads it as fp32.
      per_expert_scale: optional shape (num_experts,). Multiplied
                       element-wise into the projected logits BEFORE
                       top-k selection (so it can flip which experts
                       win). Gemma 4 stores this as
                       `router.per_expert_scale`.

    Returns:
      expert_ids:    int array, shape (batch, top_k), per-token IDs
                     ordered from highest to lowest logit
      weights:       float32 array, shape (batch, top_k), softmax
                     normalized over the top_k logits (sum to 1
                     along the last axis)

    Note: scaling is applied BEFORE top-k and softmax, so a
    per-expert scale of 0 fully suppresses an expert and a large
    positive scale boosts its selection probability. This matches
    the standard Gemma3-style router stabilizer pattern.

    Synthetic routers (Mixtral / Qwen / fake) leave both scale
    arguments as None; the multiplications are skipped so behavior
    stays byte-identical to the pre-step-24 path.
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
        # Broadcast (batch, hidden) * (hidden,) ⇒ (batch, hidden).
        # Cast to fp32 for the multiplication; safetensors_loader already
        # returns fp32 for bf16, but a defensive cast keeps synthetic
        # callers (which may pass float64) on the canonical fp32 path.
        hidden_state = hidden_state * hidden_scale.astype(np.float32)
    if per_expert_scale is not None:
        if per_expert_scale.shape != (num_experts,):
            raise ValueError(
                f"per_expert_scale shape {per_expert_scale.shape} does not "
                f"match num_experts ({num_experts},)"
            )

    # Project hidden to per-expert logits. router_weight is HF-oriented
    # ((num_experts, hidden_dim)) so we multiply against its transpose.
    logits = hidden_state @ router_weight.T  # (batch, num_experts)
    if per_expert_scale is not None:
        # Broadcast (batch, num_experts) * (num_experts,) — element-wise
        # per-expert weighting before top-k. Applied here (not after
        # softmax) so an expert with a near-zero scale can fall out of
        # the top-k entirely, which is the whole point of the parameter.
        logits = logits * per_expert_scale.astype(np.float32)

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
