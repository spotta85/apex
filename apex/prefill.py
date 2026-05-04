"""Process prompt tokens through a per-token step function to fill the KV cache.

`prefill(prompt_ids, *, step_fn)` runs `step_fn(token_id)` once per
prompt token and returns the prediction from the last token. The
step function does whatever per-token forward the caller wants (in
the canonical case: embed → N decoder layers → final RMSNorm →
tied lm_head + softcap → argmax) — `prefill` itself is stateless.
The KV cache lives inside `step_fn`'s closure, so it ends up populated
across the prompt without `prefill` needing to know about it.

Why a separate function rather than an inline loop in every caller:

  - The decode loop has the same structural shape (call `step_fn`,
    do something with the result), but the *meaning* is different:
    prefill is "process the user's input", decode is "generate the
    model's output". Naming them differently makes the two phases
    legible in demos and matches the HF transformers API.

  - With `step_fn` factored out, `prefill` is trivially testable
    against a mock — no real model required.

Why no batched-attention prefill yet:

  - This stays the "loop over prompt tokens one at a time" style
    that the step 18 spec calls for. A real batched prefill (one
    multi-token attention with causal mask, processing all prompt
    tokens in parallel) is a follow-on optimization. It cuts wall
    time substantially for long prompts but doesn't change semantics.
"""


def prefill(prompt_ids, *, step_fn) -> int:
    """Feed each prompt token through `step_fn`; return the last prediction.

    `step_fn(token_id) -> next_token_id` runs the full per-token
    forward. Pass a closure that keeps the model weights / KV cache
    / backend / metrics in scope; `prefill` only needs to see token
    ids.

    The returned id is conventionally the first generated token —
    feed it back into `step_fn` to start the decode loop.

    Raises `ValueError` on an empty prompt: a zero-token "prefill"
    would leave the cache empty and there'd be no first-generated
    id to return, so it's a programming error rather than a no-op.
    """
    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty")
    last_id = None
    for token_id in prompt_ids:
        last_id = step_fn(token_id)
    return last_id
