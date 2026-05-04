"""Token sampling primitives for the generation loop.

Two functions, both NumPy:

  - sample_argmax(logits)               deterministic greedy
  - sample(logits, *, temperature, rng) temperature-scaled categorical

`sample_argmax` is what the existing demos use; it stays here so
`apex.generation` can re-export it for back-compat without keeping
the implementation split across two modules.

`sample(temperature=0.0)` is greedy (collapses to `sample_argmax`).
`temperature > 0` scales the logits by `1/temperature` and draws
from the resulting softmax. Top-k / top-p / repetition-penalty are
deferred — they compose on top of this function and aren't needed
to ship the step 20 MVP.
"""

import numpy as np

from apex.ops import softmax


def sample_argmax(logits) -> int:
    """Greedy next-token id: argmax along the last axis, batch=1.

    Lives here (not in `apex.generation`) so the sampler family stays
    in one module; `apex.generation.sample_argmax` re-exports it so
    existing imports keep working.
    """
    return int(np.argmax(logits, axis=-1).item())


def sample(logits, *, temperature: float = 0.0, rng=None) -> int:
    """Sample one token id from `logits`.

    `temperature == 0` is greedy (calls `sample_argmax`). `temperature > 0`
    scales logits by `1/temperature` and draws categorically from the
    resulting softmax. `rng` is an optional `numpy.random.Generator`;
    one is created on demand when omitted.

    `logits` is `(1, vocab_size)` (batch=1) — same shape `final_logits`
    returns. Negative temperature is rejected; `0` is the documented
    way to ask for greedy.
    """
    if temperature < 0:
        raise ValueError(
            f"temperature must be >= 0 (0 = greedy), got {temperature}"
        )
    if temperature == 0:
        return sample_argmax(logits)
    probs = softmax(np.asarray(logits) / temperature, axis=-1)
    # `softmax` returns shape (1, vocab); flatten to a 1-D distribution.
    probs = np.asarray(probs).reshape(-1)
    if rng is None:
        rng = np.random.default_rng()
    return int(rng.choice(probs.shape[-1], p=probs))
