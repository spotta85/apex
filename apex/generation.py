"""One-token generation primitives.

Three functions, all NumPy:

  - embed_token(embedding, token_id)
  - final_logits(hidden, embedding, *, softcap=None)
  - sample_argmax(logits)

`embed_token` and `final_logits` are designed for tied-embedding
models (Gemma 4 sets `tie_word_embeddings=True`): the same
fp32 (vocab_size, hidden_size) table serves as both the input
embedding and the lm_head weight.

Final logit softcapping is Gemma's "tanh squash" applied before
sampling: `softcap * tanh(logits / softcap)`. Gemma 4 uses
`softcap=30.0` from `text_config.final_logit_softcapping`. The
post-softcap logits stay bounded in (-softcap, softcap), which
prevents the soft-max from collapsing onto a few peaky values.

`sample_argmax` is the deterministic baseline; future variants
(temperature, top-k/top-p) compose naturally on top of `final_logits`.
"""

import numpy as np

from apex.ops import linear


def embed_token(embedding, token_id: int):
    """Return a single token's embedding row, shape (1, hidden_size).

    `embedding` is fp32 (vocab_size, hidden_size). The 1-row
    leading dimension matches what the rest of the decoder pipeline
    expects (single-token batch).
    """
    return embedding[token_id : token_id + 1].astype(np.float32, copy=False)


def final_logits(hidden, embedding, *, softcap=None):
    """Tied-embedding lm_head logits with optional Gemma softcapping.

    Computes `linear(hidden, embedding)` — equivalent to
    `hidden @ embedding.T` since `apex.ops.linear` applies the HF
    orientation transpose internally. Result shape is
    `(batch, vocab_size)`.

    When `softcap` is provided, applies `softcap * tanh(logits/softcap)`
    so logits stay bounded in (-softcap, softcap). Pass `None` to skip
    (smoke against models without the cap).
    """
    logits = linear(hidden, embedding)
    if softcap is not None:
        logits = softcap * np.tanh(logits / softcap)
    return logits


def sample_argmax(logits) -> int:
    """Greedy next-token id: argmax along the last axis, batch=1."""
    return int(np.argmax(logits, axis=-1).item())
