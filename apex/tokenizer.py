"""Thin wrapper around HuggingFace's `tokenizers` library.

`Tokenizer.from_pretrained(path)` loads `<path>/tokenizer.json` via
the pure-Rust `tokenizers` package — no `transformers` dependency
(which would pull in torch). Intended for the step 19 demo where
APEX accepts a text prompt instead of hardcoded token ids.

Two design choices worth knowing:

  - BOS handling is APEX-side, not delegated to the underlying
    tokenizer. The Gemma 4 `tokenizer.json` post-processor doesn't
    prepend `<bos>` by default; the official transformers
    `GemmaTokenizer` does (`add_bos_token=True`). We replicate that
    by reading `bos_token_id` from `<path>/config.json` (Gemma 4
    nests it under `text_config`) and prepending it ourselves when
    callers ask for `add_special_tokens=True` (the default).

  - Decode strips special tokens by default — we want to print
    `Explain MoE simply.`, not `<bos>Explain MoE simply.<eos>`.
    Pass `skip_special_tokens=False` when you need them visible.
"""

import json
from pathlib import Path


class Tokenizer:
    """Wraps a `tokenizers.Tokenizer` with Gemma-style BOS handling.

    Construct via `Tokenizer.from_pretrained(checkpoint_dir)`; the
    underlying tokenizer is exposed as `.tokenizer` if a caller needs
    the full HF API. `bos_token_id` is read from the checkpoint's
    `config.json` (or `text_config.bos_token_id` for multimodal
    models like Gemma 4) and is `None` when no config is present.
    """

    def __init__(self, tokenizer, *, bos_token_id=None):
        self.tokenizer = tokenizer
        self.bos_token_id = bos_token_id

    @classmethod
    def from_pretrained(cls, path) -> "Tokenizer":
        """Load `<path>/tokenizer.json` + read BOS id from config.json.

        Raises:
            FileNotFoundError: if `tokenizer.json` is missing — the
                most common operator mistake (pointing at a path
                without tokenizer files).
            ImportError: if the `tokenizers` package isn't installed
                (`uv add tokenizers` to install).
        """
        try:
            from tokenizers import Tokenizer as _HFTokenizer
        except ImportError as e:
            raise ImportError(
                "the `tokenizers` package is required for "
                "apex.tokenizer.Tokenizer; install with `uv add tokenizers`"
            ) from e

        path = Path(path)
        tokenizer_path = path / "tokenizer.json"
        if not tokenizer_path.is_file():
            raise FileNotFoundError(
                f"no tokenizer.json at {tokenizer_path} — pass a checkpoint "
                "directory that contains both tokenizer.json and config.json"
            )
        underlying = _HFTokenizer.from_file(str(tokenizer_path))

        # BOS id is optional — synthetic tokenizers used by tests
        # don't always have a config.json next to them.
        bos_id = None
        config_path = path / "config.json"
        if config_path.is_file():
            with open(config_path) as f:
                cfg = json.load(f)
            text_cfg = cfg.get("text_config", cfg)
            bos_id = text_cfg.get("bos_token_id")

        return cls(underlying, bos_token_id=bos_id)

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list:
        """Encode `text` to a list of token ids.

        With `add_special_tokens=True` (default), prepends
        `bos_token_id` if it isn't already the first token.
        Matches the official `transformers.GemmaTokenizer` behavior
        (which sets `add_bos_token=True`).
        """
        ids = list(self.tokenizer.encode(text).ids)
        if add_special_tokens and self.bos_token_id is not None:
            # Skip double-BOS if the underlying tokenizer's
            # post-processor already prepended one.
            if not ids or ids[0] != self.bos_token_id:
                ids = [self.bos_token_id] + ids
        return ids

    def decode(
        self, token_ids, *, skip_special_tokens: bool = True,
    ) -> str:
        """Decode `token_ids` back to text.

        Default `skip_special_tokens=True` matches the standard HF
        decode-for-display behavior (no `<bos>` / `<eos>` clutter).
        """
        return self.tokenizer.decode(
            list(token_ids), skip_special_tokens=skip_special_tokens,
        )
