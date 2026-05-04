"""Thin wrapper around HuggingFace's `tokenizers` library.

`Tokenizer.from_pretrained(path)` loads `<path>/tokenizer.json` via
the pure-Rust `tokenizers` package — no `transformers` dependency
(which would pull in torch). Intended for the step 19 demo where
APEX accepts a text prompt instead of hardcoded token ids.

Three design choices worth knowing:

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

  - Chat templates render via `jinja2`, not a re-implementation.
    Gemma 4's chat_template.jinja is 347 lines (function calling,
    multimodal, tool use). Loaded from `<path>/chat_template.jinja`
    (Gemma 4 style) or `<path>/tokenizer_config.json["chat_template"]`
    (most other HF models); rendered with the `loopcontrols`
    extension since some templates use `{% break %}`. `apply_chat_template`
    returns rendered text; `encode_chat` runs it through `encode`.
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

    def __init__(self, tokenizer, *, bos_token_id=None, chat_template=None):
        self.tokenizer = tokenizer
        self.bos_token_id = bos_token_id
        # Raw Jinja2 source for the chat template, or None if the
        # checkpoint didn't ship one. Compiled lazily on first use
        # of `apply_chat_template` so the dep is paid only when
        # actually invoked (jinja2 is a small import but still).
        self.chat_template = chat_template
        self._compiled_template = None

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

        # Chat template can live in two places. Gemma 4 ships a
        # standalone chat_template.jinja; most other HF models inline
        # it in tokenizer_config.json's "chat_template" string field.
        # Standalone file wins when both exist (it's the canonical
        # source for newer multimodal templates).
        chat_template = None
        standalone = path / "chat_template.jinja"
        if standalone.is_file():
            chat_template = standalone.read_text()
        else:
            tok_cfg_path = path / "tokenizer_config.json"
            if tok_cfg_path.is_file():
                with open(tok_cfg_path) as f:
                    tok_cfg = json.load(f)
                chat_template = tok_cfg.get("chat_template")

        return cls(underlying, bos_token_id=bos_id, chat_template=chat_template)

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

    def apply_chat_template(
        self,
        messages: list,
        *,
        add_generation_prompt: bool = True,
        **template_kwargs,
    ) -> str:
        """Render this checkpoint's chat template against `messages`.

        `messages` is a list of `{"role": ..., "content": ...}` dicts
        in the standard HF format. With `add_generation_prompt=True`
        (default), the rendered text ends at the start of the model's
        turn so the next decoded tokens are the model's reply.

        Extra `template_kwargs` are forwarded to the Jinja render
        context. Common ones for Gemma 4: `enable_thinking=True`,
        `tools=[...]`. Unknown kwargs are silently ignored by the
        template (Jinja semantics) — that's the right behavior for an
        MVP: callers can add new template variables without us
        having to enumerate them.

        Raises:
            ValueError: when the checkpoint has no chat template
                (no `chat_template.jinja` and no `tokenizer_config.json`
                with a `chat_template` field).
            ImportError: when `jinja2` isn't installed (added as a
                runtime dep in step 22B; install with `uv add jinja2`).
        """
        if self.chat_template is None:
            raise ValueError(
                "this checkpoint has no chat template — pass a "
                "directory containing chat_template.jinja or "
                "tokenizer_config.json with a chat_template field"
            )
        if self._compiled_template is None:
            try:
                import jinja2
            except ImportError as e:
                raise ImportError(
                    "the `jinja2` package is required for "
                    "apply_chat_template; install with `uv add jinja2`"
                ) from e
            # `loopcontrols` so templates that use `{% break %}`
            # / `{% continue %}` (some Llama / Qwen variants) render.
            env = jinja2.Environment(
                trim_blocks=False,
                lstrip_blocks=False,
                extensions=["jinja2.ext.loopcontrols"],
            )
            self._compiled_template = env.from_string(self.chat_template)
        return self._compiled_template.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            **template_kwargs,
        )

    def encode_chat(
        self,
        messages: list,
        *,
        add_generation_prompt: bool = True,
        add_special_tokens: bool = True,
        **template_kwargs,
    ) -> list:
        """Render `messages` via the chat template and encode to token ids.

        Convenience over `encode(apply_chat_template(messages))`. The
        rendered text typically does NOT include `<bos>`, so
        `add_special_tokens=True` (default) prepends BOS the same way
        a plain `encode("Hello")` would.
        """
        text = self.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            **template_kwargs,
        )
        return self.encode(text, add_special_tokens=add_special_tokens)
