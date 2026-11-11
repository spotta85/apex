"""Encode a prompt into token ids for `apex generate`, and decode ids back to text.

Tokenization stays in Python for now (the engine has no tokenizer). The encoding is the one
dump_reference.py uses: the chat template rendered to text (it emits <bos> itself and, with
thinking off, an empty thought channel), then encoded without extra special tokens.

    uv run chat_ids.py encode "What is the capital of France?"          # prints 2,105,...
    uv run chat_ids.py encode "..." -o /tmp/ids.txt                       # or write a file
    ../build/apex generate <packed_dir> --ids /tmp/ids.txt --max-tokens 64
    uv run chat_ids.py decode 818,5279,...                                # ids from "ids:"
    uv run chat_ids.py decode --file out_ids.txt
"""

import argparse
import re
import sys
from pathlib import Path

CKPT = Path("/Users/spotta/Desktop/models/gemma4/gemma-4-26b-a4b-it-nvfp4")


def load_tokenizer(ckpt: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(ckpt))


def encode(tok, prompt: str, raw: bool) -> list[int]:
    """Token ids for one user turn, with <bos> exactly once (same as dump_reference.encode)."""
    if raw:
        return [tok.bos_token_id] + tok.encode(prompt, add_special_tokens=False)
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tok.encode(text, add_special_tokens=False)


def parse_ids(text: str) -> list[int]:
    """Ids separated by commas and/or whitespace; tolerates the engine's 'ids:' prefix."""
    text = text.replace("ids:", " ")
    return [int(t) for t in re.split(r"[,\s]+", text.strip()) if t]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--ckpt", default=str(CKPT), help="checkpoint with tokenizer.json")
    sub = parser.add_subparsers(dest="cmd", required=True)

    enc = sub.add_parser("encode", help="prompt -> comma separated ids")
    enc.add_argument("prompt")
    enc.add_argument("--raw", action="store_true", help="skip the chat template")
    enc.add_argument("-o", "--output", help="write the ids to this file instead of stdout")

    dec = sub.add_parser("decode", help="ids -> text")
    dec.add_argument("ids", nargs="?", help="comma separated ids")
    dec.add_argument("--file", help="read the ids from a file")
    dec.add_argument("--keep-special", action="store_true",
                     help="show special tokens such as <end_of_turn>")

    args = parser.parse_args()
    tok = load_tokenizer(Path(args.ckpt))

    if args.cmd == "encode":
        line = ",".join(str(i) for i in encode(tok, args.prompt, args.raw))
        if args.output:
            Path(args.output).write_text(line + "\n")
            print(f"wrote {len(line.split(','))} ids to {args.output}", file=sys.stderr)
        else:
            print(line)
        return 0

    text = Path(args.file).read_text() if args.file else (args.ids or sys.stdin.read())
    ids = parse_ids(text)
    print(tok.decode(ids, skip_special_tokens=not args.keep_special))
    return 0


if __name__ == "__main__":
    sys.exit(main())
