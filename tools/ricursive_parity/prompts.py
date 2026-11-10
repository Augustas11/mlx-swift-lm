"""Reproducible prompt token ids: fixed Swift sources from this checkout, cut at a length."""

import json
import sys
from pathlib import Path

from tokenizers import Tokenizer


def prompt_tokens(tokenizer_path, source_root, length):
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    files = sorted(Path(source_root).glob("Libraries/MLXLLM/Models/*.swift"))
    if not files:
        raise FileNotFoundError(f"no Swift sources under {source_root}")
    ids, parts = [], 0
    while len(ids) < length:
        text = files[parts % len(files)].read_text()
        ids += tokenizer.encode(text, add_special_tokens=False).ids
        parts += 1
    return ids[:length]


def main():
    """prompts.py <tokenizer.json> <source-root> <output-dir> <lengths> [<vocab-modulus>]"""
    tokenizer, root, output, lengths = sys.argv[1:5]
    modulus = int(sys.argv[5]) if len(sys.argv) > 5 else None
    Path(output).mkdir(parents=True, exist_ok=True)
    for length in map(int, lengths.split(",")):
        ids = prompt_tokens(tokenizer, root, length)
        if modulus:
            ids = [i % modulus for i in ids]
        Path(output, f"prompt-{length}.json").write_text(json.dumps(ids) + "\n")


if __name__ == "__main__":
    main()
