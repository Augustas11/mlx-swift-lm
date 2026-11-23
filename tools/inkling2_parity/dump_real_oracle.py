"""Greedy oracle for Inkling 2 Mini real weights, from mlx-vlm (Blaizzy/mlx-vlm@f1618d6d).

Loads the source checkpoint (thinkingmachines/Inkling-2-Mini @ b85c3cfa, with its
hf_quant_config.json) through mlx-vlm's own loader, then for each context length writes
`oracle-<length>.safetensors`:

- `prompt`: int32 [length], seeded random ids below the special-token range;
- `greedy`: int32 [steps], free-running greedy tokens;
- `logits`: float32 [steps, unpadded vocabulary], the logits each greedy token came from.

Prefill is fixed on both sides because the chunk size changes greedy output: chunks of
`--prefill-step` (2048) over every prompt token but the last, then the last token alone, then
one token per step. `Inkling2RealParityTests` (MLXLMTests) replays exactly this schedule.

Run on the Studio, in a window, with mlx-vlm at f1618d6d (`mlx` 0.32.4), from the fork root:

    python3 -I tools/inkling2_parity/dump_real_oracle.py \\
        --mlx-vlm <mlx-vlm clone> --model ~/lab-2467/models/Inkling-2-Mini \\
        --out ~/lab-2467/oracle-real --contexts 1024,8192 --steps 64

`--device gpu` (the default) runs mlx-vlm's Metal kernels, which the Swift port does not
reproduce bit for bit; `--device cpu` runs mlx-vlm's portable path, which the port follows op
by op. Never run this on the M5.
"""

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

PROMPT_SEED = 2467
# Ordinary text tokens: the o200k-style base range, below the special tokens at 200000+.
PROMPT_VOCABULARY = 199_998


def revision(path):
    if path is None:
        return None
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", type=Path, required=True, help="source checkpoint directory")
    parser.add_argument("--mlx-vlm", type=Path, help="mlx-vlm checkout at f1618d6d (else installed)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--contexts", default="1024,8192")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--prefill-step", type=int, default=2048)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args()
    if args.mlx_vlm is not None:
        sys.path.insert(0, str(args.mlx_vlm.resolve()))

    import mlx.core as mx
    from mlx_vlm.utils import load_model

    mx.set_default_device(mx.cpu if args.device == "cpu" else mx.gpu)
    model_dir = args.model.expanduser().resolve()
    if not (model_dir / "hf_quant_config.json").exists():
        sys.exit(f"{model_dir}/hf_quant_config.json is missing: mlx-vlm needs it for the expert formats")
    language = load_model(model_dir).language_model
    language.eval()
    config = json.loads((model_dir / "config.json").read_text())
    vocabulary = config["text_config"].get("unpadded_vocab_size") or config["text_config"]["vocab_size"]
    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "mlx": mx.__version__,
        "mlx_vlm": revision(args.mlx_vlm),
        "model": str(model_dir),
        "device": args.device,
        "prefill_step": args.prefill_step,
        "steps": args.steps,
        "prompt_seed": PROMPT_SEED,
        "contexts": [],
    }

    for length in (int(c) for c in args.contexts.split(",")):
        prompt = random.Random(PROMPT_SEED + length).choices(
            range(min(PROMPT_VOCABULARY, vocabulary)), k=length
        )
        cache = language.make_cache()
        begin = time.perf_counter()
        rest = prompt
        while len(rest) > 1:
            n = min(args.prefill_step, len(rest) - 1)
            # The decoder alone, as the Swift `prepare` runs it: no 201K-wide head per row.
            hidden = language(
                mx.array([rest[:n]]), cache=cache, skip_logits=True, return_hidden=True
            ).hidden_states[0]
            mx.eval(hidden)
            rest = rest[n:]
        prefill_seconds = time.perf_counter() - begin

        tokens, rows = [], []
        inputs = rest
        begin = time.perf_counter()
        for _ in range(args.steps):
            row = language(mx.array([inputs]), cache=cache).logits[0, -1].astype(mx.float32)
            token = mx.argmax(row)
            mx.eval(row, token)
            tokens.append(int(token.item()))
            rows.append(row)
            inputs = [tokens[-1]]
        decode_seconds = time.perf_counter() - begin

        path = out / f"oracle-{length}.safetensors"
        mx.save_safetensors(
            str(path),
            {
                "prompt": mx.array(prompt, dtype=mx.int32),
                "greedy": mx.array(tokens, dtype=mx.int32),
                "logits": mx.stack(rows),
            },
            metadata={"prefill_step": str(args.prefill_step), "device": args.device},
        )
        entry = {
            "length": length,
            "file": path.name,
            "greedy_head": tokens[:16],
            "prefill_seconds": round(prefill_seconds, 3),
            "decode_tokens_per_second": round(args.steps / decode_seconds, 3),
        }
        meta["contexts"].append(entry)
        print(json.dumps(entry), flush=True)
        del cache
        mx.clear_cache()

    (out / "oracle.json").write_text(json.dumps(meta, indent=1) + "\n")


if __name__ == "__main__":
    main()
