"""Makes the greedy token and logit oracle from the PyTorch forward.

For each prompt length, writes `oracle-<length>.safetensors` holding `prompt` (int32),
`greedy` (int32) and `logits` (float32, [steps, vocab]). Prompts come from a fixed
list of source files cut at a token count, so they are reproducible.
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import torch
from safetensors.torch import save_file
from prompts import prompt_tokens
from ricursive_torch import RicursiveTorch, load_checkpoint


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--source-root", required=True, help="mlx-swift-lm checkout")
    p.add_argument("--output", required=True)
    p.add_argument("--lengths", default="1024,8192,32768")
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--prefill-step", type=int, default=512)
    p.add_argument("--device", default="mps")
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument("--synthetic-vocab", type=int, help="dry run: map prompt ids below this")
    return p.parse_args()


def main():
    args = parse()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    start = time.time()
    config, weights = load_checkpoint(args.checkpoint, dtype, device)
    print(f"loaded {len(weights)} tensors in {time.time() - start:.0f}s", flush=True)
    model = RicursiveTorch(config, weights, device)

    for length in map(int, args.lengths.split(",")):
        target = output / f"oracle-{length}.safetensors"
        if target.exists():
            print(f"{length}: already done", flush=True)
            continue
        prompt = prompt_tokens(args.tokenizer, args.source_root, length)
        if args.synthetic_vocab:
            prompt = [token % args.synthetic_vocab for token in prompt]

        greedy, rows = [], []
        begin = time.time()

        def record(index, token, logits):
            if not torch.isfinite(logits).all():
                raise RuntimeError(f"non-finite logits at length {length} step {index}")
            greedy.append(token)
            rows.append(logits.cpu())
            if index == 0:
                print(f"{length}: prefill done in {time.time() - begin:.0f}s", flush=True)

        model.prefill_and_generate(prompt, args.steps, args.prefill_step, record)
        if device.type == "mps":
            torch.mps.empty_cache()

        # Write atomically so a re-run never trusts a partial oracle.
        partial = target.with_suffix(".partial")
        save_file(
            {
                "prompt": torch.tensor(prompt, dtype=torch.int32),
                "greedy": torch.tensor(greedy, dtype=torch.int32),
                "logits": torch.stack(rows),
            },
            partial,
        )
        partial.rename(target)
        print(f"{length}: {args.steps} tokens in {time.time() - begin:.0f}s: {greedy[:8]}", flush=True)

    (output / "oracle.json").write_text(
        json.dumps(
            {
                "lengths": args.lengths,
                "steps": args.steps,
                "dtype": args.dtype,
                "device": args.device,
                "prefill_step": args.prefill_step,
            }
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
