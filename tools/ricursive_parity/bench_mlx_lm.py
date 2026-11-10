"""mlx-lm baseline: single-stream prefill and decode through `mlx_lm.generate.generate_step`.

bench_mlx_lm.py --model DIR --prompt prompt-N.json --tokens 128 --runs 3 --output OUT.json

Timing matches the Swift bench: prefill tok/s = prompt tokens / time to the first
generated token; decode tok/s = (tokens - 1) / time from the first to the last token.
Each run starts from a fresh cache. One untimed warm-up run precedes the timed runs.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.generate import generate_step, wired_limit
from mlx_lm.utils import load_model

import ricursive_mlx


def run_once(model, prompt, tokens, prefill_step):
    begin = time.perf_counter()
    first = last = None
    count = 0
    for token, _ in generate_step(
        mx.array(prompt), model, max_tokens=tokens, prefill_step_size=prefill_step
    ):
        now = time.perf_counter()
        first = first or now
        last = now
        count += 1
    return {
        "prefill_tokens_per_second": len(prompt) / (first - begin),
        "decode_tokens_per_second": (count - 1) / (last - first),
        "generated_tokens": count,
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--prefill-step", type=int, default=2048, help="mlx-lm's default")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    prompt = json.loads(Path(args.prompt).read_text())
    model, _ = load_model(
        Path(args.model),
        get_model_classes=lambda config: (ricursive_mlx.Model, ricursive_mlx.ModelArgs),
    )
    with wired_limit(model, [mx.default_stream(mx.default_device())]):
        run_once(model, prompt, min(args.tokens, 8), args.prefill_step)
        runs = [run_once(model, prompt, args.tokens, args.prefill_step) for _ in range(args.runs)]

    result = {
        "engine": "mlx-lm",
        "prompt_tokens": len(prompt),
        "prefill_step": args.prefill_step,
        "runs": runs,
        "median_prefill_tokens_per_second": statistics.median(
            r["prefill_tokens_per_second"] for r in runs
        ),
        "median_decode_tokens_per_second": statistics.median(
            r["decode_tokens_per_second"] for r in runs
        ),
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}))


if __name__ == "__main__":
    main()
