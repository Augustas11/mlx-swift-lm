"""Greedy oracle for Inkling 2 Mini real weights, from mlx-vlm (Blaizzy/mlx-vlm@f1618d6d).

Two checkpoint layouts, detected from the index (`--layout` forces one):

- source (thinkingmachines/Inkling-2-Mini @ b85c3cfa, with its hf_quant_config.json): loaded
  through mlx-vlm's own `load_model`;
- oq (mlx-community/Inkling-2-Mini-oQ4e-mtp @ 3eea0427), which mlx-vlm cannot load itself:
  `oq_adapter.from_oq` maps the oQ text tensors onto mlx-vlm's `LanguageModel` names without
  requantizing anything (affine 8-bit attention, embedding and head; NVFP4 experts with
  `global_scale / 2688` as `weight_scale_2`; affine 4-bit experts), and the model is quantized
  per the oQ `quantization` map as `load_model` would. `make_tiny_fixture.py` checks that
  mapping as the exact inverse of its own `to_oq`.

For each context length it writes `oracle-<length>.safetensors`:

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

and with `--model <oQ4e-mtp dir> --out ~/lab-2467/oracle-real-oq` for the oQ build. `oracle.json`
records the layout; `Inkling2RealParityTests` refuses an oracle of the other layout.

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

# `python3 -I` leaves the script's directory off sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from oq_adapter import from_oq, is_text_key, load_reference  # noqa: E402

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


def detect_layout(model_dir):
    """`source` for `model.llm.*` tensors, `oq` for oMLX's `model.layers.*` names."""
    keys = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    source = any(k.startswith("model.llm.") for k in keys)
    oq = any(k.startswith("model.layers.") or k == "lm_head.weight" for k in keys)
    if source == oq:
        sys.exit(f"{model_dir}: cannot tell the layout (source keys {source}, oQ keys {oq})")
    return "source" if source else "oq"


def load_oq(mx, model_dir, config):
    """mlx-vlm's `LanguageModel` on the oQ text tensors (see the module docstring)."""
    import mlx.nn as nn
    from mlx_vlm.models.inkling.inkling import _split_gate_up
    from mlx_vlm.models.inkling2.config import TextConfig
    from mlx_vlm.models.inkling2.language import LanguageModel

    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    oq = {}
    for shard in sorted(set(index.values())):
        for key, value in mx.load(str(model_dir / shard)).items():
            if is_text_key(key):
                oq[key] = value
    text = TextConfig.from_dict(config["text_config"])
    weights, quant = from_oq(mx, oq, text, config.get("quantization"), _split_gate_up)
    del oq
    return load_reference(mx, nn, LanguageModel, text, weights, quant)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", type=Path, required=True, help="source or oQ checkpoint directory")
    parser.add_argument("--layout", choices=("auto", "source", "oq"), default="auto")
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
    layout = detect_layout(model_dir)
    if args.layout != "auto" and args.layout != layout:
        sys.exit(f"{model_dir} holds the {layout} layout, not {args.layout}")
    config = json.loads((model_dir / "config.json").read_text())
    if layout == "source":
        if not (model_dir / "hf_quant_config.json").exists():
            sys.exit(f"{model_dir}/hf_quant_config.json is missing: mlx-vlm needs it for the expert formats")
        language = load_model(model_dir).language_model
    else:
        language = load_oq(mx, model_dir, config)
    language.eval()
    vocabulary = config["text_config"].get("unpadded_vocab_size") or config["text_config"]["vocab_size"]
    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "mlx": mx.__version__,
        "mlx_vlm": revision(args.mlx_vlm),
        "model": str(model_dir),
        "layout": layout,
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
            metadata={"prefill_step": str(args.prefill_step), "device": args.device, "layout": layout},
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
