"""DeepSeek-V4 tiny-fixture oracles from a port of DeepSeek's reference.

`make_fixture.py` writes the weights (`config.json`, `model.safetensors`).
This script writes `oracle.json` from the TensorFold MLX port of DeepSeek's
`inference/model.py` + `kernel.py` (`deepseek_v4.py` in the
`DeepSeek-V4-Flash-0731-MXFP4-MLX` repository): fp32 compressor with carried
overlap state, exact flat top-k indexer, fp32 sparse attention and sinks,
fp32 router, fp32 expert SwiGLU and combine, fp32 head logits. The fixture's
bf16 tensors are widened to float32 (see `build`). Prefill is one call over
the whole prompt, then one token per step, on the CPU.

The reference's QAT simulations (fp8/fp4 activation quant-dequant, the
indexer's Hadamard rotation) are off by default: the Swift port leaves them
out, so the oracle compares the arithmetic both implement. `--qat` keeps them
(for measuring how far the port is from the full reference, not for the test).

    python make_oracle.py --reference /path/to/deepseek_v4.py [--qat]
"""

import argparse
import hashlib
import importlib.util
import json
import os

import mlx.core as mx

mx.set_default_device(mx.cpu)

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DECODE = 24
# (prompt length, Swift prefill step): the reference prefills in one call;
# the steps only choose how the Swift test chunks the same prompt.
CASES = [(300, 64), (300, 2048), (45, 16)]


def load_reference(path, qat):
    spec = importlib.util.spec_from_file_location("deepseek_v4_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not qat:
        module.act_quant_sim = lambda x, block_size=128: x
        module.fp4_act_quant_sim = lambda x, block_size=32: x
        module.rotate_activation = lambda x: x

    # The port's Block.hc_post differs from DeepSeek's (model.py:681-684):
    # it indexes `residual[:, :, None]`, which puts the source stream on the
    # output axis (y[j] = sum_i comb[i, j] * residual[j]), and expands in the
    # activation dtype. The reference mixes y[j] = sum_i comb[i, j] *
    # residual[i] in float32 (float32 post/comb promote the bf16 operands)
    # and casts once.
    def hc_post(self, x, residual, post, comb):
        y = post[..., None] * x[:, :, None, :].astype(mx.float32) + (
            comb[..., None] * residual[:, :, :, None, :].astype(mx.float32)
        ).sum(axis=2)
        return y.astype(x.dtype)

    module.Block.hc_post = hc_post
    return module


def build(ref):
    args = ref.ModelArgs.from_json(os.path.join(HERE, "config.json"))
    model = ref.Transformer(args)
    # Float32 activations: the bf16 tensors (embedding, norms, gates,
    # compressors, head) widen to float32, so the tiny random model's
    # routing does not flip on bf16 rounding noise and the oracle checks the
    # algorithm; the quantized projections stay as stored.
    weights = {
        k: v.astype(mx.float32) if v.dtype == mx.bfloat16 else v
        for k, v in mx.load(os.path.join(HERE, "model.safetensors")).items()
    }
    expected = {
        k for k in ref.expected_param_shapes(model) if not k.startswith("mtp.")
    }
    missing = expected - set(weights)
    assert not missing, f"fixture lacks reference tensors: {sorted(missing)[:8]}"
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())
    return model


def run_case(model, prompt):
    logits = model(mx.array([prompt]), start_pos=0)
    greedy, logprobs = [], []
    pos = len(prompt)
    for _ in range(DECODE):
        row = logits[0, -1].astype(mx.float32)
        lp = row - mx.logsumexp(row, axis=-1, keepdims=True)
        token = int(mx.argmax(lp))
        greedy.append(token)
        logprobs.append(np.array(lp).tolist())
        logits = model(mx.array([[token]]), start_pos=pos)
        pos += 1
    return greedy, logprobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, help="path to deepseek_v4.py")
    parser.add_argument("--qat", action="store_true", help="keep the QAT simulations")
    parser.add_argument("--out", default=os.path.join(HERE, "oracle.json"))
    args = parser.parse_args()

    with open(args.reference, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    ref = load_reference(args.reference, args.qat)
    previous = json.load(open(os.path.join(HERE, "oracle.json")))
    prompts = {}
    for case in previous["cases"]:
        prompts.setdefault(len(case["prompt"]), case["prompt"])

    cases = []
    for length, step in CASES:
        # A fresh model per case: the reference keeps its caches on the modules.
        greedy, logprobs = run_case(build(ref), prompts[length])
        cases.append({
            "indexer": "flat",
            "index_block": 0,
            "index_keep": 0,
            "prefill_step": step,
            "prompt": prompts[length],
            "greedy": greedy,
            "logprobs": logprobs,
        })
        print(length, step, greedy[:12])
    with open(args.out, "w") as f:
        json.dump({
            "reference": "TensorFold deepseek_v4.py",
            "reference_sha256": digest,
            "qat_simulation": args.qat,
            "activations": "float32",
            "decode": DECODE,
            "cases": cases,
        }, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
