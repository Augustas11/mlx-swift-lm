# Ricursive parity tools

`ricursive_torch.py` is a PyTorch transcription of the forward path in the reference
`modeling_ricursive.py`, used so that no Python from a model repository is executed.
`make_tiny_fixture.py` writes the test fixture. `make_oracle.py` and `run_job.sh`
run the full-size parity check.

## Full-size job

```bash
tools/ricursive_parity/run_job.sh <work-dir>
```

Needs 200 GB of memory, 200 GB of free disk, `uv` and a Swift toolchain, and nothing
else running on the GPU. It stops at the first failed step and resumes from it when
re-run. Logs go to `<work-dir>/logs`, reports to `<work-dir>/parity-*.json`.

It downloads, excluding repository Python:

| What | Size |
|---|---|
| `ricursiveAI/Ripple-65B-Preview` BF16 weights, config, tokenizer | 129.6 GB |
| `frogggias/Ripple-65B-Preview-MLX-4bit` weights, config | 36.4 GB |
| Python packages: torch, safetensors, tokenizers, numpy, huggingface_hub | about 1 GB |

The bar is greedy-token equality at 1024, 8192 and 32768 prompt tokens, 64 generated
tokens each, with Swift on the unquantized BF16 weights. Logit differences and KL are
reported. The 4-bit run is report-only, because quantization changes the greedy output.

`--dry-run` exercises every step on the tiny fixture checkpoint without downloading.

## Serving bench

```bash
tools/ricursive_parity/bench_job.sh <work-dir>
```

Single-stream prefill and decode tok/s on the 4-bit community build at 1024, 8192 and
32768 prompt tokens: Swift `generate()` (the serving decode loop) against mlx-lm
0.32.2's `generate_step` on the same weights and prompts. Each engine runs one
untimed warm-up and then three timed runs of 128 tokens; the median is reported.
Both use the same formulas: prefill tok/s is prompt tokens over the time to the first
token, and decode tok/s is the remaining tokens over the time from the first to the
last. Each engine keeps its default prefill chunk (2048 in both), and the
summary records both.

mlx-lm runs `ricursive_mlx.py` from this directory through `load_model(get_model_classes=...)`,
so no Python from a model repository is executed. `check_mlx_port.py` first checks it
against the fixture oracle. Set `RICURSIVE_BENCH_MODEL` to an existing download of the
4-bit build to skip its 36.4 GB download. Results: `<work-dir>/bench-summary.json`.

## Metal library

`swift_build` in `lib.sh` builds `mlx.metallib` with `build_metallib.sh` (about 20 minutes)
and copies it beside the test binaries, because a native SwiftPM build does not package
it. Set `RICURSIVE_METALLIB` to a library already built from the same mlx-swift checkout
to skip the compile. `RICURSIVE_SWIFT_BUILD_SYSTEM=native` forces that build system.
