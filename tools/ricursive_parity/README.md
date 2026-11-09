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
