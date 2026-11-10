#!/bin/bash
# Ripple-65B parity job. Run on the machine that holds the window, from an mlx-swift-lm
# checkout of the parity branch. Every step is resumable.
#
#   tools/ricursive_parity/run_job.sh <work-dir>            full run
#   tools/ricursive_parity/run_job.sh --dry-run <work-dir>  tiny synthetic checkpoint, no download
#
# Passing means greedy-token parity at every length against BF16 weights. The 4-bit
# run afterwards is report-only: quantization alone changes greedy output.
set -uo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; shift; fi
WORK="${1:?usage: run_job.sh [--dry-run] <work-dir>}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"

REFERENCE_REPO="ricursiveAI/Ripple-65B-Preview"
REFERENCE_REV="da94cc013a8fda04a502f0a185083c7099ea06c1"
QUANT_REPO="frogggias/Ripple-65B-Preview-MLX-4bit"
QUANT_REV="4a33b33f043f41a51ae7236edd440d64cfa7cc94"
MIN_MEMORY_GB=200
MIN_DISK_GB=200
if [[ $DRY_RUN -eq 1 ]]; then DEFAULT_LENGTHS="16,32,128"; DEFAULT_STEPS=8; else DEFAULT_LENGTHS="1024,8192,32768"; DEFAULT_STEPS=64; fi
LENGTHS="${RICURSIVE_LENGTHS:-$DEFAULT_LENGTHS}"
STEPS="${RICURSIVE_STEPS:-$DEFAULT_STEPS}"
DEVICE="${RICURSIVE_DEVICE:-mps}"
DTYPE="bfloat16"

mkdir -p "$WORK/logs" "$WORK/done" || exit 2
WORK="$(cd "$WORK" && pwd)"
source "$REPO/tools/ricursive_parity/lib.sh"

preflight() {
    command -v uv >/dev/null || { echo "uv is required"; return 1; }
    command -v swift >/dev/null || { echo "swift is required"; return 1; }
    if [[ $DRY_RUN -eq 1 ]]; then return 0; fi
    local memory disk
    memory=$(( $(sysctl -n hw.memsize) / 1073741824 ))
    disk=$(( $(df -k "$WORK" | awk 'NR==2{print $4}') / 1048576 ))
    echo "memory ${memory}GB, free disk ${disk}GB"
    [[ $memory -ge $MIN_MEMORY_GB ]] || { echo "need ${MIN_MEMORY_GB}GB memory"; return 1; }
    [[ $disk -ge $MIN_DISK_GB ]] || { echo "need ${MIN_DISK_GB}GB free disk"; return 1; }
}

make_venv() {
    uv venv --python 3.12 "$WORK/venv" &&
    uv pip install --python "$PY" torch safetensors tokenizers numpy
}

install_hf() { uv pip install --python "$PY" huggingface_hub; }

dry_checkpoint() {
    mkdir -p "$WORK/models/reference"
    cp "$REPO/Tests/MLXLMTests/Resources/ricursive_tiny/config.json" \
       "$REPO/Tests/MLXLMTests/Resources/ricursive_tiny/model.safetensors" \
       "$WORK/models/reference/" &&
    "$PY" -B -c '
import sys
from tokenizers import Tokenizer, models, pre_tokenizers
t = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
t.pre_tokenizer = pre_tokenizers.Whitespace()
t.add_tokens([chr(c) for c in range(33, 127)])
t.save(sys.argv[1])' "$WORK/models/reference/tokenizer.json"
}

oracle() {
    "$PY" -B "$TOOLS/make_oracle.py" \
        --checkpoint "$WORK/models/reference" \
        --tokenizer "$WORK/models/reference/tokenizer.json" \
        --source-root "$REPO" --output "$WORK/oracle" \
        --lengths "$LENGTHS" --steps "$STEPS" --device "$DEVICE" --dtype "$DTYPE" "$@"
}

# parity <exact|report> <model-dir> <report-name>
parity() {
    swift_test RicursivePytorchParityTests \
        RICURSIVE_PARITY_MODEL="$2" RICURSIVE_PARITY_ORACLE="$WORK/oracle" \
        RICURSIVE_PARITY_MODE="$1" RICURSIVE_PARITY_REPORT="$WORK/$3.json"
}

log "repo $(git -C "$REPO" rev-parse HEAD) dry_run=$DRY_RUN lengths=$LENGTHS steps=$STEPS device=$DEVICE"
run preflight preflight
run venv make_venv
if [[ $DRY_RUN -eq 1 ]]; then
    # The fixture is float32, so the oracle must be too or the comparison is meaningless.
    DTYPE="float32"
    run dry_checkpoint dry_checkpoint
    run oracle oracle --synthetic-vocab 64
else
    run install_hf install_hf
    run download_reference download "$REFERENCE_REPO" "$REFERENCE_REV" "$WORK/models/reference"
    run download_quant download "$QUANT_REPO" "$QUANT_REV" "$WORK/models/quant"
    run oracle oracle
fi
run swift_build swift_build
run parity_exact parity exact "$WORK/models/reference" parity-exact
if [[ $DRY_RUN -eq 0 ]]; then
    run parity_quant parity report "$WORK/models/quant" parity-quant
fi
log "PASS greedy parity at $LENGTHS; reports in $WORK"
