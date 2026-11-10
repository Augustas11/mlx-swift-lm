#!/bin/bash
# Ripple-65B serving bench: single-stream prefill and decode tok/s for the 4-bit community
# build at 1K, 8K and 32K prompt tokens, Swift `generate()` against mlx-lm's
# `generate_step`, on the same weights and prompts. Median of 3 runs after a warm-up.
# Run inside a paused window, from an mlx-swift-lm checkout of the parity branch.
# Every step is resumable.
#
#   tools/ricursive_parity/bench_job.sh <work-dir>            full run
#   tools/ricursive_parity/bench_job.sh --dry-run <work-dir>  tiny fixture, no download
#
# Set RICURSIVE_BENCH_MODEL to an existing download of the 4-bit build to skip the download.
set -uo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; shift; fi
WORK="${1:?usage: bench_job.sh [--dry-run] <work-dir>}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"

QUANT_REPO="frogggias/Ripple-65B-Preview-MLX-4bit"
QUANT_REV="4a33b33f043f41a51ae7236edd440d64cfa7cc94"
MLX_VERSION="0.32.4"
MLX_LM_VERSION="0.32.2"
MIN_MEMORY_GB=64
MIN_DISK_GB=45
if [[ $DRY_RUN -eq 1 ]]; then
    DEFAULT_LENGTHS="16,32,128"; DEFAULT_TOKENS=8
else
    DEFAULT_LENGTHS="1024,8192,32768"; DEFAULT_TOKENS=128
fi
LENGTHS="${RICURSIVE_LENGTHS:-$DEFAULT_LENGTHS}"
TOKENS="${RICURSIVE_BENCH_TOKENS:-$DEFAULT_TOKENS}"
RUNS="${RICURSIVE_BENCH_RUNS:-3}"

mkdir -p "$WORK/logs" "$WORK/done" "$WORK/results" "$WORK/prompts" || exit 2
WORK="$(cd "$WORK" && pwd)"
source "$REPO/tools/ricursive_parity/lib.sh"

if [[ $DRY_RUN -eq 1 ]]; then
    MODEL="$REPO/Tests/MLXLMTests/Resources/ricursive_tiny"
else
    MODEL="${RICURSIVE_BENCH_MODEL:-$WORK/models/quant}"
fi

preflight() {
    command -v uv >/dev/null || { echo "uv is required"; return 1; }
    command -v swift >/dev/null || { echo "swift is required"; return 1; }
    if [[ $DRY_RUN -eq 1 ]]; then return 0; fi
    local memory disk
    memory=$(( $(sysctl -n hw.memsize) / 1073741824 ))
    disk=$(( $(df -k "$WORK" | awk 'NR==2{print $4}') / 1048576 ))
    echo "memory ${memory}GB, free disk ${disk}GB"
    [[ $memory -ge $MIN_MEMORY_GB ]] || { echo "need ${MIN_MEMORY_GB}GB memory"; return 1; }
    if [[ ! -f "$MODEL/config.json" ]]; then
        [[ $disk -ge $MIN_DISK_GB ]] || { echo "need ${MIN_DISK_GB}GB free disk"; return 1; }
    fi
}

make_venv() {
    uv venv --python 3.12 "$WORK/venv" &&
    uv pip install --python "$PY" "mlx==$MLX_VERSION" "mlx-lm==$MLX_LM_VERSION" \
        tokenizers huggingface_hub
}

download_quant() {
    if [[ -f "$MODEL/config.json" ]]; then echo "using $MODEL"; return 0; fi
    download "$QUANT_REPO" "$QUANT_REV" "$MODEL"
}

dry_tokenizer() {
    "$PY" -B -c '
import sys
from tokenizers import Tokenizer, models, pre_tokenizers
t = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
t.pre_tokenizer = pre_tokenizers.Whitespace()
t.add_tokens([chr(c) for c in range(33, 127)])
t.save(sys.argv[1])' "$WORK/tokenizer.json"
}

make_prompts() {
    if [[ $DRY_RUN -eq 1 ]]; then
        "$PY" -B "$TOOLS/prompts.py" "$WORK/tokenizer.json" "$REPO" "$WORK/prompts" "$LENGTHS" 64
    else
        "$PY" -B "$TOOLS/prompts.py" "$MODEL/tokenizer.json" "$REPO" "$WORK/prompts" "$LENGTHS"
    fi
}

# The baseline model file must reproduce the fixture's oracle before it is timed.
check_baseline() {
    (cd "$TOOLS" && "$PY" -B check_mlx_port.py "$REPO/Tests/MLXLMTests/Resources/ricursive_tiny")
}

bench_swift() {
    swift_test RicursiveServingBenchTests \
        RICURSIVE_BENCH_MODEL="$MODEL" RICURSIVE_BENCH_PROMPT="$WORK/prompts/prompt-$1.json" \
        RICURSIVE_BENCH_REPORT="$WORK/results/swift-$1.json" \
        RICURSIVE_BENCH_TOKENS="$TOKENS" RICURSIVE_BENCH_RUNS="$RUNS" &&
    [[ -s "$WORK/results/swift-$1.json" ]]
}

bench_mlx_lm() {
    (cd "$TOOLS" && "$PY" -B bench_mlx_lm.py --model "$MODEL" \
        --prompt "$WORK/prompts/prompt-$1.json" --tokens "$TOKENS" --runs "$RUNS" \
        --output "$WORK/results/mlx-lm-$1.json")
}

summarize() {
    "$PY" -B - "$WORK/results" "$LENGTHS" "$WORK/bench-summary.json" <<'PYEOF'
import json, sys
from pathlib import Path
results, lengths, out = Path(sys.argv[1]), sys.argv[2].split(","), Path(sys.argv[3])
rows = []
for length in lengths:
    swift = json.loads((results / f"swift-{length}.json").read_text())
    mlx = json.loads((results / f"mlx-lm-{length}.json").read_text())
    rows.append({
        "prompt_tokens": int(length),
        "swift_decode_tps": swift["median_decode_tokens_per_second"],
        "mlx_lm_decode_tps": mlx["median_decode_tokens_per_second"],
        "decode_ratio": swift["median_decode_tokens_per_second"] / mlx["median_decode_tokens_per_second"],
        "swift_prefill_tps": swift["median_prefill_tokens_per_second"],
        "mlx_lm_prefill_tps": mlx["median_prefill_tokens_per_second"],
        "prefill_ratio": swift["median_prefill_tokens_per_second"] / mlx["median_prefill_tokens_per_second"],
        "swift_prefill_step": swift["prefill_step"],
        "mlx_lm_prefill_step": mlx["prefill_step"],
    })
out.write_text(json.dumps(rows, indent=2) + "\n")
for r in rows:
    print(f"{r['prompt_tokens']:>6}: decode swift {r['swift_decode_tps']:.1f} / mlx-lm {r['mlx_lm_decode_tps']:.1f} tok/s "
          f"({r['decode_ratio']:.2f}x); prefill swift {r['swift_prefill_tps']:.0f} / mlx-lm {r['mlx_lm_prefill_tps']:.0f} tok/s")
PYEOF
}

log "bench repo $(git -C "$REPO" rev-parse HEAD) dry_run=$DRY_RUN model=$MODEL lengths=$LENGTHS tokens=$TOKENS runs=$RUNS"
run preflight preflight
run venv make_venv
if [[ $DRY_RUN -eq 1 ]]; then
    run dry_tokenizer dry_tokenizer
else
    run download_quant download_quant
fi
run prompts make_prompts
run check_baseline check_baseline
run swift_build swift_build
for length in ${LENGTHS//,/ }; do
    run "swift_$length" bench_swift "$length"
    run "mlx_lm_$length" bench_mlx_lm "$length"
done
run summarize summarize
cat "$WORK/logs/summarize.log" | tee -a "$WORK/logs/job.log"
log "DONE bench; results in $WORK/bench-summary.json"
