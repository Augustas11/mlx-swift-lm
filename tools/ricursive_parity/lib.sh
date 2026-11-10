# Shared steps for run_job.sh and bench_job.sh. Source after setting WORK and REPO.

TOOLS="$REPO/tools/ricursive_parity"
PY="$WORK/venv/bin/python"
# RICURSIVE_SWIFT_BUILD_SYSTEM=native forces the build system that does not package the
# Metal library, to check the metallib step on a toolchain that defaults to swiftbuild.
SWIFT_FLAGS=(-c release -Xswiftc -enable-testing)
[[ -n "${RICURSIVE_SWIFT_BUILD_SYSTEM:-}" ]] &&
    SWIFT_FLAGS+=(--build-system "$RICURSIVE_SWIFT_BUILD_SYSTEM")

log() { echo "$(date '+%Y-%m-%d %H:%M:%S %z') $*" | tee -a "$WORK/logs/job.log"; }
fail() { log "FAIL $*"; exit 1; }

# run <name> <command...>: runs once, marks done on success, stops the job on failure.
run() {
    local name="$1"; shift
    if [[ -e "$WORK/done/$name" ]]; then log "skip $name (done)"; return 0; fi
    log "start $name"
    if "$@" >>"$WORK/logs/$name.log" 2>&1; then
        touch "$WORK/done/$name"; log "done $name"
    else
        fail "$name: see $WORK/logs/$name.log"
    fi
}

# Release build of the test bundle, then mlx.metallib next to every place MLX looks for it.
# The native SwiftPM build system does not package the Metal library, and every MLX op
# then fails with "Failed to load the default metallib".
swift_build() {
    (cd "$REPO" && swift build --build-tests "${SWIFT_FLAGS[@]}") || return 1
    local bin
    bin="$(cd "$REPO" && swift build "${SWIFT_FLAGS[@]}" --show-bin-path)" || return 1
    # About 20 minutes; reuse a library already built from the same mlx-swift checkout.
    if [[ -n "${RICURSIVE_METALLIB:-}" ]]; then
        mkdir -p "$WORK/metallib" && cp "$RICURSIVE_METALLIB" "$WORK/metallib/mlx.metallib" || return 1
    elif [[ ! -s "$WORK/metallib/mlx.metallib" ]]; then
        "$TOOLS/build_metallib.sh" "$WORK/metallib" "$REPO/.build/checkouts/mlx-swift" || return 1
    fi
    cp "$WORK/metallib/mlx.metallib" "$bin/mlx.metallib" || return 1
    local bundle placed=0
    for bundle in "$bin"/*.xctest; do
        [[ -d "$bundle/Contents/MacOS" ]] || continue
        cp "$WORK/metallib/mlx.metallib" "$bundle/Contents/MacOS/mlx.metallib" || return 1
        mkdir -p "$bundle/Contents/Resources" &&
        cp "$WORK/metallib/mlx.metallib" "$bundle/Contents/Resources/mlx.metallib" || return 1
        echo "placed mlx.metallib in $bundle"
        placed=1
    done
    [[ $placed -eq 1 ]] || { echo "no .xctest bundle under $bin"; return 1; }
}

# swift_test <filter> [VAR=value...]: runs the built tests with the given environment.
swift_test() {
    local filter="$1"; shift
    (cd "$REPO" && env "$@" swift test --skip-build "${SWIFT_FLAGS[@]}" \
        --filter "$filter")
}

# download <repo> <revision> <dir>: weights, configs and tokenizer, never repository code.
download() {
    "$WORK/venv/bin/hf" download "$1" --revision "$2" --local-dir "$3" \
        --exclude '*.py' --exclude '*.bin' --exclude '*.pth' --exclude '*.gguf' \
        --exclude 'inference/*'
}
