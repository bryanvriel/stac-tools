#!/bin/bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./run_download.sh [coverage|download|stack|all] [--dry-run]

Steps:
  coverage   Compute/resume coverage only.
  download   Query official STAC and download matching velocity granules.
  stack      Resample downloaded granules into one iceutils NetCDF Stack.
  all        Run coverage, download granules, then build the Stack (default).

Options:
  --dry-run  Write the filtered STAC inventory without downloading granules.
  -h, --help Show this help message.

Examples:
  ./run_download.sh coverage
  ./run_download.sh download
  ./run_download.sh download --dry-run
  ./run_download.sh stack
  ./run_download.sh all
EOF
}

STEP=""
DOWNLOAD_DRY_RUN=false

for argument in "$@"; do
    case "$argument" in
        coverage|download|stack|all|both)
            if [[ -n "$STEP" ]]; then
                echo "ERROR: specify only one step: coverage, download, stack, or all" >&2
                exit 2
            fi
            STEP="$argument"
            [[ "$STEP" == "both" ]] && STEP="all"
            ;;
        --dry-run)
            DOWNLOAD_DRY_RUN=true
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $argument" >&2
            usage >&2
            exit 2
            ;;
    esac
done

STEP="${STEP:-all}"
if [[ ("$STEP" == "coverage" || "$STEP" == "stack") && "$DOWNLOAD_DRY_RUN" == true ]]; then
    echo "ERROR: --dry-run applies only to the download step" >&2
    exit 2
fi

# -----------------------------
# Common settings
# -----------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

BBOX=(97.8127 -68.2938 101.97906 -65.427)

MISSION="SENTINEL-1"
MAX_PAIR_DAYS=12

WORKERS="${ITSLIVE_WORKERS:-4}"
BLOCK_SIZE="${ITSLIVE_BLOCK_SIZE:-100}"
DOWNLOAD_WORKERS="${ITSLIVE_DOWNLOAD_WORKERS:-2}"
DOWNLOAD_RETRIES="${ITSLIVE_DOWNLOAD_RETRIES:-12}"
STACK_RESOLUTION="${ITSLIVE_STACK_RESOLUTION:-120}"

# ASF's catalog polygons need a 1 km edge tolerance for this Denman grid;
# 500 m leaves a small number of valid pixels just outside the denominator.
SCENE_FOOTPRINT_BUFFER=1000

COVERAGE_START="2020-11-30"
COVERAGE_END="2020-12-31"
OUTDIR="month_denman_s1_velocity"

# -----------------------------
# Coverage calculation
# -----------------------------

run_coverage() {
    echo "Computing coverage"
    python3 -u "$SCRIPT_DIR/itslive_cube_coverage.py" \
        --bbox-lonlat "${BBOX[@]}" \
        --start "$COVERAGE_START" \
        --end "$COVERAGE_END" \
        --mission "$MISSION" \
        --max-pair-days "$MAX_PAIR_DAYS" \
        --multi-cube \
        --fraction-denominator scene-footprint \
        --cache-observations \
        --scene-footprint-buffer "$SCENE_FOOTPRINT_BUFFER" \
        --workers "$WORKERS" \
        --spatial-block-size "$BLOCK_SIZE" \
        --resume \
        --outdir "$OUTDIR"
}

# -----------------------------
# Velocity download
# -----------------------------

run_download() {
    echo "Downloading velocity granules from the official ITS_LIVE STAC catalog"
    download_args=(
        --bbox-lonlat "${BBOX[@]}"
        --start "$COVERAGE_START"
        --end "$COVERAGE_END"
        --mission "$MISSION"
        --max-pair-days "$MAX_PAIR_DAYS"
        --outdir "$OUTDIR/stac_granules"
        --workers "$DOWNLOAD_WORKERS"
        --download-retries "$DOWNLOAD_RETRIES"
    )
    if [[ "$DOWNLOAD_DRY_RUN" == false ]]; then
        download_args+=(--download-data)
    fi
    python3 -u "$SCRIPT_DIR/itslive_stac_inventory.py" "${download_args[@]}"
}

run_stack() {
    echo "Building common-grid iceutils velocity Stack"
    conda run --no-capture-output -n ice python -u \
        "$SCRIPT_DIR/itslive_granules_to_stack.py" \
        --input-dir "$OUTDIR/stac_granules" \
        --bbox-lonlat "${BBOX[@]}" \
        --resolution "$STACK_RESOLUTION" \
        --output "$OUTDIR/velocity_stack.nc" \
        --overwrite
}

case "$STEP" in
    coverage)
        run_coverage
        ;;
    download)
        run_download
        ;;
    stack)
        run_stack
        ;;
    all)
        run_coverage
        run_download
        if [[ "$DOWNLOAD_DRY_RUN" == false ]]; then
            run_stack
        else
            echo "Skipping stack build because download is a dry run"
        fi
        ;;
esac

# end of file
