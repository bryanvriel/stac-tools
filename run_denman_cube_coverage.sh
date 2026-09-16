#!/bin/bash
set -euo pipefail

# The currently cataloged Denman cubes end in December 2021. Adjust these
# dates when ITS_LIVE publishes newer cube assets.
exec conda run --no-capture-output -n stac python -u itslive_cube_coverage.py \
    --bbox-lonlat 97.8127 -68.2938 101.97906 -65.427 \
    --start 2019-01-01 \
    --end 2021-12-31 \
    --mission SENTINEL-1 \
    --max-pair-days 12 \
    --multi-cube \
    --fraction-denominator scene-footprint \
    --cache-observations \
    --scene-footprint-buffer 500 \
    --workers "${ITSLIVE_WORKERS:-4}" \
    --spatial-block-size "${ITSLIVE_BLOCK_SIZE:-100}" \
    --resume \
    --outdir denman_s1_large_cached
