#!/bin/bash
set -euo pipefail

exec conda run --no-capture-output -n stac python -u \
    itslive_inversion_diagnostics.py \
    denman_s1_large_cached \
    denman_temporal_template.h5 \
    --ridge 0 \
    --error-variable auto \
    --observation-cache require \
    --workers 2 \
    --resume
