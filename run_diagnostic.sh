#!/bin/bash
set -euo pipefail

exec conda run -n stac python -u itslive_inversion_diagnostics.py \
  denman_s1_large_cached \
  denman_temporal_template.h5 \
  --ridge 0.0 \
  --error-variable auto \
  --observation-cache require \
  --resume
