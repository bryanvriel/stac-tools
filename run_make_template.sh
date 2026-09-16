#!/bin/bash
set -euo pipefail

exec conda run -n ice python \
  make_iceutils_inversion_template.py \
  denman_temporal_template.h5 \
  --start 2018-12-15 \
  --end 2022-01-15 \
  --step-days 1 \
  --poly-order 1 \
  --isplines 16 8 4 \
  --secular-prior-std 1000 \
  --transient-prior-std 100 \
  --ridge-groups \
  --overwrite
