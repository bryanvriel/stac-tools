#!/bin/bash

conda run -n stac python plot_itslive_inversion_diagnostics.py \
  denman_s1_large_cached \
  --metrics \
    data_rank \
    effective_dof \
    log10_condition_number \
    max_prediction_std \
  --cmap log10_condition_number turbo \
  --clim max_prediction_std 0 50
