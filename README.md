# ITS_LIVE coverage and inversion diagnostics

Tools for mapping valid ITS_LIVE image-pair velocities and evaluating how well their
observation dates constrain a temporal model at each pixel.

For details on cube selection, filtering, mosaicking, caching, file formats, and all
command-line options, see [README_extended.md](README_extended.md).

## Install

The coverage and diagnostic tools use the `stac` conda environment:

```bash
conda activate stac
python -m pip install -r requirements_itslive_cube.txt
```

Template creation also requires an environment containing `iceutils`; the examples
below call it `ice`.

## 1. Create the Denman coverage map and observation cache

```bash
./run_denman_cube_coverage.sh
```

This runs the 2019–2021 Sentinel-1 example over the larger Denman region and writes
`denman_s1_large_cached/`. Each cube is checkpointed, and `--resume` allows an
interrupted run to continue. Optional concurrency settings are:

```bash
ITSLIVE_WORKERS=2 ITSLIVE_BLOCK_SIZE=100 ./run_denman_cube_coverage.sh
```

Allow roughly 8–10 GB of free space for the complete cached example.

## 2. Create a temporal-model template

The supplied example uses a linear polynomial plus integrated B-splines and explicit
prior variances:

```bash
./run_make_template.sh
```

This creates `denman_temporal_template.h5`. If `iceutils` is not installed in the
`ice` environment, run the underlying builder with:

```bash
conda run -n ice python make_iceutils_inversion_template.py \
  denman_temporal_template.h5 \
  --start 2018-12-15 --end 2022-01-15 \
  --poly-order 1 --isplines 16 8 4 \
  --secular-prior-std 1000 --transient-prior-std 100 \
  --ridge-groups \
  --iceutils-src /path/to/iceutils \
  --overwrite
```

## 3. Run the inversion diagnostics

```bash
./run_diagnostic.sh
```

Use `ITSLIVE_DIAGNOSTIC_WORKERS` to choose the number of cubes processed in parallel:

```bash
ITSLIVE_DIAGNOSTIC_WORKERS=2 ./run_diagnostic.sh
```

The diagnostics use the cached per-observation validity and uncertainty arrays and
write restartable results under
`denman_s1_large_cached/inversion_diagnostics/`. The example uses explicit prior
variances, so its scalar ridge strength is zero. The launcher uses Conda's
`--no-capture-output` mode so per-cube and per-block progress appears immediately.
Each worker owns a separate cube cache and checkpoint directory; mosaicking remains a
single parent-process step.

## 4. Plot an existing coverage result

```bash
conda run -n stac python plot_itslive_coverage.py denman_s1_large_cached \
  --count-cmap viridis --count-clim 0 500 \
  --fraction-cmap magma --fraction-clim 0 1
```

Plot all available inversion diagnostics in a shared-scale operator comparison:

```bash
conda run -n stac python plot_itslive_inversion_diagnostics.py \
  denman_s1_large_cached
```

## Run the tests

```bash
conda run -n stac python -m unittest -v \
  test_itslive_cube_coverage.py \
  test_itslive_inversion_diagnostics.py
```

## Main programs

- `itslive_cube_coverage.py`: single- or multi-cube coverage calculation.
- `itslive_inversion_diagnostics.py`: per-pixel temporal constraint diagnostics.
- `make_iceutils_inversion_template.py`: HDF5 temporal-template builder.
- `plot_itslive_coverage.py`: standalone two-panel coverage plot.
- `plot_itslive_inversion_diagnostics.py`: multi-panel diagnostic overview.
- `compare_itslive_pair_duration.py`: mission and pair-duration comparison.
