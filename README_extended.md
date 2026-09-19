# ITS_LIVE cube coverage

`itslive_cube_coverage.py` counts valid image-pair velocity measurements at every
ITS_LIVE Zarr-cube pixel whose center falls inside a requested ROI. It queries cube
metadata through STAC, reads only the selected observations and spatial chunks, and
does not download individual granule NetCDF files.

## Install

```bash
conda activate stac
python -m pip install -r requirements_itslive_cube.txt
```

The coverage, plotting, and diagnostic programs use the `stac` environment. The
template builder additionally needs an environment containing `iceutils`; the examples
below call that environment `ice`. If `iceutils` is not installed into that environment,
pass its source checkout with `--iceutils-src /path/to/iceutils`.

## Inspect first

```bash
python itslive_cube_coverage.py \
  --bbox-lonlat 99.7 -68.4 101.5 -67.9 \
  --start 2019-01-01 --end 2024-12-31 \
  --inspect-only --outdir inspect_denman
```

This writes `cube_schema.txt`, `cube_schema.json`, `cube_info.json`, and
`run_config.json`. A known Zarr URL can be supplied with `--cube-url` to bypass STAC.

## Produce a coverage map

```bash
python itslive_cube_coverage.py \
  --bbox 2320000 -480000 2380000 -420000 --epsg 3031 \
  --start 2019-01-01 --end 2024-12-31 \
  --mission SENTINEL-1 --max-pair-days 12 \
  --outdir denman_s1
```

Date-only end values include the entire named calendar day. Mission-specific filters
require both images to normalize to the same mission; mixed-mission pairs are listed
as `MIXED` and excluded. `--mission LANDSAT` groups all Landsat missions.

Successful nonempty runs write:

- `coverage_count.nc` containing `valid_count`, `valid_fraction`, and `roi_mask`;
- `coverage_count.tif` and `coverage_fraction.tif`;
- `coverage_map.png` unless `--no-plot` is used;
- `selected_observations.csv`, `summary.json`, `cube_info.json`, and `run_config.json`;
- schema reports used to make the resolved cube conventions auditable.

Pixels outside the exact ROI footprint are nodata. Pixels inside the ROI with no valid
velocity are genuine zeros and are included in summary denominators. If no observation
matches the filters, diagnostics and an empty observation CSV are written but coverage
rasters are intentionally omitted.

## Cube selection

In normal single-cube mode, the smallest footprint containing the complete ROI is
selected. Equal-area ambiguity or an ROI spanning cube boundaries produces a clear
error. Use `--cube-url` for an explicit single-cube override, or `--multi-cube` to
process and mosaic every intersecting cube.

## Large, multi-cube ROI

The ready-to-run Denman command is:

```bash
./run_denman_cube_coverage.sh
```

Its expanded form is:

```bash
python itslive_cube_coverage.py \
  --bbox-lonlat 97.8127 -68.2938 101.97906 -65.427 \
  --start 2019-01-01 --end 2021-12-31 \
  --mission SENTINEL-1 --max-pair-days 12 \
  --multi-cube --fraction-denominator scene-footprint \
  --cache-observations \
  --scene-footprint-buffer 500 \
  --workers 4 --spatial-block-size 100 --resume \
  --outdir denman_s1_large_cached
```

Each cube is checkpointed under `outdir/tiles/<cube-id>/`, and its spatial read blocks
are checkpointed under that tile's `work/` directory. Repeating the same command with
`--resume` reuses complete cubes and completed read blocks, so a transient remote
disconnect only repeats the current block. Use `--overwrite` instead of `--resume` to
intentionally recompute every cube. A failure is recorded in both the tile status and
`multi_cube_manifest.json`. The launcher accepts `ITSLIVE_WORKERS` and
`ITSLIVE_BLOCK_SIZE` environment overrides; smaller values trade speed for fewer
simultaneous remote requests.

`--cache-observations` writes a compressed HDF5 cache at
`tiles/<cube-id>/work/observation_cache.h5` while each remote block is already in
memory. It stores the selected-observation validity mask and native `v_error` values.
This increases disk use, but avoids rereading the large remote velocity arrays during
repeated inversion-diagnostic experiments. Because cache creation changes the coverage
run fingerprint, add it when starting a new output directory or use `--overwrite` for
an existing uncached run; an old completed run cannot be upgraded with `--resume` alone.
For this Denman configuration, allow roughly 3–7 GB for all compressed tile caches and
8–10 GB of free space for margin. Actual size depends mainly on the number of selected
observations and the compressibility of `v_error`.

Multi-cube output adds `selected_observation_count.tif`, which records the owning
cube's selected-pair count as a diagnostic (and is the legacy fraction denominator
when `--fraction-denominator cube` is used). The global
`selected_observations.csv` is deduplicated by granule URL and records every source
cube/index for each pair. Global count statistics use all common-grid pixel centers
inside the exact ROI.

For Sentinel-1, `--fraction-denominator scene-footprint` makes the global fraction
pixelwise: the denominator is the number of unique selected pairs whose two source
SLC footprints overlap that pixel center. Source footprints are obtained from the
official ASF Search API and cached in `sentinel1_scene_footprints.json`. The resulting
`opportunity_count.tif` is also stored in the NetCDF file. The older owning-cube
denominator remains available as `selected_observation_count.tif` for diagnostics.
Catalog polygons receive a configurable projected buffer (`--scene-footprint-buffer`,
default 500 m) to reconcile their coarse edges with valid pixels on the 120 m cube grid;
the tolerance and consistency diagnostics are recorded in `summary.json`.

Adjacent ITS_LIVE cubes have 120 m grids whose origins can differ by 40 m. The mosaic
therefore uses nearest-neighbor sampling onto an explicitly aligned 120 m output grid,
with each target cell assigned by the nominal STAC `proj:bbox` tile. The method is
recorded in NetCDF and summary metadata. Counts are never averaged or interpolated.

The example ends in 2021 because the Denman cube catalog snapshot used to develop this
workflow ended in December 2021. Inspect the current catalog before extending the dates;
a range with no matching cube observations writes diagnostics and an empty inventory,
but no global coverage rasters.

## Plot an existing result

Create a standalone two-panel count/fraction map from any completed results directory:

```bash
conda run -n stac python plot_itslive_coverage.py denman_s1_large_cached \
  --count-cmap viridis --count-clim 0 500 \
  --fraction-cmap magma --fraction-clim 0 1
```

The default output is `coverage_count_fraction.png` inside the results directory.
Use `--output`, `--title`, `--figsize`, and `--dpi` to customize it. `--cmap` or
`--clim` applies one setting to both panels; panel-specific options take precedence.

## Export selected velocity data

`itslive_cube_export.py` downloads the actual filtered velocity arrays from the cloud
cubes into a local Zarr v2 store. The safest route is to replay a completed coverage
run, which uses its saved ROI and exact cube-specific observation indices:

```bash
python itslive_cube_export.py \
  --coverage-results denman_s1_large_cached \
  --outdir denman_s1_velocity \
  --dry-run

python itslive_cube_export.py \
  --coverage-results denman_s1_large_cached \
  --outdir denman_s1_velocity \
  --workers 4
```

The dry run opens metadata, validates variables and observation identities, and prints
an uncompressed size estimate without creating output. Exports larger than 50 GiB
require `--allow-large`. If a remote read is interrupted, repeat the same command with
`--resume`; changing the ROI, observations, variables, or block sizes requires a new
output directory or `--overwrite`.

A fresh selection uses the same filters as the coverage command:

```bash
python itslive_cube_export.py \
  --bbox-lonlat 97.8127 -68.2938 101.97906 -65.427 \
  --start 2019-01-01 --end 2021-12-31 \
  --mission SENTINEL-1 --max-pair-days 12 --multi-cube \
  --outdir denman_s1_velocity
```

By default the exporter writes `vx`, `vy`, `v`, `v_error`, `vx_error`, and `vy_error`.
Repeat `--variables` to choose a different set, for example
`--variables v --variables v_error --variables interp_mask`. Output is organized as
`velocity.zarr/cubes/<cube-id>`, with each cube's native coordinates, CRS, observation
metadata, source indices, and exact `roi_mask`. Pixels outside the ROI are missing.
Open one group with:

```python
import xarray as xr

ds = xr.open_zarr(
    "denman_s1_velocity/velocity.zarr",
    group="cubes/<cube-id>",
)
```

Adjacent cube grids are intentionally not mosaicked or interpolated. The root
`manifest.json`, `export_run_config.json`, and `export_selected_observations.csv` record
provenance, selection, estimated size, completion state, and the export fingerprint.

## Assess temporal-inversion constraints

`itslive_inversion_diagnostics.py` reads a completed multi-cube result and an HDF5
temporal template, then maps posterior constraint diagnostics using the actual valid
observation dates at every pixel. A velocity template runs both physically faithful
interval averaging and midpoint sampling by default:

```bash
conda run --no-capture-output -n stac python -u itslive_inversion_diagnostics.py \
  denman_s1_large_cached denman_temporal_template.h5 \
  --ridge 0 --error-variable auto --observation-cache require \
  --workers 4 --resume
```

The HDF5 schema is:

```text
/time                 float64 [ntime]       # strictly increasing decimal years
/template_matrix      float64 [ntime,ncoef]
/coefficient_names    UTF-8   [ncoef]       # optional
/prior_variance       float64 [ncoef]       # optional; inf means unconstrained
/ridge_weights        float64 [ncoef]       # optional; default 1

attribute time_units       = "decimal_year_iceutils"
attribute template_quantity = "velocity"   # or "displacement"
```

An example builder using the local `iceutils` temporal model API is included. For a
quadratic polynomial plus 16-, 8-, and 4-function integrated B-spline sets:

For an explicit-prior model, use finite group prior standard deviations, clear the
ridge groups, and run the diagnostics with `--ridge 0`:

```bash
conda run -n ice python make_iceutils_inversion_template.py \
  denman_temporal_template.h5 \
  --start 2018-12-15 --end 2022-01-15 --step-days 1 \
  --poly-order 2 --isplines 16 8 4 \
  --secular-prior-std 1000 --transient-prior-std 100 \
  --ridge-groups
```

For ridge regularization instead, leave all prior standard deviations at their default
of infinity, select the penalized groups in the template, and choose the scalar ridge
precision when running the diagnostics:

```bash
conda run -n ice python make_iceutils_inversion_template.py \
  denman_temporal_template.h5 \
  --start 2018-12-15 --end 2022-01-15 --step-days 1 \
  --poly-order 2 --isplines 16 8 4 \
  --ridge-groups transient

conda run --no-capture-output -n stac python -u itslive_inversion_diagnostics.py \
  denman_s1_large_cached denman_temporal_template.h5 \
  --ridge 0.01 --error-variable auto --observation-cache require --resume
```

The builder stores `model.G` and `model.tdec`, labels each coefficient using its
iceutils basis representation, and translates group-specific prior standard deviations
to per-coefficient variances. Its central `build_temporal_model(...)` call is intended
to be easy to edit for more specialized iceutils collections.

For a velocity template, repeat `--observation-operator` to request
`interval-average`, `midpoint`, or both. A displacement template accepts
`endpoint-difference`. Output directories contain usable observation count, data rank,
effective degrees of freedom, log10 posterior condition number, Gaussian information
gain, and mean/maximum prediction-standard-deviation GeoTIFFs. When both velocity
operators are requested, `comparison/` contains midpoint-to-interval uncertainty
ratios. Block checkpoints allow interrupted remote reads to resume.
Compatible coverage caches are used automatically. Set `--observation-cache require`
to fail instead of falling back to remote reads, or `--observation-cache off` to ignore
the cache explicitly. `--workers N` processes independent cubes in separate processes;
the parent process mosaics their completed outputs. Start with 2–4 workers. If NumPy's
BLAS is itself multithreaded, set `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` to
avoid CPU oversubscription.

### Diagnostic metrics

Let `I = Gᵀ C_d⁻¹ G` be the data-information matrix, `P` the prior/ridge precision,
and `C_post = (I + P)⁻¹` the posterior coefficient covariance. These metrics describe
constraint geometry and assumed uncertainty; they do not measure model residuals or
guarantee that the fitted time series is accurate.

| Metric | Definition | Good and bad values |
|---|---|---|
| `usable_observation_count` | Number of selected observations with valid velocity and finite, positive uncertainty at the pixel. | Higher is generally better; zero means no data constraint. Count alone does not capture temporal distribution or measurement precision. |
| `data_rank` | Numerical rank of the normalized data-only information matrix `I`; maximum is the number of model coefficients. | Full or high rank means more independent coefficient combinations are informed by data. Low rank means temporal redundancy or missing coefficient directions; priors may still make the posterior invertible. |
| `effective_dof` | `trace(C_post I)`, the effective number of model degrees of freedom determined by the observations rather than regularization. | Higher, up to the coefficient count, means greater data control. Near zero means the solution is mostly prior/ridge controlled. Lower values are not necessarily undesirable when strong regularization is intentional. |
| `log10_condition_number` | Base-10 logarithm of the condition number of the diagonally normalized posterior precision `I + P`. | Zero is ideal; lower is better. Large values indicate poorly separated coefficient combinations and numerical sensitivity. As rough guidance, `<2` is comfortable, `2–4` deserves attention, and `>6` is very poorly conditioned. `NaN` means the posterior precision was not numerically full rank. |
| `information_gain_nats` | `0.5 [log det(I + P) − log det(P)]`, the Gaussian reduction in parameter-volume uncertainty relative to the prior. Computed only for a positive-definite prior. | Higher means the observations add more information beyond the prior; near zero means little improvement. Compare only runs using the same model and prior. `NaN` with an improper or absent prior is expected, not evidence of bad data. |
| `mean_prediction_std` | Mean over the template time grid of `sqrt(b(t)ᵀ C_post b(t))`, where `b(t)` predicts velocity from the coefficients. | Lower is better and indicates good typical-time precision. High values indicate broadly weak constraints. Interpret in the prediction's velocity units and relative to the signal scale. |
| `max_prediction_std` | Maximum of the same posterior prediction standard deviation over the template time grid. | Lower is better. High values expose the worst-constrained time, commonly near temporal edges or observation gaps; this is the conservative counterpart to the mean. |

Interpret rank, effective DOF, and information gain relative to the number of template
coefficients. Prediction uncertainties and information gain are directly comparable
between pixels or operators only when the template, error model, and regularization are
held fixed.

### Plot inversion diagnostics

Create a single overview with one row per available observation operator and one column
per diagnostic metric. Color limits are shared between operator rows and default to the
2nd–98th percentiles:

```bash
conda run -n stac python plot_itslive_inversion_diagnostics.py \
  denman_s1_large_cached
```

Select a subset and override individual metric styling as needed:

```bash
conda run -n stac python plot_itslive_inversion_diagnostics.py \
  denman_s1_large_cached \
  --metrics \
    data_rank \
    effective_dof \
    log10_condition_number \
    max_prediction_std \
  --cmap log10_condition_number turbo \
  --clim max_prediction_std 0 50
```

The default output is
`inversion_diagnostics/inversion_diagnostics_overview.png`. The positional argument may
be either the coverage-results directory or its `inversion_diagnostics/` directory.

## Run the tests

```bash
conda run -n stac python -m unittest -v \
  test_itslive_cube_coverage.py \
  test_itslive_inversion_diagnostics.py
```

## Compare pair-duration bins

Use a completed run as the source of the ROI, dates, and cube, then compute the
same four disjoint bins for Sentinel-1, Sentinel-2, and grouped Landsat:

```bash
python compare_itslive_pair_duration.py \
  --reference-run denman_s1_pilot \
  --outdir denman_pair_duration_comparison
```

The bins are `<=12`, `>12–32`, `>32–96`, and `>96` days. The comparison tool
opens the cube once and computes all nonempty maps together. It writes per-bin
NetCDF, count/fraction GeoTIFFs, observation CSVs, summaries, plus a combined
CSV, JSON, and common-color-scale PNG.
