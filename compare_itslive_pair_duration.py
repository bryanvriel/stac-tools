#!/usr/bin/env python3
"""Compare ITS_LIVE missions in common, disjoint image-pair duration bins."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import itslive_cube_coverage as coverage


BINS = (
    ("le_12_days", "≤12 days", None, 12.0),
    ("gt12_le32_days", ">12–32 days", 12.0, 32.0),
    ("gt32_le96_days", ">32–96 days", 32.0, 96.0),
    ("gt96_days", ">96 days", 96.0, None),
)
MISSIONS = ("SENTINEL-1", "SENTINEL-2", "LANDSAT")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-run",
        type=Path,
        required=True,
        help="Completed coverage run whose ROI, dates, and cube will be reused.",
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def in_bin(series, lower, upper):
    mask = series.notna()
    if lower is not None:
        mask &= series > lower
    if upper is not None:
        mask &= series <= upper
    return mask


def main() -> int:
    args = parse_args()
    deps = coverage.require_dependencies()
    np, pd, xr = deps["np"], deps["pd"], deps["xr"]
    config_path = args.reference_run / "run_config.json"
    cube_path = args.reference_run / "cube_info.json"
    if not config_path.exists() or not cube_path.exists():
        raise SystemExit("Reference run must contain run_config.json and cube_info.json.")
    config = json.loads(config_path.read_text())
    cube_info = json.loads(cube_path.read_text())
    original = config["arguments"]
    roi_args = SimpleNamespace(
        bbox=original.get("bbox"),
        bbox_lonlat=original.get("bbox_lonlat"),
        epsg=original.get("epsg", 3031),
        edge_points=original.get("edge_points", 41),
    )
    roi = coverage.construct_roi(roi_args, deps)
    cube_url = cube_info["url"]

    args.outdir.mkdir(parents=True, exist_ok=True)
    top_products = [
        args.outdir / "pair_duration_comparison.csv",
        args.outdir / "pair_duration_comparison.json",
        args.outdir / "pair_duration_comparison.png",
        args.outdir / "run_config.json",
    ]
    existing = [path for path in top_products if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit("Comparison outputs exist; use --overwrite: " + ", ".join(map(str, existing)))
    if args.overwrite:
        for path in existing:
            path.unlink()

    ds = coverage.open_cube(cube_url, deps)
    schema = coverage.ITSLiveCubeSchema(ds)
    cube_crs = schema.crs(deps)
    roi_cube = roi.in_crs(cube_crs, deps)
    table = coverage.observation_table(ds, schema, deps)
    start, end_exclusive, date_meta = coverage.parse_dates(original["start"], original["end"], pd)
    table = table[(table.mid_date >= start) & (table.mid_date < end_exclusive)].copy()
    subset, roi_mask = coverage.coordinate_subset(ds, schema, roi_cube, deps)

    jobs = []
    records = []
    for mission in MISSIONS:
        same = table.same_mission.astype("boolean").fillna(False).to_numpy(dtype=bool)
        match = table.mission.map(lambda value: coverage.mission_matches(value, mission)).to_numpy(dtype=bool)
        mission_table = table[same & match].copy()
        for slug, label, lower, upper in BINS:
            selected = mission_table[in_bin(mission_table.pair_days, lower, upper)].copy()
            output = args.outdir / mission.lower().replace("-", "_") / slug
            coverage.check_output_paths(output, args.overwrite)
            selected.to_csv(output / "selected_observations.csv", index=False)
            record = {
                "mission": mission,
                "duration_bin": label,
                "duration_bin_id": slug,
                "lower_days_exclusive": lower,
                "upper_days_inclusive": upper,
                "selected_observations": int(len(selected)),
                "output_directory": str(output),
            }
            records.append(record)
            if selected.empty:
                coverage.write_json(output / "summary.json", {**record, "status": "no_matching_observations"})
                continue
            indices = selected.observation_index.to_numpy(dtype=np.int64)
            data = subset.isel({schema.obs: indices})
            if "v" in schema.velocity:
                valid = coverage.finite_valid(data[schema.velocity["v"]], deps)
                definition = f"finite {schema.velocity['v']} excluding declared fill/missing values"
            else:
                valid = coverage.finite_valid(data[schema.velocity["vx"]], deps) & coverage.finite_valid(
                    data[schema.velocity["vy"]], deps
                )
                definition = (
                    f"finite {schema.velocity['vx']} AND {schema.velocity['vy']} "
                    "excluding declared fill/missing values"
                )
            jobs.append((record, selected, output, definition, valid.sum(dim=schema.obs, dtype=np.uint32)))

    print(f"Computing {len(jobs)} nonempty mission-duration maps together...")
    from dask import compute
    from dask.diagnostics import ProgressBar

    with ProgressBar():
        calculated = compute(*[job[4] for job in jobs])

    maps = {}
    for (record, selected, output, definition, _), result in zip(jobs, calculated):
        values = np.asarray(result.values, dtype=np.uint32)
        count_values = np.where(roi_mask, values, coverage.COUNT_NODATA).astype(np.uint32)
        fraction_values = np.where(
            roi_mask, values.astype(np.float64) / len(selected), np.nan
        ).astype(np.float32)
        coords = {schema.y: subset[schema.y], schema.x: subset[schema.x]}
        count = xr.DataArray(
            count_values,
            dims=(schema.y, schema.x),
            coords=coords,
            name="valid_count",
            attrs={
                "long_name": "number of valid ITS_LIVE velocity observations",
                "validity_definition": definition,
            },
        )
        fraction = xr.DataArray(
            fraction_values,
            dims=(schema.y, schema.x),
            coords=coords,
            name="valid_fraction",
            attrs={
                "long_name": "fraction of selected observations with valid velocity",
                "units": "1",
            },
        )
        stats = coverage.summary_statistics(count, roi_mask, len(selected), deps)
        record.update(
            {
                "roi_pixel_count": stats["roi_pixel_count"],
                "count_min": stats["valid_count"]["min"],
                "count_mean": stats["valid_count"]["mean"],
                "count_median": stats["valid_count"]["median"],
                "count_max": stats["valid_count"]["max"],
                "fraction_mean": stats["valid_count"]["mean"] / len(selected),
            }
        )
        metadata = {
            "source": "NASA ITS_LIVE",
            "cube_url": cube_url,
            "cube_id": cube_info.get("id"),
            "cube_crs": cube_crs.to_string(),
            "input_roi": config["input_roi"],
            "start": original["start"],
            "end": original["end"],
            "mission_filter": record["mission"],
            "pair_duration_bin": record["duration_bin"],
            "selected_observation_count": len(selected),
            "validity": definition,
            "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        coverage.write_netcdf(count, fraction, roi_mask, schema, metadata, output, deps)
        coverage.write_geotiffs(count, fraction, schema, cube_crs, output, deps)
        coverage.write_json(output / "summary.json", {**record, **stats, "status": "complete"})
        maps[(record["mission"], record["duration_bin_id"])] = count

    frame = pd.DataFrame(records)
    frame.to_csv(args.outdir / "pair_duration_comparison.csv", index=False)
    payload = {
        "source_run": str(args.reference_run),
        "cube_url": cube_url,
        "date_filter": date_meta,
        "bins": [
            {"id": slug, "label": label, "lower_exclusive": lower, "upper_inclusive": upper}
            for slug, label, lower, upper in BINS
        ],
        "results": records,
    }
    coverage.write_json(args.outdir / "pair_duration_comparison.json", payload)
    coverage.write_json(
        args.outdir / "run_config.json",
        {
            "reference_run": str(args.reference_run),
            "cube_url": cube_url,
            "input_roi": config["input_roi"],
            "start": original["start"],
            "end": original["end"],
            "created_utc": datetime.now(timezone.utc),
        },
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    available = [da.values[roi_mask] for da in maps.values()]
    vmax = max(float(np.max(values)) for values in available) if available else 1
    fig, axes = plt.subplots(3, 4, figsize=(17, 12), sharex=True, sharey=True, constrained_layout=True)
    mesh = None
    for row, mission in enumerate(MISSIONS):
        for col, (slug, label, _, _) in enumerate(BINS):
            ax = axes[row, col]
            record = next(r for r in records if r["mission"] == mission and r["duration_bin_id"] == slug)
            da = maps.get((mission, slug))
            if da is None:
                ax.text(0.5, 0.5, "No pairs", ha="center", va="center", transform=ax.transAxes)
            else:
                values = np.where(da.values == coverage.COUNT_NODATA, np.nan, da.values)
                mesh = ax.pcolormesh(
                    da[schema.x].values / 1000,
                    da[schema.y].values / 1000,
                    values,
                    shading="auto",
                    cmap="viridis",
                    vmin=0,
                    vmax=vmax,
                )
                ax.text(
                    0.02,
                    0.03,
                    f"N={record['selected_observations']:,}\nmean={record['count_mean']:.1f}",
                    transform=ax.transAxes,
                    color="white",
                    fontsize=9,
                    bbox={"facecolor": "black", "alpha": 0.45, "pad": 2},
                )
            if row == 0:
                ax.set_title(label)
            if col == 0:
                ax.set_ylabel(f"{mission}\ny (km)")
            if row == 2:
                ax.set_xlabel("x (km)")
            ax.set_aspect("equal")
    if mesh is not None:
        fig.colorbar(mesh, ax=axes, label="Valid observation count", shrink=0.82)
    fig.suptitle(
        f"ITS_LIVE coverage by mission and pair duration\n{original['start']} to {original['end']}",
        fontsize=16,
    )
    fig.savefig(args.outdir / "pair_duration_comparison.png", dpi=180)
    plt.close(fig)
    print(f"Comparison complete: {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
