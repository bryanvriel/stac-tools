#!/usr/bin/env python3
"""Create a compact map overview of ITS_LIVE inversion diagnostics."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any


METRICS = (
    "usable_observation_count",
    "data_rank",
    "effective_dof",
    "log10_condition_number",
    "information_gain_nats",
    "mean_prediction_std",
    "max_prediction_std",
)

METRIC_LABELS = {
    "usable_observation_count": "Usable observations",
    "data_rank": "Data rank",
    "effective_dof": "Effective DOF",
    "log10_condition_number": r"$\log_{10}$ condition number",
    "information_gain_nats": "Information gain (nats)",
    "mean_prediction_std": "Mean prediction std.",
    "max_prediction_std": "Maximum prediction std.",
}

DEFAULT_CMAPS = {
    "usable_observation_count": "viridis",
    "data_rank": "viridis",
    "effective_dof": "viridis",
    "log10_condition_number": "magma",
    "information_gain_nats": "cividis",
    "mean_prediction_std": "inferno",
    "max_prediction_std": "inferno",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot one or more operator directories from an "
            "itslive_inversion_diagnostics.py result."
        )
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Diagnostic directory, or its parent coverage-results directory.",
    )
    parser.add_argument(
        "--operators",
        nargs="+",
        help="Operators to plot (default: all directories containing diagnostic rasters).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=METRICS,
        default=list(METRICS),
        help="Metrics to plot, in column order (default: all).",
    )
    parser.add_argument(
        "--cmap",
        action="append",
        nargs=2,
        metavar=("METRIC", "NAME"),
        help="Override a metric colormap; repeat as needed.",
    )
    parser.add_argument(
        "--clim",
        action="append",
        nargs=3,
        metavar=("METRIC", "MIN", "MAX"),
        help="Override shared metric color limits; repeat as needed.",
    )
    parser.add_argument(
        "--percentiles",
        type=float,
        nargs=2,
        default=(2.0, 98.0),
        metavar=("LOW", "HIGH"),
        help="Automatic robust color-limit percentiles (default: 2 98).",
    )
    parser.add_argument("--output", type=Path, help="Output PNG path.")
    parser.add_argument("--title", help="Optional figure title.")
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches (default scales with rows and columns).",
    )
    parser.add_argument("--dpi", type=int, default=180, help="PNG resolution (default: 180).")
    parser.add_argument("--show", action="store_true", help="Also display interactively.")
    args = parser.parse_args(argv)
    low, high = args.percentiles
    if not 0 <= low < high <= 100:
        parser.error("--percentiles requires 0 <= LOW < HIGH <= 100")
    if args.dpi < 1:
        parser.error("--dpi must be positive")
    if args.figsize and any(value <= 0 for value in args.figsize):
        parser.error("--figsize values must be positive")
    args.cmaps = dict(args.cmap or [])
    args.clims = {}
    for metric, lower, upper in args.clim or []:
        if metric not in METRICS:
            parser.error(f"unknown --clim metric {metric!r}")
        try:
            limits = (float(lower), float(upper))
        except ValueError:
            parser.error(f"--clim for {metric} requires numeric MIN and MAX")
        if not all(math.isfinite(value) for value in limits) or limits[0] >= limits[1]:
            parser.error(f"--clim for {metric} requires finite MIN < MAX")
        args.clims[metric] = limits
    unknown_cmaps = set(args.cmaps) - set(METRICS)
    if unknown_cmaps:
        parser.error(f"unknown --cmap metric(s): {', '.join(sorted(unknown_cmaps))}")
    return args


def diagnostic_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    nested = path / "inversion_diagnostics"
    return nested if nested.is_dir() else path


def discover_operators(root: Path, requested: list[str] | None) -> list[str]:
    available = sorted(
        directory.name
        for directory in root.iterdir()
        if directory.is_dir() and any((directory / f"{metric}.tif").is_file() for metric in METRICS)
    )
    if not available:
        raise RuntimeError(f"No diagnostic operator directories found under {root}.")
    if requested:
        missing = sorted(set(requested) - set(available))
        if missing:
            raise RuntimeError(
                f"Requested operator(s) unavailable: {', '.join(missing)}; "
                f"available: {', '.join(available)}"
            )
        return list(dict.fromkeys(requested))
    preferred = [name for name in ("interval-average", "midpoint", "endpoint-difference") if name in available]
    return preferred or available


def read_rasters(
    root: Path, operators: list[str], metrics: list[str]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    import numpy as np
    import rasterio

    arrays: dict[str, dict[str, Any]] = {}
    reference = None
    profile = None
    for operator in operators:
        arrays[operator] = {}
        for metric in metrics:
            path = root / operator / f"{metric}.tif"
            if not path.is_file():
                raise FileNotFoundError(path)
            with rasterio.open(path) as source:
                data = np.ma.masked_invalid(source.read(1, masked=True).astype(np.float64))
                grid = (source.shape, source.crs, source.transform)
                if reference is None:
                    reference = grid
                    profile = {
                        "bounds": source.bounds,
                        "crs": source.crs,
                        "shape": source.shape,
                    }
                elif grid != reference:
                    raise RuntimeError(f"Raster grid differs from the first input: {path}")
            if data.count() == 0:
                raise RuntimeError(f"Raster contains no plottable pixels: {path}")
            arrays[operator][metric] = data
    return arrays, profile


def color_limits(
    arrays: dict[str, dict[str, Any]],
    operators: list[str],
    metrics: list[str],
    overrides: dict[str, tuple[float, float]],
    percentiles: tuple[float, float],
) -> dict[str, tuple[float, float]]:
    import numpy as np

    limits = {}
    for metric in metrics:
        if metric in overrides:
            limits[metric] = overrides[metric]
            continue
        values = np.concatenate([arrays[operator][metric].compressed() for operator in operators])
        lower, upper = np.percentile(values, percentiles)
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise RuntimeError(f"Could not determine finite color limits for {metric}.")
        if lower == upper:
            pad = max(abs(float(lower)) * 0.01, 0.5)
            lower, upper = lower - pad, upper + pad
        limits[metric] = (float(lower), float(upper))
    return limits


def plot_results(args: argparse.Namespace) -> Path:
    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import FuncFormatter

    root = diagnostic_root(args.results_dir)
    operators = discover_operators(root, args.operators)
    arrays, profile = read_rasters(root, operators, args.metrics)
    limits = color_limits(arrays, operators, args.metrics, args.clims, args.percentiles)
    nrows, ncols = len(operators), len(args.metrics)
    figsize = args.figsize or (3.25 * ncols, 3.15 * nrows + 0.6)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        squeeze=False,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    bounds = profile["bounds"]
    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    formatter = FuncFormatter(lambda value, _: f"{value / 1000:g}")
    for column, metric in enumerate(args.metrics):
        cmap_name = args.cmaps.get(metric, DEFAULT_CMAPS[metric])
        try:
            cmap = matplotlib.colormaps[cmap_name].copy()
        except KeyError as exc:
            raise ValueError(f"Unknown Matplotlib colormap: {cmap_name}") from exc
        cmap.set_bad(alpha=0)
        images = []
        for row, operator in enumerate(operators):
            axis = axes[row, column]
            image = axis.imshow(
                arrays[operator][metric],
                extent=extent,
                origin="upper",
                interpolation="nearest",
                cmap=cmap,
                vmin=limits[metric][0],
                vmax=limits[metric][1],
            )
            images.append(image)
            axis.set_aspect("equal")
            axis.set_facecolor("0.88")
            axis.xaxis.set_major_formatter(formatter)
            axis.yaxis.set_major_formatter(formatter)
            if row == 0:
                axis.set_title(METRIC_LABELS[metric])
            if column == 0:
                axis.set_ylabel(f"{operator}\nNorthing (km)")
            if row == nrows - 1:
                axis.set_xlabel("Easting (km)")
        fig.colorbar(images[0], ax=axes[:, column], shrink=0.78, pad=0.015)

    crs_label = profile["crs"].to_string() if profile["crs"] else "unknown CRS"
    fig.suptitle(args.title or f"{root.parent.name} inversion diagnostics — {crs_label}")
    output = args.output or root / "inversion_diagnostics_overview.png"
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close(fig)
    return output


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        output = plot_results(args)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
