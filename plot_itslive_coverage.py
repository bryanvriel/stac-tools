#!/usr/bin/env python3
"""Create a two-panel count/fraction map from an ITS_LIVE results directory."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot coverage_count.tif and coverage_fraction.tif side by side from "
            "an ITS_LIVE coverage results directory."
        )
    )
    parser.add_argument("results_dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PNG (default: RESULTS_DIR/coverage_count_fraction.png).",
    )
    parser.add_argument(
        "--cmap",
        help="Colormap for both panels unless overridden by a panel-specific option.",
    )
    parser.add_argument("--count-cmap", help="Count-panel colormap (default: viridis).")
    parser.add_argument("--fraction-cmap", help="Fraction-panel colormap (default: magma).")
    parser.add_argument(
        "--clim",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Color limits for both panels unless overridden panel by panel.",
    )
    parser.add_argument(
        "--count-clim",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Count-panel color limits (default: data range).",
    )
    parser.add_argument(
        "--fraction-clim",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Fraction-panel color limits (default: 0 1).",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(13.0, 6.0),
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches (default: 13 6).",
    )
    parser.add_argument("--dpi", type=int, default=200, help="PNG resolution (default: 200).")
    parser.add_argument("--title", help="Optional figure title (default: results directory name).")
    parser.add_argument("--show", action="store_true", help="Also display the figure interactively.")
    args = parser.parse_args(argv)
    for name in ("clim", "count_clim", "fraction_clim"):
        limits = getattr(args, name)
        if limits is not None and (
            not all(math.isfinite(value) for value in limits) or limits[0] >= limits[1]
        ):
            parser.error(f"--{name.replace('_', '-')} requires finite MIN < MAX")
    if args.dpi < 1:
        parser.error("--dpi must be positive")
    if any(value <= 0 for value in args.figsize):
        parser.error("--figsize values must be positive")
    return args


def read_matching_rasters(results_dir: Path) -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        import rasterio
    except ImportError as exc:
        raise SystemExit(
            f"Missing dependency {exc.name!r}; run this script in the stac conda environment."
        ) from exc

    count_path = results_dir / "coverage_count.tif"
    fraction_path = results_dir / "coverage_fraction.tif"
    missing = [str(path) for path in (count_path, fraction_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required raster(s): " + ", ".join(missing))

    with rasterio.open(count_path) as source:
        count = source.read(1, masked=True).astype(np.float64)
        profile = {
            "shape": source.shape,
            "crs": source.crs,
            "transform": source.transform,
            "bounds": source.bounds,
        }
    with rasterio.open(fraction_path) as source:
        fraction = source.read(1, masked=True).astype(np.float64)
        comparison = (source.shape, source.crs, source.transform)
    expected = (profile["shape"], profile["crs"], profile["transform"])
    if comparison != expected:
        raise RuntimeError("Count and fraction rasters do not have the same grid and CRS.")
    count = np.ma.masked_invalid(count)
    fraction = np.ma.masked_invalid(fraction)
    if count.count() == 0 or fraction.count() == 0:
        raise RuntimeError("One or both coverage rasters contain no plottable pixels.")
    return count, fraction, profile


def plot_results(args: argparse.Namespace) -> Path:
    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    count, fraction, profile = read_matching_rasters(args.results_dir)
    output = args.output or args.results_dir / "coverage_count_fraction.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    count_cmap = args.count_cmap or args.cmap or "viridis"
    fraction_cmap = args.fraction_cmap or args.cmap or "magma"
    count_clim = args.count_clim or args.clim
    fraction_clim = args.fraction_clim or args.clim or (0.0, 1.0)

    bounds = profile["bounds"]
    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    fig, axes = plt.subplots(1, 2, figsize=args.figsize, constrained_layout=True)
    panels = (
        (axes[0], count, "Coverage count", "Valid observations", count_cmap, count_clim),
        (axes[1], fraction, "Coverage fraction", "Valid fraction", fraction_cmap, fraction_clim),
    )
    for axis, data, panel_title, colorbar_label, cmap_name, limits in panels:
        try:
            cmap = matplotlib.colormaps[cmap_name].copy()
        except KeyError as exc:
            raise ValueError(f"Unknown Matplotlib colormap: {cmap_name}") from exc
        cmap.set_bad(alpha=0)
        image = axis.imshow(
            data,
            extent=extent,
            origin="upper",
            interpolation="nearest",
            cmap=cmap,
            vmin=None if limits is None else limits[0],
            vmax=None if limits is None else limits[1],
        )
        axis.set_title(panel_title)
        axis.set_aspect("equal")
        axis.set_facecolor("0.88")
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}"))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}"))
        axis.set_xlabel("Easting (km)")
        axis.set_ylabel("Northing (km)")
        fig.colorbar(image, ax=axis, label=colorbar_label, shrink=0.88)

    crs_label = profile["crs"].to_string() if profile["crs"] else "unknown CRS"
    fig.suptitle(args.title or f"{args.results_dir.name} — {crs_label}")
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close(fig)
    return output


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output = plot_results(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
