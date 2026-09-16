#!/usr/bin/env python3
"""Example builder for an inversion-diagnostic HDF5 template using iceutils."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path
from typing import Any


DEFAULT_ICEUTILS_SOURCE = Path("/Users/briel/src/iceutils")
GROUPS = ("secular", "seasonal", "transient", "step")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an iceutils velocity-basis template for "
            "itslive_inversion_diagnostics.py."
        )
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", required=True, help="First template date (YYYY-MM-DD).")
    parser.add_argument("--end", required=True, help="Last template date (YYYY-MM-DD).")
    parser.add_argument(
        "--step-days",
        type=float,
        default=1.0,
        help="Template sampling interval in days (default: 1).",
    )
    parser.add_argument("--poly-order", type=int, default=2, help="Maximum polynomial order (default: 2).")
    parser.add_argument("--min-poly-order", type=int, default=0, help="Minimum polynomial order (default: 0).")
    parser.add_argument(
        "--periods",
        type=float,
        nargs="*",
        default=[],
        metavar="YEARS",
        help="Sinusoidal periods in years, e.g. --periods 1 0.5.",
    )
    parser.add_argument(
        "--bsplines",
        type=int,
        nargs="*",
        default=[],
        metavar="COUNT",
        help="Numbers of B-splines in one or more sets, e.g. --bsplines 16 8.",
    )
    parser.add_argument(
        "--isplines",
        type=int,
        nargs="*",
        default=[],
        metavar="COUNT",
        help="Numbers of integrated B-splines, e.g. --isplines 16 8 4.",
    )
    parser.add_argument(
        "--seasonal-bspline-separation",
        type=float,
        help="Optional separation in years for the iceutils seasonal B-spline set.",
    )
    parser.add_argument(
        "--ridge-groups",
        nargs="*",
        choices=GROUPS,
        default=["seasonal", "transient"],
        help="Coefficient groups assigned ridge weight 1 (default: seasonal transient).",
    )
    for group in GROUPS:
        parser.add_argument(
            f"--{group}-prior-std",
            type=float,
            default=math.inf,
            help=f"Prior standard deviation for {group} coefficients (default: inf).",
        )
    parser.add_argument(
        "--iceutils-src",
        type=Path,
        default=DEFAULT_ICEUTILS_SOURCE,
        help=f"iceutils source checkout (default: {DEFAULT_ICEUTILS_SOURCE}).",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        args.start_date = dt.datetime.fromisoformat(args.start)
        args.end_date = dt.datetime.fromisoformat(args.end)
    except ValueError:
        parser.error("--start and --end must be ISO calendar dates")
    if args.end_date <= args.start_date:
        parser.error("--end must be later than --start")
    if args.step_days <= 0 or not math.isfinite(args.step_days):
        parser.error("--step-days must be finite and positive")
    if args.poly_order < 0 or args.min_poly_order < 0 or args.min_poly_order > args.poly_order:
        parser.error("polynomial orders require 0 <= min <= max")
    if any(value < 2 for value in args.bsplines + args.isplines):
        parser.error("each B-spline/I-spline set must contain at least two functions")
    if args.seasonal_bspline_separation is not None and args.seasonal_bspline_separation <= 0:
        parser.error("--seasonal-bspline-separation must be positive")
    for group in GROUPS:
        value = getattr(args, f"{group}_prior_std")
        if value <= 0 or math.isnan(value):
            parser.error(f"--{group}-prior-std must be positive or inf")
    if args.output.exists() and not args.overwrite:
        parser.error(f"{args.output} exists; use --overwrite to replace it")
    return args


def import_iceutils(source: Path) -> Any:
    source = source.expanduser().resolve()
    if source.exists() and str(source) not in sys.path:
        sys.path.insert(0, str(source))
    try:
        import iceutils as ice
    except ImportError as exc:
        raise RuntimeError(
            f"Could not import iceutils from {source}: {exc}. Run this builder in "
            "the conda environment used for iceutils or pass --iceutils-src."
        ) from exc
    if getattr(ice, "tseries", None) is None:
        raise RuntimeError(
            "iceutils.tseries is unavailable; its optional cvxopt, scikit-learn, "
            "and pint dependencies must be installed."
        )
    return ice


def template_dates(start: dt.datetime, end: dt.datetime, step_days: float) -> list[dt.datetime]:
    step = dt.timedelta(days=step_days)
    dates = []
    current = start
    while current <= end:
        dates.append(current)
        current += step
    if dates[-1] != end:
        dates.append(end)
    return dates


def coefficient_groups(model: Any) -> dict[str, list[int]]:
    return {
        group: [int(index) for index in getattr(model, f"i{group}")]
        for group in GROUPS
    }


def coefficient_names(model: Any) -> list[str]:
    names = []
    for index, basis in enumerate(model.collection.data):
        names.append(f"{index:03d}_{basis.fnname}_{basis!r}")
    return names


def build_template(args: argparse.Namespace) -> dict[str, Any]:
    import h5py
    import numpy as np
    import matplotlib.pyplot as plt

    ice = import_iceutils(args.iceutils_src)
    dates = template_dates(args.start_date, args.end_date, args.step_days)

    # This is the main section to edit when a custom iceutils temporal model is
    # desired. Model.G contains velocity basis functions evaluated at `dates`.
    model = ice.tseries.build_temporal_model(
        np.asarray(dates),
        poly=args.poly_order,
        min_poly=args.min_poly_order,
        periods=args.periods,
        bsplines=args.bsplines,
        isplines=args.isplines,
        seasonal_bspline_sep=args.seasonal_bspline_separation,
    )

    groups = coefficient_groups(model)
    ncoef = model.G.shape[1]
    prior_variance = np.full(ncoef, np.inf, dtype=np.float64)
    ridge_weights = np.zeros(ncoef, dtype=np.float64)
    for group, indices in groups.items():
        prior_std = getattr(args, f"{group}_prior_std")
        if math.isfinite(prior_std):
            prior_variance[indices] = prior_std**2
        if group in args.ridge_groups:
            ridge_weights[indices] = 1.0

    fig, axes = plt.subplot_mosaic("""
        a
        a
        b
        c
        """,
        figsize=(10, 9),
        layout='constrained',
    )

    axes['a'].plot(dates, model.G)
    axes['b'].plot(prior_variance)
    axes['c'].plot(ridge_weights)

    plt.show()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(args.output, "w") as h5:
        time_dataset = h5.create_dataset("time", data=np.asarray(model.tdec, dtype=np.float64))
        matrix_dataset = h5.create_dataset(
            "template_matrix", data=np.asarray(model.G, dtype=np.float64), compression="gzip"
        )
        h5.create_dataset(
            "coefficient_names",
            data=np.asarray(coefficient_names(model), dtype=string_dtype),
        )
        h5.create_dataset("prior_variance", data=prior_variance)
        h5.create_dataset("ridge_weights", data=ridge_weights)
        time_dataset.attrs["long_name"] = "iceutils decimal-year template epochs"
        matrix_dataset.attrs["long_name"] = "velocity temporal basis functions"
        h5.attrs["time_units"] = "decimal_year_iceutils"
        h5.attrs["template_quantity"] = "velocity"
        h5.attrs["iceutils_source"] = str(args.iceutils_src.expanduser().resolve())
        h5.attrs["iceutils_model"] = repr(model.collection)
        h5.attrs["template_start"] = args.start_date.isoformat()
        h5.attrs["template_end"] = args.end_date.isoformat()
        h5.attrs["requested_step_days"] = args.step_days
        h5.attrs["coefficient_groups_json"] = json.dumps(groups, sort_keys=True)

    return {
        "output": str(args.output),
        "epochs": int(len(model.tdec)),
        "coefficients": int(ncoef),
        "groups": groups,
        "finite_prior_variances": int(np.isfinite(prior_variance).sum()),
        "ridge_weighted_coefficients": int((ridge_weights > 0).sum()),
        "time_range": [float(model.tdec[0]), float(model.tdec[-1])],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = build_template(args)
    except (ImportError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
