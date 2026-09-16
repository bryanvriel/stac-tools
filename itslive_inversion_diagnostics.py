#!/usr/bin/env python3
"""Map temporal-inversion constraint diagnostics for ITS_LIVE coverage results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import itslive_cube_coverage as coverage
import itslive_multi_cube as multi


FLOAT_NODATA = -9999.0
METRICS = (
    "usable_observation_count",
    "data_rank",
    "effective_dof",
    "log10_condition_number",
    "information_gain_nats",
    "mean_prediction_std",
    "max_prediction_std",
)


@dataclass
class Template:
    time: Any
    matrix: Any
    quantity: str
    coefficient_names: list[str]
    prior_variance: Any
    ridge_weights: Any
    digest: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assess per-pixel temporal inversion constraints from a completed "
            "multi-cube ITS_LIVE coverage run and an HDF5 temporal template."
        )
    )
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("template_h5", type=Path)
    parser.add_argument(
        "--observation-operator",
        action="append",
        choices=("interval-average", "midpoint", "endpoint-difference", "all"),
        help=(
            "Repeat to request multiple operators. Velocity templates default to "
            "interval-average and midpoint; displacement templates default to "
            "endpoint-difference."
        ),
    )
    parser.add_argument("--velocity-variable", default="v", help="Cube velocity variable (default: v).")
    parser.add_argument(
        "--error-variable",
        default="auto",
        help="Diagonal 1-sigma error variable, 'auto', or 'unit' (default: auto).",
    )
    parser.add_argument(
        "--error-floor",
        type=float,
        default=1.0,
        help="Minimum positive 1-sigma error in velocity units (default: 1).",
    )
    parser.add_argument(
        "--observation-cache",
        choices=("auto", "require", "off"),
        default="auto",
        help=(
            "Use a compatible cache from the coverage run automatically, require one, "
            "or always reread remote cube arrays (default: auto)."
        ),
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=0.0,
        help="Scalar ridge precision multiplier (default: 0).",
    )
    parser.add_argument(
        "--block-size", type=int, default=25, help="Spatial block side in pixels (default: 25)."
    )
    parser.add_argument(
        "--pixel-batch-size",
        type=int,
        default=256,
        help="Pixels per batched linear-algebra operation (default: 256).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Independent cube worker processes (default: 1).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        help="Output directory (default: RESULTS_DIR/inversion_diagnostics).",
    )
    parser.add_argument("--resume", action="store_true", help="Reuse compatible block checkpoints.")
    parser.add_argument("--overwrite", action="store_true", help="Discard diagnostic checkpoints and recompute.")
    args = parser.parse_args(argv)
    if args.error_floor <= 0 or not math.isfinite(args.error_floor):
        parser.error("--error-floor must be finite and positive")
    if args.ridge < 0 or not math.isfinite(args.ridge):
        parser.error("--ridge must be finite and nonnegative")
    if args.block_size < 1 or args.pixel_batch_size < 1 or args.workers < 1:
        parser.error("--block-size, --pixel-batch-size, and --workers must be positive")
    if args.resume and args.overwrite:
        parser.error("choose either --resume or --overwrite")
    args.outdir = args.outdir or args.results_dir / "inversion_diagnostics"
    return args


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def decode_strings(values: Any) -> list[str]:
    return [
        value.decode() if isinstance(value, (bytes, bytearray)) else str(value)
        for value in values
    ]


def load_template(path: Path, deps: dict[str, Any]) -> Template:
    import h5py

    np = deps["np"]
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as h5:
        if "time" not in h5 or "template_matrix" not in h5:
            raise RuntimeError("Template HDF5 requires /time and /template_matrix datasets.")
        time_grid = np.asarray(h5["time"], dtype=np.float64)
        matrix = np.asarray(h5["template_matrix"], dtype=np.float64)
        quantity = h5.attrs.get("template_quantity", h5.attrs.get("matrix_quantity"))
        if isinstance(quantity, bytes):
            quantity = quantity.decode()
        quantity = str(quantity or "").strip().lower()
        time_units = h5.attrs.get("time_units", "")
        if isinstance(time_units, bytes):
            time_units = time_units.decode()
        if str(time_units).lower() not in {"decimal_year", "decimal_year_iceutils"}:
            raise RuntimeError(
                "Template time_units must be 'decimal_year' or 'decimal_year_iceutils'."
            )
        if quantity not in {"velocity", "displacement"}:
            raise RuntimeError("template_quantity must be 'velocity' or 'displacement'.")
        if "coefficient_names" in h5:
            names = decode_strings(h5["coefficient_names"][:])
        else:
            names = [f"coefficient_{index}" for index in range(matrix.shape[1])]
        prior = (
            np.asarray(h5["prior_variance"], dtype=np.float64)
            if "prior_variance" in h5
            else np.full(matrix.shape[1], np.inf)
        )
        ridge_weights = (
            np.asarray(h5["ridge_weights"], dtype=np.float64)
            if "ridge_weights" in h5
            else np.ones(matrix.shape[1], dtype=np.float64)
        )
    if time_grid.ndim != 1 or len(time_grid) < 2:
        raise RuntimeError("/time must be a one-dimensional array with at least two entries.")
    if matrix.ndim != 2 or matrix.shape[0] != len(time_grid):
        raise RuntimeError("/template_matrix must have shape (len(time), ncoef).")
    if not np.all(np.isfinite(time_grid)) or not np.all(np.diff(time_grid) > 0):
        raise RuntimeError("/time must be finite and strictly increasing.")
    if not np.all(np.isfinite(matrix)):
        raise RuntimeError("/template_matrix contains non-finite values.")
    ncoef = matrix.shape[1]
    if len(names) != ncoef or prior.shape != (ncoef,) or ridge_weights.shape != (ncoef,):
        raise RuntimeError("Coefficient names, prior variances, and ridge weights must match ncoef.")
    if np.any((prior <= 0) & np.isfinite(prior)):
        raise RuntimeError("Finite prior variances must be positive; use inf for no prior.")
    if np.any(~np.isfinite(ridge_weights)) or np.any(ridge_weights < 0):
        raise RuntimeError("Ridge weights must be finite and nonnegative.")
    return Template(time_grid, matrix, quantity, names, prior, ridge_weights, file_digest(path))


def resolve_operators(template: Template, requested: list[str] | None) -> list[str]:
    if not requested or "all" in requested:
        operators = (
            ["interval-average", "midpoint"]
            if template.quantity == "velocity"
            else ["endpoint-difference"]
        )
    else:
        operators = list(dict.fromkeys(requested))
    allowed = {
        "velocity": {"interval-average", "midpoint"},
        "displacement": {"endpoint-difference"},
    }[template.quantity]
    invalid = set(operators) - allowed
    if invalid:
        raise RuntimeError(
            f"Operators {sorted(invalid)} are incompatible with a {template.quantity} template; "
            f"allowed: {sorted(allowed)}."
        )
    return operators


def decimal_year(values: Any, deps: dict[str, Any]) -> Any:
    pd, np = deps["pd"], deps["np"]
    timestamps = pd.DatetimeIndex(pd.to_datetime(values, utc=True))
    years = timestamps.year.to_numpy()
    starts = pd.to_datetime(years.astype(str) + "-01-01", utc=True)
    ends = pd.to_datetime((years + 1).astype(str) + "-01-01", utc=True)
    fraction = (timestamps - starts).total_seconds().to_numpy() / (
        (ends - starts).total_seconds().to_numpy()
    )
    return years.astype(np.float64) + fraction


def interpolate_columns(time_grid: Any, values: Any, query: Any, np: Any) -> Any:
    if np.any(query < time_grid[0]) or np.any(query > time_grid[-1]):
        raise RuntimeError(
            f"Observation time range [{query.min():.8f}, {query.max():.8f}] exceeds "
            f"template range [{time_grid[0]:.8f}, {time_grid[-1]:.8f}]."
        )
    return np.column_stack(
        [np.interp(query, time_grid, values[:, column]) for column in range(values.shape[1])]
    )


def build_design_matrices(
    template: Template, observations: Any, operators: list[str], deps: dict[str, Any]
) -> tuple[dict[str, Any], Any]:
    np = deps["np"]
    t1 = decimal_year(observations["date_img1"], deps)
    t2 = decimal_year(observations["date_img2"], deps)
    if np.any(t2 <= t1):
        raise RuntimeError("All selected observations must have date_img2 > date_img1.")
    midpoint = 0.5 * (t1 + t2)
    matrices = {}
    if "midpoint" in operators:
        matrices["midpoint"] = interpolate_columns(
            template.time, template.matrix, midpoint, np
        )
    if "interval-average" in operators:
        delta = np.diff(template.time)
        cumulative = np.zeros_like(template.matrix)
        cumulative[1:] = np.cumsum(
            0.5 * (template.matrix[1:] + template.matrix[:-1]) * delta[:, None],
            axis=0,
        )
        first = interpolate_columns(template.time, cumulative, t1, np)
        second = interpolate_columns(template.time, cumulative, t2, np)
        matrices["interval-average"] = (second - first) / (t2 - t1)[:, None]
    if "endpoint-difference" in operators:
        first = interpolate_columns(template.time, template.matrix, t1, np)
        second = interpolate_columns(template.time, template.matrix, t2, np)
        matrices["endpoint-difference"] = (second - first) / (t2 - t1)[:, None]
    prediction_matrix = (
        template.matrix
        if template.quantity == "velocity"
        else np.gradient(
            template.matrix,
            template.time,
            axis=0,
            edge_order=2 if len(template.time) >= 3 else 1,
        )
    )
    return matrices, prediction_matrix


def prior_precision(template: Template, ridge: float, np: Any) -> Any:
    precision = ridge * template.ridge_weights.astype(np.float64)
    finite = np.isfinite(template.prior_variance)
    precision = precision.copy()
    precision[finite] += 1.0 / template.prior_variance[finite]
    return np.diag(precision)


def diagnose_pixels(
    design: Any,
    prediction: Any,
    valid: Any,
    sigma: Any,
    precision: Any,
    pixel_batch_size: int,
    np: Any,
) -> dict[str, Any]:
    """Compute posterior diagnostics for arrays shaped (observation, y, x)."""
    nobs, ny, nx = valid.shape
    ncoef = design.shape[1]
    npixel = ny * nx
    usable = valid & np.isfinite(sigma) & (sigma > 0)
    weights = np.where(usable, 1.0 / np.square(sigma), 0.0).reshape(nobs, npixel).T
    output = {name: np.full(npixel, np.nan, dtype=np.float64) for name in METRICS}
    output["usable_observation_count"] = usable.sum(axis=0).reshape(-1).astype(np.float64)
    sign_prior, logdet_prior = np.linalg.slogdet(precision)
    proper_prior = sign_prior > 0
    for start in range(0, npixel, pixel_batch_size):
        stop = min(start + pixel_batch_size, npixel)
        w = weights[start:stop]
        information = np.einsum("bi,ij,ik->bjk", w, design, design, optimize=True)
        posterior_precision = information + precision[None, :, :]

        diagonal_data = np.diagonal(information, axis1=1, axis2=2)
        scale_data = np.sqrt(np.maximum(diagonal_data, 0.0))
        denom_data = scale_data[:, :, None] * scale_data[:, None, :]
        normalized_data = np.divide(
            information,
            denom_data,
            out=np.zeros_like(information),
            where=denom_data > 0,
        )
        data_eigenvalues = np.linalg.eigvalsh(normalized_data)
        data_tolerance = (
            np.maximum(data_eigenvalues[:, -1], 0.0) * ncoef * np.finfo(float).eps * 100
        )
        output["data_rank"][start:stop] = (
            data_eigenvalues > data_tolerance[:, None]
        ).sum(axis=1)

        diagonal = np.diagonal(posterior_precision, axis1=1, axis2=2)
        scale = np.sqrt(np.maximum(diagonal, 0.0))
        denom = scale[:, :, None] * scale[:, None, :]
        normalized = np.divide(
            posterior_precision,
            denom,
            out=np.zeros_like(posterior_precision),
            where=denom > 0,
        )
        eigenvalues = np.linalg.eigvalsh(normalized)
        tolerance = np.maximum(eigenvalues[:, -1], 0.0) * ncoef * np.finfo(float).eps * 100
        full_rank = eigenvalues[:, 0] > tolerance
        condition = np.full(stop - start, np.nan)
        condition[full_rank] = eigenvalues[full_rank, -1] / eigenvalues[full_rank, 0]
        output["log10_condition_number"][start:stop] = np.log10(condition)
        if not np.any(full_rank):
            continue
        indices = np.flatnonzero(full_rank)
        covariance = np.linalg.inv(posterior_precision[indices])
        info_selected = information[indices]
        output["effective_dof"][start + indices] = np.einsum(
            "bij,bji->b", covariance, info_selected, optimize=True
        )
        if proper_prior:
            _, logdet = np.linalg.slogdet(posterior_precision[indices])
            output["information_gain_nats"][start + indices] = 0.5 * (
                logdet - logdet_prior
            )
        prediction_sum = np.zeros(len(indices), dtype=np.float64)
        prediction_max = np.zeros(len(indices), dtype=np.float64)
        prediction_count = 0
        for tstart in range(0, prediction.shape[0], 256):
            basis = prediction[tstart : tstart + 256]
            variance = np.einsum(
                "ti,bij,tj->bt", basis, covariance, basis, optimize=True
            )
            std = np.sqrt(np.maximum(variance, 0.0))
            prediction_sum += std.sum(axis=1)
            prediction_max = np.maximum(prediction_max, std.max(axis=1))
            prediction_count += basis.shape[0]
        output["mean_prediction_std"][start + indices] = prediction_sum / prediction_count
        output["max_prediction_std"][start + indices] = prediction_max
    return {name: values.reshape(ny, nx).astype("float32") for name, values in output.items()}


def resolve_error_variable(ds: Any, schema: Any, args: argparse.Namespace) -> str | None:
    if args.error_variable == "unit":
        return None
    if args.error_variable != "auto":
        if args.error_variable not in ds:
            raise RuntimeError(f"Error variable {args.error_variable!r} is absent from cube.")
        return args.error_variable
    candidate = f"{args.velocity_variable}_error"
    if candidate not in ds:
        raise RuntimeError(
            f"Automatic error variable {candidate!r} is absent; specify --error-variable."
        )
    return candidate


def block_arrays(
    subset: Any,
    schema: Any,
    observations: Any,
    yslice: slice,
    xslice: slice,
    args: argparse.Namespace,
    deps: dict[str, Any],
) -> tuple[Any, Any]:
    np = deps["np"]
    indices = observations["observation_index"].to_numpy(dtype=np.int64)
    block = subset.isel(
        {
            schema.obs: indices,
            schema.y: yslice,
            schema.x: xslice,
        }
    )
    if args.velocity_variable not in block:
        raise RuntimeError(f"Velocity variable {args.velocity_variable!r} is absent from cube.")
    velocity = block[args.velocity_variable].transpose(schema.obs, schema.y, schema.x)
    valid = coverage.finite_valid(velocity, deps)
    error_name = resolve_error_variable(ds=subset, schema=schema, args=args)
    if error_name is None:
        sigma = None
        tasks = [valid.data]
    else:
        error = block[error_name]
        if schema.y in error.dims and schema.x in error.dims:
            error = error.transpose(schema.obs, schema.y, schema.x)
            tasks = [valid.data, error.data]
        else:
            error = error.transpose(schema.obs)
            tasks = [valid.data, error.data]
    computed = deps["dask_array"].compute(*tasks)
    valid_values = np.asarray(computed[0], dtype=bool)
    if error_name is None:
        sigma_values = np.ones(valid_values.shape, dtype=np.float64)
    else:
        error_values = np.asarray(computed[1], dtype=np.float64)
        if error_values.ndim == 1:
            error_values = error_values[:, None, None]
        sigma_values = np.broadcast_to(error_values, valid_values.shape).copy()
        sigma_values = np.maximum(np.abs(sigma_values), args.error_floor)
    return valid_values, sigma_values


def open_observation_cache(
    path: Path,
    subset: Any,
    schema: Any,
    observations: Any,
    args: argparse.Namespace,
    deps: dict[str, Any],
) -> tuple[Any | None, str]:
    """Open and validate a coverage-produced observation cache."""
    if args.observation_cache == "off":
        return None, "disabled"
    if not path.is_file():
        return None, "cache file is absent"
    import h5py

    np = deps["np"]
    try:
        cache = h5py.File(path, "r")
    except OSError as exc:
        return None, f"cache cannot be opened: {exc}"
    try:
        expected_indices = observations["observation_index"].to_numpy(dtype=np.int64)
        if int(cache.attrs.get("cache_version", -1)) != 1:
            raise ValueError("unsupported cache version")
        if str(cache.attrs.get("velocity_variable", "")) != args.velocity_variable:
            raise ValueError("velocity variable differs")
        if "validity" not in cache or "observation_index" not in cache:
            raise ValueError("required datasets are absent")
        if not np.array_equal(cache["observation_index"][:], expected_indices):
            raise ValueError("selected observations differ")
        expected_shape = (
            len(expected_indices),
            int(subset.sizes[schema.y]),
            int(subset.sizes[schema.x]),
        )
        if cache["validity"].shape != expected_shape:
            raise ValueError("spatial shape differs")
        if not np.allclose(cache["x"][:], subset[schema.x].values) or not np.allclose(
            cache["y"][:], subset[schema.y].values
        ):
            raise ValueError("spatial coordinates differ")
        if args.error_variable != "unit":
            expected_error = (
                f"{args.velocity_variable}_error"
                if args.error_variable == "auto"
                else args.error_variable
            )
            if str(cache.attrs.get("error_variable", "")) != expected_error:
                raise ValueError("uncertainty variable differs")
            if "uncertainty" not in cache:
                raise ValueError("uncertainty dataset is absent")
    except (KeyError, TypeError, ValueError) as exc:
        cache.close()
        return None, str(exc)
    return cache, "compatible"


def cached_block_arrays(
    cache: Any,
    yslice: slice,
    xslice: slice,
    args: argparse.Namespace,
    np: Any,
) -> tuple[Any, Any]:
    valid = np.asarray(cache["validity"][:, yslice, xslice], dtype=bool)
    if args.error_variable == "unit":
        sigma = np.ones(valid.shape, dtype=np.float64)
    else:
        uncertainty = cache["uncertainty"]
        if uncertainty.ndim == 1:
            values = np.asarray(uncertainty[:], dtype=np.float64)[:, None, None]
        else:
            values = np.asarray(uncertainty[:, yslice, xslice], dtype=np.float64)
        sigma = np.broadcast_to(values, valid.shape).copy()
        sigma = np.maximum(np.abs(sigma), args.error_floor)
    return valid, sigma


def write_float_tif(values: Any, x: Any, y: Any, crs: Any, path: Path, np: Any) -> None:
    import rasterio
    from rasterio.transform import from_origin

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    data = np.asarray(values, dtype=np.float32)
    if x[0] > x[-1]:
        x = x[::-1]
        data = data[:, ::-1]
    if y[0] < y[-1]:
        y = y[::-1]
        data = data[::-1, :]
    dx = float(np.median(np.diff(x)))
    dy = abs(float(np.median(np.diff(y))))
    transform = from_origin(x[0] - dx / 2, y[0] + dy / 2, dx, dy)
    filled = np.where(np.isfinite(data), data, FLOAT_NODATA).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=filled.shape[1],
        height=filled.shape[0],
        count=1,
        dtype="float32",
        nodata=FLOAT_NODATA,
        crs=crs,
        transform=transform,
        compress="deflate",
        tiled=True,
    ) as destination:
        destination.write(filled, 1)


def initialize_work_arrays(
    work_dir: Path,
    fingerprint_value: str,
    operators: list[str],
    shape: tuple[int, int],
    resume: bool,
    np: Any,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    status_path = work_dir / "block_status.json"
    state = None
    if resume and status_path.exists():
        try:
            candidate = json.loads(status_path.read_text())
            paths_exist = all(
                (work_dir / operator / f"{metric}.npy").exists()
                for operator in operators
                for metric in METRICS
            )
            if candidate.get("fingerprint") == fingerprint_value and paths_exist:
                state = candidate
        except (OSError, json.JSONDecodeError):
            pass
    arrays: dict[str, dict[str, Any]] = {}
    if state is None:
        state = {
            "status": "running",
            "fingerprint": fingerprint_value,
            "shape": list(shape),
            "completed_blocks": [],
        }
        for operator in operators:
            arrays[operator] = {}
            operator_dir = work_dir / operator
            operator_dir.mkdir(parents=True, exist_ok=True)
            for metric in METRICS:
                array = np.lib.format.open_memmap(
                    operator_dir / f"{metric}.npy",
                    mode="w+",
                    dtype=np.float32,
                    shape=shape,
                )
                array[:] = np.nan
                array.flush()
                arrays[operator][metric] = array
        multi.atomic_write_json(status_path, state)
    else:
        for operator in operators:
            arrays[operator] = {
                metric: np.lib.format.open_memmap(
                    work_dir / operator / f"{metric}.npy", mode="r+"
                )
                for metric in METRICS
            }
    return arrays, state


def process_tile(
    tile: dict[str, Any],
    roi: coverage.ROI,
    template: Template,
    operators: list[str],
    precision: Any,
    args: argparse.Namespace,
    deps: dict[str, Any],
) -> dict[str, Any]:
    np, pd = deps["np"], deps["pd"]
    cube = tile["cube"]
    cube_id = cube["id"]
    output_dir = args.outdir / "tiles" / cube_id
    work_dir = output_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    source_tile_dir = args.results_dir / "tiles" / cube_id
    observations = pd.read_csv(source_tile_dir / "selected_observations.csv")
    matrices, prediction = build_design_matrices(template, observations, operators, deps)
    fingerprint_value = multi.fingerprint(
        {
            "template": template.digest,
            "operators": operators,
            "ridge": args.ridge,
            "velocity_variable": args.velocity_variable,
            "error_variable": args.error_variable,
            "error_floor": args.error_floor,
            "block_size": args.block_size,
            "cube_id": cube_id,
            "observation_keys": observations["image_pair_id"].astype(str).tolist(),
        }
    )
    print(f"[{cube_id}] opening cube metadata")
    for attempt in range(1, 7):
        try:
            ds = coverage.open_cube(cube["url"], deps)
            break
        except Exception as exc:
            if not multi.transient_remote_error(exc) or attempt == 6:
                raise
            delay = min(30, 2**attempt)
            print(
                f"[{cube_id}] metadata retry in {delay}s "
                f"({attempt + 1}/6): {type(exc).__name__}"
            )
            time.sleep(delay)
    schema = coverage.ITSLiveCubeSchema(ds)
    cube_crs = schema.crs(deps)
    subset, roi_mask = coverage.coordinate_subset(ds, schema, roi.in_crs(cube_crs, deps), deps)
    shape = (int(subset.sizes[schema.y]), int(subset.sizes[schema.x]))
    cache_path = source_tile_dir / "work" / "observation_cache.h5"
    cache, cache_reason = open_observation_cache(
        cache_path, subset, schema, observations, args, deps
    )
    if cache is not None:
        print(f"[{cube_id}] using local observation cache")
    elif args.observation_cache == "require":
        raise RuntimeError(
            f"[{cube_id}] compatible observation cache required: {cache_reason}"
        )
    elif args.observation_cache == "auto":
        print(f"[{cube_id}] observation cache unavailable ({cache_reason}); reading remote arrays")
    arrays, state = initialize_work_arrays(
        work_dir, fingerprint_value, operators, shape, args.resume, np
    )
    completed = set(state.get("completed_blocks", []))
    blocks = [
        (y0, min(y0 + args.block_size, shape[0]), x0, min(x0 + args.block_size, shape[1]))
        for y0 in range(0, shape[0], args.block_size)
        for x0 in range(0, shape[1], args.block_size)
    ]
    if completed:
        print(f"[{cube_id}] resuming {len(completed)}/{len(blocks)} blocks")
    try:
        for number, (y0, y1, x0, x1) in enumerate(blocks, start=1):
            block_id = f"{y0}:{y1},{x0}:{x1}"
            if block_id in completed:
                continue
            if cache is not None:
                valid, sigma = cached_block_arrays(
                    cache, slice(y0, y1), slice(x0, x1), args, np
                )
            else:
                for attempt in range(1, 7):
                    try:
                        valid, sigma = block_arrays(
                            subset,
                            schema,
                            observations,
                            slice(y0, y1),
                            slice(x0, x1),
                            args,
                            deps,
                        )
                        break
                    except Exception as exc:
                        if not multi.transient_remote_error(exc) or attempt == 6:
                            raise
                        delay = min(30, 2**attempt)
                        print(f"[{cube_id}] block {number} retry in {delay}s: {type(exc).__name__}")
                        time.sleep(delay)
            for operator in operators:
                metrics = diagnose_pixels(
                    matrices[operator],
                    prediction,
                    valid,
                    sigma,
                    precision,
                    args.pixel_batch_size,
                    np,
                )
                for metric, values in metrics.items():
                    arrays[operator][metric][y0:y1, x0:x1] = values
                    arrays[operator][metric].flush()
            completed.add(block_id)
            state.update(
                {
                    "completed_blocks": sorted(completed),
                    "completed_count": len(completed),
                    "total_blocks": len(blocks),
                }
            )
            multi.atomic_write_json(work_dir / "block_status.json", state)
            print(f"[{cube_id}] diagnostic block {number}/{len(blocks)} complete")
    finally:
        if cache is not None:
            cache.close()

    source_dirs = {}
    for operator in operators:
        source_dir = output_dir / operator
        source_dirs[operator] = str(source_dir)
        for metric in METRICS:
            values = np.array(arrays[operator][metric], copy=True)
            write_float_tif(
                values,
                subset[schema.x].values,
                subset[schema.y].values,
                cube_crs,
                source_dir / f"{metric}.tif",
                np,
            )
    state["status"] = "complete"
    multi.atomic_write_json(work_dir / "block_status.json", state)
    result = {"cube": cube, "source_dirs": source_dirs, "roi_pixel_count": int(roi_mask.sum())}
    coverage.write_json(output_dir / "tile_status.json", {"status": "complete", "result": result})
    return result


def process_tile_isolated(
    tile: dict[str, Any],
    source_arguments: dict[str, Any],
    diagnostic_arguments: dict[str, Any],
) -> dict[str, Any]:
    """Process one cube in a fresh process with independently opened resources."""
    deps = coverage.require_dependencies()
    args = SimpleNamespace(**diagnostic_arguments)
    args.results_dir = Path(args.results_dir)
    args.template_h5 = Path(args.template_h5)
    args.outdir = Path(args.outdir)
    source_args = SimpleNamespace(**source_arguments)
    source_args.outdir = Path(source_args.outdir)
    roi = coverage.construct_roi(source_args, deps)
    template = load_template(args.template_h5, deps)
    operators = resolve_operators(template, args.observation_operator)
    precision = prior_precision(template, args.ridge, deps["np"])
    return process_tile(tile, roi, template, operators, precision, args, deps)


def mosaic_metric(
    tile_results: list[dict[str, Any]],
    operator: str,
    metric: str,
    reference_path: Path,
    output_path: Path,
    deps: dict[str, Any],
) -> None:
    import rasterio

    np = deps["np"]
    with rasterio.open(reference_path) as reference:
        profile = reference.profile.copy()
        reference_values = reference.read(1, masked=True)
        roi_mask = ~np.ma.getmaskarray(reference_values)
        x = reference.transform.c + reference.transform.a * (np.arange(reference.width) + 0.5)
        y = reference.transform.f + reference.transform.e * (np.arange(reference.height) + 0.5)
    values = np.full(roi_mask.shape, FLOAT_NODATA, dtype=np.float32)
    for result in tile_results:
        bbox = result["cube"].get("proj_bbox")
        if not bbox:
            raise RuntimeError(f"Cube {result['cube']['id']} lacks proj:bbox.")
        xmin, ymin, xmax, ymax = map(float, bbox)
        xi = np.flatnonzero((x >= xmin) & (x < xmax))
        yi = np.flatnonzero((y >= ymin) & (y < ymax))
        if not xi.size or not yi.size:
            continue
        sampled, nodata = multi.nearest_tile_sample(
            Path(result["source_dirs"][operator]) / f"{metric}.tif", x[xi], y[yi], np
        )
        local_roi = roi_mask[np.ix_(yi, xi)]
        view = values[np.ix_(yi, xi)]
        valid = local_roi & (sampled != nodata)
        view[valid] = sampled[valid]
        values[np.ix_(yi, xi)] = view
    values[~roi_mask] = FLOAT_NODATA
    profile.update(dtype="float32", nodata=FLOAT_NODATA, compress="deflate", tiled=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as destination:
        destination.write(values, 1)


def write_comparisons(outdir: Path, operators: list[str], reference: Path) -> list[str]:
    if "interval-average" not in operators or "midpoint" not in operators:
        return []
    import numpy as np
    import rasterio

    comparison_dir = outdir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for metric in ("mean_prediction_std", "max_prediction_std"):
        with rasterio.open(outdir / "interval-average" / f"{metric}.tif") as source:
            interval = source.read(1, masked=True)
            profile = source.profile.copy()
        with rasterio.open(outdir / "midpoint" / f"{metric}.tif") as source:
            midpoint = source.read(1, masked=True)
        ratio = np.ma.divide(midpoint, interval)
        path = comparison_dir / f"midpoint_to_interval_{metric}_ratio.tif"
        with rasterio.open(path, "w", **profile) as destination:
            destination.write(ratio.filled(FLOAT_NODATA).astype(np.float32), 1)
        written.append(str(path))
    return written


def run(args: argparse.Namespace) -> int:
    deps = coverage.require_dependencies()
    template = load_template(args.template_h5, deps)
    operators = resolve_operators(template, args.observation_operator)
    precision = prior_precision(template, args.ridge, deps["np"])
    manifest_path = args.results_dir / "multi_cube_manifest.json"
    run_config_path = args.results_dir / "run_config.json"
    if not manifest_path.is_file() or not run_config_path.is_file():
        raise RuntimeError("results_dir must contain a completed multi-cube manifest and run_config.json.")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete":
        raise RuntimeError("The source multi-cube coverage run is not complete.")
    tiles = manifest.get("tiles", [])
    config = json.loads(run_config_path.read_text())
    source_args = SimpleNamespace(**config["arguments"])
    source_args.outdir = Path(source_args.outdir)
    roi = coverage.construct_roi(source_args, deps)
    if args.outdir.exists() and any(args.outdir.iterdir()) and not (
        args.resume or args.overwrite
    ):
        raise RuntimeError(
            f"{args.outdir} is not empty; use --resume or --overwrite."
        )
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.workers == 1 or len(tiles) <= 1:
        results = [
            process_tile(tile, roi, template, operators, precision, args, deps)
            for tile in tiles
        ]
    else:
        worker_count = min(args.workers, len(tiles))
        print(f"Processing {len(tiles)} cubes with {worker_count} worker processes...")
        source_arguments = coverage.json_value(config["arguments"])
        diagnostic_arguments = coverage.json_value(vars(args))
        results_by_cube: dict[str, dict[str, Any]] = {}
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            futures = {
                pool.submit(
                    process_tile_isolated,
                    tile,
                    source_arguments,
                    diagnostic_arguments,
                ): tile["cube"]["id"]
                for tile in tiles
            }
            for future in as_completed(futures):
                cube_id = futures[future]
                try:
                    results_by_cube[cube_id] = future.result()
                except Exception as exc:
                    for pending in futures:
                        pending.cancel()
                    raise RuntimeError(
                        f"Diagnostic worker failed for cube {cube_id}: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                print(
                    f"[{cube_id}] cube diagnostics complete "
                    f"({len(results_by_cube)}/{len(tiles)})"
                )
        results = [results_by_cube[tile["cube"]["id"]] for tile in tiles]
    reference = args.results_dir / "coverage_count.tif"
    for operator in operators:
        print(f"Mosaicking {operator} diagnostic maps...")
        for metric in METRICS:
            mosaic_metric(
                results,
                operator,
                metric,
                reference,
                args.outdir / operator / f"{metric}.tif",
                deps,
            )
    comparisons = write_comparisons(args.outdir, operators, reference)
    summary = {
        "status": "complete",
        "source_results_dir": str(args.results_dir),
        "template_h5": str(args.template_h5),
        "template_sha256": template.digest,
        "template_quantity": template.quantity,
        "time_range": [float(template.time[0]), float(template.time[-1])],
        "median_time_step": float(deps["np"].median(deps["np"].diff(template.time))),
        "uniform_time_grid": bool(
            deps["np"].allclose(
                deps["np"].diff(template.time),
                deps["np"].diff(template.time)[0],
                rtol=1e-6,
                atol=1e-10,
            )
        ),
        "coefficient_count": int(template.matrix.shape[1]),
        "coefficient_names": template.coefficient_names,
        "operators": operators,
        "velocity_variable": args.velocity_variable,
        "error_variable": args.error_variable,
        "error_floor": args.error_floor,
        "ridge": args.ridge,
        "workers": args.workers,
        "prior_variance": coverage.json_value(template.prior_variance.tolist()),
        "ridge_weights": coverage.json_value(template.ridge_weights.tolist()),
        "metrics": list(METRICS),
        "comparison_outputs": comparisons,
    }
    coverage.write_json(args.outdir / "summary.json", summary)
    print(f"Inversion diagnostics complete: {args.outdir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
