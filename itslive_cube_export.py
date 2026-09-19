#!/usr/bin/env python3
"""Export filtered ITS_LIVE cube velocities to a resumable grouped Zarr store."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import itslive_cube_coverage as coverage
import itslive_multi_cube as multi


DEFAULT_VARIABLES = ("vx", "vy", "v", "v_error", "vx_error", "vy_error")
DEFAULT_SIZE_LIMIT = 50 * 1024**3
STORE_NAME = "velocity.zarr"
MANIFEST_NAME = "manifest.json"
EXPORT_CONFIG_NAME = "export_run_config.json"
EXPORT_INVENTORY_NAME = "export_selected_observations.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export selected ITS_LIVE velocities to a resumable grouped Zarr v2 store."
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--coverage-results", type=Path, help="Replay a completed coverage run.")
    mode.add_argument("--bbox", type=float, nargs=4, metavar=("XMIN", "YMIN", "XMAX", "YMAX"))
    mode.add_argument(
        "--bbox-lonlat", type=float, nargs=4, metavar=("WEST", "SOUTH", "EAST", "NORTH")
    )
    p.add_argument("--epsg", type=int, default=3031)
    p.add_argument("--start", help="Inclusive start date or timestamp.")
    p.add_argument("--end", help="Inclusive end date or timestamp.")
    p.add_argument("--mission")
    p.add_argument("--sensor")
    p.add_argument("--min-pair-days", type=float)
    p.add_argument("--max-pair-days", type=float)
    p.add_argument("--cube-url")
    p.add_argument("--multi-cube", action="store_true")
    p.add_argument("--edge-points", type=int, default=41)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument(
        "--variables", action="append", metavar="NAME",
        help="Variable to export; repeat for multiple variables (default: analysis-ready set).",
    )
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--observation-block-size", type=int, default=16)
    p.add_argument("--spatial-block-size", type=int, default=100)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--allow-large", action="store_true")
    args = p.parse_args(argv)

    if args.coverage_results is None and (not args.start or not args.end):
        p.error("standalone mode requires --start and --end")
    if args.coverage_results is not None:
        forbidden = {
            "start": args.start, "end": args.end, "mission": args.mission,
            "sensor": args.sensor, "min_pair_days": args.min_pair_days,
            "max_pair_days": args.max_pair_days, "cube_url": args.cube_url,
            "multi_cube": args.multi_cube,
        }
        used = [f"--{name.replace('_', '-')}" for name, value in forbidden.items() if value not in (None, False)]
        if used:
            p.error("--coverage-results cannot be combined with selection overrides: " + ", ".join(used))
    if args.cube_url and args.multi_cube:
        p.error("--cube-url cannot be combined with --multi-cube")
    if args.resume and args.overwrite:
        p.error("choose either --resume or --overwrite")
    if args.workers < 1 or args.observation_block_size < 1 or args.spatial_block_size < 1:
        p.error("worker and block sizes must be positive")
    if args.edge_points < 2:
        p.error("--edge-points must be at least 2")
    if args.min_pair_days is not None and args.min_pair_days < 0:
        p.error("--min-pair-days must be nonnegative")
    if args.max_pair_days is not None and args.max_pair_days < 0:
        p.error("--max-pair-days must be nonnegative")
    if (
        args.min_pair_days is not None and args.max_pair_days is not None
        and args.min_pair_days > args.max_pair_days
    ):
        p.error("--min-pair-days cannot exceed --max-pair-days")
    args.variables = list(dict.fromkeys(args.variables or DEFAULT_VARIABLES))
    return args


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in {path}: {exc}") from exc


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    coverage.write_json(temporary, payload)
    temporary.replace(path)


def hash_payload(payload: Any) -> str:
    encoded = json.dumps(coverage.json_value(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def human_bytes(number: int) -> str:
    value = float(number)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}" if unit != "B" else f"{value:.0f} B"
        value /= 1024
    raise AssertionError


def enforce_size_limit(estimated: int, allow_large: bool) -> None:
    if estimated > DEFAULT_SIZE_LIMIT and not allow_large:
        raise RuntimeError(
            f"Estimated export exceeds the 50 GiB safety limit ({human_bytes(estimated)}); "
            "narrow the selection or pass --allow-large."
        )


def json_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    return {str(k): coverage.json_value(v) for k, v in attrs.items() if k != "_ARRAY_DIMENSIONS"}


def roi_from_saved_config(config: dict[str, Any], deps: dict[str, Any]) -> coverage.ROI:
    saved = config.get("input_roi")
    if isinstance(saved, dict) and "geometry" in saved and "crs" in saved:
        return coverage.ROI(
            deps["shape"](saved["geometry"]), deps["CRS"].from_user_input(saved["crs"]),
            str(saved.get("kind", "saved")), list(saved.get("bounds", [])),
        )
    arguments = config.get("arguments", {})
    if arguments.get("bbox") is not None or arguments.get("bbox_lonlat") is not None:
        return coverage.construct_roi(
            argparse.Namespace(
                bbox=arguments.get("bbox"), bbox_lonlat=arguments.get("bbox_lonlat"),
                epsg=int(arguments.get("epsg", 3031)),
                edge_points=int(arguments.get("edge_points", 41)),
            ),
            deps,
        )
    raise RuntimeError("Coverage run_config.json does not contain a reusable input ROI.")


def cube_id(cube: dict[str, Any], position: int = 0) -> str:
    return str(cube.get("id") or f"cube-{position:04d}")


def standalone_selection(
    args: argparse.Namespace, deps: dict[str, Any]
) -> tuple[coverage.ROI, list[dict[str, Any]], dict[str, Any] | None]:
    roi = coverage.construct_roi(args, deps)
    if args.cube_url:
        cubes = [{"id": "cube-0000", "url": args.cube_url, "selection": "--cube-url override"}]
    elif args.multi_cube:
        cubes = multi.discover_intersecting_cubes(roi, deps)
    else:
        url, info = coverage.discover_cube(roi, deps)
        cubes = [{**info, "url": url}]
    return roi, cubes, None


def replay_selection(
    results: Path, deps: dict[str, Any]
) -> tuple[coverage.ROI, list[dict[str, Any]], dict[str, Any]]:
    config = read_json(results / "run_config.json")
    # Exporter versions before the output files were namespaced could overwrite
    # the coverage run_config.json when both stages shared one directory. Recover
    # the original coverage arguments embedded in that export configuration.
    embedded_source = config.get("coverage_source")
    if isinstance(embedded_source, dict) and isinstance(embedded_source.get("arguments"), dict):
        config = {
            "arguments": embedded_source["arguments"],
            "fingerprint": embedded_source.get("fingerprint"),
            "input_roi": config.get("input_roi"),
        }
    roi = roi_from_saved_config(config, deps)
    args_saved = config.get("arguments", {})
    source_fingerprint = config.get("fingerprint")
    if args_saved.get("multi_cube"):
        manifest_path = results / "multi_cube_manifest.json"
        manifest = read_json(manifest_path)
        if manifest.get("status") not in (None, "complete"):
            raise RuntimeError(f"Coverage manifest is not complete: {manifest_path}")
        source_fingerprint = source_fingerprint or manifest.get("fingerprint")
        cubes = manifest.get("configuration", {}).get("cubes") or config.get("configuration", {}).get("cubes")
        if not cubes:
            raise RuntimeError("Multi-cube coverage metadata contains no cube list.")
    else:
        info = read_json(results / "cube_info.json")
        if not info.get("url"):
            raise RuntimeError("Coverage cube_info.json contains no cube URL.")
        cubes = [info]
    return roi, list(cubes), {
        "directory": str(results.resolve()),
        "fingerprint": source_fingerprint,
        "arguments": args_saved,
    }


def replay_indices(results: Path, cubes: list[dict[str, Any]], deps: dict[str, Any]) -> dict[str, dict[str, Any]]:
    pd = deps["pd"]
    table = pd.read_csv(results / "selected_observations.csv")
    result = {
        cube_id(cube, i): {"indices": [], "expected_pair_ids": {}, "url": cube.get("url")}
        for i, cube in enumerate(cubes)
    }
    if table.empty:
        return result
    if "cube_observations" in table:
        for _, row in table.dropna(subset=["cube_observations"]).iterrows():
            pair_value = row.get("image_pair_id", "")
            pair_id = "" if pd.isna(pair_value) else str(pair_value)
            for entry in json.loads(row["cube_observations"]):
                cid = str(entry["cube_id"])
                if cid not in result:
                    raise RuntimeError(f"Coverage inventory references unknown cube {cid!r}.")
                index = int(entry["observation_index"])
                result[cid]["indices"].append(index)
                result[cid]["expected_pair_ids"][str(index)] = pair_id
    elif len(cubes) > 1 and {"source_cube_id", "observation_index"}.issubset(table.columns):
        # Recover inventories written by exporter versions that overwrote the
        # coverage CSV with one row per cube-observation record.
        for _, row in table.iterrows():
            cid = str(row["source_cube_id"])
            if cid not in result:
                raise RuntimeError(f"Export inventory references unknown cube {cid!r}.")
            index = int(row["observation_index"])
            pair_value = row.get("image_pair_id", "")
            pair_id = "" if pd.isna(pair_value) else str(pair_value)
            result[cid]["indices"].append(index)
            result[cid]["expected_pair_ids"][str(index)] = pair_id
    else:
        cid = cube_id(cubes[0])
        if "observation_index" not in table:
            raise RuntimeError("Coverage observation inventory has no observation_index column.")
        result[cid]["indices"] = table["observation_index"].astype(int).tolist()
        if "image_pair_id" in table:
            result[cid]["expected_pair_ids"] = {
                str(int(index)): ("" if pd.isna(pair_id) else str(pair_id))
                for index, pair_id in zip(table["observation_index"], table["image_pair_id"])
            }
    for selection in result.values():
        selection["indices"] = list(dict.fromkeys(selection["indices"]))
    return result


def selected_for_cube(
    ds: Any,
    schema: coverage.ITSLiveCubeSchema,
    args: argparse.Namespace,
    deps: dict[str, Any],
    replay: dict[str, Any] | None,
) -> Any:
    table = coverage.observation_table(ds, schema, deps)
    if replay is None:
        selected, _, _ = coverage.filter_observations(table, args, deps)
        return selected
    indices = list(map(int, replay["indices"]))
    if len(indices) != len(set(indices)):
        raise RuntimeError("Replay observation indices are not unique within a cube.")
    if any(index < 0 or index >= len(table) for index in indices):
        raise RuntimeError("Replay observation index is outside the source cube.")
    selected = table.iloc[indices].copy() if indices else table.iloc[0:0].copy()
    expected = replay.get("expected_pair_ids", {})
    for _, row in selected.iterrows():
        wanted = expected.get(str(int(row["observation_index"])), "")
        actual = str(row.get("image_pair_id", "") or "")
        if wanted and actual != wanted:
            raise RuntimeError(
                f"Replay validation failed at observation {int(row['observation_index'])}: "
                f"saved pair ID {wanted!r} does not match source {actual!r}."
            )
    return selected


def variable_spec(da: Any, schema: coverage.ITSLiveCubeSchema) -> dict[str, Any]:
    dims = tuple(da.dims)
    if dims == (schema.obs,):
        kind = "observation"
        out_dims = ["observation"]
    elif set(dims) == {schema.obs, schema.y, schema.x} and len(dims) == 3:
        kind = "spatial"
        out_dims = ["observation", "y", "x"]
    else:
        raise RuntimeError(
            f"Requested variable {da.name!r} has unsupported dimensions {dims}; expected "
            f"({schema.obs!r},) or an observation/y/x array."
        )
    dtype = da.dtype
    if dtype.kind not in "biufc":
        raise RuntimeError(f"Requested variable {da.name!r} has unsupported dtype {dtype}.")
    return {
        "name": str(da.name), "kind": kind, "dimensions": out_dims,
        "source_dimensions": list(dims), "dtype": dtype.str,
        "attributes": json_attrs(dict(da.attrs)),
    }


def _preflight_cube_once(
    cube: dict[str, Any], position: int, roi: coverage.ROI, args: argparse.Namespace,
    deps: dict[str, Any], replay: dict[str, Any] | None,
) -> tuple[dict[str, Any], Any]:
    cid = cube_id(cube, position)
    url = cube.get("url")
    if not url:
        raise RuntimeError(f"Cube {cid!r} has no URL.")
    ds = coverage.open_cube(url, deps)
    try:
        schema = coverage.ITSLiveCubeSchema(ds)
        missing = [name for name in args.variables if name not in ds]
        if missing:
            raise RuntimeError(f"Cube {cid} is missing requested variables: {', '.join(missing)}")
        specs = [variable_spec(ds[name], schema) for name in args.variables]
        selected = selected_for_cube(ds, schema, args, deps, replay)
        cube_crs = schema.crs(deps)
        subset, mask = coverage.coordinate_subset(ds, schema, roi.in_crs(cube_crs, deps), deps)
        nobs, ny, nx = len(selected), int(subset.sizes[schema.y]), int(subset.sizes[schema.x])
        estimated = 0
        for spec in specs:
            count = nobs * ny * nx if spec["kind"] == "spatial" else nobs
            estimated += count * deps["np"].dtype(spec["dtype"]).itemsize
        estimated += ny * nx  # uint8 ROI mask
        rows = selected.copy()
        rows.insert(0, "source_cube_url", url)
        rows.insert(0, "source_cube_id", cid)
        plan = {
            "id": cid, "url": url, "crs": cube_crs.to_string(),
            "observation_indices": selected["observation_index"].astype(int).tolist(),
            "shape": {"observation": nobs, "y": ny, "x": nx},
            "x_min": float(subset[schema.x].values.min()),
            "x_max": float(subset[schema.x].values.max()),
            "y_min": float(subset[schema.y].values.min()),
            "y_max": float(subset[schema.y].values.max()),
            "variables": specs, "estimated_uncompressed_bytes": estimated,
        }
        return plan, rows
    finally:
        ds.close()


def preflight_cube(
    cube: dict[str, Any], position: int, roi: coverage.ROI, args: argparse.Namespace,
    deps: dict[str, Any], replay: dict[str, Any] | None,
) -> tuple[dict[str, Any], Any]:
    """Preflight one cube, reopening metadata after transient remote failures."""
    cid = cube_id(cube, position)
    max_attempts = 6
    for attempt in range(1, max_attempts + 1):
        try:
            return _preflight_cube_once(cube, position, roi, args, deps, replay)
        except Exception as exc:
            if not multi.transient_remote_error(exc) or attempt == max_attempts:
                raise
            reset_remote_filesystems()
            delay = min(30, 2**attempt)
            print(
                f"[{cid}] transient metadata {type(exc).__name__}; "
                f"retrying in {delay}s (attempt {attempt + 1}/{max_attempts})",
                flush=True,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def build_preflight(
    roi: coverage.ROI, cubes: list[dict[str, Any]], args: argparse.Namespace,
    deps: dict[str, Any], replay: dict[str, dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], Any, int]:
    frames, plans = [], []
    for position, cube in enumerate(cubes):
        cid = cube_id(cube, position)
        plan, rows = preflight_cube(
            cube, position, roi, args, deps, None if replay is None else replay.get(cid, [])
        )
        plans.append(plan)
        if len(rows):
            frames.append(rows)
    selected = deps["pd"].concat(frames, ignore_index=True) if frames else deps["pd"].DataFrame()
    if len(selected):
        pair_id = selected["image_pair_id"].fillna("").astype(str).str.strip()
        fallback_columns = [
            "mid_date", "date_img1", "date_img2", "pair_days",
            "mission_img1_normalized", "mission_img2_normalized",
            "satellite_img1", "satellite_img2",
        ]
        fallback = selected[fallback_columns].fillna("").astype(str).agg("|".join, axis=1)
        selected["observation_key"] = pair_id.where(pair_id.ne(""), "metadata:" + fallback)
        selected["cube_observation"] = [
            json.dumps(
                {"cube_id": str(cid), "observation_index": int(index)},
                separators=(",", ":"),
            )
            for cid, index in zip(selected["source_cube_id"], selected["observation_index"])
        ]
        selected["observation_cube_count"] = selected.groupby("observation_key")[
            "source_cube_id"
        ].transform("nunique")
    total = sum(plan["estimated_uncompressed_bytes"] for plan in plans)
    return plans, selected, total


def plan_fingerprint(
    plans: list[dict[str, Any]], roi: coverage.ROI, args: argparse.Namespace, deps: dict[str, Any]
) -> str:
    return hash_payload({
        "format": 1,
        "cubes": plans,
        "roi": {"crs": roi.crs.to_string(), "geometry": deps["mapping"](roi.geometry)},
        "variables": args.variables,
        "observation_block_size": args.observation_block_size,
        "spatial_block_size": args.spatial_block_size,
        "mask_outside_roi": True,
        "zarr_format": 2,
    })


def block_keys(shape: dict[str, int], obs_block: int, spatial_block: int) -> Iterable[tuple[int, int, int, int, int, int]]:
    for o0 in range(0, shape["observation"], obs_block):
        for y0 in range(0, shape["y"], spatial_block):
            for x0 in range(0, shape["x"], spatial_block):
                yield (
                    o0, min(o0 + obs_block, shape["observation"]),
                    y0, min(y0 + spatial_block, shape["y"]),
                    x0, min(x0 + spatial_block, shape["x"]),
                )


def block_name(block: tuple[int, int, int, int, int, int]) -> str:
    return ":".join(map(str, block))


def missing_value(dtype: Any, attrs: dict[str, Any], np: Any) -> Any:
    dtype = np.dtype(dtype)
    if dtype.kind in "fc":
        return np.nan
    for key in ("_FillValue", "missing_value"):
        if key in attrs:
            try:
                return np.asarray(attrs[key], dtype=dtype).item()
            except (TypeError, ValueError):
                pass
    if dtype.kind == "b":
        return False
    if dtype.kind in "iu":
        return np.iinfo(dtype).min
    raise RuntimeError(f"No missing value policy for dtype {dtype}")


def create_array(group: Any, name: str, data: Any, dimensions: list[str], **kwargs: Any) -> Any:
    attrs = dict(kwargs.pop("attributes", {}))
    attrs["_ARRAY_DIMENSIONS"] = dimensions
    kwargs.setdefault("fill_value", None)
    return group.create_array(name, data=data, attributes=attrs, **kwargs)


def initialize_group(
    group: Any, plan: dict[str, Any], subset: Any, schema: coverage.ITSLiveCubeSchema,
    selected: Any, mask: Any, args: argparse.Namespace, np: Any, compressor: Any,
) -> None:
    nobs, ny, nx = (plan["shape"][key] for key in ("observation", "y", "x"))
    group.attrs.update({
        "source_cube_id": plan["id"], "source_cube_url": plan["url"],
        "crs": plan["crs"], "masking": "values outside roi_mask are missing",
    })
    create_array(group, "observation", np.arange(nobs, dtype=np.int64), ["observation"])
    create_array(group, "source_observation_index", np.asarray(plan["observation_indices"], dtype=np.int64), ["observation"])
    create_array(group, "x", np.asarray(subset[schema.x].values), ["x"], attributes=json_attrs(dict(subset[schema.x].attrs)))
    create_array(group, "y", np.asarray(subset[schema.y].values), ["y"], attributes=json_attrs(dict(subset[schema.y].attrs)))
    create_array(group, "roi_mask", np.asarray(mask, dtype=np.uint8), ["y", "x"], chunks=(min(args.spatial_block_size, ny), min(args.spatial_block_size, nx)), compressor=compressor, attributes={"long_name": "pixel center is inside requested ROI"})

    metadata = {
        "mid_date": ("mid_date", "datetime64[ns]"),
        "date_img1": ("date_img1", "datetime64[ns]"),
        "date_img2": ("date_img2", "datetime64[ns]"),
        "pair_days": ("pair_days", "float64"),
        "mission_img1": ("mission_img1", None), "mission_img2": ("mission_img2", None),
        "satellite_img1": ("satellite_img1", None), "satellite_img2": ("satellite_img2", None),
        "sensor_img1": ("sensor_img1", None), "sensor_img2": ("sensor_img2", None),
        "mission": ("mission", None), "image_pair_id": ("granule_url", None),
    }
    for column, (output_name, forced_dtype) in metadata.items():
        if column not in selected:
            continue
        values = selected[column].to_numpy()
        if forced_dtype is not None:
            values = values.astype(forced_dtype)
        elif values.dtype.kind not in "SU":
            strings = ["" if value is None or (isinstance(value, float) and math.isnan(value)) else str(value) for value in values]
            width = max([1] + [len(value) for value in strings])
            values = np.asarray(strings, dtype=f"<U{width}")
        create_array(group, output_name, values, ["observation"])

    for spec in plan["variables"]:
        dtype = np.dtype(spec["dtype"])
        fill = missing_value(dtype, spec["attributes"], np)
        shape = (nobs, ny, nx) if spec["kind"] == "spatial" else (nobs,)
        chunks = (
            (min(args.observation_block_size, max(1, nobs)), min(args.spatial_block_size, ny), min(args.spatial_block_size, nx))
            if spec["kind"] == "spatial" else (min(args.observation_block_size, max(1, nobs)),)
        )
        attrs = dict(spec["attributes"])
        attrs["source_dimensions"] = spec["source_dimensions"]
        group.create_array(
            spec["name"], shape=shape, dtype=dtype, chunks=chunks,
            compressor=compressor, fill_value=fill,
            attributes={**attrs, "_ARRAY_DIMENSIONS": spec["dimensions"]},
        )


def _export_cube_once(payload: dict[str, Any]) -> dict[str, Any]:
    args = argparse.Namespace(**payload["args"])
    plan = payload["plan"]
    fingerprint = payload["fingerprint"]
    outdir = Path(payload["outdir"])
    deps = coverage.require_dependencies()
    np = deps["np"]
    import zarr
    from numcodecs import Blosc

    status_path = outdir / "checkpoints" / f"{plan['id']}.json"
    state = None
    if status_path.exists():
        candidate = read_json(status_path)
        if candidate.get("fingerprint") != fingerprint:
            raise RuntimeError(f"Checkpoint fingerprint mismatch for cube {plan['id']}.")
        state = candidate
    completed = set(state.get("completed_blocks", [])) if state else set()

    ds = coverage.open_cube(plan["url"], deps)
    try:
        schema = coverage.ITSLiveCubeSchema(ds)
        roi = coverage.ROI(
            deps["shape"](payload["roi"]["geometry"]),
            deps["CRS"].from_user_input(payload["roi"]["crs"]), "saved", [],
        )
        subset, mask = coverage.coordinate_subset(ds, schema, roi.in_crs(schema.crs(deps), deps), deps)
        table = coverage.observation_table(ds, schema, deps)
        indices = np.asarray(plan["observation_indices"], dtype=np.int64)
        selected = table.iloc[indices].copy() if len(indices) else table.iloc[0:0].copy()

        root = zarr.open_group(outdir / STORE_NAME, mode="a", zarr_format=2)
        cubes_group = root.require_group("cubes")
        group_exists = plan["id"] in cubes_group
        if state is not None and not group_exists:
            state = None
            completed = set()
        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
        if state is None or not state.get("initialized", False):
            if plan["id"] in cubes_group:
                del cubes_group[plan["id"]]
            group = cubes_group.create_group(plan["id"])
            initialize_group(group, plan, subset, schema, selected, mask, args, np, compressor)
            state = {
                "status": "running", "fingerprint": fingerprint,
                "cube_id": plan["id"], "initialized": True,
                "completed_blocks": [], "updated_utc": now_utc(),
            }
            completed = set()
            atomic_json(status_path, state)
        else:
            group = cubes_group[plan["id"]]

        spatial_specs = [spec for spec in plan["variables"] if spec["kind"] == "spatial"]
        observation_specs = [spec for spec in plan["variables"] if spec["kind"] == "observation"]
        for spec in observation_specs:
            key = f"observation:{spec['name']}"
            if key not in completed:
                values = ds[spec["name"]].isel({schema.obs: indices}).values
                group[spec["name"]][:] = values
                completed.add(key)
                state["completed_blocks"] = sorted(completed)
                state["updated_utc"] = now_utc()
                atomic_json(status_path, state)

        shape = plan["shape"]
        for y0 in range(0, shape["y"], args.spatial_block_size):
            y1 = min(y0 + args.spatial_block_size, shape["y"])
            for x0 in range(0, shape["x"], args.spatial_block_size):
                x1 = min(x0 + args.spatial_block_size, shape["x"])
                pending = []
                for o0 in range(0, shape["observation"], args.observation_block_size):
                    o1 = min(o0 + args.observation_block_size, shape["observation"])
                    block = (o0, o1, y0, y1, x0, x1)
                    if block_name(block) not in completed:
                        pending.append(block)
                if not pending:
                    continue
                block_mask = np.asarray(mask[y0:y1, x0:x1], dtype=bool)
                # Remote ITS_LIVE velocity chunks span the complete observation
                # dimension. Read every selected observation once for this spatial
                # block, then split only the local writes into observation chunks.
                for spec in spatial_specs:
                    source = subset[spec["name"]].transpose(schema.obs, schema.y, schema.x)
                    values = source.isel({
                        schema.obs: indices,
                        schema.y: slice(y0, y1),
                        schema.x: slice(x0, x1),
                    }).values
                    fill = missing_value(values.dtype, spec["attributes"], np)
                    values = np.where(block_mask[None, :, :], values, fill)
                    for o0, o1, _, _, _, _ in pending:
                        group[spec["name"]][o0:o1, y0:y1, x0:x1] = values[o0:o1]
                for block in pending:
                    completed.add(block_name(block))
                state["completed_blocks"] = sorted(completed)
                state["updated_utc"] = now_utc()
                atomic_json(status_path, state)

        state.update({"status": "complete", "completed_utc": now_utc()})
        atomic_json(status_path, state)
        return {"id": plan["id"], "status": "complete", "completed_blocks": len(completed)}
    except Exception as exc:
        failure = state or {"fingerprint": fingerprint, "cube_id": plan["id"], "completed_blocks": []}
        failure.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}", "updated_utc": now_utc()})
        atomic_json(status_path, failure)
        raise
    finally:
        ds.close()


def reset_remote_filesystems() -> None:
    """Discard cached fsspec clients after an asynchronous transport failure."""
    try:
        import fsspec

        fsspec.AbstractFileSystem.clear_instance_cache()
    except Exception:
        # Cache clearing is best-effort; the next open still gets a fresh Dataset.
        pass


def export_cube_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Export one cube, reopening it after transient remote failures."""
    plan = payload["plan"]
    max_attempts = 6
    for attempt in range(1, max_attempts + 1):
        try:
            return _export_cube_once(payload)
        except Exception as exc:
            if not multi.transient_remote_error(exc) or attempt == max_attempts:
                raise
            reset_remote_filesystems()
            delay = min(30, 2**attempt)
            print(
                f"[{plan['id']}] transient {type(exc).__name__}; "
                f"resuming from its last completed block in {delay}s "
                f"(attempt {attempt + 1}/{max_attempts})",
                flush=True,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def prepare_output(outdir: Path, overwrite: bool, resume: bool) -> None:
    known = [
        outdir / STORE_NAME,
        outdir / MANIFEST_NAME,
        outdir / EXPORT_CONFIG_NAME,
        outdir / EXPORT_INVENTORY_NAME,
        outdir / "checkpoints",
    ]
    existing = [path for path in known if path.exists()]
    if existing and not (overwrite or resume):
        raise RuntimeError("Output products already exist; use --resume or --overwrite: " + ", ".join(map(str, existing)))
    if overwrite:
        for path in existing:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    outdir.mkdir(parents=True, exist_ok=True)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        deps = coverage.require_dependencies()
        if args.coverage_results is not None:
            roi, cubes, source = replay_selection(args.coverage_results, deps)
            replay = replay_indices(args.coverage_results, cubes, deps)
        else:
            roi, cubes, source = standalone_selection(args, deps)
            replay = None
        if not cubes:
            raise RuntimeError("No ITS_LIVE cubes intersect the requested ROI.")

        print(f"Preflighting {len(cubes)} cube(s)...")
        plans, selected, estimated = build_preflight(roi, cubes, args, deps, replay)
        print(f"Selected cube-observation records: {len(selected):,}")
        print(f"Estimated uncompressed export size: {human_bytes(estimated)}")
        enforce_size_limit(estimated, args.allow_large)
        fingerprint = plan_fingerprint(plans, roi, args, deps)
        if args.dry_run:
            print(f"Dry run complete. Export fingerprint: {fingerprint}")
            return 0

        prepare_output(args.outdir, args.overwrite, args.resume)
        existing_manifest = args.outdir / MANIFEST_NAME
        if args.resume and existing_manifest.exists():
            previous = read_json(existing_manifest)
            if previous.get("fingerprint") != fingerprint:
                raise RuntimeError("Existing export is incompatible with this request; use a new outdir or --overwrite.")
            if previous.get("store") and not (args.outdir / STORE_NAME).is_dir():
                raise RuntimeError("The export manifest exists but velocity.zarr is missing; use --overwrite.")

        run_config = {
            "arguments": vars(args), "created_utc": now_utc(), "fingerprint": fingerprint,
            "input_roi": {"kind": roi.input_kind, "bounds": roi.input_bounds, "crs": roi.crs.to_string(), "geometry": deps["mapping"](roi.geometry)},
            "coverage_source": source,
        }
        manifest = {
            "status": "empty" if len(selected) == 0 else "running",
            "fingerprint": fingerprint, "created_utc": now_utc(),
            "estimated_uncompressed_bytes": estimated,
            "coverage_source_fingerprint": None if source is None else source.get("fingerprint"),
            "store": None if len(selected) == 0 else STORE_NAME,
            "cubes": plans, "results": [],
        }
        atomic_json(args.outdir / EXPORT_CONFIG_NAME, run_config)
        selected.to_csv(args.outdir / EXPORT_INVENTORY_NAME, index=False)
        atomic_json(existing_manifest, manifest)
        if len(selected) == 0:
            print("No observations selected; wrote metadata without creating a Zarr store.")
            return 0

        import zarr
        root = zarr.open_group(args.outdir / STORE_NAME, mode="a", zarr_format=2)
        root.attrs.update({
            "title": "Filtered ITS_LIVE image-pair velocities",
            "export_fingerprint": fingerprint, "created_utc": now_utc(),
            "group_layout": "cubes/<source cube id>",
        })
        root.require_group("cubes")
        payload_args = {
            "variables": args.variables,
            "observation_block_size": args.observation_block_size,
            "spatial_block_size": args.spatial_block_size,
        }
        roi_payload = {"crs": roi.crs.to_string(), "geometry": deps["mapping"](roi.geometry)}
        payloads = [{
            "plan": plan, "args": payload_args, "fingerprint": fingerprint,
            "outdir": str(args.outdir), "roi": roi_payload,
        } for plan in plans if plan["shape"]["observation"] > 0]
        results, failures = [], []
        with ProcessPoolExecutor(
            max_workers=args.workers,
            max_tasks_per_child=1,
        ) as executor:
            futures = {executor.submit(export_cube_worker, payload): payload["plan"]["id"] for payload in payloads}
            for future in as_completed(futures):
                cid = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"[{cid}] complete")
                except Exception as exc:
                    failures.append({"id": cid, "error": f"{type(exc).__name__}: {exc}"})
                    print(f"[{cid}] failed: {exc}", file=sys.stderr)
        manifest["results"] = sorted(results, key=lambda item: item["id"])
        manifest["failures"] = sorted(failures, key=lambda item: item["id"])
        if failures:
            manifest["status"] = "failed"
            manifest["updated_utc"] = now_utc()
            atomic_json(existing_manifest, manifest)
            raise RuntimeError(f"{len(failures)} cube export(s) failed; rerun the same command with --resume.")
        zarr.consolidate_metadata(args.outdir / STORE_NAME, zarr_format=2)
        manifest["status"] = "complete"
        manifest["completed_utc"] = now_utc()
        atomic_json(existing_manifest, manifest)
        print(f"Complete. Wrote {args.outdir / STORE_NAME}")
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
