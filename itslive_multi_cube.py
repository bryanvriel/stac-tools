"""Resumable multi-cube orchestration for :mod:`itslive_cube_coverage`."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import itslive_cube_coverage as coverage


ASF_SEARCH_URL = "https://api.daac.asf.alaska.edu/services/search/param"
SENTINEL1_SCENE_PATTERN = re.compile(
    r"S1[AB]_[A-Z]{2}_SLC_{1,2}1S[A-Z]{2}_"
    r"\d{8}T\d{6}_\d{8}T\d{6}_\d{6}_[0-9A-F]{6}_[0-9A-F]{4}"
)


def fingerprint(payload: Any) -> str:
    encoded = json.dumps(coverage.json_value(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def discover_intersecting_cubes(roi: coverage.ROI, deps: dict[str, Any]) -> list[dict[str, Any]]:
    from shapely.geometry import box

    wgs84 = deps["CRS"].from_epsg(4326)
    roi_wgs84 = roi.in_crs(wgs84, deps)
    # STAC cube footprints use sparse WGS84 corner polygons. Near a projected
    # 100 km tile edge, their straight lon/lat chords can miss a genuinely
    # intersecting narrow ROI. Broaden only candidate discovery, then perform
    # the authoritative test against proj:bbox in the cube CRS below.
    candidate_geometry = roi_wgs84.buffer(0.15)
    catalog = deps["Client"].open(coverage.STAC_URL)
    items = list(
        catalog.search(
            collections=[coverage.COLLECTION],
            intersects=deps["mapping"](candidate_geometry),
        ).items()
    )
    cubes = []
    for item in items:
        proj_bbox = item.properties.get("proj:bbox")
        proj_code = item.properties.get("proj:code")
        if proj_bbox and proj_code:
            item_crs = deps["CRS"].from_user_input(proj_code)
            exact_roi = roi.in_crs(item_crs, deps)
            intersects = box(*proj_bbox).intersection(exact_roi).area > 0
        else:
            intersects = deps["shape"](item.geometry).intersects(roi_wgs84)
        if not intersects:
            continue
        cubes.append(
            {
                "id": item.id,
                "url": coverage.item_asset_url(item),
                "bbox_wgs84": item.bbox,
                "geometry_wgs84": item.geometry,
                "proj_bbox": proj_bbox,
                "proj_code": proj_code,
                "granule_count": item.properties.get("granule_count"),
                "start_datetime": item.properties.get("start_datetime"),
                "end_datetime": item.properties.get("end_datetime"),
            }
        )
    return sorted(cubes, key=lambda cube: cube["id"])


def tile_expected_paths(tile_dir: Path) -> list[Path]:
    return [
        tile_dir / "coverage_count.nc",
        tile_dir / "coverage_count.tif",
        tile_dir / "coverage_fraction.tif",
        tile_dir / "selected_observation_count.tif",
        tile_dir / "selected_observations.csv",
        tile_dir / "summary.json",
        tile_dir / "cube_info.json",
        tile_dir / "tile_status.json",
        tile_dir / "mosaic_source" / "coverage_count.tif",
        tile_dir / "mosaic_source" / "selected_observation_count.tif",
    ]


def checkpoint_result(tile_dir: Path, expected_fingerprint: str) -> dict[str, Any] | None:
    status_path = tile_dir / "tile_status.json"
    if not status_path.exists():
        return None
    try:
        status = json.loads(status_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if status.get("status") != "complete" or status.get("fingerprint") != expected_fingerprint:
        return None
    if not all(path.exists() for path in tile_expected_paths(tile_dir)):
        return None
    cache_path = status.get("result", {}).get("observation_cache")
    if cache_path and not Path(cache_path).is_file():
        return None
    return status["result"]


def clean_tile(tile_dir: Path, preserve_work: bool = False) -> None:
    tile_dir.mkdir(parents=True, exist_ok=True)
    for path in tile_expected_paths(tile_dir):
        if path.exists():
            path.unlink()
    for name in ("cube_schema.json", "cube_schema.txt", "coverage_map.png"):
        path = tile_dir / name
        if path.exists():
            path.unlink()
    if not preserve_work:
        work_dir = tile_dir / "work"
        for name in (
            "coverage_count.npy",
            "block_status.json",
            "block_status.json.tmp",
            "observation_cache.h5",
        ):
            path = work_dir / name
            if path.exists():
                path.unlink()


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write a restart checkpoint without exposing a partially written JSON file."""
    temporary = path.with_name(path.name + ".tmp")
    coverage.write_json(temporary, payload)
    temporary.replace(path)


def transient_remote_error(exc: Exception) -> bool:
    return isinstance(exc, (OSError, TimeoutError)) or type(exc).__name__ in {
        "ClientConnectionError",
        "ClientConnectionResetError",
        "ClientOSError",
        "ClientPayloadError",
        "ContentLengthError",
        "CubeOpenError",
        "ServerDisconnectedError",
        "FSTimeoutError",
        "IncompleteCubeMetadataError",
    }


def scalar_count_geotiff(count_path: Path, output_path: Path, value: int) -> None:
    import numpy as np
    import rasterio

    with rasterio.open(count_path) as source:
        count = source.read(1)
        valid = count != source.nodata
        result = np.full(source.shape, coverage.COUNT_NODATA, dtype=np.uint32)
        result[valid] = np.uint32(value)
        profile = source.profile.copy()
        profile.update(dtype="uint32", nodata=coverage.COUNT_NODATA)
    with rasterio.open(output_path, "w", **profile) as destination:
        destination.write(result, 1)


def zero_coverage(subset: Any, schema: Any, roi_mask: Any, deps: dict[str, Any]) -> tuple[Any, Any]:
    np, xr = deps["np"], deps["xr"]
    count_values = np.where(roi_mask, 0, coverage.COUNT_NODATA).astype(np.uint32)
    fraction_values = np.full(roi_mask.shape, np.nan, dtype=np.float32)
    coords = {schema.y: subset[schema.y], schema.x: subset[schema.x]}
    count = xr.DataArray(
        count_values,
        dims=(schema.y, schema.x),
        coords=coords,
        name="valid_count",
        attrs={
            "long_name": "number of valid ITS_LIVE velocity observations",
            "validity_definition": "no observations selected; count is zero inside ROI",
        },
    )
    fraction = xr.DataArray(
        fraction_values,
        dims=(schema.y, schema.x),
        coords=coords,
        name="valid_fraction",
        attrs={"long_name": "fraction undefined because no observations were selected", "units": "1"},
    )
    return count, fraction


def apply_output_mask(count: Any, fraction: Any, roi_mask: Any, deps: dict[str, Any]) -> tuple[Any, Any]:
    np, xr = deps["np"], deps["xr"]
    count_values = np.where(roi_mask, count.values, coverage.COUNT_NODATA).astype(np.uint32)
    fraction_values = np.where(roi_mask, fraction.values, np.nan).astype(np.float32)
    masked_count = xr.DataArray(
        count_values,
        dims=count.dims,
        coords=count.coords,
        name=count.name,
        attrs=count.attrs,
    )
    masked_fraction = xr.DataArray(
        fraction_values,
        dims=fraction.dims,
        coords=fraction.coords,
        name=fraction.name,
        attrs=fraction.attrs,
    )
    return masked_count, masked_fraction


def compute_coverage_blocked(
    subset: Any,
    schema: Any,
    selected: Any,
    tile_dir: Path,
    tile_fingerprint: str,
    block_size: int,
    deps: dict[str, Any],
    cache_observations: bool = False,
) -> tuple[Any, Any]:
    """Compute a tile in durable spatial blocks, retrying only failed reads."""
    np, xr = deps["np"], deps["xr"]
    ny = int(subset.sizes[schema.y])
    nx = int(subset.sizes[schema.x])
    work_dir = tile_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    count_path = work_dir / "coverage_count.npy"
    status_path = work_dir / "block_status.json"
    cache_path = work_dir / "observation_cache.h5"
    velocity_name = (
        schema.velocity["v"] if "v" in schema.velocity else None
    )
    error_name = (
        f"{velocity_name}_error"
        if velocity_name is not None and f"{velocity_name}_error" in subset
        else None
    )
    work_fingerprint = fingerprint(
        {
            "tile": tile_fingerprint,
            "shape": [ny, nx],
            "selected_observation_indices": selected["observation_index"].tolist(),
            "block_size": block_size,
            "cache_observations": cache_observations,
            "cache_velocity_variable": velocity_name,
            "cache_error_variable": error_name,
        }
    )
    state = None
    if status_path.exists() and count_path.exists() and (
        not cache_observations or cache_path.exists()
    ):
        try:
            candidate = json.loads(status_path.read_text())
            if candidate.get("fingerprint") == work_fingerprint:
                state = candidate
        except (OSError, json.JSONDecodeError):
            pass
    if state is None:
        count_memmap = np.lib.format.open_memmap(
            count_path, mode="w+", dtype=np.uint32, shape=(ny, nx)
        )
        count_memmap[:] = 0
        count_memmap.flush()
        state = {
            "status": "running",
            "fingerprint": work_fingerprint,
            "shape": [ny, nx],
            "block_size": block_size,
            "completed_blocks": [],
        }
        atomic_write_json(status_path, state)
    else:
        count_memmap = np.lib.format.open_memmap(count_path, mode="r+")

    cache = None
    cache_recreated = False
    if cache_observations:
        if velocity_name is None:
            raise RuntimeError(
                "Observation caching requires the scalar ITS_LIVE velocity variable 'v'."
            )
        import h5py

        cache = h5py.File(cache_path, "a")
        if cache.attrs.get("fingerprint", "") != work_fingerprint:
            cache_recreated = True
            cache.close()
            cache_path.unlink(missing_ok=True)
            cache = h5py.File(cache_path, "w")
            cache.attrs.update(
                {
                    "cache_version": 1,
                    "fingerprint": work_fingerprint,
                    "velocity_variable": velocity_name,
                    "error_variable": error_name or "",
                    "observation_count": len(selected),
                }
            )
            cache.create_dataset(
                "observation_index",
                data=selected["observation_index"].to_numpy(dtype=np.int64),
            )
            cache.create_dataset("x", data=np.asarray(subset[schema.x].values))
            cache.create_dataset("y", data=np.asarray(subset[schema.y].values))
            cache.create_dataset(
                "validity",
                shape=(len(selected), ny, nx),
                dtype=np.bool_,
                chunks=(min(32, len(selected)), min(block_size, ny), min(block_size, nx)),
                compression="gzip",
                compression_opts=1,
                shuffle=True,
            )
            if error_name is not None:
                error = subset[error_name]
                spatial_error = schema.y in error.dims and schema.x in error.dims
                cache.attrs["error_is_spatial"] = spatial_error
                error_shape = (len(selected), ny, nx) if spatial_error else (len(selected),)
                error_chunks = (
                    (min(32, len(selected)), min(block_size, ny), min(block_size, nx))
                    if spatial_error
                    else None
                )
                cache.create_dataset(
                    "uncertainty",
                    shape=error_shape,
                    dtype=np.float32,
                    chunks=error_chunks,
                    compression="gzip",
                    compression_opts=1,
                    shuffle=True,
                )
                if not spatial_error:
                    indices = selected["observation_index"].to_numpy(dtype=np.int64)
                    values = deps["dask_array"].compute(
                        error.isel({schema.obs: indices}).transpose(schema.obs).data
                    )[0]
                    cache["uncertainty"][:] = np.asarray(values, dtype=np.float32)
            cache.flush()
        if cache_recreated and state.get("completed_blocks"):
            state["status"] = "running"
            state["completed_blocks"] = []
            state["completed_count"] = 0
            atomic_write_json(status_path, state)

    blocks = [
        (y0, min(y0 + block_size, ny), x0, min(x0 + block_size, nx))
        for y0 in range(0, ny, block_size)
        for x0 in range(0, nx, block_size)
    ]
    completed = set(state.get("completed_blocks", []))
    if completed:
        print(
            f"[{tile_dir.name}] resuming {len(completed):,}/{len(blocks):,} "
            "completed spatial blocks"
        )
    count_attrs = None
    try:
        for number, (y0, y1, x0, x1) in enumerate(blocks, start=1):
            block_id = f"{y0}:{y1},{x0}:{x1}"
            if block_id in completed:
                continue
            block = subset.isel(
                {schema.y: slice(y0, y1), schema.x: slice(x0, x1)}
            )
            block_mask = np.ones((y1 - y0, x1 - x0), dtype=bool)
            for attempt in range(1, 7):
                try:
                    if cache is None:
                        block_count, _ = coverage.compute_coverage(
                            block,
                            schema,
                            selected,
                            block_mask,
                            deps,
                            show_progress=False,
                        )
                    else:
                        indices = selected["observation_index"].to_numpy(dtype=np.int64)
                        selected_block = block.isel({schema.obs: indices})
                        velocity = selected_block[velocity_name].transpose(
                            schema.obs, schema.y, schema.x
                        )
                        valid = coverage.finite_valid(velocity, deps)
                        tasks = [valid.data]
                        spatial_error = False
                        if error_name is not None:
                            error = selected_block[error_name]
                            spatial_error = schema.y in error.dims and schema.x in error.dims
                            if spatial_error:
                                error = error.transpose(schema.obs, schema.y, schema.x)
                                tasks.append(error.data)
                        computed = deps["dask_array"].compute(*tasks)
                        valid_values = np.asarray(computed[0], dtype=bool)
                        count_values = valid_values.sum(axis=0, dtype=np.uint32)
                        block_count = xr.DataArray(
                            count_values,
                            dims=(schema.y, schema.x),
                            coords={schema.y: block[schema.y], schema.x: block[schema.x]},
                            attrs={
                                "long_name": "number of valid ITS_LIVE velocity observations",
                                "validity_definition": (
                                    f"finite {velocity_name} excluding declared fill/missing values"
                                ),
                            },
                        )
                        cache["validity"][:, y0:y1, x0:x1] = valid_values
                        if error_name is not None and spatial_error:
                            cache["uncertainty"][:, y0:y1, x0:x1] = np.asarray(
                                computed[1], dtype=np.float32
                            )
                        cache.flush()
                    break
                except Exception as exc:
                    if not transient_remote_error(exc) or attempt == 6:
                        raise
                    delay = min(30, 2**attempt)
                    print(
                        f"[{tile_dir.name}] block {number}/{len(blocks)} transient "
                        f"{type(exc).__name__}; retrying in {delay}s "
                        f"(attempt {attempt + 1}/6)"
                    )
                    time.sleep(delay)
            count_memmap[y0:y1, x0:x1] = block_count.values
            count_memmap.flush()
            count_attrs = block_count.attrs
            completed.add(block_id)
            state["completed_blocks"] = sorted(completed)
            state["completed_count"] = len(completed)
            state["total_blocks"] = len(blocks)
            atomic_write_json(status_path, state)
            print(f"[{tile_dir.name}] spatial block {number}/{len(blocks)} complete")
    finally:
        if cache is not None:
            cache.close()

    count_values = np.array(count_memmap, dtype=np.uint32, copy=True)
    del count_memmap
    if count_attrs is None:
        if "v" in schema.velocity:
            definition = (
                f"finite {schema.velocity['v']} excluding declared fill/missing values"
            )
        else:
            definition = (
                f"finite {schema.velocity['vx']} AND {schema.velocity['vy']} "
                "excluding fill/missing values"
            )
        count_attrs = {
            "long_name": "number of valid ITS_LIVE velocity observations",
            "validity_definition": definition,
        }
    coords = {schema.y: subset[schema.y], schema.x: subset[schema.x]}
    count = xr.DataArray(
        count_values,
        dims=(schema.y, schema.x),
        coords=coords,
        name="valid_count",
        attrs=count_attrs,
    )
    fraction = xr.DataArray(
        (count_values.astype(np.float64) / len(selected)).astype(np.float32),
        dims=(schema.y, schema.x),
        coords=coords,
        name="valid_fraction",
        attrs={
            "long_name": "fraction of selected observations with valid velocity",
            "units": "1",
        },
    )
    state["status"] = "complete"
    atomic_write_json(status_path, state)
    return count, fraction


def process_tile(
    cube: dict[str, Any],
    args: Any,
    roi: coverage.ROI,
    tile_dir: Path,
    tile_fingerprint: str,
    deps: dict[str, Any],
) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    coverage.write_json(
        tile_dir / "tile_status.json",
        {
            "status": "running",
            "fingerprint": tile_fingerprint,
            "cube_id": cube["id"],
            "started_utc": started,
        },
    )
    try:
        print(f"[{cube['id']}] opening metadata")
        ds = coverage.open_cube(cube["url"], deps)
        schema = coverage.ITSLiveCubeSchema(ds)
        cube_crs = schema.crs(deps)
        roi_cube = roi.in_crs(cube_crs, deps)
        table = coverage.observation_table(ds, schema, deps)
        selected, filter_counts, date_meta = coverage.filter_observations(table, args, deps)
        selected.insert(0, "source_cube_id", cube["id"])
        selected.insert(1, "source_cube_url", cube["url"])
        selected.to_csv(tile_dir / "selected_observations.csv", index=False)

        subset, roi_mask = coverage.coordinate_subset(ds, schema, roi_cube, deps)
        envelope_mask = deps["np"].ones_like(roi_mask, dtype=bool)
        if selected.empty:
            source_count, source_fraction = zero_coverage(
                subset, schema, envelope_mask, deps
            )
        else:
            source_count, source_fraction = compute_coverage_blocked(
                subset,
                schema,
                selected,
                tile_dir,
                tile_fingerprint,
                args.spatial_block_size,
                deps,
                cache_observations=args.cache_observations,
            )
        count, fraction = apply_output_mask(
            source_count, source_fraction, roi_mask, deps
        )
        stats = coverage.summary_statistics(count, roi_mask, len(selected), deps)
        metadata = {
            "source": "NASA ITS_LIVE",
            "cube_url": cube["url"],
            "cube_id": cube["id"],
            "cube_crs": cube_crs.to_string(),
            "input_roi": {
                "kind": roi.input_kind,
                "bounds": roi.input_bounds,
                "crs": roi.crs.to_string(),
            },
            "start": args.start,
            "end": args.end,
            "mission_filter": args.mission or "all",
            "sensor_filter": args.sensor or "all",
            "min_pair_days": args.min_pair_days,
            "max_pair_days": args.max_pair_days,
            "validity": count.attrs["validity_definition"],
            "selected_observation_count": len(selected),
            "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        coverage.write_netcdf(
            count, fraction, roi_mask, schema, metadata, tile_dir, deps
        )
        coverage.write_geotiffs(
            count, fraction, schema, cube_crs, tile_dir, deps
        )
        scalar_count_geotiff(
            tile_dir / "coverage_count.tif",
            tile_dir / "selected_observation_count.tif",
            len(selected),
        )
        mosaic_source_dir = tile_dir / "mosaic_source"
        mosaic_source_dir.mkdir(parents=True, exist_ok=True)
        coverage.write_geotiffs(
            source_count,
            source_fraction,
            schema,
            cube_crs,
            mosaic_source_dir,
            deps,
        )
        scalar_count_geotiff(
            mosaic_source_dir / "coverage_count.tif",
            mosaic_source_dir / "selected_observation_count.tif",
            len(selected),
        )
        summary = {
            "status": "complete",
            "cube": cube,
            "resolved_schema": schema.summary(deps),
            "filters": filter_counts,
            "date_filter": date_meta,
            **stats,
        }
        coverage.write_json(tile_dir / "summary.json", summary)
        coverage.write_json(
            tile_dir / "cube_info.json",
            {**cube, "resolved_schema": schema.summary(deps)},
        )
        result = {
            "cube": cube,
            "tile_dir": str(tile_dir),
            "crs": cube_crs.to_string(),
            "selected_observations": int(len(selected)),
            "roi_pixel_count": stats["roi_pixel_count"],
            "elapsed_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
        }
        cache_path = tile_dir / "work" / "observation_cache.h5"
        if args.cache_observations and cache_path.is_file():
            result["observation_cache"] = str(cache_path)
        coverage.write_json(
            tile_dir / "tile_status.json",
            {
                "status": "complete",
                "fingerprint": tile_fingerprint,
                "cube_id": cube["id"],
                "started_utc": started,
                "completed_utc": datetime.now(timezone.utc),
                "result": result,
            },
        )
        print(f"[{cube['id']}] complete ({len(selected):,} selected observations)")
        return result
    except Exception as exc:
        coverage.write_json(
            tile_dir / "tile_status.json",
            {
                "status": "failed",
                "fingerprint": tile_fingerprint,
                "cube_id": cube["id"],
                "started_utc": started,
                "failed_utc": datetime.now(timezone.utc),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        )
        raise


def process_tile_isolated(
    cube: dict[str, Any],
    arguments: dict[str, Any],
    tile_dir: str,
    tile_fingerprint: str,
) -> dict[str, Any]:
    """Process one cube in its own process to isolate HDF5 and fsspec state."""
    args = SimpleNamespace(**arguments)
    args.outdir = Path(args.outdir)
    deps = coverage.require_dependencies()
    roi = coverage.construct_roi(args, deps)
    clean_tile(Path(tile_dir), preserve_work=args.resume)
    last_error = None
    max_attempts = 6
    for attempt in range(1, max_attempts + 1):
        try:
            return process_tile(
                cube,
                args,
                roi,
                Path(tile_dir),
                tile_fingerprint,
                deps,
            )
        except Exception as exc:
            last_error = exc
            transient = transient_remote_error(exc)
            if not transient or attempt == max_attempts:
                raise
            delay = min(30, 2**attempt)
            print(
                f"[{cube['id']}] transient {type(exc).__name__}; "
                f"retrying tile in {delay}s "
                f"(attempt {attempt + 1}/{max_attempts})"
            )
            time.sleep(delay)
    raise last_error  # pragma: no cover


def deduplicate_observations(tile_results: list[dict[str, Any]], output: Path, deps: dict[str, Any]) -> dict[str, int]:
    pd = deps["pd"]
    frames = []
    for result in tile_results:
        path = Path(result["tile_dir"]) / "selected_observations.csv"
        frame = pd.read_csv(path)
        if len(frame):
            frames.append(frame)
    if not frames:
        pd.DataFrame(columns=["observation_key", "cube_count", "cube_ids", "cube_observations"]).to_csv(
            output, index=False
        )
        return {"cube_observation_records": 0, "unique_observations": 0}
    combined = pd.concat(frames, ignore_index=True)
    pair_id = combined["image_pair_id"].fillna("").astype(str).str.strip()
    fallback_columns = [
        "mid_date",
        "date_img1",
        "date_img2",
        "pair_days",
        "mission_img1_normalized",
        "mission_img2_normalized",
        "satellite_img1",
        "satellite_img2",
    ]
    fallback = combined[fallback_columns].fillna("").astype(str).agg("|".join, axis=1)
    combined["observation_key"] = pair_id.where(pair_id.ne(""), "metadata:" + fallback)
    rows = []
    for key, group in combined.groupby("observation_key", sort=True, dropna=False):
        row = group.iloc[0].to_dict()
        observations = [
            {"cube_id": str(cube_id), "observation_index": int(index)}
            for cube_id, index in zip(group.source_cube_id, group.observation_index)
        ]
        row["observation_key"] = key
        row["cube_count"] = int(group.source_cube_id.nunique())
        row["cube_ids"] = ";".join(sorted(group.source_cube_id.astype(str).unique()))
        row["cube_observations"] = json.dumps(observations, separators=(",", ":"))
        rows.append(row)
    deduplicated = pd.DataFrame(rows)
    deduplicated.to_csv(output, index=False)
    return {
        "cube_observation_records": int(len(combined)),
        "unique_observations": int(len(deduplicated)),
    }


def nearest_tile_sample(source_path: Path, target_x: Any, target_y: Any, np: Any) -> Any:
    import rasterio

    with rasterio.open(source_path) as source:
        data = source.read(1)
        x0 = source.transform.c + source.transform.a / 2
        y0 = source.transform.f + source.transform.e / 2
        cols = np.rint((target_x - x0) / source.transform.a).astype(int)
        rows = np.rint((target_y - y0) / source.transform.e).astype(int)
        cols = np.clip(cols, 0, source.width - 1)
        rows = np.clip(rows, 0, source.height - 1)
        return data[np.ix_(rows, cols)], source.nodata


def sentinel1_pair_scenes(pair_id: str) -> tuple[str, str]:
    scenes = SENTINEL1_SCENE_PATTERN.findall(str(pair_id))
    if len(scenes) != 2:
        raise RuntimeError(
            "Could not extract exactly two Sentinel-1 SLC identifiers from "
            f"image pair ID: {pair_id}"
        )
    return scenes[0], scenes[1]


def fetch_sentinel1_scene_footprints(
    scene_names: set[str], cache_path: Path
) -> dict[str, Any]:
    """Fetch ASF scene polygons in restartable batches."""
    import requests

    cached: dict[str, Any] = {}
    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text())
            cached = payload.get("scenes", {})
        except (OSError, json.JSONDecodeError):
            pass
    missing = sorted(scene_names - cached.keys())
    if cached:
        print(f"Loaded {len(cached):,} cached Sentinel-1 scene footprints.")
    for start in range(0, len(missing), 100):
        batch = missing[start : start + 100]
        response = None
        for attempt in range(1, 7):
            try:
                response = requests.post(
                    ASF_SEARCH_URL,
                    data={
                        "granule_list": ",".join(batch),
                        "processingLevel": "SLC",
                        "output": "geojson",
                    },
                    timeout=120,
                )
                response.raise_for_status()
                result = response.json()
                break
            except (requests.RequestException, ValueError) as exc:
                if attempt == 6:
                    raise RuntimeError(
                        f"ASF footprint query failed after 6 attempts: {exc}"
                    ) from exc
                delay = min(30, 2**attempt)
                print(
                    f"ASF footprint batch transient {type(exc).__name__}; "
                    f"retrying in {delay}s (attempt {attempt + 1}/6)"
                )
                time.sleep(delay)
        for feature in result.get("features", []):
            properties = feature.get("properties", {})
            scene = properties.get("sceneName")
            geometry = feature.get("geometry")
            if scene in scene_names and geometry:
                cached[scene] = geometry
        atomic_write_json(
            cache_path,
            {
                "source": ASF_SEARCH_URL,
                "description": "Sentinel-1 SLC scene footprints used for opportunity counts",
                "scenes": cached,
            },
        )
        print(
            f"Fetched Sentinel-1 footprint batch "
            f"{start // 100 + 1}/{math.ceil(len(missing) / 100)}."
        )
    unresolved = sorted(scene_names - cached.keys())
    if unresolved:
        raise RuntimeError(
            f"ASF returned no footprint for {len(unresolved)} Sentinel-1 scene(s); "
            f"first missing scene: {unresolved[0]}"
        )
    return cached


def scene_footprint_opportunity(
    inventory_path: Path,
    count: Any,
    roi_mask: Any,
    crs: Any,
    outdir: Path,
    footprint_buffer: float,
    deps: dict[str, Any],
) -> tuple[Any, Any, dict[str, Any]]:
    """Count unique Sentinel-1 pair scene-overlap footprints at every pixel."""
    import rasterio
    from rasterio.enums import MergeAlg
    from rasterio.features import rasterize
    from rasterio.transform import from_origin

    np, xr, pd = deps["np"], deps["xr"], deps["pd"]
    inventory = pd.read_csv(inventory_path)
    pair_scenes = [sentinel1_pair_scenes(value) for value in inventory.image_pair_id]
    scenes = {scene for pair in pair_scenes for scene in pair}
    footprints = fetch_sentinel1_scene_footprints(
        scenes, outdir / "sentinel1_scene_footprints.json"
    )
    pair_geometries = []
    transform_to_cube = deps["Transformer"].from_crs(
        deps["CRS"].from_epsg(4326), crs, always_xy=True
    )
    for first, second in pair_scenes:
        overlap = deps["shape"](footprints[first]).intersection(
            deps["shape"](footprints[second])
        )
        if overlap.is_empty:
            continue
        projected = deps["transform_geometry"](
            transform_to_cube.transform, overlap
        ).buffer(footprint_buffer)
        pair_geometries.append(projected)
    x, y = count.x.values, count.y.values
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    transform = from_origin(x[0] - dx / 2, y[0] + abs(dy) / 2, dx, abs(dy))
    opportunity_values = rasterize(
        ((deps["mapping"](geometry), 1) for geometry in pair_geometries),
        out_shape=count.shape,
        transform=transform,
        fill=0,
        dtype="uint32",
        all_touched=False,
        merge_alg=MergeAlg.add,
    )
    valid_count = np.asarray(count.values)
    violations = roi_mask & (valid_count != coverage.COUNT_NODATA) & (
        valid_count > opportunity_values
    )
    diagnostics = {
        "unique_pairs": int(len(pair_scenes)),
        "unique_source_scenes": int(len(scenes)),
        "nonempty_pair_overlaps": int(len(pair_geometries)),
        "scene_footprint_buffer_crs_units": float(footprint_buffer),
        "pixels_where_valid_count_exceeds_opportunity": int(violations.sum()),
    }
    if violations.any():
        worst = int((valid_count[violations] - opportunity_values[violations]).max())
        diagnostics["maximum_count_excess"] = worst
        raise RuntimeError(
            f"Scene-footprint denominator is smaller than valid_count at "
            f"{int(violations.sum()):,} ROI pixels (maximum excess {worst}); "
            "refusing to write an invalid fraction."
        )
    fraction_values = np.full(count.shape, np.nan, dtype=np.float32)
    defined = roi_mask & (opportunity_values > 0)
    fraction_values[defined] = valid_count[defined] / opportunity_values[defined]
    opportunity_values = np.where(
        roi_mask, opportunity_values, coverage.COUNT_NODATA
    ).astype(np.uint32)
    coords = {"y": count.y, "x": count.x}
    opportunity = xr.DataArray(
        opportunity_values,
        dims=("y", "x"),
        coords=coords,
        name="opportunity_count",
        attrs={
            "long_name": "selected unique Sentinel-1 pair scene-overlap opportunities",
            "definition": "pixel center lies in intersection of both source SLC footprints",
            "footprint_source": ASF_SEARCH_URL,
            "footprint_buffer_crs_units": float(footprint_buffer),
        },
    )
    fraction = xr.DataArray(
        fraction_values,
        dims=("y", "x"),
        coords=coords,
        name="valid_fraction",
        attrs={
            "long_name": "valid velocity count divided by scene-footprint opportunity count",
            "units": "1",
        },
    )
    return opportunity, fraction, diagnostics


def mosaic_tiles(
    tile_results: list[dict[str, Any]],
    roi: coverage.ROI,
    resolution: float,
    outdir: Path,
    deps: dict[str, Any],
) -> tuple[Any, Any, Any, Any, Any]:
    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.transform import from_origin

    np, xr = deps["np"], deps["xr"]
    crs_values = {result["crs"] for result in tile_results}
    if len(crs_values) != 1:
        raise RuntimeError(f"Multi-CRS mosaics are not supported: {sorted(crs_values)}")
    crs = deps["CRS"].from_user_input(next(iter(crs_values)))
    if not crs.is_projected:
        raise RuntimeError("Multi-cube mosaicking currently requires a projected cube CRS.")
    roi_cube = roi.in_crs(crs, deps)
    xmin, ymin, xmax, ymax = roi_cube.bounds
    left = math.floor(xmin / resolution) * resolution
    right = math.ceil(xmax / resolution) * resolution
    bottom = math.floor(ymin / resolution) * resolution
    top = math.ceil(ymax / resolution) * resolution
    width = int(round((right - left) / resolution))
    height = int(round((top - bottom) / resolution))
    transform = from_origin(left, top, resolution, resolution)
    x = left + resolution * (np.arange(width) + 0.5)
    y = top - resolution * (np.arange(height) + 0.5)
    roi_mask = geometry_mask(
        [deps["mapping"](roi_cube)],
        out_shape=(height, width),
        transform=transform,
        invert=True,
        all_touched=False,
    )
    count_values = np.full((height, width), coverage.COUNT_NODATA, dtype=np.uint32)
    denominator_values = np.full((height, width), coverage.COUNT_NODATA, dtype=np.uint32)

    for result in tile_results:
        cube = result["cube"]
        proj_bbox = cube.get("proj_bbox")
        if not proj_bbox:
            raise RuntimeError(f"Cube {cube['id']} lacks proj:bbox required for unambiguous mosaicking.")
        bxmin, bymin, bxmax, bymax = map(float, proj_bbox)
        x_indices = np.flatnonzero((x >= bxmin) & (x < bxmax))
        y_indices = np.flatnonzero((y >= bymin) & (y < bymax))
        if not x_indices.size or not y_indices.size:
            continue
        target_x, target_y = x[x_indices], y[y_indices]
        tile_dir = Path(result["tile_dir"])
        tile_count, count_nodata = nearest_tile_sample(
            tile_dir / "mosaic_source" / "coverage_count.tif", target_x, target_y, np
        )
        tile_denominator, denominator_nodata = nearest_tile_sample(
            tile_dir / "mosaic_source" / "selected_observation_count.tif", target_x, target_y, np
        )
        local_roi = roi_mask[np.ix_(y_indices, x_indices)]
        valid = local_roi & (tile_count != count_nodata)
        count_view = count_values[np.ix_(y_indices, x_indices)]
        denominator_view = denominator_values[np.ix_(y_indices, x_indices)]
        count_view[valid] = tile_count[valid]
        denominator_view[valid] = tile_denominator[valid]
        count_values[np.ix_(y_indices, x_indices)] = count_view
        denominator_values[np.ix_(y_indices, x_indices)] = denominator_view

    unfilled = roi_mask & (count_values == coverage.COUNT_NODATA)
    if unfilled.any():
        raise RuntimeError(
            f"Mosaic has {int(unfilled.sum()):,} ROI pixels not assigned to any cube tile."
        )
    fraction_values = np.full((height, width), np.nan, dtype=np.float32)
    defined = roi_mask & (denominator_values != coverage.COUNT_NODATA) & (denominator_values > 0)
    fraction_values[defined] = count_values[defined] / denominator_values[defined]
    coords = {"y": y, "x": x}
    count = xr.DataArray(
        count_values,
        dims=("y", "x"),
        coords=coords,
        name="valid_count",
        attrs={
            "long_name": "number of valid ITS_LIVE velocity observations",
            "mosaic_resampling": "nearest source pixel within each nominal STAC proj:bbox tile",
        },
    )
    fraction = xr.DataArray(
        fraction_values,
        dims=("y", "x"),
        coords=coords,
        name="valid_fraction",
        attrs={
            "long_name": "valid count divided by selected observations in the owning cube",
            "units": "1",
        },
    )
    denominator = xr.DataArray(
        denominator_values,
        dims=("y", "x"),
        coords=coords,
        name="selected_observation_count",
        attrs={
            "long_name": "selected observation count in the owning ITS_LIVE cube",
            "units": "1",
        },
    )
    schema = SimpleNamespace(x="x", y="y")
    return count, fraction, denominator, roi_mask, (schema, crs)


def write_scalar_global_tif(data: Any, path: Path, crs: Any, nodata: int) -> None:
    import rasterio
    from rasterio.transform import from_origin

    x = data.x.values
    y = data.y.values
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    transform = from_origin(x[0] - dx / 2, y[0] - dy / 2, dx, abs(dy))
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=len(x),
        height=len(y),
        count=1,
        dtype="uint32",
        nodata=nodata,
        crs=crs,
        transform=transform,
        compress="deflate",
        tiled=True,
    ) as destination:
        destination.write(data.values.astype("uint32"), 1)


def run_multi_cube(args: Any, deps: dict[str, Any]) -> int:
    try:
        roi = coverage.construct_roi(args, deps)
        args.outdir.mkdir(parents=True, exist_ok=True)
        global_names = (
            "coverage_count.nc",
            "coverage_count.tif",
            "coverage_fraction.tif",
            "opportunity_count.tif",
            "selected_observation_count.tif",
            "coverage_map.png",
            "selected_observations.csv",
            "summary.json",
            "run_config.json",
        )
        if args.overwrite:
            for name in global_names:
                path = args.outdir / name
                if path.exists():
                    path.unlink()
        elif args.no_plot:
            stale_plot = args.outdir / "coverage_map.png"
            if stale_plot.exists():
                stale_plot.unlink()
        cubes = discover_intersecting_cubes(roi, deps)
        if not cubes:
            raise RuntimeError("No ITS_LIVE Zarr cube intersects the requested ROI.")
        print(f"Found {len(cubes)} intersecting ITS_LIVE cubes for multi-cube processing.")
        run_payload = {
            "roi": {
                "kind": roi.input_kind,
                "bounds": roi.input_bounds,
                "crs": roi.crs.to_string(),
            },
            "start": args.start,
            "end": args.end,
            "mission": args.mission,
            "sensor": args.sensor,
            "min_pair_days": args.min_pair_days,
            "max_pair_days": args.max_pair_days,
            "validity": args.validity,
            "mosaic_resolution": args.mosaic_resolution,
            "cache_observations": args.cache_observations,
            "cubes": cubes,
        }
        run_fingerprint = fingerprint(run_payload)
        manifest_path = args.outdir / "multi_cube_manifest.json"
        if manifest_path.exists() and not (args.resume or args.overwrite):
            raise RuntimeError(
                f"{manifest_path} exists; use --resume to reuse tiles or --overwrite to rerun them."
            )
        if args.resume and manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if previous.get("fingerprint") != run_fingerprint:
                raise RuntimeError(
                    "Existing multi-cube run is incompatible with these arguments; use a new outdir or --overwrite."
                )
        coverage.write_json(
            manifest_path,
            {
                "status": "running",
                "fingerprint": run_fingerprint,
                "created_utc": datetime.now(timezone.utc),
                "configuration": run_payload,
            },
        )

        results = []
        pending = []
        for cube in cubes:
            tile_dir = args.outdir / "tiles" / cube["id"]
            tile_fp = fingerprint({"run": run_fingerprint, "cube": cube})
            cached = checkpoint_result(tile_dir, tile_fp) if args.resume else None
            if cached:
                print(f"[{cube['id']}] resumed from completed checkpoint")
                results.append(cached)
            else:
                pending.append((cube, tile_dir, tile_fp))

        failures = []
        if pending:
            worker_arguments = coverage.json_value(vars(args))
            with ProcessPoolExecutor(max_workers=min(args.workers, len(pending))) as pool:
                futures = {
                    pool.submit(
                        process_tile_isolated,
                        cube,
                        worker_arguments,
                        str(tile_dir),
                        tile_fp,
                    ): cube
                    for cube, tile_dir, tile_fp in pending
                }
                for future in as_completed(futures):
                    cube = futures[future]
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        failures.append({"cube_id": cube["id"], "error": f"{type(exc).__name__}: {exc}"})
        if failures:
            coverage.write_json(
                manifest_path,
                {
                    "status": "failed",
                    "fingerprint": run_fingerprint,
                    "configuration": run_payload,
                    "completed_tiles": results,
                    "failures": failures,
                    "resume_command_hint": "rerun the same command with --resume",
                },
            )
            raise RuntimeError(
                f"{len(failures)} cube(s) failed. See {manifest_path}; rerun with --resume after resolving the error."
            )

        results.sort(key=lambda result: result["cube"]["id"])
        inventory = deduplicate_observations(
            results, args.outdir / "selected_observations.csv", deps
        )
        if inventory["unique_observations"] == 0:
            summary = {
                "status": "no_matching_observations",
                "cube_count": len(results),
                "observation_inventory": inventory,
                "per_cube": results,
            }
            coverage.write_json(args.outdir / "summary.json", summary)
            coverage.write_json(
                args.outdir / "run_config.json",
                {
                    "arguments": vars(args),
                    "fingerprint": run_fingerprint,
                    "configuration": run_payload,
                },
            )
            coverage.write_json(
                manifest_path,
                {
                    "status": "complete",
                    "result": "no_matching_observations",
                    "fingerprint": run_fingerprint,
                    "completed_utc": datetime.now(timezone.utc),
                    "configuration": run_payload,
                    "tiles": results,
                    "global_outputs": {
                        "selected_observations": str(args.outdir / "selected_observations.csv"),
                        "summary": str(args.outdir / "summary.json"),
                    },
                },
            )
            print("No observations match the requested filters in any cube; wrote diagnostics without global rasters.")
            return 0
        print("Mosaicking completed cube checkpoints...")
        count, fraction, cube_denominator, roi_mask, schema_crs = mosaic_tiles(
            results, roi, args.mosaic_resolution, args.outdir, deps
        )
        schema, crs = schema_crs
        opportunity = None
        opportunity_diagnostics = None
        if args.fraction_denominator == "scene-footprint":
            if not args.mission or not coverage.mission_matches(
                "SENTINEL-1", args.mission
            ):
                raise RuntimeError(
                    "--fraction-denominator scene-footprint currently requires "
                    "--mission SENTINEL-1."
                )
            print("Building pixelwise opportunity count from Sentinel-1 scene footprints...")
            opportunity, fraction, opportunity_diagnostics = scene_footprint_opportunity(
                args.outdir / "selected_observations.csv",
                count,
                roi_mask,
                crs,
                args.outdir,
                args.scene_footprint_buffer,
                deps,
            )
            fraction_denominator = (
                "opportunity_count: unique selected pairs whose two source "
                "Sentinel-1 SLC footprints overlap the pixel center"
            )
        else:
            fraction_denominator = "selected_observation_count in owning cube"
            stale_opportunity = args.outdir / "opportunity_count.tif"
            if stale_opportunity.exists():
                stale_opportunity.unlink()
        stats = coverage.summary_statistics(count, roi_mask, inventory["unique_observations"], deps)
        metadata = {
            "source": "NASA ITS_LIVE",
            "cube_count": len(results),
            "cube_ids": [result["cube"]["id"] for result in results],
            "cube_crs": crs.to_string(),
            "input_roi": run_payload["roi"],
            "start": args.start,
            "end": args.end,
            "mission_filter": args.mission or "all",
            "sensor_filter": args.sensor or "all",
            "min_pair_days": args.min_pair_days,
            "max_pair_days": args.max_pair_days,
            "validity": args.validity,
            "unique_selected_observations": inventory["unique_observations"],
            "cube_observation_records": inventory["cube_observation_records"],
            "fraction_denominator": fraction_denominator,
            "mosaic_resolution": args.mosaic_resolution,
            "mosaic_resampling": "nearest source pixel partitioned by nominal STAC proj:bbox",
            "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        coverage.write_netcdf(
            count,
            fraction,
            roi_mask,
            schema,
            metadata,
            args.outdir,
            deps,
            selected_count_map=cube_denominator,
            opportunity_count_map=opportunity,
        )
        coverage.write_geotiffs(count, fraction, schema, crs, args.outdir, deps)
        write_scalar_global_tif(
            cube_denominator,
            args.outdir / "selected_observation_count.tif",
            crs,
            coverage.COUNT_NODATA,
        )
        if opportunity is not None:
            write_scalar_global_tif(
                opportunity,
                args.outdir / "opportunity_count.tif",
                crs,
                coverage.COUNT_NODATA,
            )
        if not args.no_plot:
            coverage.plot_map(
                count,
                schema,
                crs,
                args,
                inventory["unique_observations"],
                args.outdir,
                deps,
            )
        summary = {
            "status": "complete",
            "cube_count": len(results),
            "mosaic": {
                "crs": crs.to_string(),
                "resolution": args.mosaic_resolution,
                "shape": list(count.shape),
                "resampling": metadata["mosaic_resampling"],
                "fraction_denominator": metadata["fraction_denominator"],
            },
            "observation_inventory": inventory,
            "opportunity_diagnostics": opportunity_diagnostics,
            "per_cube": results,
            **stats,
        }
        coverage.write_json(args.outdir / "summary.json", summary)
        coverage.write_json(
            args.outdir / "run_config.json",
            {
                "arguments": vars(args),
                "fingerprint": run_fingerprint,
                "configuration": run_payload,
            },
        )
        coverage.write_json(
            manifest_path,
            {
                "status": "complete",
                "fingerprint": run_fingerprint,
                "completed_utc": datetime.now(timezone.utc),
                "configuration": run_payload,
                "tiles": results,
                "global_outputs": {
                    "coverage_count": str(args.outdir / "coverage_count.tif"),
                    "coverage_fraction": str(args.outdir / "coverage_fraction.tif"),
                    "selected_observation_count": str(args.outdir / "selected_observation_count.tif"),
                    "opportunity_count": (
                        str(args.outdir / "opportunity_count.tif")
                        if opportunity is not None
                        else None
                    ),
                    "summary": str(args.outdir / "summary.json"),
                },
            },
        )
        print(f"Multi-cube coverage complete: {args.outdir}")
        return 0
    except (ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=__import__("sys").stderr)
        return 2
