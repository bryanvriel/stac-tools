#!/usr/bin/env python3
"""Map valid ITS_LIVE image-pair observations from a cloud Zarr cube."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


STAC_URL = "https://stac.itslive.cloud"
COLLECTION = "itslive-cubes"
COUNT_NODATA = 2**32 - 1
FRACTION_NODATA = -9999.0


class CubeOpenError(RuntimeError):
    """The remote Zarr metadata could not be read completely."""


class IncompleteCubeMetadataError(RuntimeError):
    """A remote metadata response produced an incomplete dataset."""


def require_dependencies() -> dict[str, Any]:
    """Import runtime dependencies with one actionable error message."""
    try:
        import numpy as np
        import pandas as pd
        import xarray as xr
        import dask.array as dask_array
        from pyproj import CRS, Geod, Transformer
        from pystac_client import Client
        from shapely.geometry import MultiPolygon, Polygon, mapping, shape
        from shapely.ops import transform as transform_geometry
    except ImportError as exc:
        raise SystemExit(
            f"Missing dependency {exc.name!r}. Install requirements_itslive_cube.txt "
            "before running this tool."
        ) from exc
    return locals()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Count valid ITS_LIVE Zarr-cube velocity observations per ROI pixel."
    )
    roi = p.add_mutually_exclusive_group(required=True)
    roi.add_argument(
        "--bbox", type=float, nargs=4, metavar=("XMIN", "YMIN", "XMAX", "YMAX")
    )
    roi.add_argument(
        "--bbox-lonlat",
        type=float,
        nargs=4,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
    )
    p.add_argument("--epsg", type=int, default=3031, help="CRS of --bbox (default 3031).")
    p.add_argument("--start", required=True, help="Inclusive start date or timestamp.")
    p.add_argument("--end", required=True, help="Inclusive date or timestamp.")
    p.add_argument("--mission", help="Mission, e.g. SENTINEL-1 or LANDSAT.")
    p.add_argument("--sensor", help="Optional sensor string; both images must match.")
    p.add_argument("--min-pair-days", type=float)
    p.add_argument("--max-pair-days", type=float)
    p.add_argument("--validity", choices=["velocity"], default="velocity")
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--inspect-only", action="store_true")
    p.add_argument("--cube-url", help="Open this Zarr store instead of using STAC discovery.")
    p.add_argument(
        "--multi-cube",
        action="store_true",
        help="Process every intersecting cube and create a resumable mosaic.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="With --multi-cube, reuse completed compatible per-cube checkpoints.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Concurrent cube workers used by --multi-cube (default 2).",
    )
    p.add_argument(
        "--mosaic-resolution",
        type=float,
        default=120.0,
        help="Multi-cube output grid spacing in cube CRS units (default 120).",
    )
    p.add_argument(
        "--fraction-denominator",
        choices=("cube", "scene-footprint"),
        default="cube",
        help=(
            "Multi-cube fraction denominator: selected records in the owning cube, "
            "or unique source-scene pair overlaps per pixel (Sentinel-1 only)."
        ),
    )
    p.add_argument(
        "--scene-footprint-buffer",
        type=float,
        default=500.0,
        help=(
            "Projected-metre tolerance around Sentinel-1 pair footprints "
            "to reconcile catalog polygons with the 120 m cube grid (default 500)."
        ),
    )
    p.add_argument(
        "--spatial-block-size",
        type=int,
        default=100,
        help=(
            "Pixels per side of each restartable multi-cube read block "
            "(default 100)."
        ),
    )
    p.add_argument(
        "--cache-observations",
        action="store_true",
        help=(
            "With --multi-cube, cache the selected per-observation velocity-validity "
            "mask and native uncertainty values for later inversion diagnostics."
        ),
    )
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--edge-points", type=int, default=41)
    args = p.parse_args(argv)
    if args.min_pair_days is not None and args.min_pair_days < 0:
        p.error("--min-pair-days must be nonnegative")
    if args.max_pair_days is not None and args.max_pair_days < 0:
        p.error("--max-pair-days must be nonnegative")
    if (
        args.min_pair_days is not None
        and args.max_pair_days is not None
        and args.min_pair_days > args.max_pair_days
    ):
        p.error("--min-pair-days cannot exceed --max-pair-days")
    if args.edge_points < 2:
        p.error("--edge-points must be at least 2")
    if args.workers < 1:
        p.error("--workers must be at least 1")
    if args.mosaic_resolution <= 0:
        p.error("--mosaic-resolution must be positive")
    if args.spatial_block_size < 1:
        p.error("--spatial-block-size must be at least 1")
    if args.scene_footprint_buffer < 0:
        p.error("--scene-footprint-buffer must be nonnegative")
    if args.multi_cube and args.cube_url:
        p.error("--multi-cube discovers its cube set; it cannot be combined with --cube-url")
    if args.multi_cube and args.inspect_only:
        p.error("--inspect-only is a single-cube operation and cannot be combined with --multi-cube")
    if args.resume and not args.multi_cube:
        p.error("--resume is only valid with --multi-cube")
    if args.resume and args.overwrite:
        p.error("Choose either --resume or --overwrite, not both")
    if args.cache_observations and not args.multi_cube:
        p.error("--cache-observations currently requires --multi-cube")
    return args


def json_value(value: Any) -> Any:
    """Convert numpy, datetime, CRS, and other metadata values to JSON values."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return json_value(value.item())
        except (ValueError, TypeError):
            pass
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_value(v) for v in value]
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(json_value(payload), indent=2, sort_keys=True) + "\n")


def densified_ring(bounds: Iterable[float], n: int) -> list[tuple[float, float]]:
    xmin, ymin, xmax, ymax = map(float, bounds)
    if not (xmin < xmax and ymin < ymax):
        raise ValueError("Expected XMIN < XMAX and YMIN < YMAX.")

    def edge(x0: float, y0: float, x1: float, y1: float) -> list[tuple[float, float]]:
        return [(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n) for i in range(n)]

    points = edge(xmin, ymin, xmax, ymin)
    points += edge(xmax, ymin, xmax, ymax)
    points += edge(xmax, ymax, xmin, ymax)
    points += edge(xmin, ymax, xmin, ymin)
    return points + [points[0]]


@dataclass
class ROI:
    geometry: Any
    crs: Any
    input_kind: str
    input_bounds: list[float]

    def in_crs(self, target_crs: Any, deps: dict[str, Any]) -> Any:
        if self.crs == target_crs:
            return self.geometry
        transformer = deps["Transformer"].from_crs(self.crs, target_crs, always_xy=True)
        return deps["transform_geometry"](transformer.transform, self.geometry)


def construct_roi(args: argparse.Namespace, deps: dict[str, Any]) -> ROI:
    CRS, Polygon, MultiPolygon = deps["CRS"], deps["Polygon"], deps["MultiPolygon"]
    if args.bbox is not None:
        crs = CRS.from_epsg(args.epsg)
        return ROI(Polygon(densified_ring(args.bbox, args.edge_points)), crs, "bbox", list(args.bbox))

    west, south, east, north = args.bbox_lonlat
    if not (-180 <= west <= 180 and -180 <= east <= 180):
        raise ValueError("Longitude must be in [-180, 180].")
    if not (-90 <= south < north <= 90):
        raise ValueError("Expected -90 <= SOUTH < NORTH <= 90.")
    if west == east:
        raise ValueError("WEST and EAST must differ.")
    if west < east:
        geom = Polygon(densified_ring((west, south, east, north), args.edge_points))
    else:
        # A STAC-safe antimeridian-crossing ROI represented as two polygons.
        geom = MultiPolygon(
            [
                Polygon(densified_ring((west, south, 180, north), args.edge_points)),
                Polygon(densified_ring((-180, south, east, north), args.edge_points)),
            ]
        )
    return ROI(geom, CRS.from_epsg(4326), "bbox-lonlat", list(args.bbox_lonlat))


def item_asset_url(item: Any) -> str:
    for key in ("zarr", "v", "vx", "vy"):
        if key in item.assets:
            return item.assets[key].href
    for asset in item.assets.values():
        if ".zarr" in asset.href:
            return asset.href
    raise RuntimeError(f"Cube item {item.id} has no recognizable Zarr asset.")


def discover_cube(roi: ROI, deps: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Choose the smallest STAC cube whose footprint contains the ROI."""
    wgs84_roi = roi.in_crs(deps["CRS"].from_epsg(4326), deps)
    print("[1/7] Finding ITS_LIVE cube...")
    catalog = deps["Client"].open(STAC_URL)
    items = list(catalog.search(collections=[COLLECTION], intersects=deps["mapping"](wgs84_roi)).items())
    print(f"Found {len(items)} ITS_LIVE cube(s) intersecting requested ROI.")
    if not items:
        raise RuntimeError("No ITS_LIVE Zarr cube intersects the requested ROI.")

    containing: list[tuple[float, Any]] = []
    geod = deps["Geod"](ellps="WGS84")
    for item in items:
        geom = deps["shape"](item.geometry)
        area = abs(geod.geometry_area_perimeter(geom)[0])
        print(f"  {item.id}: bbox={item.bbox}, contains_roi={geom.covers(wgs84_roi)}")
        if geom.covers(wgs84_roi):
            containing.append((area, item))

    if not containing:
        raise RuntimeError(
            f"ROI intersects {len(items)} cubes and no single cube contains the full ROI. "
            "Multi-cube coverage is not implemented in v1; use a smaller ROI or --cube-url."
        )
    containing.sort(key=lambda pair: (pair[0], pair[1].id))
    best_area = containing[0][0]
    tied = [item for area, item in containing if math.isclose(area, best_area, rel_tol=1e-9)]
    if len(tied) != 1:
        ids = ", ".join(item.id for item in tied)
        raise RuntimeError(
            f"Multiple equally small cubes contain the ROI ({ids}). Use --cube-url to choose one."
        )
    item = tied[0]
    return item_asset_url(item), {
        "id": item.id,
        "bbox": item.bbox,
        "geometry": item.geometry,
        "properties": item.properties,
        "assets": {k: {"href": v.href, "type": v.media_type} for k, v in item.assets.items()},
        "selection": "smallest containing footprint",
    }


def open_cube(url: str, deps: dict[str, Any]) -> Any:
    print("[2/7] Opening Zarr metadata lazily...")
    xr = deps["xr"]
    errors: list[str] = []
    for consolidated in (True, None, False):
        try:
            ds = xr.open_zarr(
                url, consolidated=consolidated, chunks="auto", decode_cf=True
            )
            # An interrupted unconsolidated listing can look like a successful
            # but empty Zarr group. Treat it as a retryable remote read failure.
            if not ds.variables:
                ds.close()
                raise IncompleteCubeMetadataError(
                    "remote Zarr listing returned no variables"
                )
            return ds
        except Exception as exc:  # preserve useful attempts for cloud/Zarr variants
            errors.append(f"consolidated={consolidated}: {type(exc).__name__}: {exc}")
    raise CubeOpenError("Could not open Zarr store:\n  " + "\n  ".join(errors))


def array_chunks(da: Any) -> Any:
    chunks = getattr(da.data, "chunks", None)
    if chunks is not None:
        return [list(map(int, c)) for c in chunks]
    return json_value(da.encoding.get("chunks") or da.encoding.get("preferred_chunks"))


def inspect_cube(ds: Any, outdir: Path) -> dict[str, Any]:
    """Save detailed metadata while reading only small coordinate arrays."""
    report: dict[str, Any] = {
        "dimensions": dict(ds.sizes),
        "global_attributes": dict(ds.attrs),
        "coordinates": {},
        "data_variables": {},
    }
    text: list[str] = [str(ds), "", "DIMENSIONS"]
    text.extend(f"  {name}: {size}" for name, size in ds.sizes.items())
    text.append("\nCOORDINATES")
    for name, da in ds.coords.items():
        entry: dict[str, Any] = {
            "dtype": str(da.dtype),
            "shape": list(da.shape),
            "dimensions": list(da.dims),
            "attributes": dict(da.attrs),
            "encoding": dict(da.encoding),
            "chunks": array_chunks(da),
        }
        try:
            vals = da.values if da.size <= 1_000_000 else da.isel({da.dims[0]: slice(0, 8)}).values
            flat = vals.reshape(-1)
            entry["first_values"] = flat[:8].tolist()
            if da.size <= 1_000_000 and flat.size:
                entry["min"] = flat.min()
                entry["max"] = flat.max()
        except Exception as exc:
            entry["sample_error"] = f"{type(exc).__name__}: {exc}"
        report["coordinates"][name] = entry
        text.append(f"  {name}: dims={da.dims} shape={da.shape} dtype={da.dtype} attrs={da.attrs}")
    text.append("\nDATA VARIABLES")
    for name, da in ds.data_vars.items():
        entry = {
            "dtype": str(da.dtype),
            "shape": list(da.shape),
            "dimensions": list(da.dims),
            "chunks": array_chunks(da),
            "fill_value": da.encoding.get("_FillValue", da.attrs.get("_FillValue")),
            "missing_value": da.encoding.get("missing_value", da.attrs.get("missing_value")),
            "attributes": dict(da.attrs),
        }
        report["data_variables"][name] = entry
        text.append(
            f"  {name}: dims={da.dims} shape={da.shape} dtype={da.dtype} "
            f"chunks={entry['chunks']} attrs={da.attrs}"
        )
    text.append("\nGLOBAL ATTRIBUTES")
    text.extend(f"  {k}: {v}" for k, v in ds.attrs.items())
    write_json(outdir / "cube_schema.json", report)
    (outdir / "cube_schema.txt").write_text("\n".join(text) + "\n")
    return report


def first_existing(ds: Any, names: Iterable[str]) -> str | None:
    lower = {name.lower(): name for name in ds.variables}
    for candidate in names:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


class ITSLiveCubeSchema:
    """The only place where ITS_LIVE-specific names are resolved."""

    def __init__(self, ds: Any):
        self.ds = ds
        self.x, self.y = self._spatial_dims()
        self.velocity = self._velocity_variables()
        self.obs = self._observation_dim()

    def _spatial_dims(self) -> tuple[str, str]:
        x = first_existing(self.ds, ["x", "projection_x_coordinate", "lon", "longitude"])
        y = first_existing(self.ds, ["y", "projection_y_coordinate", "lat", "latitude"])
        for name, coord in self.ds.coords.items():
            std = str(coord.attrs.get("standard_name", "")).lower()
            axis = str(coord.attrs.get("axis", "")).upper()
            if std in ("projection_x_coordinate", "longitude") or axis == "X":
                x = name
            if std in ("projection_y_coordinate", "latitude") or axis == "Y":
                y = name
        if not x or not y or self.ds[x].ndim != 1 or self.ds[y].ndim != 1:
            if not self.ds.coords and not self.ds.data_vars:
                raise IncompleteCubeMetadataError(
                    "Remote Zarr metadata was incomplete: no coordinates or variables returned."
                )
            raise RuntimeError(f"Could not resolve 1-D spatial coordinates. Coordinates: {list(self.ds.coords)}")
        return x, y

    def _velocity_variables(self) -> dict[str, str]:
        found = {key: first_existing(self.ds, names) for key, names in {
            "v": ["v", "velocity", "speed"],
            "vx": ["vx", "v_x", "velocity_x"],
            "vy": ["vy", "v_y", "velocity_y"],
        }.items()}
        found = {k: v for k, v in found.items() if v is not None}
        if "v" not in found and not {"vx", "vy"}.issubset(found):
            candidates = [n for n, da in self.ds.data_vars.items() if self.x in da.dims and self.y in da.dims]
            raise RuntimeError(f"Missing velocity magnitude or vx/vy pair. Spatial variables: {candidates}")
        return found

    def _observation_dim(self) -> str:
        source = self.ds[self.velocity["v"] if "v" in self.velocity else self.velocity["vx"]]
        dims = [d for d in source.dims if d not in (self.x, self.y)]
        if len(dims) != 1:
            raise RuntimeError(f"Expected one observation dimension in {source.name}{source.dims}; found {dims}")
        return dims[0]

    def date_names(self) -> dict[str, str | None]:
        return {
            "mid_date": first_existing(self.ds, ["date_center", "mid_date", "time"]),
            "date_img1": first_existing(self.ds, ["acquisition_date_img1", "date_img1", "date_1"]),
            "date_img2": first_existing(self.ds, ["acquisition_date_img2", "date_img2", "date_2"]),
        }

    def pair_days_name(self) -> str | None:
        return first_existing(self.ds, ["date_dt", "pair_dt", "pair_interval", "pair_days"])

    def pair_metadata_names(self) -> dict[str, str | None]:
        return {
            "mission_img1": first_existing(self.ds, ["mission_img1", "mission_1"]),
            "mission_img2": first_existing(self.ds, ["mission_img2", "mission_2"]),
            "satellite_img1": first_existing(self.ds, ["satellite_img1", "platform_img1"]),
            "satellite_img2": first_existing(self.ds, ["satellite_img2", "platform_img2"]),
            "sensor_img1": first_existing(self.ds, ["sensor_img1", "sensor_1"]),
            "sensor_img2": first_existing(self.ds, ["sensor_img2", "sensor_2"]),
            "image_pair_id": first_existing(self.ds, ["granule_url", "image_pair_id", "pair_id"]),
        }

    def crs(self, deps: dict[str, Any]) -> Any:
        CRS = deps["CRS"]
        candidates: list[Any] = []
        primary = self.velocity["v"] if "v" in self.velocity else self.velocity["vx"]
        mapping_name = self.ds[primary].attrs.get("grid_mapping")
        if mapping_name and mapping_name in self.ds:
            attrs = self.ds[mapping_name].attrs
            candidates += [attrs.get(k) for k in ("spatial_ref", "crs_wkt", "spatial_epsg")]
        candidates += [self.ds.attrs.get(k) for k in ("projection", "spatial_epsg", "crs", "crs_wkt")]
        for value in candidates:
            if value is None:
                continue
            try:
                if str(value).isdigit():
                    return CRS.from_epsg(int(value))
                return CRS.from_user_input(value)
            except Exception:
                continue
        raise RuntimeError("Could not determine cube CRS from grid-mapping or global metadata.")

    def summary(self, deps: dict[str, Any]) -> dict[str, Any]:
        dates = self.date_names()
        return {
            "spatial_coordinates": {"x": self.x, "y": self.y},
            "observation_dimension": self.obs,
            "velocity_variables": self.velocity,
            "date_variables": dates,
            "pair_days_variable": self.pair_days_name(),
            "pair_metadata_variables": self.pair_metadata_names(),
            "crs": self.crs(deps).to_string(),
        }


def parse_dates(start: str, end: str, pd: Any) -> tuple[Any, Any, dict[str, str]]:
    date_only = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is not None:
        start_ts = start_ts.tz_convert("UTC").tz_localize(None)
    if end_ts.tzinfo is not None:
        end_ts = end_ts.tz_convert("UTC").tz_localize(None)
    if date_only.match(end):
        end_exclusive = end_ts + pd.Timedelta(days=1)
        end_description = f"{end_ts.date()} inclusive (through 23:59:59.999999999)"
    else:
        end_exclusive = end_ts + pd.Timedelta(nanoseconds=1)
        end_description = f"{end_ts.isoformat()} inclusive"
    if start_ts >= end_exclusive:
        raise ValueError("--start must not be after --end")
    return start_ts, end_exclusive, {
        "start_inclusive": start_ts.isoformat(),
        "end_input": end,
        "end_interpretation": end_description,
        "end_exclusive_internal": end_exclusive.isoformat(),
    }


def normalize_mission(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "UNKNOWN"
    s = re.sub(r"[^A-Z0-9]+", "-", str(value).strip().upper()).strip("-")
    aliases = {
        "S1": "SENTINEL-1", "SENTINEL1": "SENTINEL-1",
        "S2": "SENTINEL-2", "SENTINEL2": "SENTINEL-2",
        "L4": "LANDSAT-4", "L5": "LANDSAT-5", "L7": "LANDSAT-7",
        "L8": "LANDSAT-8", "L9": "LANDSAT-9",
    }
    if s in aliases:
        return aliases[s]
    match = re.fullmatch(r"LANDSAT-?([45789])", s)
    return f"LANDSAT-{match.group(1)}" if match else s


def mission_matches(value: str, requested: str) -> bool:
    requested = normalize_mission(requested)
    return value.startswith("LANDSAT-") if requested == "LANDSAT" else value == requested


def normalize_pair_mission(mission: Any, satellite: Any) -> str:
    """Expand ITS_LIVE's compact mission/satellite codes into mission groups."""
    normalized = normalize_mission(mission)
    satellite_code = re.sub(r"[^A-Z0-9]+", "", str(satellite or "").upper())
    # Current v2 cubes use mission S with satellite 1A/1B or 2A/2B, and
    # mission L with a numeric Landsat platform. Keep already explicit values.
    if normalized in {"S", "SENTINEL"}:
        if re.fullmatch(r"1[A-Z]?", satellite_code):
            return "SENTINEL-1"
        if re.fullmatch(r"2[A-Z]?", satellite_code):
            return "SENTINEL-2"
    if normalized in {"L", "LANDSAT"}:
        match = re.match(r"([45789])", satellite_code)
        return f"LANDSAT-{match.group(1)}" if match else "LANDSAT"
    if normalized == "UNKNOWN" and satellite_code:
        return normalize_mission(satellite_code)
    return normalized


def to_series(ds: Any, name: str | None, obs: str, pd: Any, n: int) -> Any:
    if not name:
        return pd.Series([None] * n)
    da = ds[name]
    if da.dims != (obs,):
        raise RuntimeError(f"Observation metadata {name!r} has unexpected dimensions {da.dims}.")
    return pd.Series(da.values)


def observation_table(ds: Any, schema: ITSLiveCubeSchema, deps: dict[str, Any]) -> Any:
    pd, np = deps["pd"], deps["np"]
    n = ds.sizes[schema.obs]
    dates, metadata = schema.date_names(), schema.pair_metadata_names()
    table = pd.DataFrame({"observation_index": np.arange(n, dtype=np.int64)})
    for key, name in {**dates, **metadata}.items():
        table[key] = to_series(ds, name, schema.obs, pd, n)

    if dates["mid_date"]:
        table["mid_date"] = pd.to_datetime(table["mid_date"], errors="coerce")
    if dates["date_img1"]:
        table["date_img1"] = pd.to_datetime(table["date_img1"], errors="coerce")
    if dates["date_img2"]:
        table["date_img2"] = pd.to_datetime(table["date_img2"], errors="coerce")
    if not dates["mid_date"] and dates["date_img1"] and dates["date_img2"]:
        table["mid_date"] = table["date_img1"] + (table["date_img2"] - table["date_img1"]) / 2
    if "mid_date" not in table or table["mid_date"].isna().all():
        raise RuntimeError(f"Could not resolve observation midpoint dates. Candidates: {dates}")

    pair_name = schema.pair_days_name()
    if pair_name:
        table["pair_days"] = pd.to_numeric(to_series(ds, pair_name, schema.obs, pd, n), errors="coerce").abs()
    elif dates["date_img1"] and dates["date_img2"]:
        table["pair_days"] = (table["date_img2"] - table["date_img1"]).dt.total_seconds().abs() / 86400
    else:
        table["pair_days"] = np.nan

    table["mission_img1_normalized"] = [
        normalize_pair_mission(mission, satellite)
        for mission, satellite in zip(table["mission_img1"], table["satellite_img1"])
    ]
    table["mission_img2_normalized"] = [
        normalize_pair_mission(mission, satellite)
        for mission, satellite in zip(table["mission_img2"], table["satellite_img2"])
    ]
    same = table["mission_img1_normalized"] == table["mission_img2_normalized"]
    table["mission"] = table["mission_img1_normalized"].where(same, "MIXED")
    table["same_mission"] = same
    return table


def filter_observations(table: Any, args: argparse.Namespace, deps: dict[str, Any]) -> tuple[Any, dict[str, int], dict[str, str]]:
    pd = deps["pd"]
    start, end_exclusive, date_meta = parse_dates(args.start, args.end, pd)
    counts = {"observations_in_cube": int(len(table))}
    selected = table[(table["mid_date"] >= start) & (table["mid_date"] < end_exclusive)].copy()
    counts["within_time_range"] = int(len(selected))
    if args.mission:
        known = selected.loc[
            selected["same_mission"].astype("boolean").fillna(False), "mission"
        ]
        if len(known) and known.eq("UNKNOWN").all():
            raw1 = sorted(table["mission_img1"].dropna().astype(str).unique().tolist())
            raw2 = sorted(table["mission_img2"].dropna().astype(str).unique().tolist())
            raise RuntimeError(
                "Could not interpret mission metadata. "
                f"mission_img1 values={raw1}; mission_img2 values={raw2}."
            )
        same_mission = selected["same_mission"].astype("boolean").fillna(False).to_numpy(dtype=bool)
        requested_mission = selected["mission"].map(
            lambda value: mission_matches(value, args.mission)
        ).to_numpy(dtype=bool)
        selected = selected[same_mission & requested_mission].copy()
    counts["matching_mission"] = int(len(selected))
    if args.sensor:
        if not table["sensor_img1"].notna().any() or not table["sensor_img2"].notna().any():
            raise RuntimeError("--sensor was requested, but paired sensor metadata are unavailable.")
        wanted = str(args.sensor).strip().upper()
        s1 = selected["sensor_img1"].fillna("").astype(str).str.strip().str.upper()
        s2 = selected["sensor_img2"].fillna("").astype(str).str.strip().str.upper()
        selected = selected[(s1 == wanted) & (s2 == wanted)].copy()
    counts["matching_sensor"] = int(len(selected))
    if (args.min_pair_days is not None or args.max_pair_days is not None) and not table["pair_days"].notna().any():
        raise RuntimeError("Pair-duration filtering was requested, but pair separation cannot be resolved.")
    if args.min_pair_days is not None:
        selected = selected[selected["pair_days"] >= args.min_pair_days].copy()
    if args.max_pair_days is not None:
        selected = selected[selected["pair_days"] <= args.max_pair_days].copy()
    counts["matching_pair_duration"] = int(len(selected))
    return selected, counts, date_meta


def coordinate_subset(ds: Any, schema: ITSLiveCubeSchema, roi_cube: Any, deps: dict[str, Any]) -> tuple[Any, Any]:
    np = deps["np"]
    xmin, ymin, xmax, ymax = roi_cube.bounds
    xv, yv = np.asarray(ds[schema.x].values), np.asarray(ds[schema.y].values)
    xi = np.flatnonzero((xv >= xmin) & (xv <= xmax))
    yi = np.flatnonzero((yv >= ymin) & (yv <= ymax))
    if not xi.size or not yi.size:
        raise RuntimeError("ROI contains no cube-grid pixel centers.")
    subset = ds.isel({schema.x: slice(int(xi.min()), int(xi.max()) + 1), schema.y: slice(int(yi.min()), int(yi.max()) + 1)})
    x = np.asarray(subset[schema.x].values)
    y = np.asarray(subset[schema.y].values)
    xx, yy = np.meshgrid(x, y)
    try:
        from shapely import intersects_xy
        mask = intersects_xy(roi_cube, xx, yy)
    except ImportError:  # Shapely < 2 fallback
        from shapely.geometry import Point
        mask = np.fromiter((roi_cube.covers(Point(a, b)) for a, b in zip(xx.ravel(), yy.ravel())), bool).reshape(xx.shape)
    if not mask.any():
        raise RuntimeError("ROI contains no cube-grid pixel centers.")
    return subset, mask


def finite_valid(da: Any, deps: dict[str, Any]) -> Any:
    np = deps["np"]
    valid = np.isfinite(da)
    values = []
    for source in (da.attrs, da.encoding):
        for key in ("_FillValue", "missing_value"):
            if source.get(key) is not None:
                raw = source[key]
                values.extend(raw if isinstance(raw, (list, tuple)) else [raw])
    for value in values:
        try:
            if not (isinstance(value, float) and math.isnan(value)):
                valid = valid & (da != value)
        except TypeError:
            pass
    return valid


def compute_coverage(
    ds: Any,
    schema: ITSLiveCubeSchema,
    selected: Any,
    roi_mask: Any,
    deps: dict[str, Any],
    show_progress: bool = True,
) -> tuple[Any, Any]:
    np, xr = deps["np"], deps["xr"]
    obs_indices = selected["observation_index"].to_numpy(dtype=np.int64)
    data = ds.isel({schema.obs: obs_indices})
    if "v" in schema.velocity:
        valid = finite_valid(data[schema.velocity["v"]], deps)
        definition = f"finite {schema.velocity['v']} excluding declared fill/missing values"
    else:
        valid = finite_valid(data[schema.velocity["vx"]], deps) & finite_valid(data[schema.velocity["vy"]], deps)
        definition = f"finite {schema.velocity['vx']} AND {schema.velocity['vy']} excluding fill/missing values"
    count = valid.sum(dim=schema.obs, dtype=np.uint32)
    if show_progress:
        print("[6/7] Reading required Zarr chunks and counting valid pixels...")
    try:
        from dask.diagnostics import ProgressBar
        if not show_progress:
            count_values = np.asarray(count.compute().values, dtype=np.uint32)
        else:
            with ProgressBar():
                count_values = np.asarray(count.compute().values, dtype=np.uint32)
    except ImportError:
        count_values = np.asarray(count.compute().values, dtype=np.uint32)
    count_values = np.where(roi_mask, count_values, COUNT_NODATA).astype(np.uint32)
    fraction_values = np.where(roi_mask, count_values.astype(np.float64) / len(selected), np.nan).astype(np.float32)
    coords = {schema.y: ds[schema.y], schema.x: ds[schema.x]}
    count_da = xr.DataArray(count_values, dims=(schema.y, schema.x), coords=coords, name="valid_count")
    count_da.attrs.update({"long_name": "number of valid ITS_LIVE velocity observations", "validity_definition": definition})
    fraction_da = xr.DataArray(fraction_values, dims=(schema.y, schema.x), coords=coords, name="valid_fraction")
    fraction_da.attrs.update({"long_name": "fraction of selected observations with valid velocity", "units": "1"})
    return count_da, fraction_da


def summary_statistics(count: Any, roi_mask: Any, selected_count: int, deps: dict[str, Any]) -> dict[str, Any]:
    np = deps["np"]
    values = np.asarray(count.values)[roi_mask]
    return {
        "selected_observations": selected_count,
        "roi_pixel_count": int(values.size),
        "valid_count": {
            "min": int(values.min()), "max": int(values.max()),
            "mean": float(values.mean()), "median": float(np.median(values)),
            "percentile_25": float(np.percentile(values, 25)),
            "percentile_75": float(np.percentile(values, 75)),
        },
        "roi_fraction_at_or_above": {
            str(threshold): float((values >= threshold).mean()) for threshold in (1, 5, 10, 20, 50)
        },
    }


def regular_spacing(values: Any, np: Any, name: str) -> float:
    delta = np.diff(np.asarray(values, dtype=float))
    if not delta.size or not np.allclose(delta, delta[0], rtol=1e-7, atol=1e-7):
        raise RuntimeError(f"{name} coordinates are not a regular grid.")
    return float(delta[0])


def north_up_arrays(count: Any, fraction: Any, schema: ITSLiveCubeSchema, deps: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    np = deps["np"]
    x, y = np.asarray(count[schema.x].values), np.asarray(count[schema.y].values)
    c, f = np.asarray(count.values), np.asarray(fraction.values)
    if x[0] > x[-1]:
        x, c, f = x[::-1], c[:, ::-1], f[:, ::-1]
    if y[0] < y[-1]:
        y, c, f = y[::-1], c[::-1, :], f[::-1, :]
    return x, y, c, f


def write_geotiffs(count: Any, fraction: Any, schema: ITSLiveCubeSchema, crs: Any, outdir: Path, deps: dict[str, Any]) -> None:
    try:
        import rasterio
        from rasterio.transform import Affine
    except ImportError as exc:
        raise RuntimeError("Rasterio is required to write GeoTIFF outputs.") from exc
    np = deps["np"]
    x, y, c, f = north_up_arrays(count, fraction, schema, deps)
    dx, dy = regular_spacing(x, np, "x"), regular_spacing(y, np, "y")
    transform = Affine.translation(x[0] - dx / 2, y[0] - dy / 2) * Affine.scale(dx, dy)
    common = dict(driver="GTiff", width=len(x), height=len(y), count=1, crs=crs, transform=transform, compress="deflate", tiled=True)
    with rasterio.open(outdir / "coverage_count.tif", "w", dtype="uint32", nodata=COUNT_NODATA, **common) as dst:
        dst.write(c.astype(np.uint32), 1)
    filled_fraction = np.where(np.isfinite(f), f, FRACTION_NODATA).astype(np.float32)
    with rasterio.open(outdir / "coverage_fraction.tif", "w", dtype="float32", nodata=FRACTION_NODATA, **common) as dst:
        dst.write(filled_fraction, 1)


def write_netcdf(
    count: Any,
    fraction: Any,
    roi_mask: Any,
    schema: ITSLiveCubeSchema,
    attrs: dict[str, Any],
    outdir: Path,
    deps: dict[str, Any],
    selected_count_map: Any | None = None,
    opportunity_count_map: Any | None = None,
) -> None:
    xr, np = deps["xr"], deps["np"]
    mask_da = xr.DataArray(roi_mask, dims=(schema.y, schema.x), coords={schema.y: count[schema.y], schema.x: count[schema.x]}, name="roi_mask")
    mask_da.attrs.update({"long_name": "pixel center falls inside requested ROI", "flag_values": [0, 1]})
    output = xr.Dataset({"valid_count": count, "valid_fraction": fraction, "roi_mask": mask_da})
    if selected_count_map is not None:
        output["selected_observation_count"] = selected_count_map
    if opportunity_count_map is not None:
        output["opportunity_count"] = opportunity_count_map
    def netcdf_attr(value: Any) -> Any:
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(json_value(value))
        if value is None:
            return "null"
        if isinstance(value, bool):
            return int(value)
        return json_value(value)

    output.attrs.update({k: netcdf_attr(v) for k, v in attrs.items()})
    encoding = {
        "valid_count": {"dtype": "uint32", "_FillValue": np.uint32(COUNT_NODATA), "zlib": True},
        "valid_fraction": {"dtype": "float32", "_FillValue": np.float32(FRACTION_NODATA), "zlib": True},
        "roi_mask": {"dtype": "uint8", "_FillValue": None, "zlib": True},
    }
    if selected_count_map is not None:
        encoding["selected_observation_count"] = {
            "dtype": "uint32",
            "_FillValue": np.uint32(COUNT_NODATA),
            "zlib": True,
        }
    if opportunity_count_map is not None:
        encoding["opportunity_count"] = {
            "dtype": "uint32",
            "_FillValue": np.uint32(COUNT_NODATA),
            "zlib": True,
        }
    output.to_netcdf(outdir / "coverage_count.nc", encoding=encoding)


def plot_map(count: Any, schema: ITSLiveCubeSchema, crs: Any, args: argparse.Namespace, selected_count: int, outdir: Path, deps: dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    np = deps["np"]
    values = np.where(np.asarray(count.values) == COUNT_NODATA, np.nan, count.values)
    x, y = np.asarray(count[schema.x]), np.asarray(count[schema.y])
    scale = 1000.0 if crs.is_projected else 1.0
    unit = "km" if crs.is_projected else "degrees"
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    mesh = ax.pcolormesh(x / scale, y / scale, values, shading="auto", cmap="viridis")
    fig.colorbar(mesh, ax=ax, label="Valid observation count")
    mission = args.mission or "all missions"
    ax.set_title(f"ITS_LIVE valid observation count\n{mission} | {args.start} to {args.end} | N={selected_count:,}")
    ax.set_xlabel(f"{schema.x} ({unit})")
    ax.set_ylabel(f"{schema.y} ({unit})")
    ax.set_aspect("equal")
    fig.savefig(outdir / "coverage_map.png", dpi=180)
    plt.close(fig)


def check_output_paths(outdir: Path, overwrite: bool) -> None:
    products = [
        "coverage_count.nc", "coverage_count.tif", "coverage_fraction.tif", "coverage_map.png",
        "summary.json", "selected_observations.csv", "cube_info.json", "run_config.json",
        "cube_schema.txt", "cube_schema.json",
    ]
    existing = [outdir / p for p in products if (outdir / p).exists()]
    if existing and not overwrite:
        raise RuntimeError("Output files already exist; use --overwrite: " + ", ".join(map(str, existing)))
    outdir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        # Prevent stale rasters/plots when a replacement run has no observations
        # or uses --no-plot. Targets are deliberately restricted to known products.
        for path in existing:
            path.unlink()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    deps = require_dependencies()
    if args.multi_cube:
        from itslive_multi_cube import run_multi_cube

        return run_multi_cube(args, deps)
    try:
        check_output_paths(args.outdir, args.overwrite)
        roi = construct_roi(args, deps)
        start, end_exclusive, date_meta = parse_dates(args.start, args.end, deps["pd"])
        config = {
            "arguments": vars(args),
            "created_utc": datetime.now(timezone.utc),
            "date_filter": date_meta,
            "input_roi": {"kind": roi.input_kind, "bounds": roi.input_bounds, "crs": roi.crs.to_string(), "geometry": deps["mapping"](roi.geometry)},
        }
        write_json(args.outdir / "run_config.json", config)

        if args.cube_url:
            cube_url = args.cube_url
            cube_info = {"id": None, "url": cube_url, "selection": "--cube-url override"}
        else:
            cube_url, cube_info = discover_cube(roi, deps)
            cube_info["url"] = cube_url
        write_json(args.outdir / "cube_info.json", cube_info)

        ds = open_cube(cube_url, deps)
        print("[3/7] Resolving cube schema...")
        inspect_cube(ds, args.outdir)
        schema = ITSLiveCubeSchema(ds)
        schema_info = schema.summary(deps)
        cube_info["resolved_schema"] = schema_info
        write_json(args.outdir / "cube_info.json", cube_info)
        print(json.dumps(json_value(schema_info), indent=2))
        if args.inspect_only:
            print(f"Inspection complete: {args.outdir / 'cube_schema.json'}")
            return 0

        cube_crs = schema.crs(deps)
        roi_cube = roi.in_crs(cube_crs, deps)
        print("[4/7] Filtering observations...")
        table = observation_table(ds, schema, deps)
        selected, filter_counts, date_meta = filter_observations(table, args, deps)
        print("Mission inventory in requested time range:")
        time_start, time_end, _ = parse_dates(args.start, args.end, deps["pd"])
        inventory = table[(table.mid_date >= time_start) & (table.mid_date < time_end)].groupby("mission").size().sort_values(ascending=False)
        for mission, number in inventory.items():
            print(f"  {mission}: {number:,}")
        for key, value in filter_counts.items():
            print(f"  {key.replace('_', ' ')}: {value:,}")
        selected.to_csv(args.outdir / "selected_observations.csv", index=False)

        base_summary = {
            "status": "no_matching_observations" if selected.empty else "complete",
            "filters": filter_counts,
            "mission_inventory_within_time_range": {str(k): int(v) for k, v in inventory.items()},
            "date_filter": date_meta,
            "cube": {"id": cube_info.get("id"), "url": cube_url, "crs": cube_crs.to_string()},
            "roi": {"input": config["input_roi"], "cube_crs": cube_crs.to_string(), "geometry_in_cube_crs": deps["mapping"](roi_cube)},
        }
        if selected.empty:
            write_json(args.outdir / "summary.json", base_summary)
            print("No observations match the requested filters. Wrote diagnostics; no coverage rasters were created.")
            return 0

        print("[5/7] Subsetting spatially and constructing exact ROI mask...")
        subset, roi_mask = coordinate_subset(ds, schema, roi_cube, deps)
        ny, nx = subset.sizes[schema.y], subset.sizes[schema.x]
        samples = len(selected) * ny * nx
        velocity_name = schema.velocity.get("v", schema.velocity["vx"])
        itemsize = subset[velocity_name].dtype.itemsize
        print(f"ROI grid envelope: {nx} x {ny} pixels; {int(roi_mask.sum()):,} centers inside ROI")
        print(f"Selected observations: {len(selected):,}")
        print(f"Logical subset: {samples:,} samples (~{samples * itemsize / 2**30:.2f} GiB uncompressed)")
        count, fraction = compute_coverage(subset, schema, selected, roi_mask, deps)
        stats = summary_statistics(count, roi_mask, len(selected), deps)
        summary = {**base_summary, **stats}

        print("[7/7] Writing outputs...")
        metadata = {
            "source": "NASA ITS_LIVE", "cube_url": cube_url, "cube_id": cube_info.get("id"),
            "cube_crs": cube_crs.to_string(), "input_roi": config["input_roi"],
            "roi_in_cube_crs": deps["mapping"](roi_cube), "start": args.start, "end": args.end,
            "mission_filter": args.mission or "all", "sensor_filter": args.sensor or "all",
            "min_pair_days": args.min_pair_days, "max_pair_days": args.max_pair_days,
            "validity": count.attrs["validity_definition"], "selected_observation_count": len(selected),
            "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        write_netcdf(count, fraction, roi_mask, schema, metadata, args.outdir, deps)
        write_geotiffs(count, fraction, schema, cube_crs, args.outdir, deps)
        if not args.no_plot:
            plot_map(count, schema, cube_crs, args, len(selected), args.outdir, deps)
        write_json(args.outdir / "summary.json", summary)
        print(f"Complete. Outputs written to {args.outdir}")
        return 0
    except (ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
