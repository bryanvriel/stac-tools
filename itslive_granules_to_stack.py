#!/usr/bin/env python3
"""Resample downloaded ITS_LIVE granules into an iceutils-ready NetCDF Stack."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import iceutils as ice
import numpy as np
import xarray as xr
from pyproj import CRS, Transformer


VARIABLES = ("vx", "vy", "v_error")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crop and resample downloaded ITS_LIVE image-pair NetCDF files onto "
            "one grid and write an iceutils-compatible NetCDF Stack."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="STAC inventory directory containing items.csv and data/*.nc.",
    )
    roi = parser.add_mutually_exclusive_group()
    roi.add_argument(
        "--bbox-lonlat",
        nargs=4,
        type=float,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="Original WGS84 bbox. Defaults to input-dir/summary.json.",
    )
    roi.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
        help="Bounding box already in the target CRS.",
    )
    parser.add_argument("--target-epsg", type=int, default=3031)
    parser.add_argument("--resolution", type=float, default=120.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--resampling",
        choices=("nearest", "bilinear", "cubic"),
        default="bilinear",
    )
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Build a stack from available files even if inventory downloads are missing.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_bbox(args: argparse.Namespace) -> tuple[str, list[float]]:
    if args.bbox_lonlat is not None:
        return "lonlat", list(map(float, args.bbox_lonlat))
    if args.bbox is not None:
        return "projected", list(map(float, args.bbox))
    summary_path = args.input_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            "No bbox supplied and STAC summary is missing: " + str(summary_path)
        )
    summary = json.loads(summary_path.read_text())
    bbox = summary.get("input_bbox")
    bbox_type = summary.get("input_bbox_type")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"Invalid input_bbox in {summary_path}")
    if bbox_type == "lonlat":
        return "lonlat", list(map(float, bbox))
    if bbox_type == "projected":
        source_epsg = int(summary.get("input_epsg", args.target_epsg))
        if source_epsg != args.target_epsg:
            raise ValueError(
                "The saved projected bbox uses EPSG:"
                f"{source_epsg}; pass --bbox in EPSG:{args.target_epsg} explicitly."
            )
        return "projected", list(map(float, bbox))
    raise ValueError(f"Unrecognized input_bbox_type in {summary_path}: {bbox_type!r}")


def densified_lonlat_bbox(bbox: list[float], points: int = 101) -> tuple[np.ndarray, np.ndarray]:
    west, south, east, north = bbox
    if not (west < east and south < north):
        raise ValueError("Expected WEST < EAST and SOUTH < NORTH")
    horizontal = np.linspace(west, east, points)
    vertical = np.linspace(south, north, points)
    lon = np.concatenate((horizontal, np.full(points, east), horizontal[::-1], np.full(points, west)))
    lat = np.concatenate((np.full(points, south), vertical, np.full(points, north), vertical[::-1]))
    return lon, lat


def projected_envelope(
    bbox_type: str, bbox: list[float], target_epsg: int
) -> tuple[float, float, float, float]:
    if bbox_type == "projected":
        xmin, ymin, xmax, ymax = bbox
        if not (xmin < xmax and ymin < ymax):
            raise ValueError("Expected XMIN < XMAX and YMIN < YMAX")
        return xmin, ymin, xmax, ymax
    lon, lat = densified_lonlat_bbox(bbox)
    x, y = Transformer.from_crs(4326, target_epsg, always_xy=True).transform(lon, lat)
    return float(np.min(x)), float(np.min(y)), float(np.max(x)), float(np.max(y))


def discover_inputs(input_dir: Path, allow_missing: bool) -> list[Path]:
    data_dir = input_dir / "data"
    paths = sorted(data_dir.glob("*.nc"))
    inventory_path = input_dir / "items.csv"
    if inventory_path.exists():
        with inventory_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        expected = {
            Path(str(row["data_href"])).name
            for row in rows
            if row.get("data_href")
        }
        available = {path.name for path in paths}
        missing = sorted(expected - available)
        if missing and not allow_missing:
            preview = ", ".join(missing[:3])
            raise RuntimeError(
                f"{len(missing)} of {len(expected)} inventoried downloads are missing "
                f"({preview}). Rerun the download or pass --allow-missing."
            )
    if not paths:
        raise RuntimeError(f"No NetCDF granules found under {data_dir}")
    return paths


def source_info(path: Path, target_epsg: int) -> dict[str, Any]:
    with xr.open_dataset(path, engine="h5netcdf", decode_coords="all") as ds:
        missing = [name for name in ("x", "y", "time", *VARIABLES) if name not in ds]
        if missing:
            raise ValueError(f"{path.name} is missing: {', '.join(missing)}")
        if ds.sizes.get("time") != 1:
            raise ValueError(f"{path.name} must contain exactly one time value")
        epsg = None
        if "mapping" in ds:
            epsg = ds["mapping"].attrs.get("spatial_epsg")
        if epsg is None:
            epsg = ds.attrs.get("EPSG")
        if epsg is None and "mapping" in ds:
            wkt = ds["mapping"].attrs.get("spatial_ref") or ds["mapping"].attrs.get("crs_wkt")
            if wkt:
                epsg = CRS.from_wkt(str(wkt)).to_epsg()
        if int(epsg or -1) != target_epsg:
            raise ValueError(
                f"{path.name} uses EPSG:{epsg}; direct component resampling to "
                f"EPSG:{target_epsg} would require vector rotation."
            )
        time_value = np.asarray(ds["time"].values).reshape(-1)[0].astype("datetime64[ns]")
        return {
            "path": path,
            "time": time_value,
            "x0": float(ds.x.values[0]),
            "y0": float(ds.y.values[0]),
            "dx": float(ds.x.values[1] - ds.x.values[0]),
            "dy": float(ds.y.values[1] - ds.y.values[0]),
        }


def aligned_coordinates(
    envelope: tuple[float, float, float, float],
    resolution: float,
    x_phase: float,
    y_phase: float,
) -> tuple[np.ndarray, np.ndarray]:
    xmin, ymin, xmax, ymax = envelope
    x_start = x_phase + math.floor((xmin - x_phase) / resolution) * resolution
    x_stop = x_phase + math.ceil((xmax - x_phase) / resolution) * resolution
    y_start = y_phase + math.ceil((ymax - y_phase) / resolution) * resolution
    y_stop = y_phase + math.floor((ymin - y_phase) / resolution) * resolution
    x = np.arange(x_start, x_stop + 0.5 * resolution, resolution, dtype=np.float64)
    y = np.arange(y_start, y_stop - 0.5 * resolution, -resolution, dtype=np.float64)
    return x, y


def roi_mask(
    x: np.ndarray,
    y: np.ndarray,
    bbox_type: str,
    bbox: list[float],
    target_epsg: int,
) -> np.ndarray:
    xx, yy = np.meshgrid(x, y)
    if bbox_type == "projected":
        xmin, ymin, xmax, ymax = bbox
        return (xx >= xmin) & (xx <= xmax) & (yy >= ymin) & (yy <= ymax)
    west, south, east, north = bbox
    lon, lat = Transformer.from_crs(target_epsg, 4326, always_xy=True).transform(xx, yy)
    return (lon >= west) & (lon <= east) & (lat >= south) & (lat <= north)


def datetime64_to_tdec(value: np.datetime64) -> float:
    nanoseconds = int(value.astype("datetime64[ns]").astype(np.int64))
    date = datetime.fromtimestamp(nanoseconds / 1.0e9, tz=timezone.utc).replace(tzinfo=None)
    return float(ice.datestr2tdec(dateobj=date))


def resample_granule(
    path: Path,
    target_hdr: Any,
    target_mask: np.ndarray,
    order: int,
    margin: float,
) -> dict[str, np.ndarray]:
    xmin, ymin, xmax, ymax = target_hdr.bounds
    with xr.open_dataset(path, engine="h5netcdf", decode_coords="all") as ds:
        xvals = np.asarray(ds.x.values)
        yvals = np.asarray(ds.y.values)
        xi = np.flatnonzero((xvals >= xmin - margin) & (xvals <= xmax + margin))
        yi = np.flatnonzero((yvals >= ymin - margin) & (yvals <= ymax + margin))
        if not xi.size or not yi.size:
            return {
                name: np.full(target_hdr.shape, np.nan, dtype=np.float32)
                for name in VARIABLES
            }
        subset = ds.isel(
            x=slice(int(xi.min()), int(xi.max()) + 1),
            y=slice(int(yi.min()), int(yi.max()) + 1),
            time=0,
        ).load()

    sx = np.asarray(subset.x.values)
    sy = np.asarray(subset.y.values)
    source_x, source_y = np.meshgrid(sx, sy)
    source_hdr = ice.RasterInfo(X=source_x, Y=source_y, epsg=target_hdr.epsg)
    output: dict[str, np.ndarray] = {}
    for name in VARIABLES:
        values = np.asarray(subset[name].values, dtype=np.float32)
        raster = ice.Raster(data=values, hdr=source_hdr)
        raster.nodataval = np.nan
        raster.resample(
            target_hdr,
            order=order,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
        result = np.asarray(raster.data, dtype=np.float32)
        result[~target_mask] = np.nan
        output[name] = result
    return output


def build_stack(args: argparse.Namespace) -> None:
    if args.resolution <= 0 or args.chunk_size < 1:
        raise ValueError("--resolution and --chunk-size must be positive")
    bbox_type, bbox = load_bbox(args)
    paths = discover_inputs(args.input_dir, args.allow_missing)
    infos = [source_info(path, args.target_epsg) for path in paths]
    infos.sort(key=lambda item: (item["time"], item["path"].name))
    first = infos[0]
    tolerance = max(1.0e-6, args.resolution * 1.0e-6)
    for info in infos:
        if abs(abs(info["dx"]) - args.resolution) > tolerance or abs(abs(info["dy"]) - args.resolution) > tolerance:
            raise ValueError(
                f"{info['path'].name} has {info['dx']} x {info['dy']} m pixels; "
                f"expected {args.resolution} m."
            )

    envelope = projected_envelope(bbox_type, bbox, args.target_epsg)
    x, y = aligned_coordinates(envelope, args.resolution, first["x0"], first["y0"])
    xx, yy = np.meshgrid(x, y)
    target_hdr = ice.RasterInfo(X=xx, Y=yy, epsg=args.target_epsg)
    target_mask = roi_mask(x, y, bbox_type, bbox, args.target_epsg)
    tdec = np.asarray([datetime64_to_tdec(info["time"]) for info in infos])

    output = args.output.resolve()
    partial = output.with_name(output.name + ".partial")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output}; pass --overwrite to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    if partial.exists():
        partial.unlink()

    chunks = (1, min(args.chunk_size, len(y)), min(args.chunk_size, len(x)))
    order = {"nearest": 0, "bilinear": 1, "cubic": 3}[args.resampling]
    print(
        f"Building {len(infos)}-observation stack on EPSG:{args.target_epsg} "
        f"grid {len(y)} x {len(x)} at {args.resolution:g} m"
    )
    stack = ice.Stack(str(partial), mode="w", init_tdec=tdec, init_rasterinfo=target_hdr)
    try:
        for name in VARIABLES:
            variable = stack.create_dataset(
                name,
                (len(infos), len(y), len(x)),
                dtype="f4",
                chunks=chunks,
                fillvalue=np.nan,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
            variable.attrs["units"] = "meter/year"
        stack.fid.attrs["source"] = "ITS_LIVE image-pair velocity granules"
        stack.fid.attrs["target_epsg"] = args.target_epsg
        stack.fid.attrs["resolution_m"] = args.resolution
        stack.fid.attrs["input_bbox_type"] = bbox_type
        stack.fid.attrs["input_bbox"] = json.dumps(bbox)
        stack.fid.attrs["resampling"] = args.resampling

        for index, info in enumerate(infos):
            arrays = resample_granule(
                info["path"], target_hdr, target_mask, order, 2 * args.resolution
            )
            for name, array in arrays.items():
                stack.set_slice(index, array, key=name)
            finite = int(np.count_nonzero(np.isfinite(arrays["vx"])))
            print(
                f"[{index + 1}/{len(infos)}] {info['path'].name}: "
                f"{finite:,} finite ROI pixels"
            )
    finally:
        stack.close()
    os.replace(partial, output)
    print(f"Wrote iceutils-ready stack: {output}")


def main() -> None:
    args = parse_args()
    try:
        build_stack(args)
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
