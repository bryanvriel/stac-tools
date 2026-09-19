#!/usr/bin/env python3
"""
Query ITS_LIVE image-pair velocity granules for an ROI and optionally download them.

The ROI can be supplied in either:
  * projected coordinates, e.g. Antarctic Polar Stereographic EPSG:3031:
        --bbox XMIN YMIN XMAX YMAX [--epsg 3031]
  * longitude/latitude:
        --bbox-lonlat WEST SOUTH EAST NORTH

The script:
  1. Converts the ROI to WGS84 GeoJSON suitable for a STAC `intersects` query.
  2. Queries the ITS_LIVE STAC API over a requested time range.
  3. Inventories useful metadata.
  4. Estimates the total download size:
       - first from STAC `file:size` metadata, where available;
       - then, by default, via lightweight HTTP HEAD/range probes for missing sizes;
       - if some sizes remain unknown, extrapolates using the median known file size
         and clearly labels the result as an estimate.
  5. Writes:
       - query_roi_wgs84.geojson
       - items.csv
       - items.ndjson
       - summary.json
  6. Optionally downloads each retained item's primary data asset.

Metadata-only is the default. Use --download-data to fetch the image-pair files.

Examples
--------
Projected Antarctic Polar Stereographic bounding box:

    python itslive_stac_inventory_v2.py \
        --bbox -450000 350000 -350000 450000 \
        --start 2019-01-01 \
        --end 2024-12-31 \
        --outdir denman_itslive

Longitude/latitude bounding box:

    python itslive_stac_inventory_v2.py \
        --bbox-lonlat 99.0 -67.5 103.0 -65.5 \
        --start 2019-01-01 \
        --end 2024-12-31 \
        --outdir denman_itslive

Dependencies
------------
pip install pystac-client pyproj requests

Notes
-----
* STAC spatial queries are WGS84 longitude/latitude. A projected bounding box
  must therefore be transformed before querying.
* For projected boxes, the rectangle boundary is densified before reprojection
  to better preserve the intended footprint.
* `--bbox-lonlat` also supports a box crossing the antimeridian by specifying
  WEST > EAST, for example: 170 -80 -170 -75.
* The ITS_LIVE STAC `percent_valid_pixels` property, when present, describes
  the granule overall. It is not the same as the valid-data fraction inside
  your requested ROI.
* ROI-specific valid coverage requires inspecting the actual velocity arrays
  (or, preferably in many cases, the cloud-optimized ITS_LIVE Zarr cubes).
* If an HTTPS asset requires NASA Earthdata authentication, set EDL_TOKEN in
  your environment. The downloader and size probes will add it as a Bearer token.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Any
from urllib.parse import urlparse

import requests
from pyproj import Transformer
from pystac_client import Client
from pystac_client.stac_api_io import StacApiIO
from tqdm import tqdm


STAC_URL = "https://stac.itslive.cloud"
COLLECTION = "itslive-granules"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Inventory/download ITS_LIVE image-pair velocity granules over a "
            "projected or lon/lat bounding box."
        )
    )

    roi = p.add_mutually_exclusive_group(required=True)
    roi.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
        help=(
            "Bounding box in --epsg coordinates. By default this is Antarctic "
            "Polar Stereographic EPSG:3031, in meters."
        ),
    )
    roi.add_argument(
        "--bbox-lonlat",
        type=float,
        nargs=4,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help=(
            "Bounding box directly in WGS84 lon/lat degrees. WEST > EAST is "
            "allowed for a box crossing the antimeridian."
        ),
    )

    p.add_argument(
        "--epsg",
        type=int,
        default=3031,
        help="EPSG code of --bbox. Ignored with --bbox-lonlat. Default: 3031.",
    )
    p.add_argument("--start", required=True, help="Start date/time, e.g. 2019-01-01.")
    p.add_argument("--end", required=True, help="End date/time, e.g. 2024-12-31.")
    p.add_argument(
        "--mission",
        help=(
            "Optional mission filter, e.g. SENTINEL-1, SENTINEL-2, LANDSAT, "
            "LANDSAT-8, S1A, or S1B. Matching uses the STAC platform field."
        ),
    )
    p.add_argument("--min-pair-days", type=float, default=None)
    p.add_argument("--max-pair-days", type=float, default=None)
    p.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Output directory.",
    )
    p.add_argument(
        "--edge-points",
        type=int,
        default=25,
        help=(
            "Samples per projected bbox edge before reprojection. "
            "Used only with --bbox. Default: 25."
        ),
    )
    p.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional maximum number of STAC items to return. Default: all matches.",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the STAC query progress bar.",
    )
    p.add_argument(
        "--min-valid-percent",
        type=float,
        default=None,
        help=(
            "Optional post-filter using STAC percent_valid_pixels "
            "(or a compatible property if present). This is granule-wide, "
            "not ROI-specific."
        ),
    )
    p.add_argument(
        "--skip-size-probe",
        action="store_true",
        help=(
            "Do not make HTTP HEAD/range requests for assets whose size is absent "
            "from STAC metadata. Size estimates will use only catalog metadata."
        ),
    )
    p.add_argument(
        "--size-workers",
        type=int,
        default=8,
        help="Parallel workers for lightweight asset-size probes. Default: 8.",
    )
    p.add_argument(
        "--download-data",
        action="store_true",
        help="Download the primary data asset for each retained item.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel download workers used with --download-data. Default: 4.",
    )
    p.add_argument(
        "--download-retries",
        type=int,
        default=10,
        help="Attempts per download, retaining partial files between attempts. Default: 10.",
    )
    p.add_argument(
        "--retry-backoff",
        type=float,
        default=5.0,
        help="Initial retry delay in seconds (exponential, capped at 60). Default: 5.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files that already exist.",
    )
    return p.parse_args()


def parse_datetime(value: str) -> datetime:
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def time_bounds(start: str, end: str) -> tuple[datetime, datetime]:
    start_dt = parse_datetime(start)
    end_dt = parse_datetime(end)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(end).strip()):
        end_exclusive = end_dt + timedelta(days=1)
    else:
        end_exclusive = end_dt + timedelta(microseconds=1)
    if start_dt >= end_exclusive:
        raise ValueError("--start must not be after --end")
    return start_dt, end_exclusive


def platform_matches(platform: Any, requested: str | None) -> bool:
    if not requested:
        return True
    value = re.sub(r"[^A-Z0-9]", "", str(platform or "").upper())
    wanted = re.sub(r"[^A-Z0-9]", "", str(requested).upper())
    aliases = {
        "SENTINEL1": "S1",
        "SENTINEL2": "S2",
        "LANDSAT4": "L4",
        "LANDSAT5": "L5",
        "LANDSAT7": "L7",
        "LANDSAT8": "L8",
        "LANDSAT9": "L9",
    }
    wanted = aliases.get(wanted, wanted)
    if wanted == "LANDSAT":
        return value.startswith(("LC", "LE", "LT", "LO", "LM"))
    if wanted.startswith("L") and len(wanted) == 2 and wanted[1].isdigit():
        return value.startswith(
            ("LC" + wanted[1], "LE" + wanted[1], "LT" + wanted[1], "LO" + wanted[1])
        )
    return value.startswith(wanted)


def record_matches(record: dict[str, Any], args: argparse.Namespace) -> bool:
    midpoint = record.get("datetime") or record.get("mid_datetime")
    if not midpoint:
        return False
    start_dt, end_exclusive = time_bounds(args.start, args.end)
    try:
        mid_dt = parse_datetime(str(midpoint))
    except (TypeError, ValueError):
        return False
    if not (start_dt <= mid_dt < end_exclusive):
        return False
    if not platform_matches(record.get("platform"), args.mission):
        return False
    pair_days = numeric_or_none(record.get("pair_interval_days"))
    if args.min_pair_days is not None and (
        pair_days is None or abs(pair_days) < args.min_pair_days
    ):
        return False
    if args.max_pair_days is not None and (
        pair_days is None or abs(pair_days) > args.max_pair_days
    ):
        return False
    return True


def stac_cql_filter(args: argparse.Namespace) -> dict[str, Any] | None:
    """Build an equivalent server-side filter; local filtering remains authoritative."""
    clauses: list[dict[str, Any]] = []
    if args.mission:
        wanted = re.sub(r"[^A-Z0-9]", "", str(args.mission).upper())
        platforms: list[str] | None = {
            "SENTINEL1": ["S1A", "S1B"],
            "S1": ["S1A", "S1B"],
            "SENTINEL2": ["S2A", "S2B"],
            "S2": ["S2A", "S2B"],
            "LANDSAT8": ["LC08", "LO08"],
            "L8": ["LC08", "LO08"],
            "LANDSAT9": ["LC09", "LO09"],
            "L9": ["LC09", "LO09"],
        }.get(wanted)
        if platforms:
            clauses.append(
                {"op": "in", "args": [{"property": "platform"}, platforms]}
            )
        elif wanted in {"S1A", "S1B", "S2A", "S2B", "LC08", "LC09"}:
            clauses.append(
                {"op": "=", "args": [{"property": "platform"}, wanted]}
            )
    if args.min_pair_days is not None:
        clauses.append(
            {"op": ">=", "args": [{"property": "date_dt"}, args.min_pair_days]}
        )
    if args.max_pair_days is not None:
        clauses.append(
            {"op": "<=", "args": [{"property": "date_dt"}, args.max_pair_days]}
        )
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"op": "and", "args": clauses}


def densified_bbox_ring(
    bbox: tuple[float, float, float, float], points_per_edge: int
) -> list[tuple[float, float]]:
    """Return a densified closed ring around a projected rectangle."""
    xmin, ymin, xmax, ymax = bbox
    if not (xmin < xmax and ymin < ymax):
        raise ValueError("Expected XMIN < XMAX and YMIN < YMAX.")
    if points_per_edge < 2:
        raise ValueError("--edge-points must be >= 2.")

    def segment(
        x0: float, y0: float, x1: float, y1: float, n: int
    ) -> list[tuple[float, float]]:
        # Exclude the segment endpoint to avoid duplicate vertices at corners.
        return [
            (x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n)
            for i in range(n)
        ]

    ring: list[tuple[float, float]] = []
    ring += segment(xmin, ymin, xmax, ymin, points_per_edge)
    ring += segment(xmax, ymin, xmax, ymax, points_per_edge)
    ring += segment(xmax, ymax, xmin, ymax, points_per_edge)
    ring += segment(xmin, ymax, xmin, ymin, points_per_edge)
    ring.append(ring[0])
    return ring


def projected_bbox_to_wgs84_geojson(
    bbox: tuple[float, float, float, float],
    epsg: int,
    points_per_edge: int,
) -> dict[str, Any]:
    """Reproject a densified projected bbox boundary to WGS84 GeoJSON."""
    ring_xy = densified_bbox_ring(bbox, points_per_edge)
    transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)

    ring_ll: list[list[float]] = []
    for x, y in ring_xy:
        lon, lat = transformer.transform(x, y)
        if not (math.isfinite(lon) and math.isfinite(lat)):
            raise ValueError(f"Non-finite lon/lat produced from ({x}, {y}).")
        ring_ll.append([float(lon), float(lat)])

    return {"type": "Polygon", "coordinates": [ring_ll]}


def lonlat_rectangle_polygon(
    west: float, south: float, east: float, north: float
) -> dict[str, Any]:
    """Construct a WGS84 rectangle, splitting at the antimeridian if needed."""
    if not (-180.0 <= west <= 180.0 and -180.0 <= east <= 180.0):
        raise ValueError("WEST and EAST must be in [-180, 180].")
    if not (-90.0 <= south <= 90.0 and -90.0 <= north <= 90.0):
        raise ValueError("SOUTH and NORTH must be in [-90, 90].")
    if not south < north:
        raise ValueError("Expected SOUTH < NORTH.")
    if west == east:
        raise ValueError("WEST and EAST must differ.")

    def ring(w: float, e: float) -> list[list[float]]:
        return [
            [w, south],
            [e, south],
            [e, north],
            [w, north],
            [w, south],
        ]

    if west < east:
        return {"type": "Polygon", "coordinates": [ring(west, east)]}

    # WEST > EAST means the requested box crosses +/-180 degrees.
    return {
        "type": "MultiPolygon",
        "coordinates": [
            [ring(west, 180.0)],
            [ring(-180.0, east)],
        ],
    }


def get_roi_geojson_and_metadata(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if args.bbox_lonlat is not None:
        bbox = tuple(float(v) for v in args.bbox_lonlat)
        west, south, east, north = bbox
        geometry = lonlat_rectangle_polygon(west, south, east, north)
        metadata = {
            "input_bbox_type": "lonlat",
            "input_crs": "EPSG:4326",
            "input_epsg": 4326,
            "input_bbox": list(bbox),
        }
        return geometry, metadata

    bbox = tuple(float(v) for v in args.bbox)
    geometry = projected_bbox_to_wgs84_geojson(
        bbox=bbox,
        epsg=args.epsg,
        points_per_edge=args.edge_points,
    )
    metadata = {
        "input_bbox_type": "projected",
        "input_crs": f"EPSG:{args.epsg}",
        "input_epsg": args.epsg,
        "input_bbox": list(bbox),
    }
    return geometry, metadata


def first_present(mapping: dict[str, Any], names: list[str]) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return None


def choose_data_asset(item) -> tuple[str | None, Any | None]:
    """Prefer asset 'data'; otherwise choose the most NetCDF-like asset."""
    if "data" in item.assets:
        return "data", item.assets["data"]

    for key, asset in item.assets.items():
        href = (asset.href or "").lower()
        media_type = (asset.media_type or "").lower()
        if href.endswith((".nc", ".nc4", ".netcdf")) or "netcdf" in media_type:
            return key, asset

    if item.assets:
        key = next(iter(item.assets))
        return key, item.assets[key]

    return None, None


def alternate_s3_href(asset) -> str | None:
    if asset is None:
        return None
    alt = asset.extra_fields.get("alternate", {})
    if isinstance(alt, dict):
        s3 = alt.get("s3", {})
        if isinstance(s3, dict):
            return s3.get("href")
    return None


def asset_size_from_stac(asset) -> int | None:
    """Return asset size from common STAC metadata fields, if present."""
    if asset is None:
        return None

    candidates = [
        asset.extra_fields.get("file:size"),
        asset.extra_fields.get("size"),
        asset.extra_fields.get("content_length"),
        asset.extra_fields.get("content-length"),
    ]
    for value in candidates:
        try:
            n = int(value)
        except (TypeError, ValueError):
            continue
        if n >= 0:
            return n
    return None


def item_record(item) -> dict[str, Any]:
    p = item.properties
    asset_key, asset = choose_data_asset(item)

    percent_valid = first_present(
        p,
        [
            "percent_valid_pixels",
            "roi_valid_percentage",
            "valid_pixel_percent",
            "valid_pixels_percent",
        ],
    )
    pair_days = first_present(
        p,
        [
            "date_dt",
            "pair_interval_days",
            "img_pair_dt",
            "temporal_baseline_days",
        ],
    )
    platform = first_present(
        p,
        [
            "platform",
            "sat:platform",
            "mission",
            "constellation",
        ],
    )
    proj_code = first_present(
        p,
        [
            "proj:code",
            "proj:epsg",
        ],
    )

    bbox = item.bbox or [None, None, None, None]
    if len(bbox) != 4:
        bbox = [None, None, None, None]

    size_bytes = asset_size_from_stac(asset)

    return {
        "item_id": item.id,
        "datetime": p.get("datetime"),
        "start_datetime": p.get("start_datetime"),
        "end_datetime": p.get("end_datetime"),
        "mid_datetime": p.get("mid_datetime"),
        "platform": platform,
        "pair_interval_days": pair_days,
        "percent_valid_pixels": percent_valid,
        "proj_code": proj_code,
        "bbox_west": bbox[0],
        "bbox_south": bbox[1],
        "bbox_east": bbox[2],
        "bbox_north": bbox[3],
        "asset_key": asset_key,
        "asset_type": None if asset is None else asset.media_type,
        "data_href": None if asset is None else asset.href,
        "data_s3_href": alternate_s3_href(asset),
        "asset_size_bytes": size_bytes,
        "asset_size_mib": None if size_bytes is None else size_bytes / 1024**2,
        "asset_size_source": None if size_bytes is None else "stac",
    }


def numeric_or_none(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def int_or_none(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def human_bytes(n: float | int | None) -> str:
    if n is None:
        return "unknown"
    value = float(n)
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    i = 0
    while abs(value) >= 1024.0 and i < len(units) - 1:
        value /= 1024.0
        i += 1
    if i == 0:
        return f"{value:.0f} {units[i]}"
    if value >= 100:
        return f"{value:.0f} {units[i]}"
    if value >= 10:
        return f"{value:.1f} {units[i]}"
    return f"{value:.2f} {units[i]}"


def parse_total_from_content_range(value: str | None) -> int | None:
    if not value:
        return None
    # Expected form: "bytes 0-0/12345678"
    m = re.search(r"/(\d+)\s*$", value)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def request_headers(bearer_token: str | None) -> dict[str, str]:
    headers = {"User-Agent": "itslive-stac-inventory/2.0"}
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    return headers


def probe_http_size(
    href: str,
    bearer_token: str | None,
) -> tuple[int | None, str | None]:
    """
    Determine remote file size without downloading the full file.

    First try HEAD. If that does not expose Content-Length, issue a streamed
    GET with Range: bytes=0-0 and inspect Content-Range/Content-Length.
    The streamed response body is not consumed.
    """
    headers = request_headers(bearer_token)

    try:
        with requests.head(
            href,
            allow_redirects=True,
            timeout=(15, 45),
            headers=headers,
        ) as r:
            if r.ok:
                value = r.headers.get("Content-Length")
                if value is not None:
                    try:
                        return int(value), "http-head"
                    except ValueError:
                        pass
    except requests.RequestException:
        pass

    range_headers = dict(headers)
    range_headers["Range"] = "bytes=0-0"

    try:
        with requests.get(
            href,
            allow_redirects=True,
            stream=True,
            timeout=(15, 45),
            headers=range_headers,
        ) as r:
            if not r.ok:
                return None, None

            total = parse_total_from_content_range(r.headers.get("Content-Range"))
            if total is not None:
                return total, "http-range"

            # If the server ignored Range and returned 200 with a normal
            # Content-Length, that header still describes the full object.
            value = r.headers.get("Content-Length")
            if value is not None and r.status_code == 200:
                try:
                    return int(value), "http-get-header"
                except ValueError:
                    pass
    except requests.RequestException:
        pass

    return None, None


def populate_missing_sizes(
    records: list[dict[str, Any]],
    workers: int,
) -> None:
    """Probe remote asset sizes for records that lack STAC file-size metadata."""
    missing = [
        r for r in records
        if r.get("asset_size_bytes") is None and r.get("data_href")
    ]
    if not missing:
        return

    token = os.environ.get("EDL_TOKEN")
    print(
        f"Asset size missing from STAC for {len(missing)} item(s); "
        f"probing remote headers with {max(1, workers)} worker(s)..."
    )

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = {
            ex.submit(probe_http_size, str(r["data_href"]), token): r
            for r in missing
        }
        completed = 0
        resolved = 0
        for fut in as_completed(futures):
            r = futures[fut]
            completed += 1
            try:
                size_bytes, source = fut.result()
            except Exception:
                size_bytes, source = None, None

            if size_bytes is not None:
                r["asset_size_bytes"] = size_bytes
                r["asset_size_mib"] = size_bytes / 1024**2
                r["asset_size_source"] = source
                resolved += 1

            # Lightweight progress without printing hundreds of item IDs.
            if completed % 100 == 0:
                print(
                    f"  size probes: {completed}/{len(missing)} checked, "
                    f"{resolved} resolved"
                )

    print(f"Resolved sizes for {resolved}/{len(missing)} probed item(s).")


def compute_size_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    known_sizes = [
        int_or_none(r.get("asset_size_bytes"))
        for r in records
    ]
    known_sizes = [n for n in known_sizes if n is not None and n >= 0]

    known_count = len(known_sizes)
    total_count = len(records)
    unknown_count = total_count - known_count
    known_total = sum(known_sizes)

    median_size = int(median(known_sizes)) if known_sizes else None

    if unknown_count == 0:
        estimated_total = known_total
        estimate_method = "exact_sum_of_known_asset_sizes"
    elif median_size is not None:
        estimated_total = known_total + unknown_count * median_size
        estimate_method = "known_sizes_plus_median_known_size_for_unknown_assets"
    else:
        estimated_total = None
        estimate_method = "unavailable"

    return {
        "asset_count": total_count,
        "asset_size_known_count": known_count,
        "asset_size_unknown_count": unknown_count,
        "known_download_bytes": known_total,
        "known_download_human": human_bytes(known_total),
        "median_known_asset_bytes": median_size,
        "median_known_asset_human": human_bytes(median_size),
        "estimated_download_bytes": estimated_total,
        "estimated_download_human": human_bytes(estimated_total),
        "download_estimate_method": estimate_method,
        "download_estimate_is_extrapolated": unknown_count > 0 and estimated_total is not None,
    }


def write_outputs(
    outdir: Path,
    roi_geojson: dict[str, Any],
    roi_metadata: dict[str, Any],
    items: list,
    records: list[dict[str, Any]],
    returned_item_count: int,
    args: argparse.Namespace,
    size_summary: dict[str, Any],
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    with (outdir / "query_roi_wgs84.geojson").open("w") as f:
        json.dump(
            {
                "type": "Feature",
                "properties": roi_metadata,
                "geometry": roi_geojson,
            },
            f,
            indent=2,
        )

    fieldnames = [
        "item_id",
        "datetime",
        "start_datetime",
        "end_datetime",
        "mid_datetime",
        "platform",
        "pair_interval_days",
        "percent_valid_pixels",
        "proj_code",
        "bbox_west",
        "bbox_south",
        "bbox_east",
        "bbox_north",
        "asset_key",
        "asset_type",
        "asset_size_bytes",
        "asset_size_mib",
        "asset_size_source",
        "data_href",
        "data_s3_href",
    ]
    with (outdir / "items.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(records)

    with (outdir / "items.ndjson").open("w") as f:
        for item in items:
            f.write(json.dumps(item.to_dict(), separators=(",", ":")) + "\n")

    valid_values = [
        numeric_or_none(r["percent_valid_pixels"]) for r in records
    ]
    valid_values = [v for v in valid_values if v is not None]

    platforms = sorted(
        {str(r["platform"]) for r in records if r["platform"] not in (None, "")}
    )

    summary = {
        "stac_url": STAC_URL,
        "collection": COLLECTION,
        **roi_metadata,
        "start": args.start,
        "end": args.end,
        "mission_filter": args.mission,
        "min_pair_days": args.min_pair_days,
        "max_pair_days": args.max_pair_days,
        "returned_item_count": returned_item_count,
        "retained_item_count": len(records),
        "min_valid_percent_filter": args.min_valid_percent,
        "platforms_seen": platforms,
        "granule_valid_percent_min": min(valid_values) if valid_values else None,
        "granule_valid_percent_median": (
            median(valid_values) if valid_values else None
        ),
        "granule_valid_percent_max": max(valid_values) if valid_values else None,
        **size_summary,
        "important_note": (
            "STAC percent_valid_pixels is granule-wide. ROI-specific valid-data "
            "coverage requires reading the velocity arrays or using ITS_LIVE Zarr cubes."
        ),
    }
    with (outdir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)


def destination_name(item_id: str, href: str) -> str:
    name = Path(urlparse(href).path).name
    if name:
        return name
    return f"{item_id}.nc"


def download_one(
    record: dict[str, Any],
    download_dir: Path,
    overwrite: bool,
    bearer_token: str | None,
    retries: int,
    retry_backoff: float,
) -> tuple[str, str]:
    href = record.get("data_href")
    item_id = str(record["item_id"])
    if not href:
        return item_id, "NO_ASSET"

    dest = download_dir / destination_name(item_id, href)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if overwrite and tmp.exists():
        tmp.unlink()
    expected_size = int_or_none(record.get("asset_size_bytes"))
    if dest.exists() and not overwrite:
        if expected_size is None or dest.stat().st_size == expected_size:
            return item_id, f"SKIP {dest.name}"
        if not tmp.exists() or dest.stat().st_size > tmp.stat().st_size:
            dest.replace(tmp)

    headers = request_headers(bearer_token)

    last_error: Exception | None = None
    for attempt in range(1, max(1, retries) + 1):
        offset = tmp.stat().st_size if tmp.exists() else 0
        attempt_headers = dict(headers)
        if offset:
            attempt_headers["Range"] = f"bytes={offset}-"
        try:
            with requests.get(
                href,
                stream=True,
                timeout=(30, 300),
                headers=attempt_headers,
                allow_redirects=True,
            ) as r:
                r.raise_for_status()
                append = offset > 0 and r.status_code == 206
                mode = "ab" if append else "wb"
                with tmp.open(mode) as f:
                    for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            f.write(chunk)
            actual_size = tmp.stat().st_size
            if expected_size is not None and actual_size != expected_size:
                raise IOError(
                    f"size mismatch for {dest.name}: got {actual_size}, expected {expected_size}"
                )
            tmp.replace(dest)
            return item_id, f"OK {dest.name}"
        except (requests.RequestException, OSError) as exc:
            last_error = exc
            if attempt >= max(1, retries):
                break
            delay = min(60.0, max(0.0, retry_backoff) * (2 ** (attempt - 1)))
            print(
                f"{item_id}: retry {attempt}/{retries} after {type(exc).__name__}; "
                f"{tmp.stat().st_size if tmp.exists() else 0} bytes saved, waiting {delay:g}s",
                file=sys.stderr,
            )
            time.sleep(delay)
    raise RuntimeError(f"{item_id}: download failed after {retries} attempts: {last_error}")


def download_assets(
    records: list[dict[str, Any]],
    outdir: Path,
    workers: int,
    overwrite: bool,
    retries: int,
    retry_backoff: float,
) -> None:
    download_dir = outdir / "data"
    download_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("EDL_TOKEN")

    print(f"Downloading {len(records)} asset(s) with {workers} worker(s)...")
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = [
            ex.submit(
                download_one,
                r,
                download_dir,
                overwrite,
                token,
                retries,
                retry_backoff,
            )
            for r in records
        ]
        failures: list[str] = []
        for fut in as_completed(futures):
            try:
                item_id, status = fut.result()
                print(f"{item_id}: {status}")
            except Exception as e:
                print(f"DOWNLOAD ERROR: {e}", file=sys.stderr)
                failures.append(str(e))
    if failures:
        raise RuntimeError(
            f"{len(failures)} download(s) remain incomplete; rerun the same command to resume."
        )


def main() -> None:
    args = parse_args()

    if args.min_pair_days is not None and args.min_pair_days < 0:
        raise SystemExit("ERROR: --min-pair-days must be nonnegative")
    if args.max_pair_days is not None and args.max_pair_days < 0:
        raise SystemExit("ERROR: --max-pair-days must be nonnegative")
    if (
        args.min_pair_days is not None
        and args.max_pair_days is not None
        and args.min_pair_days > args.max_pair_days
    ):
        raise SystemExit("ERROR: --min-pair-days cannot exceed --max-pair-days")
    if args.workers < 1 or args.download_retries < 1:
        raise SystemExit("ERROR: --workers and --download-retries must be at least 1")

    roi_geojson, roi_metadata = get_roi_geojson_and_metadata(args)
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Save the ROI before querying so it can be inspected even if the query fails.
    with (args.outdir / "query_roi_wgs84.geojson").open("w") as f:
        json.dump(
            {
                "type": "Feature",
                "properties": roi_metadata,
                "geometry": roi_geojson,
            },
            f,
            indent=2,
        )

    print(f"Opening ITS_LIVE STAC: {STAC_URL}")
    catalog = Client.open(
        STAC_URL,
        stac_io=StacApiIO(timeout=(30, 120), max_retries=10),
    )

    search_kwargs: dict[str, Any] = {
        "collections": [COLLECTION],
        "intersects": roi_geojson,
        "datetime": f"{args.start}/{args.end}",
    }
    cql_filter = stac_cql_filter(args)
    if cql_filter is not None:
        search_kwargs["filter"] = cql_filter
        search_kwargs["filter_lang"] = "cql2-json"
    if args.max_items is not None:
        search_kwargs["max_items"] = args.max_items

    print("Preparing STAC search...")
    search = catalog.search(**search_kwargs)

    # If the server supports the STAC Context Extension, PySTAC Client can ask
    # how many items match before we retrieve all paginated results.
    matched = None
    try:
        print("Counting matching granules...")
        matched = search.matched()
    except Exception:
        # Some STAC APIs do not expose a total match count.
        matched = None

    if matched is not None:
        expected = matched
        if args.max_items is not None:
            expected = min(expected, args.max_items)
        print(f"Query matches {matched:,} granule(s); retrieving {expected:,}.")
    else:
        expected = args.max_items
        print("Server did not provide a total match count; retrieving results...")

    items_all = []
    iterator = search.items()

    if args.no_progress:
        for item in iterator:
            items_all.append(item)
            if len(items_all) % 100 == 0:
                if expected is None:
                    print(f"  retrieved {len(items_all):,} granule(s)...")
                else:
                    print(
                        f"  retrieved {len(items_all):,}/{expected:,} granule(s)..."
                    )
    else:
        with tqdm(
            total=expected,
            unit="granule",
            desc="STAC query",
            dynamic_ncols=True,
        ) as progress:
            for item in iterator:
                items_all.append(item)
                progress.update(1)

    print(f"STAC retrieval complete: {len(items_all):,} item(s).")

    records_all = [item_record(item) for item in items_all]

    keep_mask = []
    for rec in records_all:
        keep = record_matches(rec, args)
        if keep and args.min_valid_percent is not None:
            v = numeric_or_none(rec["percent_valid_pixels"])
            keep = v is not None and v >= args.min_valid_percent
        keep_mask.append(keep)

    kept_items = [item for item, keep in zip(items_all, keep_mask) if keep]
    records = [rec for rec, keep in zip(records_all, keep_mask) if keep]

    print(
        f"Retained {len(records)} item(s) after midpoint-date, mission, "
        "pair-duration, and optional validity filters."
    )

    if not args.skip_size_probe:
        populate_missing_sizes(records, workers=args.size_workers)

    size_summary = compute_size_summary(records)

    print("")
    print("Download-size estimate")
    print("----------------------")
    print(
        f"Assets with known size: "
        f"{size_summary['asset_size_known_count']}/{size_summary['asset_count']}"
    )
    print(f"Known-size subtotal:    {size_summary['known_download_human']}")

    if size_summary["estimated_download_bytes"] is not None:
        label = (
            "Estimated total:        "
            if size_summary["download_estimate_is_extrapolated"]
            else "Total download size:    "
        )
        print(f"{label}{size_summary['estimated_download_human']}")
        if size_summary["download_estimate_is_extrapolated"]:
            print(
                f"  (extrapolated for {size_summary['asset_size_unknown_count']} "
                f"unknown asset(s) using median known size "
                f"{size_summary['median_known_asset_human']})"
            )
    else:
        print("Estimated total:        unavailable")

    write_outputs(
        outdir=args.outdir,
        roi_geojson=roi_geojson,
        roi_metadata=roi_metadata,
        items=kept_items,
        records=records,
        returned_item_count=len(items_all),
        args=args,
        size_summary=size_summary,
    )

    print("")
    print(f"Wrote metadata to: {args.outdir}")
    print("  query_roi_wgs84.geojson")
    print("  items.csv")
    print("  items.ndjson")
    print("  summary.json")

    if args.download_data:
        download_assets(
            records=records,
            outdir=args.outdir,
            workers=args.workers,
            overwrite=args.overwrite,
            retries=args.download_retries,
            retry_backoff=args.retry_backoff,
        )
    else:
        print("Data files were NOT downloaded. Add --download-data when ready.")


if __name__ == "__main__":
    main()
