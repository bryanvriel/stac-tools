import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS
from shapely.geometry import Polygon

import itslive_cube_coverage as tool
import itslive_multi_cube as multi


class CoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deps = tool.require_dependencies()

    def dataset(self):
        # Four observations, three rows, four columns. The first Sentinel pair is
        # valid everywhere except one declared missing value. The second is mixed.
        v = np.array(
            [
                [[1, 1, 1, 1], [1, -32767, 1, 1], [1, 1, 1, 1]],
                [[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]],
                [[3, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3]],
                [[4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4]],
            ],
            dtype=np.int16,
        )
        ds = xr.Dataset(
            {
                "v": (("mid_date", "y", "x"), v, {"missing_value": -32767, "grid_mapping": "mapping"}),
                "v_error": (
                    ("mid_date", "y", "x"),
                    np.broadcast_to(
                        np.asarray([1.0, 2.0, 3.0, 4.0])[:, None, None], v.shape
                    ).copy(),
                ),
                "date_center": ("mid_date", pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2021-01-01"])),
                "acquisition_date_img1": ("mid_date", pd.to_datetime(["2019-12-26", "2019-12-27", "2019-12-20", "2020-12-25"])),
                "acquisition_date_img2": ("mid_date", pd.to_datetime(["2020-01-07", "2020-01-08", "2020-01-17", "2021-01-08"])),
                "date_dt": ("mid_date", [12.0, 12.0, 28.0, 14.0]),
                "mission_img1": ("mid_date", ["S1", "S1", "L8", "S1"]),
                "mission_img2": ("mid_date", ["S1", "L8", "L8", "S1"]),
                "sensor_img1": ("mid_date", ["C-SAR", "C-SAR", "OLI", "C-SAR"]),
                "sensor_img2": ("mid_date", ["C-SAR", "OLI", "OLI", "C-SAR"]),
                "mapping": ((), "", {"spatial_epsg": 3031}),
            },
            coords={"mid_date": np.arange(4), "x": [0.0, 10.0, 20.0, 30.0], "y": [20.0, 10.0, 0.0]},
        )
        return ds

    def args(self, **changes):
        values = dict(
            start="2020-01-01", end="2020-01-03", mission="SENTINEL-1", sensor=None,
            min_pair_days=None, max_pair_days=12.0,
        )
        values.update(changes)
        return Namespace(**values)

    def test_schema_filter_mask_and_count(self):
        ds = self.dataset()
        schema = tool.ITSLiveCubeSchema(ds)
        self.assertEqual(schema.obs, "mid_date")
        self.assertEqual(schema.crs(self.deps), CRS.from_epsg(3031))
        table = tool.observation_table(ds, schema, self.deps)
        self.assertEqual(table.mission.tolist(), ["SENTINEL-1", "MIXED", "LANDSAT-8", "SENTINEL-1"])
        selected, counts, _ = tool.filter_observations(table, self.args(), self.deps)
        self.assertEqual(selected.observation_index.tolist(), [0])
        self.assertEqual(counts["within_time_range"], 3)  # end date is inclusive

        # Covers centers (0,0), (10,0), (0,10), and (10,10), including boundary.
        roi = Polygon([(-1, -1), (10, -1), (10, 10), (-1, 10), (-1, -1)])
        subset, mask = tool.coordinate_subset(ds, schema, roi, self.deps)
        self.assertEqual(int(mask.sum()), 4)
        count, fraction = tool.compute_coverage(subset, schema, selected, mask, self.deps)
        inside = count.values[mask]
        self.assertEqual(sorted(inside.tolist()), [0, 1, 1, 1])
        self.assertTrue(np.all((fraction.values[mask] >= 0) & (fraction.values[mask] <= 1)))

    def test_outputs_are_georeferenced_and_masked(self):
        ds = self.dataset()
        schema = tool.ITSLiveCubeSchema(ds)
        table = tool.observation_table(ds, schema, self.deps)
        selected, _, _ = tool.filter_observations(table, self.args(), self.deps)
        roi = Polygon([(-1, -1), (21, -1), (21, 11), (-1, 11), (-1, -1)])
        subset, mask = tool.coordinate_subset(ds, schema, roi, self.deps)
        count, fraction = tool.compute_coverage(subset, schema, selected, mask, self.deps)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            tool.write_netcdf(count, fraction, mask, schema, {"test": True, "optional": None}, out, self.deps)
            tool.write_geotiffs(count, fraction, schema, CRS.from_epsg(3031), out, self.deps)
            loaded = xr.open_dataset(out / "coverage_count.nc")
            self.assertEqual(int(loaded.roi_mask.sum()), int(mask.sum()))
            import rasterio
            with rasterio.open(out / "coverage_count.tif") as src:
                self.assertEqual(src.crs.to_epsg(), 3031)
                self.assertEqual(src.nodata, tool.COUNT_NODATA)
                self.assertLess(src.transform.e, 0)

    def test_grouped_landsat_and_zero_result(self):
        table = tool.observation_table(self.dataset(), tool.ITSLiveCubeSchema(self.dataset()), self.deps)
        selected, _, _ = tool.filter_observations(
            table, self.args(mission="LANDSAT", max_pair_days=None), self.deps
        )
        self.assertEqual(selected.observation_index.tolist(), [2])
        empty, _, _ = tool.filter_observations(
            table, self.args(start="2030-01-01", end="2030-12-31"), self.deps
        )
        self.assertTrue(empty.empty)

    def test_compact_real_cube_mission_codes(self):
        self.assertEqual(tool.normalize_pair_mission("S", "1A"), "SENTINEL-1")
        self.assertEqual(tool.normalize_pair_mission("S", "1B"), "SENTINEL-1")
        self.assertEqual(tool.normalize_pair_mission("S", "2A"), "SENTINEL-2")
        self.assertEqual(tool.normalize_pair_mission("L", "8"), "LANDSAT-8")

    def test_remote_metadata_failures_are_retryable(self):
        self.assertTrue(multi.transient_remote_error(tool.CubeOpenError("failed")))
        self.assertTrue(
            multi.transient_remote_error(
                tool.IncompleteCubeMetadataError("empty response")
            )
        )

    def test_sentinel1_pair_scene_parser(self):
        pair_id = (
            "https://example.invalid/S1B_IW_SLC__1SSH_20210104T135720_"
            "20210104T135750_025005_02F9E5_56FB_X_S1B_IW_SLC__1SSH_"
            "20210116T135720_20210116T135750_025180_02FF85_9D21_G0120V02_P098.nc"
        )
        first, second = multi.sentinel1_pair_scenes(pair_id)
        self.assertTrue(first.startswith("S1B_IW_SLC__1SSH_20210104"))
        self.assertTrue(second.startswith("S1B_IW_SLC__1SSH_20210116"))

    def test_overwrite_removes_stale_products(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            stale = out / "coverage_count.tif"
            stale.write_bytes(b"old result")
            tool.check_output_paths(out, overwrite=True)
            self.assertFalse(stale.exists())

    def test_phase_offset_multi_cube_mosaic(self):
        import rasterio
        from rasterio.transform import from_origin

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = []
            for number, (cube_id, bbox, left, value, denominator) in enumerate(
                [
                    ("left", [0, 0, 100, 100], 2.0, 10, 100),
                    ("right", [100, 0, 200, 100], 102.0, 20, 200),
                ]
            ):
                tile = root / cube_id / "mosaic_source"
                tile.mkdir(parents=True)
                profile = dict(
                    driver="GTiff", width=8, height=8, count=1, dtype="uint32",
                    nodata=tool.COUNT_NODATA, crs="EPSG:3031",
                    transform=from_origin(left, 98, 12, 12),
                )
                for name, scalar in [
                    ("coverage_count.tif", value),
                    ("selected_observation_count.tif", denominator),
                ]:
                    with rasterio.open(tile / name, "w", **profile) as dst:
                        dst.write(np.full((8, 8), scalar, dtype=np.uint32), 1)
                results.append(
                    {
                        "crs": "EPSG:3031",
                        "tile_dir": str(root / cube_id),
                        "cube": {"id": cube_id, "proj_bbox": bbox},
                    }
                )
            roi = tool.ROI(
                Polygon([(0, 0), (200, 0), (200, 100), (0, 100), (0, 0)]),
                CRS.from_epsg(3031), "bbox", [0, 0, 200, 100],
            )
            count, fraction, denominator, mask, _ = multi.mosaic_tiles(
                results, roi, 12, root, self.deps
            )
            self.assertFalse(np.any(count.values[mask] == tool.COUNT_NODATA))
            left_values = count.values[:, count.x.values < 100][mask[:, count.x.values < 100]]
            right_values = count.values[:, count.x.values >= 100][mask[:, count.x.values >= 100]]
            self.assertTrue(np.all(left_values == 10))
            self.assertTrue(np.all(right_values == 20))
            self.assertTrue(np.allclose(fraction.values[mask], 0.1))

    def test_blocked_coverage_resumes_completed_blocks(self):
        ds = self.dataset()
        schema = tool.ITSLiveCubeSchema(ds)
        table = tool.observation_table(ds, schema, self.deps)
        selected, _, _ = tool.filter_observations(table, self.args(), self.deps)
        with tempfile.TemporaryDirectory() as tmp:
            tile_dir = Path(tmp) / "tile"
            count, fraction = multi.compute_coverage_blocked(
                ds, schema, selected, tile_dir, "test-fingerprint", 2, self.deps
            )
            state_path = tile_dir / "work" / "block_status.json"
            state = __import__("json").loads(state_path.read_text())
            self.assertEqual(state["status"], "complete")
            self.assertEqual(state["completed_count"], 4)
            resumed_count, resumed_fraction = multi.compute_coverage_blocked(
                ds, schema, selected, tile_dir, "test-fingerprint", 2, self.deps
            )
            np.testing.assert_array_equal(resumed_count.values, count.values)
            np.testing.assert_array_equal(resumed_fraction.values, fraction.values)

    def test_blocked_coverage_writes_observation_cache(self):
        import h5py

        ds = self.dataset()
        schema = tool.ITSLiveCubeSchema(ds)
        table = tool.observation_table(ds, schema, self.deps)
        selected, _, _ = tool.filter_observations(table, self.args(), self.deps)
        with tempfile.TemporaryDirectory() as tmp:
            tile_dir = Path(tmp) / "tile"
            count, _ = multi.compute_coverage_blocked(
                ds,
                schema,
                selected,
                tile_dir,
                "test-fingerprint",
                2,
                self.deps,
                cache_observations=True,
            )
            with h5py.File(tile_dir / "work" / "observation_cache.h5", "r") as cache:
                self.assertEqual(cache.attrs["velocity_variable"], "v")
                self.assertEqual(cache.attrs["error_variable"], "v_error")
                self.assertEqual(cache["validity"].shape, (1, 3, 4))
                self.assertEqual(cache["uncertainty"].shape, (1, 3, 4))
                np.testing.assert_array_equal(
                    cache["validity"][:].sum(axis=0), count.values
                )
                np.testing.assert_allclose(cache["uncertainty"][:], 1.0)


if __name__ == "__main__":
    unittest.main()
