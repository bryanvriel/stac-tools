import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS
from shapely.geometry import Polygon

import itslive_cube_coverage as coverage
import itslive_cube_export as export


class ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deps = coverage.require_dependencies()

    def dataset(self):
        values = np.arange(3 * 3 * 4, dtype=np.float32).reshape(3, 3, 4)
        values[0, 1, 1] = -32767
        spatial_attrs = {"missing_value": -32767.0, "grid_mapping": "mapping", "units": "meter/year"}
        return xr.Dataset(
            {
                "v": (("mid_date", "y", "x"), values, spatial_attrs),
                "vx": (("mid_date", "y", "x"), values + 100, spatial_attrs),
                "vy": (("mid_date", "y", "x"), values + 200, spatial_attrs),
                "v_error": (("mid_date", "y", "x"), np.ones_like(values), spatial_attrs),
                "vx_error": ("mid_date", np.asarray([1, 2, 3], dtype=np.float32)),
                "vy_error": ("mid_date", np.asarray([4, 5, 6], dtype=np.float32)),
                "date_center": ("mid_date", pd.to_datetime(["2020-01-01", "2020-01-02", "2021-01-01"])),
                "acquisition_date_img1": ("mid_date", pd.to_datetime(["2019-12-26", "2019-12-27", "2020-12-25"])),
                "acquisition_date_img2": ("mid_date", pd.to_datetime(["2020-01-07", "2020-01-08", "2021-01-06"])),
                "date_dt": ("mid_date", np.asarray([12, 12, 12], dtype=np.float32)),
                "mission_img1": ("mid_date", ["S1", "S1", "S1"]),
                "mission_img2": ("mid_date", ["S1", "S1", "S1"]),
                "satellite_img1": ("mid_date", ["1A", "1A", "1A"]),
                "satellite_img2": ("mid_date", ["1A", "1B", "1A"]),
                "sensor_img1": ("mid_date", ["C", "C", "C"]),
                "sensor_img2": ("mid_date", ["C", "C", "C"]),
                "granule_url": ("mid_date", ["https://example/0.nc", "https://example/1.nc", "https://example/2.nc"]),
                "mapping": ((), "", {"spatial_epsg": 3031}),
            },
            coords={"mid_date": np.arange(3), "x": [0.0, 10.0, 20.0, 30.0], "y": [20.0, 10.0, 0.0]},
        )

    def args(self, outdir, **changes):
        values = dict(
            start="2020-01-01", end="2020-12-31", mission="SENTINEL-1", sensor=None,
            min_pair_days=None, max_pair_days=12.0, variables=list(export.DEFAULT_VARIABLES),
            observation_block_size=1, spatial_block_size=2, outdir=Path(outdir),
        )
        values.update(changes)
        return Namespace(**values)

    def test_preflight_export_mask_metadata_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.zarr"
            out = root / "output"
            self.dataset().to_zarr(source, mode="w", zarr_format=2)
            roi = coverage.ROI(
                Polygon([(-1, -1), (11, -1), (-1, 11), (-1, -1)]),
                CRS.from_epsg(3031), "bbox", [-1, -1, 11, 11],
            )
            args = self.args(out)
            plan, selected = export.preflight_cube(
                {"id": "synthetic", "url": str(source)}, 0, roi, args, self.deps, None
            )
            self.assertEqual(plan["shape"], {"observation": 2, "y": 2, "x": 2})
            self.assertEqual(len(selected), 2)
            plans, inventory, _ = export.build_preflight(
                roi, [{"id": "synthetic", "url": str(source)}], args, self.deps, None
            )
            self.assertEqual(plans[0]["id"], "synthetic")
            self.assertIn("observation_key", inventory)
            self.assertIn("cube_observation", inventory)
            fingerprint = export.plan_fingerprint([plan], roi, args, self.deps)
            export.prepare_output(out, overwrite=False, resume=False)
            import zarr
            zarr.open_group(out / export.STORE_NAME, mode="a", zarr_format=2).require_group("cubes")
            payload = {
                "plan": plan,
                "args": {
                    "variables": args.variables,
                    "observation_block_size": 1,
                    "spatial_block_size": 2,
                },
                "fingerprint": fingerprint,
                "outdir": str(out),
                "roi": {"crs": roi.crs.to_string(), "geometry": self.deps["mapping"](roi.geometry)},
            }
            first = export.export_cube_worker(payload)
            second = export.export_cube_worker(payload)
            self.assertEqual(first["status"], "complete")
            self.assertEqual(second["status"], "complete")

            loaded = xr.open_zarr(out / export.STORE_NAME, group="cubes/synthetic", consolidated=False)
            self.assertEqual(dict(loaded.sizes), {"observation": 2, "y": 2, "x": 2})
            np.testing.assert_array_equal(loaded.source_observation_index.values, [0, 1])
            self.assertEqual(int(loaded.roi_mask.values.sum()), 3)
            self.assertTrue(np.all(np.isnan(loaded.v_error.values[:, loaded.roi_mask.values == 0])))
            self.assertTrue(np.all(loaded.v_error.values[:, loaded.roi_mask.values == 1] == 1.0))
            self.assertEqual(loaded.attrs["crs"], "EPSG:3031")
            self.assertEqual(loaded.granule_url.values.tolist(), ["https://example/0.nc", "https://example/1.nc"])
            state = json.loads((out / "checkpoints" / "synthetic.json").read_text())
            self.assertEqual(state["status"], "complete")

    def test_masking_and_size_limit_helpers(self):
        fill = export.missing_value(np.dtype("float32"), {"missing_value": -32767}, np)
        self.assertTrue(np.isnan(fill))
        self.assertIn("GiB", export.human_bytes(export.DEFAULT_SIZE_LIMIT))
        with self.assertRaises(RuntimeError):
            export.enforce_size_limit(export.DEFAULT_SIZE_LIMIT + 1, allow_large=False)
        export.enforce_size_limit(export.DEFAULT_SIZE_LIMIT + 1, allow_large=True)

    def test_preflight_rejects_missing_requested_variable(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.zarr"
            self.dataset().drop_vars("vy_error").to_zarr(source, mode="w", zarr_format=2)
            roi = coverage.ROI(
                Polygon([(-1, -1), (11, -1), (11, 11), (-1, 11), (-1, -1)]),
                CRS.from_epsg(3031), "bbox", [-1, -1, 11, 11],
            )
            with self.assertRaisesRegex(RuntimeError, "missing requested variables: vy_error"):
                export.preflight_cube(
                    {"id": "synthetic", "url": str(source)}, 0, roi,
                    self.args(tmp), self.deps, None,
                )

    def test_replay_indices_expands_multi_cube_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame(
                {"cube_observations": [json.dumps([
                    {"cube_id": "a", "observation_index": 4},
                    {"cube_id": "b", "observation_index": 7},
                ])]}
            ).to_csv(root / "selected_observations.csv", index=False)
            result = export.replay_indices(
                root, [{"id": "a"}, {"id": "b"}], self.deps
            )
            self.assertEqual(result["a"]["indices"], [4])
            self.assertEqual(result["b"]["indices"], [7])

    def test_saved_multi_cube_roi_is_reconstructed_from_arguments(self):
        roi = export.roi_from_saved_config(
            {"arguments": {
                "bbox": None, "bbox_lonlat": [99.0, -68.0, 100.0, -67.0],
                "epsg": 3031, "edge_points": 41,
            }},
            self.deps,
        )
        self.assertEqual(roi.crs, CRS.from_epsg(4326))
        self.assertEqual(roi.input_bounds, [99.0, -68.0, 100.0, -67.0])

    def test_replay_indices_recovers_legacy_export_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame({
                "source_cube_id": ["a", "b"],
                "observation_index": [4, 7],
                "image_pair_id": ["pair-a", "pair-b"],
            }).to_csv(root / "selected_observations.csv", index=False)
            result = export.replay_indices(
                root, [{"id": "a"}, {"id": "b"}], self.deps
            )
            self.assertEqual(result["a"]["indices"], [4])
            self.assertEqual(result["b"]["indices"], [7])

    def test_prepare_output_does_not_touch_coverage_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            coverage_config = root / "run_config.json"
            coverage_inventory = root / "selected_observations.csv"
            coverage_config.write_text("coverage config")
            coverage_inventory.write_text("coverage inventory")
            export.prepare_output(root, overwrite=True, resume=False)
            self.assertEqual(coverage_config.read_text(), "coverage config")
            self.assertEqual(coverage_inventory.read_text(), "coverage inventory")

    def test_cli_rejects_replay_filter_overrides(self):
        with self.assertRaises(SystemExit):
            export.parse_args([
                "--coverage-results", "coverage", "--start", "2020-01-01", "--outdir", "out"
            ])

    def test_worker_retries_transient_failure(self):
        payload = {"plan": {"id": "synthetic"}}
        with (
            mock.patch.object(
                export, "_export_cube_once",
                side_effect=[TimeoutError(), {"id": "synthetic", "status": "complete"}],
            ) as run,
            mock.patch.object(export, "reset_remote_filesystems") as reset,
            mock.patch.object(export.time, "sleep") as sleep,
        ):
            result = export.export_cube_worker(payload)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(run.call_count, 2)
        reset.assert_called_once()
        sleep.assert_called_once_with(2)

    def test_worker_does_not_retry_nontransient_failure(self):
        payload = {"plan": {"id": "synthetic"}}
        with mock.patch.object(
            export, "_export_cube_once", side_effect=ValueError("bad schema")
        ) as run:
            with self.assertRaisesRegex(ValueError, "bad schema"):
                export.export_cube_worker(payload)
        self.assertEqual(run.call_count, 1)

    def test_preflight_retries_transient_metadata_failure(self):
        expected = ({"id": "synthetic"}, pd.DataFrame())
        with (
            mock.patch.object(
                export, "_preflight_cube_once",
                side_effect=[coverage.CubeOpenError("disconnected"), expected],
            ) as run,
            mock.patch.object(export, "reset_remote_filesystems") as reset,
            mock.patch.object(export.time, "sleep") as sleep,
        ):
            result = export.preflight_cube(
                {"id": "synthetic", "url": "https://example.invalid/cube.zarr"},
                0, mock.sentinel.roi, mock.sentinel.args, self.deps, None,
            )
        self.assertIs(result, expected)
        self.assertEqual(run.call_count, 2)
        reset.assert_called_once()
        sleep.assert_called_once_with(2)

    def test_truncated_http_payload_is_retryable(self):
        ClientPayloadError = type("ClientPayloadError", (Exception,), {})
        ContentLengthError = type("ContentLengthError", (Exception,), {})
        self.assertTrue(export.multi.transient_remote_error(ClientPayloadError()))
        self.assertTrue(export.multi.transient_remote_error(ContentLengthError()))


if __name__ == "__main__":
    unittest.main()
