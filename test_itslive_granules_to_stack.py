import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path

import iceutils as ice
import numpy as np
import xarray as xr

import itslive_granules_to_stack as converter


class GranulesToStackTests(unittest.TestCase):
    def write_granule(self, path, x0, date, offset):
        x = x0 + np.arange(4) * 120.0
        y = 360.0 - np.arange(4) * 120.0
        values = np.arange(16, dtype=np.float32).reshape(1, 4, 4) + offset
        dataset = xr.Dataset(
            {
                "vx": (("time", "y", "x"), values),
                "vy": (("time", "y", "x"), values + 10),
                "v_error": (("time", "y", "x"), values + 1),
                "mapping": ((), np.float32(0), {"spatial_epsg": 3031}),
            },
            coords={
                "time": np.asarray([date], dtype="datetime64[ns]"),
                "x": x,
                "y": y,
            },
        )
        for name in converter.VARIABLES:
            dataset[name].attrs["grid_mapping"] = "mapping"
        dataset.to_netcdf(path, engine="h5netcdf")

    def test_builds_time_sorted_common_grid_stack(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            self.write_granule(data / "later.nc", 60.0, "2020-01-02", 100)
            self.write_granule(data / "earlier.nc", 0.0, "2020-01-01", 0)
            with (root / "items.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["data_href"])
                writer.writeheader()
                writer.writerow({"data_href": "https://example/earlier.nc"})
                writer.writerow({"data_href": "https://example/later.nc"})
            (root / "summary.json").write_text(
                json.dumps(
                    {
                        "input_bbox_type": "projected",
                        "input_bbox": [0, 0, 420, 360],
                        "input_epsg": 3031,
                    }
                )
            )
            output = root / "stack.nc"
            converter.build_stack(
                argparse.Namespace(
                    input_dir=root,
                    bbox_lonlat=None,
                    bbox=[0, 0, 420, 360],
                    target_epsg=3031,
                    resolution=120.0,
                    output=output,
                    resampling="bilinear",
                    chunk_size=2,
                    allow_missing=False,
                    overwrite=False,
                )
            )

            stack = ice.Stack(str(output))
            try:
                self.assertEqual(stack.shape, (2, 4, 5))
                self.assertEqual(stack.hdr.epsg, 3031)
                self.assertTrue(
                    {"vx", "vy", "v_error"}.issubset(stack.ds.data_vars)
                )
                self.assertLess(stack.ds.time.values[0], stack.ds.time.values[1])
                self.assertEqual(np.isfinite(stack["vx"].values).sum(), 28)
                self.assertTrue(np.isnan(stack["vx"].values[:, :, -1]).any())
            finally:
                stack.close()


if __name__ == "__main__":
    unittest.main()
