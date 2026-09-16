import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import itslive_cube_coverage as coverage
import itslive_inversion_diagnostics as diagnostics


class InversionDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deps = coverage.require_dependencies()

    def write_template(self, path: Path) -> None:
        time = np.linspace(2019.0, 2022.0, 1097)
        centered = time - 2020.0
        matrix = np.column_stack([np.ones_like(time), centered, centered**2])
        with h5py.File(path, "w") as h5:
            h5["time"] = time
            h5["template_matrix"] = matrix
            h5["coefficient_names"] = np.asarray([b"constant", b"linear", b"quadratic"])
            h5["prior_variance"] = [100.0, 25.0, 4.0]
            h5["ridge_weights"] = [0.0, 1.0, 1.0]
            h5.attrs["time_units"] = "decimal_year_iceutils"
            h5.attrs["template_quantity"] = "velocity"

    def test_worker_argument(self):
        args = diagnostics.parse_args(["results", "template.h5", "--workers", "3"])
        self.assertEqual(args.workers, 3)
        with self.assertRaises(SystemExit):
            diagnostics.parse_args(["results", "template.h5", "--workers", "0"])

    def test_template_and_velocity_operators(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "template.h5"
            self.write_template(path)
            template = diagnostics.load_template(path, self.deps)
            operators = diagnostics.resolve_operators(template, None)
            self.assertEqual(operators, ["interval-average", "midpoint"])
            observations = pd.DataFrame(
                {
                    "date_img1": ["2020-01-01", "2020-01-01"],
                    "date_img2": ["2020-01-13", "2021-01-01"],
                }
            )
            matrices, prediction = diagnostics.build_design_matrices(
                template, observations, operators, self.deps
            )
            np.testing.assert_allclose(
                matrices["interval-average"][:, :2],
                matrices["midpoint"][:, :2],
                atol=2e-5,
            )
            self.assertGreater(
                abs(matrices["interval-average"][1, 2] - matrices["midpoint"][1, 2]),
                0.05,
            )
            self.assertEqual(prediction.shape, template.matrix.shape)

    def test_posterior_metrics_distinguish_rank(self):
        design = np.asarray([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
        valid = np.asarray(
            [
                [[True, True]],
                [[True, False]],
                [[True, False]],
            ]
        )
        sigma = np.ones_like(valid, dtype=float)
        precision = np.eye(2) * 0.1
        metrics = diagnostics.diagnose_pixels(
            design, design, valid, sigma, precision, 2, np
        )
        self.assertEqual(metrics["data_rank"].tolist(), [[2.0, 1.0]])
        self.assertGreater(metrics["effective_dof"][0, 0], metrics["effective_dof"][0, 1])
        self.assertTrue(np.all(np.isfinite(metrics["mean_prediction_std"])))
        self.assertTrue(np.all(np.isfinite(metrics["information_gain_nats"])))

    def test_displacement_endpoint_difference(self):
        template = diagnostics.Template(
            time=np.asarray([2020.0, 2021.0, 2022.0]),
            matrix=np.column_stack(
                [np.asarray([0.0, 1.0, 2.0]), np.asarray([0.0, 1.0, 4.0])]
            ),
            quantity="displacement",
            coefficient_names=["linear", "quadratic"],
            prior_variance=np.asarray([np.inf, np.inf]),
            ridge_weights=np.ones(2),
            digest="test",
        )
        observations = pd.DataFrame(
            {"date_img1": ["2020-01-01"], "date_img2": ["2021-01-01"]}
        )
        matrices, prediction = diagnostics.build_design_matrices(
            template, observations, ["endpoint-difference"], self.deps
        )
        np.testing.assert_allclose(matrices["endpoint-difference"], [[1.0, 1.0]])
        self.assertEqual(prediction.shape, (3, 2))

    def test_reads_compatible_observation_cache(self):
        import xarray as xr

        observations = pd.DataFrame({"observation_index": [3, 8]})
        subset = xr.Dataset(coords={"mid_date": np.arange(10), "x": [1.0, 2.0], "y": [4.0]})
        schema = type("Schema", (), {"obs": "mid_date", "x": "x", "y": "y"})()
        args = Namespace(
            observation_cache="require",
            velocity_variable="v",
            error_variable="auto",
            error_floor=1.5,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "observation_cache.h5"
            with h5py.File(path, "w") as cache:
                cache.attrs["cache_version"] = 1
                cache.attrs["velocity_variable"] = "v"
                cache.attrs["error_variable"] = "v_error"
                cache["observation_index"] = [3, 8]
                cache["x"] = [1.0, 2.0]
                cache["y"] = [4.0]
                cache["validity"] = np.asarray([[[True, False]], [[True, True]]])
                cache["uncertainty"] = np.asarray([[[1.0, 2.0]], [[3.0, 4.0]]])
            cache, reason = diagnostics.open_observation_cache(
                path, subset, schema, observations, args, self.deps
            )
            self.assertEqual(reason, "compatible")
            try:
                valid, sigma = diagnostics.cached_block_arrays(
                    cache, slice(0, 1), slice(0, 2), args, np
                )
            finally:
                cache.close()
            self.assertEqual(valid.tolist(), [[[True, False]], [[True, True]]])
            np.testing.assert_allclose(sigma, [[[1.5, 2.0]], [[3.0, 4.0]]])


if __name__ == "__main__":
    unittest.main()
