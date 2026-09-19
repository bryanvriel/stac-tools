import argparse
import unittest

import itslive_stac_inventory as inventory


class StacInventoryTests(unittest.TestCase):
    def args(self, **overrides):
        values = dict(
            start="2020-11-30",
            end="2020-12-31",
            mission="SENTINEL-1",
            min_pair_days=None,
            max_pair_days=12.0,
        )
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_current_denman_filter(self):
        args = self.args()
        self.assertTrue(
            inventory.record_matches(
                {
                    "datetime": "2020-12-20T12:00:00Z",
                    "platform": "S1A",
                    "pair_interval_days": 12,
                },
                args,
            )
        )
        self.assertFalse(
            inventory.record_matches(
                {
                    "datetime": "2020-12-20T12:00:00Z",
                    "platform": "LC08",
                    "pair_interval_days": 12,
                },
                args,
            )
        )
        self.assertFalse(
            inventory.record_matches(
                {
                    "datetime": "2020-12-20T12:00:00Z",
                    "platform": "S1B",
                    "pair_interval_days": 24,
                },
                args,
            )
        )

    def test_date_only_end_is_inclusive(self):
        args = self.args()
        record = {
            "datetime": "2020-12-31T23:59:59Z",
            "platform": "S1A",
            "pair_interval_days": 6,
        }
        self.assertTrue(inventory.record_matches(record, args))
        record["datetime"] = "2021-01-01T00:00:00Z"
        self.assertFalse(inventory.record_matches(record, args))

    def test_cql_filter_reduces_query(self):
        result = inventory.stac_cql_filter(self.args())
        self.assertEqual(result["op"], "and")
        self.assertEqual(result["args"][0]["op"], "in")
        self.assertEqual(result["args"][0]["args"][1], ["S1A", "S1B"])
        self.assertEqual(result["args"][1]["op"], "<=")


if __name__ == "__main__":
    unittest.main()
