"""Protect the exact scientific values and labels in S16/S17 plotting inputs."""

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "paper_figures" / "make_external_comparison_s16_s17.py"
SPEC = importlib.util.spec_from_file_location("external_comparison_figures", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
import sys
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ExternalFigureValues(unittest.TestCase):
    def test_submitted_categories_and_values(self):
        expected = {
            "Chengdu": (
                (0.9534681605, 0.9085112676, 0.9470670972, 0.9711862478, 0.9665837912,
                 0.9837656573, 0.9823142333),
                0.9930810211,
            ),
            "Hangzhou": (
                (0.9768221445, 0.9616681272, 0.9619359975, 0.9833981121, 0.9793283792,
                 0.9868215870, 0.9862973653),
                0.9944119350,
            ),
        }
        comparisons = MODULE.load_comparisons()
        self.assertEqual(len(comparisons), 2)
        for comparison in comparisons:
            values, reference = expected[comparison.target]
            self.assertEqual(comparison.categories, ("BJ", "KM", "SZ", "WH", "XA", "POOLED5", "ENSEMBLE5"))
            for actual, frozen in zip(comparison.values, values):
                self.assertAlmostEqual(actual, frozen, places=10)
            self.assertAlmostEqual(comparison.same_city_reference, reference, places=10)


if __name__ == "__main__":
    unittest.main()
