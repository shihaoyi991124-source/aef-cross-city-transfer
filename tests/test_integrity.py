"""Low-cost static and canonical-value checks; no model fitting or data downloads."""

import csv
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    with (ROOT / path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class PublicationIntegrity(unittest.TestCase):
    def assert_close(self, actual, expected):
        self.assertAlmostEqual(float(actual), expected, places=6)

    def test_directed_transfer(self):
        pairs = rows("results/canonical_tables/stage07_pair_mechanism.csv")
        self.assertEqual(len(pairs), 20)
        by_pair = {(r["source_city"], r["target_city"]): r for r in pairs}
        for source, target, macro, expansion in [
            ("Kunming", "Shenzhen", 0.986252, 0.982396),
            ("Shenzhen", "Kunming", 0.892298, 0.864645),
            ("Wuhan", "Kunming", 0.800110, 0.721628),
        ]:
            rec = by_pair[source, target]
            self.assert_close(rec["cross_macro_f1_mean"], macro)
            self.assert_close(rec["cross_expansion_f1_mean"], expansion)
        self.assertTrue(all(r["model"] == "XGBoost" and r["feature_set"] == "AEF-192" for r in pairs))

    def test_qap_and_external_risk(self):
        qap = rows("results/canonical_tables/stage07b_qap_results.csv")
        rec = next(r for r in qap if r["distance_column"] == "support_distance_mean" and r["outcome_column"] == "macro_f1_gap_mean")
        self.assert_close(rec["observed_statistic"], 0.621053)
        self.assert_close(rec["exact_qap_one_sided_p"], 0.008333)
        risk = rows("results/canonical_tables/stage09b_external_risk_validation.csv")
        r4 = next(r for r in risk if r["risk_model"] == "R4")
        self.assert_close(r4["mae"], 0.013756)
        self.assert_close(r4["spearman_rho"], 0.793939)
        self.assertEqual(int(r4["n_external_pairs"]), 10)

    def test_pooled_gains(self):
        dev = rows("results/canonical_tables/stage08_multisource_comparison.csv")
        ext = rows("results/canonical_tables/stage09b_external_summary.csv")
        self.assertEqual(len(dev), 5)
        self.assertEqual(len(ext), 2)
        self.assert_close(sum(float(r["pooled_minus_mean_single_macro_f1"]) for r in dev) / 5, 0.030725)
        self.assert_close(sum(float(r["pooled_minus_mean_single_macro_f1"]) for r in ext) / 2, 0.024297)
        self.assertTrue(all(float(r["pooled_minus_mean_single_macro_f1"]) > 0 for r in dev + ext))

    def test_no_legacy_or_sample_rows(self):
        paths = list((ROOT / "results").rglob("*.csv"))
        self.assertEqual(len(list((ROOT / "results/supplementary_tables").glob("TableS*.csv"))), 11)
        self.assertFalse(any("baseline_" in p.name or "stage10a" in p.name or "spatial_splits" in p.name for p in paths))
        for path in paths:
            with path.open(encoding="utf-8-sig") as handle:
                header = handle.readline().lower()
            self.assertNotIn("sample_id", header)
            self.assertNotIn("utm_x", header)
            self.assertNotIn("utm_y", header)

    def test_no_local_path_or_private_project_id(self):
        for path in list((ROOT / "scripts").rglob("*.py")) + list((ROOT / "gee").glob("*.js")):
            content = path.read_text(encoding="utf-8-sig")
            self.assertNotRegex(content, r"[A-Za-z]:\\(?:Users|Documents|Desktop)\\")
            self.assertNotRegex(content, r"projects/(?!sat-io|YOUR_EE_PROJECT_[AB])[^/\s'\"]+/assets/")

    def test_final_map_and_citation_metadata(self):
        self.assertTrue((ROOT / "scripts/figures/make_figure2_spatial_context.py").is_file())
        windows = rows("results/figure_source_tables/figure2_spatial_window_selection.csv")
        self.assertEqual({r["city"] for r in windows}, {"Shenzhen", "Kunming", "Chengdu"})
        self.assertTrue(all(int(r["xmax"]) - int(r["xmin"]) == 6000 for r in windows))
        citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
        manuscript = (ROOT.parent / "GeoAI_AEF" / "docs" / "manuscript" / "Full_Manuscript_Integrated_v3_4_author_metadata.md")
        expected_title = "Cross-City Transfer Risk and Pooled-Source Training with Frozen AlphaEarth Foundations Embeddings for Urban Expansion Mapping"
        expected_authors = [
            ("Shi", "Haoyi", "0009-0008-2395-7119"),
            ("Zuo", "Xiaoqing", "0000-0001-9153-1920"),
            ("Li", "Yongfa", "0000-0002-6444-3332"),
            ("Dong", "Yujuan", "0009-0007-8610-8811"),
        ]
        self.assertIn(f'title: "{expected_title}"', citation)
        records = re.findall(r"  - family-names: (\w+)\n    given-names: (\w+)\n    orcid: \"https://orcid.org/([0-9X-]+)\"", citation)
        self.assertEqual(records, expected_authors)
        for _, _, orcid in records:
            digits = orcid.replace("-", "")
            total = 0
            for digit in digits[:-1]:
                total = (total + int(digit)) * 2
            check = (12 - total % 11) % 11
            self.assertEqual("X" if check == 10 else str(check), digits[-1])
        self.assertEqual(citation.count("orcid: "), 4)
        self.assertNotIn("doi:", citation.lower())
        self.assertNotIn("date-released:", citation.lower())
        self.assertNotRegex(citation.lower(), r"(?m)^(year|volume|repository-doi):")
        self.assertIn("zxq@kust.edu.cn", (ROOT / "README.md").read_text(encoding="utf-8"))
        if manuscript.is_file():
            text = manuscript.read_text(encoding="utf-8")
            self.assertIn(expected_title, text)
            self.assertIn("Haoyi Shi; Xiaoqing Zuo (corresponding author); Yongfa Li; Yujuan Dong", text)
            self.assertIn("zxq@kust.edu.cn", text)


if __name__ == "__main__":
    unittest.main()
