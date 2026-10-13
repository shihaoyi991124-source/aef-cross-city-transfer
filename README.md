# Cross-City Transfer Risk and Pooled-Source Training with Frozen AlphaEarth Foundations Embeddings for Urban Expansion Mapping

Research code and selected aggregate result tables for a seven-city urban-expansion benchmark. Five development cities are Shenzhen, Kunming, Wuhan, Xi'an and Beijing; Chengdu and Hangzhou are frozen external targets. The study evaluates spatially held-out same-city and directed cross-city performance with 192 frozen annual AlphaEarth embedding features, diagnosis of representation mismatch, label-free-at-target-inference transfer-risk estimates, limited-label adaptation and pooled-source training.

This repository contains research code, reproduction instructions, and selected aggregate results associated with the study. It contains no raw embedding matrix, sample-level coordinates or labels, source raster product, manuscript draft, credential or repository DOI. The paper's scientific content and data-availability statement are maintained separately.

## Layout and scope

| Path | Contents |
|---|---|
| `scripts/` | Original calculation interfaces retained so cross-script imports and frozen output names remain traceable. Names reflect provenance; they are not a proposed new experimental order. |
| `scripts/figures/` and `scripts/paper_figures/` | Plotting code for main Figures 1–6 and supplementary Figures S16–S20; Figure 2 requires omitted source data. |
| `results/` | Selected aggregate canonical, figure-source and supplementary tables. |
| `outputs/` | Compatibility copies of aggregate tables at paths required by plotting scripts; generated images are not included. |
| `docs/` | Source, output and result-to-code documentation. |
| `tests/` | Low-cost integrity checks; these do not retrain models. |

## Installation

Python 3.11+ is recommended. Versions in `environment.yml` reflect the tested local environment, not a guarantee that every historical run used those exact versions.

```powershell
conda env create -f environment.yml
conda activate geoai-aef-public
python -m unittest discover -s tests -v
```

`requirements.txt` is an alternative for a Python virtual environment. `geopandas` is needed only for the spatial-context map. This repository does **not** include Earth Engine JavaScript extraction code. For Figure 2, download the cited Natural Earth archive yourself and verify its recorded SHA-256 before use.

## Data prerequisites

Full benchmark reconstruction requires users to obtain the public Earth-observation products from their providers, prepare authorized city boundaries and point assets, and reproduce the documented point sampling and annual exports in their own Earth Engine environment. The recorded product identifiers, study years, export fields, class labels and general procedure are in [`data/README.md`](data/README.md) and [`docs/data_sources.md`](docs/data_sources.md). No Earth Engine extraction script or private asset is bundled.

No raw CSV, Parquet, Earth Engine asset, external frozen prediction file or Natural Earth archive is bundled. Most model-retraining commands therefore require the researcher to regenerate the documented prerequisites. An *exact historical* external Phase-A replay additionally requires byte-identical processed Parquets matching hashes embedded in the frozen script; equivalent recomputed Parquets may be scientifically similar but will intentionally fail that gate. The external outcomes were not used when its risk specification and predictions were fixed. Predictive entropy (R4) was selected **post hoc during development**, before the external outcome reveal.

## Reproduction sequence

Run from the repository root after supplying the documented exports. The retained filenames and output paths preserve tested cross-script imports; this is deliberately not represented as one-command reproduction.

```powershell
python scripts/01_build_dataset.py
python scripts/02_baseline_experiment.py
python scripts/03_domain_shift_audit.py
python scripts/04_fewshot_adaptation.py
python scripts/05_transfer_asymmetry_analysis.py
python scripts/06_multicity_dataset_integration.py
python scripts/07_multicity_transfer_matrix.py
python scripts/07b_transfer_decomposition.py
python scripts/08_label_free_transfer_risk.py
python scripts/09a_external_dataset_audit.py
python scripts/09b_phaseA_freeze_external_predictions.py
# Restore the separate, previously frozen external outcome files only after Phase A.
python scripts/09b_phaseB_reveal_external_outcomes.py
```

The two-city baseline is a dependency of later scripts, but its historical full-target cross-city scores are **not** the final paper's held-out-target-test scores. The final directed results come from the five-city transfer calculation. Split seeds are 20261008–20261012. Spatial blocks, source-only scaling, held-out target test blocks and before-reveal external separation must be preserved. XGBoost settings are in the checked-in `config.json` (400 trees, depth 6, learning rate 0.08, subsample/column sample 0.85, 30 early-stopping rounds).

The packaged aggregate tables allow result-integrity tests and most main-chart rendering without model retraining:

```powershell
python -m unittest discover -s tests -v
python scripts/paper_figures/figure1_framework_v3.py
python scripts/paper_figures/figure4_transfer_matrix_v2.py
python scripts/paper_figures/figure5_mismatch_gap_v2.py
python scripts/paper_figures/figure8_transfer_risk_v2.py
python scripts/paper_figures/figure9_multisource_v2.py
python scripts/paper_figures/make_external_comparison_s16_s17.py
```

Figure 2 uses `scripts/figures/make_figure2_spatial_context.py` and needs seven processed city Parquets as well as a separately downloaded Natural Earth administrative-boundary archive (the script checks SHA-256). Its frozen spatial-window summary is included, but the sampled point clouds are not. The main-text numbering is Figure 1 = framework; Figure 2 = study geography; Figure 3 = directed transfer (`figure4_...`); Figure 4 = mismatch (`figure5_...`); Figure 5 = transfer risk (`figure8_...`); Figure 6 = pooled source (`figure9_...`). The script basenames preserve earlier artwork numbering; [`docs/PAPER_TO_CODE_MAP.md`](docs/PAPER_TO_CODE_MAP.md) gives the authoritative mapping. Supplementary S1–S15 require omitted detailed inputs; S16–S20 have retained plotting scripts and aggregate inputs. S16/S17 are clean scientific re-renderings from the recorded final aggregates, not pixel-identical copies of submitted artwork. See [`docs/expected_outputs.md`](docs/expected_outputs.md) for figure-by-figure reproduction tiers. Do not treat bundled aggregate CSVs as a complete benchmark dataset.

## Code, aggregate results and source-data terms

The [MIT License](LICENSE) applies to this repository's project research code only. `results/` contains selected derived research outputs associated with the paper; the code license is **not** a license for those CSV tables or any upstream dataset. No separate Creative Commons license is assigned to result tables here. Cite the associated paper and this repository when using released research outputs; author names and ORCIDs are in `CITATION.cff`. A paper DOI and publication date have not been assigned.

Source-provider data are not redistributed. This includes AlphaEarth embedding matrices, Dynamic World and ESRI/Impact Observatory products, GHSL and Natural Earth source copies, point-level labels/coordinates and processed city Parquets. Upstream terms remain governed by their respective providers; see [`docs/data_sources.md`](docs/data_sources.md). The `results/` tables are aggregate outputs, not a substitute for the sample-level benchmark. For correspondence, contact Xiaoqing Zuo at zxq@kust.edu.cn.
