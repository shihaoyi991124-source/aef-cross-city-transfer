# Paper-to-code map

Final paper numbering is used below. Script filenames preserve earlier figure or experiment numbering to avoid changing validated computational paths. `outputs/tables/` and `outputs/paper/` are code-compatible locations; corresponding selected public tables are under `results/`. Earth Engine JavaScript and private assets are not bundled; source products and export schema are documented in `data/README.md` and `data_sources.md`.

| Scientific step | Source script(s) | Required input | Output / paper role |
|---|---|---|---|
| Benchmark construction and label sampling | Researcher-produced Earth Engine point assets (not bundled); `scripts/01_build_dataset.py`, `06_multicity_dataset_integration.py`, `09a_external_dataset_audit.py` | Dynamic World, Impact Observatory/ESRI check, GHSL context; authorized city boundaries; documented 3-class export schema | Balanced 3-class city datasets; Table 1 / S10. Exact historical point identities are not guaranteed by new exports. |
| Annual AEF 2020, 2022, 2024 and AEF-192 | Researcher-produced annual Earth Engine CSV exports (no bundled JavaScript); `scripts/01_build_dataset.py`, `06_multicity_dataset_integration.py`, `09a_external_dataset_audit.py` | Three annual 64-band exports/city using the documented columns | AEF-192 frozen input; AEF-195 audit-only variant |
| Spatial block assignment and same-city evaluation | `scripts/01_build_dataset.py`, `02_baseline_experiment.py`, `07_multicity_transfer_matrix.py` | UTM point coordinates; 2 km blocks; five seeds | Same-city held-out tests; Table 2 / S2. Historical two-city baseline is not final transfer scoring. |
| Directed cross-city evaluation / target-normalized gap | `scripts/03_domain_shift_audit.py`, `07_multicity_transfer_matrix.py` | Same target held-out blocks for target-trained and source-trained models | 20 development directions, pair table; Figure 3 / Tables 2, S1 |
| Geographic domain classification | `scripts/03_domain_shift_audit.py`, `07_multicity_transfer_matrix.py` | AEF-192 only, group-aware splits; no city/coordinate/class input features | Supplementary S1–S2, S18 |
| Support distance and directional asymmetry | `scripts/05_transfer_asymmetry_analysis.py`, `07_multicity_transfer_matrix.py` | Class-specific train/test embeddings; target labels for post-hoc true-class diagnostic | Figure 4 / S3–S5, S9; Table S4 |
| QAP, leave-one-city-out and source/target decomposition | `scripts/07b_transfer_decomposition.py` | 20 directed pair aggregates | Exact 120-permutation QAP; Table 3 / S5–S7; S8–S11, S19 |
| Few-shot adaptation | `scripts/04_fewshot_adaptation.py` | Target train/validation spatial blocks, unchanged target test blocks | Table S3 / S20 |
| R1/R4 development transfer-risk analysis | `scripts/08_label_free_transfer_risk.py` | Frozen source models, label-free target inference features, 20 pair outcomes for development calibration | Table 4 / S8; Figure 5 / S12–S13. R4 selected post hoc during development. |
| External before-reveal evaluation | `scripts/09b_phaseA_freeze_external_predictions.py`, then `09b_phaseB_reveal_external_outcomes.py` | Byte-identical processed Parquets for exact replay; separated external outcomes | Ten external directions; Table 4 / S9–S10; Figure 5 / S15–S17 |
| Pooled-source training | `scripts/08_label_free_transfer_risk.py`, `09b_phaseA_freeze_external_predictions.py`, `09b_phaseB_reveal_external_outcomes.py` | Four development sources per development target; five sources for external targets | Table 5; Figure 6 / S14, S16–S17 |

## Main artwork and tables

| Final item | Rendering / calculation code | Frozen input | Public result file |
|---|---|---|---|
| Figure 1 | `scripts/paper_figures/figure1_framework_v3.py` | Protocol constants only | generated `outputs/paper/figures/Figure1_framework_v3.*` |
| Figure 2 | `scripts/figures/make_figure2_spatial_context.py` | Seven city processed Parquets + `Figure2_study_area_v2_source.csv` + separately downloaded Natural Earth archive; frozen selected window summary in `results/figure_source_tables/figure2_spatial_window_selection.csv` | generated `outputs/figures/final/Figure2_spatial_context.*` |
| Figure 3 | `scripts/paper_figures/figure4_transfer_matrix_v2.py` | `Figure4_source.csv`, transfer matrices | generated `Figure4_transfer_matrix_v2.*` |
| Figure 4 | `scripts/paper_figures/figure5_mismatch_gap_v2.py` | `Figure5_source.csv`, pair and QAP tables | generated `Figure5_mismatch_gap_v2.*` |
| Figure 5 | `scripts/paper_figures/figure8_transfer_risk_v2.py` | `Figure8_source.csv`, frozen Table 4 | generated `Figure8_transfer_risk_v2.*` |
| Figure 6 | `scripts/paper_figures/figure9_multisource_v2.py` | `Figure9_source.csv`, frozen Table 5 | generated `Figure9_multisource_v2.*` |
| Tables 1–5 | `scripts/01_build_dataset.py`, `07_multicity_transfer_matrix.py`, `07b_transfer_decomposition.py`, `08_label_free_transfer_risk.py`, `09b_phaseB_reveal_external_outcomes.py` | Final aggregate outputs | `outputs/paper/tables/Table1_manuscript_v2.csv` … `Table5_manuscript_v2.csv` (display); full-precision companion tables in `results/` |

## Supplementary items

| Final item | Generating code | Input/output mapping |
|---|---|---|
| S1–S2 | `scripts/03_domain_shift_audit.py` | Two-city processed Parquets → PCA plots |
| S3–S5 | `scripts/05_transfer_asymmetry_analysis.py` | Support-distance and error diagnostics → plots |
| S6–S7 | `scripts/07_multicity_transfer_matrix.py` | Source/target aggregate metrics → plots |
| S8–S11 | `scripts/07b_transfer_decomposition.py` | QAP, pair asymmetry, adjusted association and diversity → plots |
| S12–S14 | `scripts/08_label_free_transfer_risk.py` | Risk/pooled aggregate metrics → plots |
| S15 | `scripts/09b_phaseB_reveal_external_outcomes.py` | Original external reveal diagnostic; omitted frozen intermediates mean historical replay is not turnkey. |
| S16–S17 | `scripts/paper_figures/make_external_comparison_s16_s17.py` | `results/supplementary_tables/TableS9.csv` plus `results/canonical_tables/stage09b_external_summary.csv` → scientifically matching clean seven-category comparison, including same-city reference. No metric recomputation or model fitting. |
| S18 | `scripts/paper_figures/figure3_domain_signature_v2.py` | `Figure3_source.csv` + city-domain aggregate table |
| S19 | `scripts/paper_figures/figure6_robustness_v2.py` | `Figure6_source.csv` + LOCO/fixed-city tables |
| S20 | `scripts/paper_figures/figure7_fewshot_v2.py` | `Figure7_source.csv` + few-shot summary |
| Tables S1–S11 | The corresponding calculation scripts above | Final full-precision CSVs in `results/supplementary_tables/TableS1.csv` … `TableS11.csv`. |

No final result in this map is assigned to the old two-city full-target cross-city scores. Supplementary S1–S15 require omitted point-level features or frozen intermediates; S16–S17 now render directly from packaged final aggregates. Exact pixel identity with submitted figures is not claimed. See `expected_outputs.md` for item-level tiers.
