# Expected outputs and availability

| Output | Generation path | Present in release candidate? |
|---|---|---|
| Five-city held-out-transfer and same-city tables | `scripts/07_multicity_transfer_matrix.py` | Selected aggregate tables only |
| Pair mismatch, QAP and decomposition | `scripts/07b_transfer_decomposition.py` | Selected aggregate tables only |
| Development risk and pooled-source comparisons | `scripts/08_label_free_transfer_risk.py` | Selected aggregate tables only |
| External risk and pooled-source outcomes | `scripts/09b_phaseB_reveal_external_outcomes.py` after separate Phase A | Selected aggregate tables only |
| Main Figure 1 | `scripts/paper_figures/figure1_framework_v3.py` | Code; generated artwork omitted |
| Main Figure 2 | `scripts/figures/make_figure2_spatial_context.py` | Code + city-level source points/window summary; boundary archive and seven processed Parquets omitted |
| Main Figures 3–6 | `figure4_transfer_matrix_v2.py`, `figure5_mismatch_gap_v2.py`, `figure8_transfer_risk_v2.py`, `figure9_multisource_v2.py` | Code + input aggregate CSVs; artwork omitted |
| Supplementary Tables S1–S11 | Calculation scripts mapped in `PAPER_TO_CODE_MAP.md` | Full-precision final CSVs present |
| Supplementary Figures S1–S15 | Python calculation/diagnostic scripts mapped in `PAPER_TO_CODE_MAP.md` | Python code retained; row-level inputs, Earth Engine export scripts and some intermediate artifacts omitted |
| Supplementary Figures S16–S17 | `scripts/paper_figures/make_external_comparison_s16_s17.py` | Code and final aggregate inputs; scientifically matching clean artwork can be regenerated |
| Supplementary Figures S18–S20 | `figure3_domain_signature_v2.py`, `figure6_robustness_v2.py`, `figure7_fewshot_v2.py` | Code and selected aggregate inputs |

## Figure-by-figure reproducibility tier

`DIRECTLY REPRODUCIBLE FROM PACKAGED AGGREGATES` includes Figure 1, whose framework schematic requires only packaged code and protocol constants, not a result table. `REPRODUCIBLE AFTER DOCUMENTED SOURCE-DATA REGENERATION` means the Python computation/plotting path is supplied but users must independently create the documented Earth Engine exports and row-level dataset from provider products; no GEE JavaScript is bundled. Newly sampled points need not reproduce historical identities. `TRACEABLE BUT INPUTS NOT REDISTRIBUTED` means the final item and its creating step can be identified, but exact historical rendering also depends on omitted intermediate files or byte-identical frozen inputs; the package does not promise one-command historical replay.

| Final figure | Tier | Boundary |
|---|---|---|
| Main 1 | DIRECTLY REPRODUCIBLE FROM PACKAGED AGGREGATES | Framework schematic; code/protocol constants only. |
| Main 2 | REPRODUCIBLE AFTER DOCUMENTED SOURCE-DATA REGENERATION | Seven processed city Parquets and separately obtained verified Natural Earth archive required. |
| Main 3–6 | DIRECTLY REPRODUCIBLE FROM PACKAGED AGGREGATES | Final plotting code and aggregate source CSVs bundled. |
| Supplementary S1–S2 | REPRODUCIBLE AFTER DOCUMENTED SOURCE-DATA REGENERATION | PCA/domain diagnostics need processed point-level AEF features. |
| Supplementary S3–S5 | REPRODUCIBLE AFTER DOCUMENTED SOURCE-DATA REGENERATION | Support/error diagnostics need point-level embeddings and labels. |
| Supplementary S6–S7 | REPRODUCIBLE AFTER DOCUMENTED SOURCE-DATA REGENERATION | Original diagnostic-generation path requires recreated run intermediates. |
| Supplementary S8–S11 | REPRODUCIBLE AFTER DOCUMENTED SOURCE-DATA REGENERATION | Original decomposition diagnostics require recreated pair/run intermediates. |
| Supplementary S12–S14 | REPRODUCIBLE AFTER DOCUMENTED SOURCE-DATA REGENERATION | Original risk/pooled diagnostics require recreated run intermediates. |
| Supplementary S15 | TRACEABLE BUT INPUTS NOT REDISTRIBUTED | Original external reveal diagnostic depends on omitted frozen prediction/outcome intermediates; exact historical replay also requires byte-identical Phase-A Parquets. |
| Supplementary S16–S17 | DIRECTLY REPRODUCIBLE FROM PACKAGED AGGREGATES | New clean scientific renderings use only final TableS9 and external summary aggregates; checked against submitted categories, values and reference lines. |
| Supplementary S18–S20 | DIRECTLY REPRODUCIBLE FROM PACKAGED AGGREGATES | Final standalone plotting scripts and aggregate source tables bundled. |

Bundled CSV copies under `outputs/` exist only to satisfy stable script-relative input paths. `results/` is the human-facing release selection. The test suite checks canonical values and does not assert availability of omitted full datasets.
