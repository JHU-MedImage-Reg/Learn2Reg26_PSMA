# PSMAReg — Learn2Reg 2026

[![Static Badge](https://img.shields.io/badge/MICCAI-SIG_BIR-%2337677e?style=flat&labelColor=%23ececec&link=https%3A%2F%2Fmiccai.org%2Findex.php%2Fspecial-interest-groups%2Fbir%2F)](https://miccai.org/index.php/special-interest-groups/bir/) [![Static Badge](https://img.shields.io/badge/MICCAI-Learn2Reg-%23214f5f?labelColor=%23ececec)](https://learn2reg.grand-challenge.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official repository of the **PSMAReg** task of Learn2Reg 2026: longitudinal
registration of whole-body PSMA PET/CT, scored on anatomical accuracy, lesion
biomarker preservation (MTV, TLG) and deformation regularity. Please visit
[**learn2reg.grand-challenge.org**](https://learn2reg.grand-challenge.org/) for
the challenge description, data access and submission rules.

## 🏆 Test-phase results

The official ranking is out: **[Test-phase results](results/README.md)** —
final scores with bootstrap rank intervals, per-metric significance scores, an
explanation of the ranking rule, organizer baselines, and qualitative examples
of every method on hard test pairs.

| # | Team | Final score | | # | Team | Final score |
|---:|---|---:|---|---:|---|---:|
| 1 | neeldey | 71.4 | | 9 | jbgeb | 25.9 |
| 2 | MIAgent | 37.3 | | 10 | adidukre | 25.8 |
| 3 | tinymilky | 35.6 | | 11 | longlai0000 | 25.0 |
| 4 | shishuyue | 33.4 | | 12 | bailiangj | 19.9 |
| 5 | koegl | 31.2 | | 13 | housheng | 18.1 |
| 6 | lukasf98 | 28.1 | | 14 | dutchmasters | 13.1 |
| 7 | abakei | 26.9 | | 15 | royxue07 | 8.2 |
| 8 | mbeguin | 26.6 | | 16 | kavehsfv | 6.2 |

Test set: 80 patients (160 registration pairs) from Johns Hopkins University,
evaluated by the organizers from the submitted containers.

## What is in this repository

- **`baselines/`** — organizer baselines (ANTs affine + ConvexAdam-MIND, the
  released docker example; VoxelMorph and TransMorph with/without instance
  optimization) with training scripts and pretrained weights.
- **`docker/`** — the submission container template used in the test phase.
- **`evaluation/`** — the evaluation metrics (CT/PET organ DSC and HD95, MTV
  and TLG preservation, % non-diffeomorphic volume) and the ranking script
  (significance scores, Holm correction, cap/gate rules, bootstrap).
- **`results/`** — the test-phase results, figures and `ranking_table.csv`.

## Data

Training and validation data are the PSMA PET/CT cohort of
[autoPET-III](https://autopet-iii.grand-challenge.org/) (LMU Munich; 378
patients, 166 with longitudinal studies), preprocessed to a common grid of
2.7344 × 2.7344 × 3.27 mm, 192 × 192 × 288 voxels, with TotalSegmentator organ
labels and expert lesion annotations. See the challenge website for download.

## Citation

If you use the data, baselines or results, please cite the Learn2Reg 2026
PSMAReg challenge (citation to follow) and the autoPET-III dataset
(Gatidis et al., *Sci Data* 2022; Jeblick et al., TCIA 2024).

## Contact

Junyu Chen, Johns Hopkins University — jchen245@jhmi.edu
