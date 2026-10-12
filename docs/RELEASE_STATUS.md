# Numerical validation

The numerical package contains the matched-clean, 248-dimensional LightGBM experiment, including execution metadata, features, models, partitions, calibration records, test predictions and development results.

## Recorded checks

| Check | Coverage |
| --- | --- |
| Saved-output reconstruction | All 8,295 executions, including clean pairing, parser/geometry outcomes, canonical categories and final features |
| Scene mapping | All executions mapped through 1,195 unique inputs; zero overlap across the 312/104/104 scene partitions |
| Frozen LightGBM scoring | All calibration, ADE-only test and composite test scores |
| Final detector fit | Training-scene stopping, 446 selected iterations, final fit and three calibration-selected thresholds |
| Development summaries | All 31 candidate summaries and 1,080 saved operating-metric rows |
| Fold replay | One recorded fold for the selected representation and each of its three LightGBM group-removal variants |
| Classifier controls | Calibration and test scores for logistic regression, random forest and scalar-score controls |
| Test uncertainty | 1,000 scene-bootstrap resamples and 42 confidence-interval entries |
| Paired ablation uncertainty | 2,000 paired scene-bootstrap draws |
| Output-score collection | Eight statistics, 31 positions and early-termination handling with synthetic CPU tensors |
| Adapter integrity | File SHA256 checks and safetensors tensor offsets |
| Package portability | Full numerical archive extracted to a new directory and tested from a separate working directory |

The recorded CPU environment uses Python 3.9 on Windows x64 with the dependencies in `requirements-lock.txt`. Machine-readable results appear in `results/reproduction_validation.json`. Generator usage and GPU requirements are documented separately in `GENERATION_INPUTS.md`.

## Integrity records

`MANIFEST.json` and `SHA256SUMS.txt` identify the source/numerical files. The task-adapter asset uses `models/qwen_lora/integrity.json`. Native LightGBM scoring preserves the feature order in `models/lightgbm/feature_list.txt`.

The full source/numerical package contains the development feature matrix and out-of-fold predictions needed for repeated-fold analysis. The smaller review archive identifies its contents and omitted large files in `REVIEW_CORE_SCOPE.md`.
