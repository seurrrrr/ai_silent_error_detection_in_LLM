# Reproduction

## Numerical detector artifacts

The default numerical artifacts support three stages:

1. `verify`: verify hashes, data identities, scene separation, feature counts, saved predictions, metrics, and threshold selection on calibration scores.
2. `score`: load the frozen native LightGBM model and reproduce scores from the 248-dimensional feature matrix.
3. `refit`: reproduce training-only stopping, the final fit on all training rows, and the calibration-selected thresholds in the recorded CPU environment.

Output paths are configurable, and generated results are separate from the frozen artifacts. The recorded dependencies and feature order support numerical comparisons with the saved models. Verification commands report the maximum absolute score difference for each population.

`results/development/` contains the 31 candidate group-subset search, repetition metrics, fold roles, and pooled out-of-fold predictions. `results/ablation/` contains the final 248-D representation and its three leave-one-group-out variants evaluated in training-scene folds. Development candidate features are supplied only for the training partition. Global calibration and test rows are not inputs to feature selection.

Classifier controls include logistic regression, random forest and scalar score statistics. Their saved models, calibration records and test scores accompany the LightGBM results.

Recompute development summaries and uncertainty directly from saved predictions:

```sh
python scripts/reproduce_statistics.py summaries
python scripts/reproduce_statistics.py bootstrap-test
python scripts/reproduce_statistics.py bootstrap-ablation
```

Replay a recorded fitting fold and compare its predictions with the saved OOF records:

```sh
python scripts/reproduce_development.py search --candidate G07 --repetition 0 --fold 0 --check-saved
python scripts/reproduce_development.py ablation --candidate SELECTED --repetition 0 --fold 0 --check-saved
python scripts/reproduce_development.py ablation --candidate MINUS_SCORE_MAGNITUDE --check-saved
```

Use `--all-folds` to replay every recorded fold for a requested candidate. Search candidates are `G01` through `G31`; ablation candidates are `SELECTED`, `MINUS_SCORE_MAGNITUDE`, `MINUS_TOKEN_COMPETITION`, `MINUS_NUMERIC_TOKEN_EVIDENCE` and `RF_SAME_FEATURES`. Each replay writes to its requested output directory and compares predictions against the recorded out-of-fold results.

## Generating trajectories and injecting faults

Generator commands are supplied separately from the CPU detector workflow. Obtain the Qwen base weights and a compatible GPU software environment, install the optional dependencies, and provide inputs under their applicable data terms. Consult `models/qwen_lora/MODEL_CARD.md` before choosing a base revision.

```sh
python src/generation/run_v7_clean_candidates.py --help
python src/generation/run_v7_fault_injection_b18_dense_logits.py --help
python src/generation/build_qwen_sft_dataset.py --help
python src/generation/train_qwen_trajectory_lora.py --help
```

The runners take input/output paths as command-line arguments. Set the output-score monitoring positions to the 31 values listed in `configs/output_feature_contract.json`. The dense runner resolves its companion fault-injection module relative to its own file.

The raw runner records dataset-reference errors for execution diagnostics. ADE-only labels use displacement from the matched clean-reference trajectory after the structure and geometry checks pass. The canonical rule is `ADE(faulted, matched_clean) > 1.0 m`, implemented by the label-reconstruction scripts.

The recorded 7,100 fault-case plans and monitoring dependencies are described in `GENERATION_INPUTS.md`. Scene prompts and upstream reference trajectories are acquired through the data-preparation workflow in `DATA_ACCESS.md`.

`scripts/reconstruct_labels.py` computes this canonical category from supplied saved trajectory/status records. `src/generation/output_rules.py` contains the trajectory parser and geometry predicates extracted from the runner, without importing the language model.

For authorized raw-data holders, `data/reconstruction_source_manifest.csv` specifies the source-file layout and execution keys. Rebuild the parser, geometry checks, clean pairing, categories, and final numerical features with:

```sh
python scripts/rebuild_canonical.py --sources-root PATH_TO_AUTHORIZED_SOURCE_ROOT --output generated/rebuilt
```

This checks raw/compact feature equality, event/trajectory consistency, matched-clean qualification, injection counts and canonical categories. It compares the rebuilt features with the numerical package. Supply the required source records locally using the layout in the reconstruction manifest.

The task-adapter asset supplies the saved trajectory-generation parameters used for this experiment. Data preparation and fine-tuning entry points support additional adaptation runs with user-specified inputs. Generator runs use the base model, adapter, GPU dependencies and scene inputs; numerical detector reproduction uses the distributed feature matrices and detector models.

## Region-distribution plot

The region-distribution plot can be regenerated from the supplied numerical source table:

```sh
python -m pip install -r requirements-figures.txt
python scripts/plot_region_sensitivity_final.py
```

Outputs are written under `generated/figure2/`. The script checks the 235 ADE-only and 1,405 structural/parsing totals, region order, label placement and legend overlap.

## Integrity and release scope

`MANIFEST.json` and `SHA256SUMS.txt` cover the files in the corresponding source/numerical archive. The separate Qwen adapter asset is covered by `models/qwen_lora/integrity.json`. The review archive has its own manifest and supports final detector verification and scoring; development fold replay uses the full source/numerical archive. Reproduction commands preserve the distributed artifacts and write generated outputs under the requested directory. Package instructions appear in [RELEASE_ASSETS.md](RELEASE_ASSETS.md).

The data roles, selection folds and operating-point rules are specified in `PROTOCOL.md`. Follow those assignments when reproducing the development comparisons and final evaluations.
