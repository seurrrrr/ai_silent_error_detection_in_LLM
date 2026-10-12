# Task-Level Silent Error Detection for LLM-Based Trajectory Generation

Code, models and numerical artifacts for **Task-Level Silent Error Detection for Large-Language-Model-Based Trajectory Generation in Autonomous Driving**.

The framework studies bit-level inference faults in a Qwen3.5-9B trajectory generator adapted using low-rank adaptation (LoRA). A composite task-level detector combines deterministic output-format and geometry checks with supervised average displacement error (ADE)-only detection. ADE-only labels compare structurally and geometrically valid faulted trajectories with qualified matched clean-reference trajectories under the same input, model and inference configuration.

## Final detector and results

The LightGBM detector uses 248 output-score features collected at 31 decoding positions. The representation contains 62 score-magnitude features, 124 token-competition features and 62 numeric-token-evidence features. [Feature definitions](docs/FEATURES.md) specify the statistics, feature order and missing-value handling.

The execution dataset contains 8,295 records, including 235 ADE-only failures. The scene-disjoint split assigns 312 scenes to training, 104 to calibration and 104 to testing. Training uses all 145 eligible positives and 3,650 negatives.

The primary threshold is `0.54456291148702074`, selected by maximizing calibration recall subject to a 1% calibration false positive rate (FPR) budget.

| Test population | TP | FN | FP | TN | Recall | FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ADE-only | 29 | 24 | 10 | 1,370 | 54.72% | 0.725% |
| Composite task-level | 296 | 24 | 10 | 1,416 | 92.50% | 0.701% |

ADE-only detection has an area under the receiver operating characteristic curve of 0.9641 and an area under the precision-recall curve of 0.7202. The two test populations and their category definitions are specified in the [experimental protocol](docs/PROTOCOL.md).

## Installation and quick start

The recorded CPU environment uses Python 3.9 and LightGBM 4.6.0. Run the commands below from the root of the repository. Install the numerical dependencies and verify the full source/numerical package:

```sh
python -m pip install -r requirements-lock.txt
python scripts/reproduce.py verify
python scripts/reproduce.py score
```

These commands check artifact hashes, scene assignments, feature counts, scores, metrics and calibration threshold selection. Numerical detector reproduction runs on CPU using the supplied feature matrices and model.

Reproduce the final fit, including training-scene early stopping and calibration:

```sh
python scripts/reproduce.py refit --output generated/refit
```

Outputs are written to `generated/refit`, preserving the distributed frozen model. The reproduction checks compare the fitted model and its operating points with the saved artifacts.

## Repository layout

| Directory | Contents |
| --- | --- |
| `data/` | Execution metadata, output-score features and scene/row partitions |
| `models/lightgbm/` | Frozen detector and ordered feature list |
| `models/controls/` | Classifier and scalar-score comparison models |
| `models/qwen_lora/` | Task-adapter configuration, tokenizer, model card and integrity record |
| `configs/` | Training/calibration parameters, feature definitions, fold roles and injection plans |
| `results/` | Scores, operating points, development comparisons, ablation and uncertainty estimates |
| `scripts/` | Numerical reproduction and verification tools |
| `src/generation/` | Data preparation, adaptation, clean generation, fault injection and monitoring |
| `docs/` | Data schema, protocol, feature definitions and usage instructions |

The root-level `qwen3_5_9b_4bit.py` provides Qwen inference and layerwise fault-injection utilities; `analyze_layer_sensitivity.py` summarizes layer-sensitivity logs. Trajectory generation and numerical detector reproduction use the entry points under `src/generation/` and `scripts/`, respectively.

## Qwen trajectory-generation adapter

The task adapter uses LoRA rank 32, alpha 64 and dropout 0.05. Obtain the base model from [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) and extract the separate adapter asset into `models/qwen_lora/`.

```sh
python scripts/check_adapter.py
```

The adapter asset contains the LoRA weight, configuration, tokenizer and chat template. Generator execution uses the additional dependencies in `requirements-generator.in` and a compatible GPU environment. Record the selected base-model revision and software versions for each generation run. See the [model card](models/qwen_lora/MODEL_CARD.md) and [generator input guide](docs/GENERATION_INPUTS.md) for usage.

## Data and upstream resources

The numerical package contains model-output statistics, execution annotations, partitions and detector results. nuScenes data and GPT-Driver preparation resources are obtained from their original providers:

- [nuScenes dataset and downloads](https://www.nuscenes.org/)
- [GPT-Driver data preparation](https://github.com/PointsCoder/GPT-Driver#data-preparation)

The source/numerical package supports detector scoring and fitting independently of these upstream downloads. Trajectory generation and raw-output reconstruction use separately acquired scene inputs. [Data access](docs/DATA_ACCESS.md) describes the file interfaces and source terms.

## Reproduction and integrity

The [reproduction guide](docs/REPRODUCTION.md) covers saved-output reconstruction, development fold replay, classifier controls, scene-bootstrap statistics and Figure 2 plotting. Numerical checks and the recorded environment are summarized in [validation coverage](docs/RELEASE_STATUS.md).

`MANIFEST.json` and `SHA256SUMS.txt` identify the repository's source/numerical files. The adapter has a separate integrity record in `models/qwen_lora/integrity.json`. The smaller review archive is a subset for inspection; full development reproduction uses the complete source/numerical package. Each archive has an integrity manifest for its own contents. See [release assets](docs/RELEASE_ASSETS.md) for package selection and extraction instructions.

## License and citation

Project code is licensed under [Apache-2.0](LICENSE). Model, data and bundled third-party components follow the terms listed in [third-party notices](THIRD_PARTY_NOTICES.md). Use `CITATION.cff` when citing this work. This repository supports research and offline evaluation of inference-fault detection.
