# Release assets

The source/numerical archive and the trajectory-generation adapter are separate assets. Use the repository's [Releases page](https://github.com/seurrrrr/ai_silent_error_detection_in_LLM/releases) to obtain the corresponding version.

| Asset | Contents | Use |
| --- | --- | --- |
| `LLM_SILENT_ERROR_FINAL_248D_REPRODUCTION.zip` | Source code, configurations, final features, detector models, execution metadata, partitions, predictions, development matrices and out-of-fold results | Complete numerical detector reproduction and development analysis |
| `QWEN_TRAJECTORY_LORA_ADAPTER.zip` | LoRA weight, configuration, tokenizer, chat template, model card, integrity record and upstream license | Trajectory generation with the Qwen base model and separately acquired scene inputs |
| `LLM_SILENT_ERROR_REVIEW_CORE.zip` | Final detector features and models, partitions, calibration/test scores, operating points and development summaries | Smaller artifact-review subset and final detector scoring/fitting |

## Source/numerical archive

Extract the full source/numerical archive into a directory. From that directory, follow the installation and commands in the root README:

```sh
python -m pip install -r requirements-lock.txt
python scripts/reproduce.py verify
python scripts/reproduce.py score
```

Detector reproduction uses CPU dependencies and the supplied numerical feature matrices. The Qwen base weights and task adapter are not required for these stages.

## Task adapter

Extract the adapter archive at the same root as the source/numerical archive. It contains a `models/qwen_lora/` path, so the weight will be placed alongside its configuration. Then run:

```sh
python scripts/check_adapter.py
```

The base model is obtained from [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B). Generator dependencies and input preparation are described in [GENERATION_INPUTS.md](GENERATION_INPUTS.md) and [DATA_ACCESS.md](DATA_ACCESS.md).

## Review subset

Extract the review archive into its own directory. Its `MANIFEST.json` and `SHA256SUMS.txt` cover the files included in that subset, and `REVIEW_CORE_SCOPE.md` identifies the omitted development matrices and tokenizer. The `verify`, `score` and final-detector `refit` stages use its included numerical artifacts. Complete fold replay and bootstrap analysis use the full source/numerical archive.

## Checksums and source terms

`RELEASE_CHECKSUMS.txt` lists the SHA256 of the ZIP assets. Each source/numerical archive includes its own file manifest; adapter files are identified in `models/qwen_lora/integrity.json`. Keep upstream license and attribution notices with their assets. Dataset access follows [DATA_ACCESS.md](DATA_ACCESS.md), and code licensing follows the root LICENSE and [third-party notices](../THIRD_PARTY_NOTICES.md).
