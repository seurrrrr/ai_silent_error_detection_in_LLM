# Qwen3.5-9B trajectory-generation task adapter

## Model description

This LoRA adapter maps structured driving-scene prompts to a JSON trajectory containing six future two-dimensional coordinate pairs. It provides the trajectory-generation workload for research on bit-level inference faults and task-level detection.

## Configuration

| Setting | Value |
| --- | --- |
| Base model | `Qwen/Qwen3.5-9B` |
| Adapter type | Low-rank adaptation (LoRA) |
| Rank | 32 |
| Alpha | 64 |
| Dropout | 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |

`adapter_config.json` contains the serialized adapter configuration. The weight, tokenizer and chat template accompany the separate model asset. `integrity.json` specifies the corresponding file sizes and SHA256 values.

## Usage

Obtain the base weights from [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B). Extract the adapter asset into this directory. From the repository root, check the extracted adapter files with:

```sh
python scripts/check_adapter.py
```

Pass the local base-model directory and adapter directory to the generation runner. Use a compatible Transformers, PEFT and quantization environment, recording the selected base revision and dependency versions with each run. Input preparation and runner arguments are described in `docs/GENERATION_INPUTS.md`.

Matched clean-reference trajectories provide the nominal model baseline for fault-induced displacement measurements. Clean qualification uses the format, geometry and clean-to-dataset-reference ADE criteria specified in `docs/PROTOCOL.md`.

## Adaptation metadata and entry points

`configs/qwen_adapter_training_config.json` provides saved training parameters and identifies source-script defaults separately. Stored settings include learning rate 0.0001, weight decay 0, per-device batch size 1, gradient accumulation 8, one epoch and seed 42.

`src/generation/build_qwen_sft_dataset.py` prepares supervised input/output records. `src/generation/train_qwen_trajectory_lora.py` provides the fine-tuning entry point for additional adaptation runs. Dataset paths and run settings are supplied through its command-line arguments.

## Intended use and terms

The adapter supports research and offline evaluation of trajectory-generation behavior under numerical faults. Qwen assets follow the accompanying upstream license; driving-scene resources follow their providers' data terms. Source links and notices appear in `THIRD_PARTY_NOTICES.md` and `docs/DATA_ACCESS.md`.
