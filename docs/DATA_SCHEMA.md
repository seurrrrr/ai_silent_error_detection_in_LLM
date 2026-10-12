# Numerical data schema

`data/splits/sample_to_scene.csv` preserves the source mapping for 1,195 unique inputs, including nuScenes sample/scene identities and the upstream split index. Every released execution's sample/scene pair is checked against this mapping. Detector partitions are defined separately by `scene_split_manifest.csv`.

All feature and result tables use `execution_id` as the execution-level join key. `sample_id` identifies a driving input; `scene_token` is the scene-group identifier. Repeated executions from one scene share a partition.

`data/execution_metadata.csv.gz` contains all 8,295 execution identities, partitions, categories, structure/geometry status, injection status, and fault-campaign descriptors. The binary label is 1 for ADE-only and 0 for the eligible benign-fault/clean-reference categories; categories outside the binary population are distinguished explicitly and must not be treated as binary training negatives.

`data/output_features_248.csv.gz` contains `execution_id` and the ordered final 248 output features for all executions. Empty/nonfinite feature values are represented as missing values. Convert to float32 and preserve the order in `models/lightgbm/feature_list.txt` before model scoring.

`data/development_features_311.csv.gz` contains only the 3,795 eligible training executions and the frozen five-group candidate block used in development selection. It is not the final detector input. Its labels, identities, and order match the saved development fold-task indices.

`data/splits/scene_split_manifest.csv` and `row_partition_manifest.csv` provide the immutable scene/row partition assignment. Stopping and development fold indices refer to their associated ordered training tables, not arbitrary CSV row order.

Binary test results contain 53 positives and 1,380 negatives. Composite results contain 320 positives and 1,426 negatives, excluding two unknown test executions. The extra 46 composite negatives are no-effective-injection cases. Do not compare binary/composite FPRs using a shared denominator.

The numerical package uses execution identities and output-score statistics for detector reproduction. Scene prompts, coordinate trajectories and upstream caches follow the separate source-data workflow described in `DATA_ACCESS.md`.
