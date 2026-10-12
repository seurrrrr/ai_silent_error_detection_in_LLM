# Experimental protocol

## Trajectory outcomes

Clean-reference trajectories are generated without injected faults under the matched input/model/inference configuration and qualified using the structure, geometry, and clean-to-dataset-reference error checks. Fault-induced ADE-only labels use **fault-to-matched-clean ADE**, not clean qualification error or fault-to-dataset-reference error.

Effective faulted executions are categorized as structural/parsing failures when a complete finite six-point trajectory cannot be extracted, geometry-related failures when the geometry rules flag a parsed trajectory, ADE-only failures when these checks pass and matched-clean ADE exceeds 1.0 m, and benign faulted executions otherwise. No-effective-injection and unknown outcomes are retained separately. Unknown test outcomes are excluded from the primary composite population.

The parser and geometry predicates are provided in `src/generation/output_rules.py`. Collision proxies interpolate surrounding-agent positions to each predicted waypoint and compare distances with the corresponding radii. Missing surrounding-agent evidence leaves the collision proxy unobserved; it does not trigger an alarm. Self-intersection checks nonadjacent trajectory segments.

## Partitions and development

All executions from a scene remain in one partition. The fixed 520-scene split is 312/104/104, with split seed 18757. Only the 312 training scenes enter feature-subset comparison, repeated folds, preprocessing, model fitting, and stopping. The 104 calibration scenes determine operating thresholds. The 104 test scenes provide fixed scene-disjoint evaluation after final fitting and calibration; they were retained from earlier development.

Development evaluates 31 nonempty subsets of five candidate groups using ten five-fold repetitions. Pooled out-of-fold average precision is computed per repetition. The highest arithmetic mean chooses the final subset; ties within 1e-12 prefer fewer features and then a lower integer mask. The classifier recipe remains LightGBM. The selected subset comprises score magnitude, token competition, and numeric-token evidence.

Final-set leave-one-group-out ablations use another ten fixed five-fold partitions of the same training scenes. Paired scene bootstrap conditions on the saved OOF predictions. This measures partition robustness within the development data; it does not create an independent confirmation dataset.

## Final fitting and operating points

The final fit retains all 145 training positives and 3,650 eligible negatives. Positive-class weight is the negative-to-positive ratio within the current fitting role. There is no final negative downsampling.

Early stopping uses 249 fitting and 63 stopping scenes from the training partition, stopping seed 20261928, binary log loss, at most 2,000 estimators, and patience 80. The selected 446 iterations are refitted on all 312 training scenes before calibration. No refit follows threshold selection.

LightGBM parameters are stored in `configs/detector_protocol.json`. The primary operating point maximizes calibration recall subject to calibration FPR at most 1%; ties prefer lower FPR, then higher threshold. Prediction is positive when score is at least the threshold. The additional 2% and 5% budgets follow the same rule. A calibration budget is not a guarantee of the observed test FPR.

## Composite alarm and uncertainty

After trajectory generation, the composite decision is the logical OR of the output-format check, geometry checks, and supervised detector decision. Features from unreached monitor positions remain missing. Six waypoints represent predicted future motion; detection does not wait for the vehicle to travel through them.

The binary test contains 53 positives and 1,380 negatives. The composite test contains 256 structural/parsing, 11 geometry-related, and 53 ADE-only positives, with 1,426 negatives. Deterministic category coverage follows from the checks defining those categories.

Test confidence intervals resample 104 scenes 1,000 times with seed 20260928, holding model and thresholds fixed. Paired development intervals resample 312 training scenes 2,000 times with seed 20261002, averaging across saved repetitions. The release includes all computed classifier controls and candidate subsets, irrespective of performance ordering.
