# Final output-score feature contract

The final input is 31 positions x 8 statistics = 248 features. The ordered list is `models/lightgbm/feature_list.txt`; the position list and group mapping are in `configs/output_feature_contract.json` and `configs/final_feature_group_map.csv`.

At each retained decoding position, the final-token projection hook detaches the vocabulary-score vector and converts it to float32. Each position has its own feature entries; no across-position aggregation is applied to the LightGBM input. The first observation for each slot is retained.

| Group | Statistic | Features | Implemented definition |
|---|---|---:|---|
| Score magnitude | RMS | 31 | Square root of mean squared finite scores. |
| Score magnitude | Maximum absolute score | 31 | Maximum absolute finite score. |
| Token competition | Highest score | 31 | Largest finite score. |
| Token competition | Second-highest score | 31 | Second-largest finite score; missing if fewer than two finite entries exist. |
| Token competition | Top-two margin | 31 | Highest minus second-highest score. |
| Token competition | Top-20 entropy proxy | 31 | Sum of `-p_i log(p_i + 1e-12)` at the returned top-20 indices, with `p = softmax(clamp(z, -80, 80))`; selected probability mass is not renormalized. |
| Numeric-token evidence | Numeric probability mass | 31 | Sum of these full-vocabulary probabilities over the numeric-token set. |
| Numeric-token evidence | Maximum numeric score | 31 | Maximum original score over the numeric-token set. |

The numeric-token set uses the tokenizer's raw token strings: a token qualifies if it contains a digit, `.`, `-`, or `+`. The collector includes a character-encoding fallback if vocabulary enumeration fails.

The exact collector is supplied in `src/generation/output_score_monitor.py`, including behavior for nonfinite entries. Finite-score magnitude/top-k operations use the filtered finite vector; if it is empty, a one-element zero vector is substituted. Entropy indexing follows the indices returned from that operation and the original full-vector softmax. Nonfinite probability-derived results become missing in the detector matrix.

Generation may terminate before a retained position. Unobserved slots remain missing; the detector does not wait for all 31 positions. Matrix construction coerces nonfinite values to NaN and preserves native LightGBM missing-value handling.

`scripts/check_feature_monitor.py` checks collection, statistic definitions and missing-slot handling with synthetic CPU tensors.
