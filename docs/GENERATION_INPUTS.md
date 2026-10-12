# Generator and fault-campaign interfaces

`configs/campaigns/` contains the 6,500-case broad campaign and the 500-case and 100-case additional campaigns. Each plan records the campaign/sample identity, per-case random seed, target module, timing, bit policy and selector policy. Campaign type fields identify the broad and targeted construction sources. Detector partitions are defined by the scene manifest, independently of campaign grouping.

The supplied monitor target list and clean-profile statistics are the numerical dependencies used by the dense fault runner. The 31 output-score monitoring positions are defined in `configs/output_feature_contract.json`. The final detector consumes the corresponding 248-dimensional output-score matrix.

Generator inputs are acquired separately under the upstream data terms. The clean-bank JSONL uses `sample_id`, structured prompt fields, and reference trajectories as defined by the supplied clean/fault runners. The planned `sample_id` links campaign cases to clean-bank inputs. The raw reconstruction manifest specifies the event, compact-feature, raw-feature, and clean-bank records used in the canonical dataset.

For a new GPU environment, start with a small clean/fault input set and check the runner's outputs and monitor collection. Record the base-model revision, library versions and inference settings alongside generated records. The CPU reproduction workflow uses saved execution features; the generator entry points produce new trajectories and monitoring records.
