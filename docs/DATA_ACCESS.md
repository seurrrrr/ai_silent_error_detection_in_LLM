# Data access

## Numerical detector data

The numerical package provides output-score features, execution identities, outcome annotations, fault metadata, scene partitions and detector results. Execution IDs connect feature matrices, fold roles and per-execution scores. The tables and their join keys are defined in [DATA_SCHEMA.md](DATA_SCHEMA.md).

The supplied matrices support detector scoring, fitting and development analysis directly. Raw sensor recordings, scene prompts, coordinate trajectories and upstream caches are acquired separately for generator workflows.

## Upstream driving-scene resources

Driving-scene preparation follows the GPT-Driver workflow using nuScenes information and model-derived perception/prediction caches. Obtain the source materials from their original providers:

| Resource | Provider and instructions |
| --- | --- |
| nuScenes data | [Dataset website](https://www.nuscenes.org/), [download page](https://www.nuscenes.org/download), [Terms of Use](https://www.nuscenes.org/terms-of-use) |
| nuScenes setup | [Official devkit instructions](https://github.com/nutonomy/nuscenes-devkit#nuscenes-setup) |
| GPT-Driver prepared inputs | [Data-preparation instructions](https://github.com/PointsCoder/GPT-Driver#data-preparation) |

Follow each provider's access and use terms when downloading and preparing its materials. Upstream data retains its source terms; the project code license applies to this repository's code.

## Generator and reconstruction interfaces

The clean-bank input uses `sample_id`, structured prompt fields and reference trajectories. Campaign plans link each injection case to an input using `sample_id`. [GENERATION_INPUTS.md](GENERATION_INPUTS.md) describes the generator entry points and monitoring configuration.

`data/reconstruction_source_manifest.csv` specifies the event, raw-feature, compact-feature and clean-bank records consumed by the reconstruction script. With those records arranged under a source root, run:

```sh
python scripts/rebuild_canonical.py --sources-root PATH_TO_AUTHORIZED_SOURCE_ROOT --output generated/rebuilt
```

This workflow recomputes trajectory validity, geometry outcomes, matched-clean ADE labels and output-score features. It compares the reconstructed values with the released numerical data.
