# Included dynamic identification data

This directory contains the four dynamic training recordings used for the
published shared model and its separate validation recording. Each record
includes raw feedback (`capture.csv`), acquisition metadata, planned trajectory
and acquisition quality report. The CSV records actual joint position, velocity,
and feedback torque; the target columns are retained for auditing and are not
used by identification.

`training_03/quality.json` reports one long feedback gap. The samples and report
are retained as collected; the identification preprocessor segments across the
gap. The other included records passed their acquisition quality checks.

The public URDF keeps the original joint frames and inertial fields but uses
different visual and collision geometry. Its SHA256 is recorded in each
metadata file. The original URDF SHA256 is retained as `original_urdf_sha256`.
The kinematic regressor used by the identified model is unchanged by the
geometry substitution.

Re-fit the shared model and evaluate the held-out recording with:

```bash
uv run --locked --extra identify arm-id-identify-many \
  --train data/dynamic_identification/training_01/capture.csv \
  --train data/dynamic_identification/training_02/capture.csv \
  --train data/dynamic_identification/training_03/capture.csv \
  --train data/dynamic_identification/training_04/capture.csv \
  --validation data/dynamic_identification/validation/capture.csv \
  --output results/reproduced_fit
```

Hashes and sample counts are listed in [`index.json`](index.json). The separate
validation recording was held out from coefficient fitting, but its results
have already been inspected in this repository.
