# mitra-classifier-dataset-validator

DIMER dataset validator for the Mitra classifier pipeline. It checks that an uploaded
tabular-classification dataset zip meets the CSV contract before fine-tuning runs.

- Runs as a CPU Kubernetes Job.
- DIMER builds the root `Dockerfile` into an ECR image and runs `validator.py`.
- Pairs with `mitra-classifier-finetuner`.

The complete pipeline documentation, dataset specification, and the fine-tuner are in the
[mitra-classifier-pipeline](https://github.com/kurtvalcorza/mitra-classifier-pipeline) project.

## Contract summary

The dataset zip must contain a `train.csv` with a categorical `target` column whose distinct
values are the class labels (2–10 classes); every other non-dropped column is a feature. The
validator reports pass/fail per check in `result.json`. See the project's dataset specification
for the full rules.
