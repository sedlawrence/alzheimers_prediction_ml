# ML Engineer Take-Home

This repository contains the full workflow for the two Alzheimer’s disease progression tasks based on longitudinal blood DNA methylation.

## Setup

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

PyTorch may need a platform-specific install on some systems. On Apple Silicon, the `mps` device can be used for the deep learning scripts.

## Running the workflow

The numbered scripts in `scripts/` are intended to be run in order.

```bash
cd scripts
bash ../commands.sh
```

The `commands.sh` file contains the full ordered run list for:

1. Task 1 data description and exploratory analysis
2. Task 1 baselines, XGBoost, linear models, and deep learning
3. Task 2 data description and exploratory analysis
4. Task 2 baselines, XGBoost, linear models, and deep learning
5. The joint Task 1 + Task 2 deep learning optimisation pass

Individual scripts can also be run directly if only one stage needs to be repeated.

## Repository layout

- `data/` contains the HDF5 methylation matrices and the CpG annotation file.
- `results/` contains the model outputs, summaries, fold-level metrics, and figures.
- `reports/` contains the final PDF report and the report figures used in the write-up.
- `scripts/` contains the modelling code and shared helpers.

## Outputs

Model comparison tables, fold results, and figure assets are written under `results/task1/` and `results/task2/`. The final report PDF is stored in `reports/FinalReport.pdf`.
