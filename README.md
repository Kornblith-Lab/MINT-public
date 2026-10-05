# Learning the electronic health record at the minute scale

## A prognostic study of sixteen years of visits at five health systems

This repository contains the code for **MINT** (MINute-scale Trajectory model), a decoder-only foundation model for learning from pediatric emergency-department trajectories at minute-level resolution. MINT represents time-stamped clinical events—including triage information, vital signs, medications, procedures, respiratory support, abnormal laboratory results, and disposition—and forecasts near-term changes in clinical risk.

The accompanying manuscript evaluates MINT using 766,733 pediatric emergency-department visits across ten hospitals and five health systems. The study assesses minute-scale forecasting, external generalization, out-of-vocabulary outcomes, dynamic monitoring, individualized explanations, and simulated physiologic associations. MINT is an associative research model; it is not a causal model, medical device, or a substitute for clinical judgment.

## Data and model availability

The clinical data used in this study are not publicly available. The University of California does not permit release of the de-identified source data. Model-weight distribution is subject to institutional and regulatory policy. This repository therefore does not include patient-level data, trained checkpoints, or generated study artifacts.

## Repository contents

- `mint/tokenizer_v2/` contains the YAML-driven pipeline that transforms source clinical tables into minute-scale trajectories.
- `train.py`, `model.py`, `utils.py`, and `config/train_mint.py` contain the MINT training implementation and primary training configuration.
- `mint/five/` contains analyses and figure-generation workflows for the manuscript.
- `tokenizer_config.yaml` specifies the study token vocabulary and preprocessing rules.

## Installation

Create a Python 3.11 environment and install the core dependencies:

```bash
conda create -n mint python=3.11
conda activate mint
```

Some analyses have additional dependencies, such as scikit-learn and XGBoost. Install those packages when running the corresponding figure workflow.

## Preparing trajectories

The tokenizer expects source tables and column names matching `tokenizer_config.yaml`. The study data are not included; use only appropriately governed data with a locally adapted configuration.

```bash
python -m mint.tokenizer_v2.tokenize \
    --config tokenizer_config.yaml \
    --data-dir /path/to/source_tables \
    --output-dir output
```

The tokenizer writes the trajectory and vocabulary outputs consumed by training. See [the tokenizer documentation](mint/tokenizer_v2/README.md) for configuration details and input-table expectations.

## Training

After tokenization, train the foundation model with the study configuration:

```bash
python train.py config/train_mint.py --device=cuda --out_dir=output/mint
```

The supplied configuration uses a 12-layer decoder-only transformer and is configured for 100,000 optimization steps.

## Analyses and figures

The figure workflows under `mint/five/` reproduce the manuscript analyses when supplied with the governed input data, trained checkpoint, and any required intermediate files. Outputs are written below `artifacts/`, which is excluded from version control. Individual modules document their expected inputs and command-line options.

## Contact

Kush Narang — [kushnarang@stanford.edu](mailto:kushnarang@stanford.edu)

## Acknowledgements

The authors acknowledge the use of the University of California, San Francisco (UCSF), Information Commons computational research platform, developed and supported by UCSF Bakar Computational Health Sciences Institute. The authors thank the Center for Data-driven Insights and Innovation at UC Health (CDI2; https://www.ucop.edu/uc-health/departments/center-for-data-driven-insights-and-innovations-cdi2.html), for its analytical and technical support related to use of the UC Health Data Warehouse and related data assets.


Agentic coding tools were used to assist software development and related research workflows.
