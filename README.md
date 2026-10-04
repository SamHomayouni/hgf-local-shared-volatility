# Local vs Shared Volatility in Hierarchical Gaussian Filter Models

This repository contains the current computational modelling pipeline for studying how beliefs about volatility are represented and transferred across multiple information streams in Hierarchical Gaussian Filter (HGF) models.

The central modelling question is:

> If one information stream becomes more volatile while another remains objectively stable, does a model with a shared or global volatility representation increase its learning rate in the stable stream as well?

The project uses synthetic two-stream environments and several candidate HGF architectures implemented with `pyhgf`. The current repository is primarily a simulation, model-comparison, and recovery pipeline. Scientific parameter choices are still being refined, so the repository distinguishes stable pipeline infrastructure from provisional modelling choices.

---

## Current pipeline

```text
environment configuration
        |
        v
two-stream Gaussian random-walk environment
        |
        v
local-only / global-only / hybrid HGF models
        |
        +--------------------+
        |                    |
        v                    v
learning-rate analysis   synthetic responses
        |                    |
        v                    v
CTI / plots            parameter and model recovery
```

The normal runnable pipeline lives under `scripts/`. Recovery and identifiability experiments live under `experiments/`. An earlier changepoint-based environment is retained under `archive/` for provenance but is not part of the current pipeline.

---

## 1. Synthetic environments

The current environment generator is:

```text
src/environments/two_stream_hgf_random_walk.py
```

It generates two continuous streams, A and B, whose latent values evolve as Gaussian random walks.

For stream s in {A, B},

\[
x_s(k)=\lambda_x x_s(k-1)+ho_x+\sqrt{Q_s(k)}\,\epsilon_s(k)
\]

with observations

\[
u_s(k)=x_s(k)+\sigma_{\mathrm{obs}}\eta_s(k),
\]

where

\[
\epsilon_s(k),\eta_s(k)\sim\mathcal N(0,1).
\]

The current environments use:

```text
lambda_x = 1.0
rho_x = 0.0
q_low = 0.0004
q_high = 0.01
observation_noise_sd = 0.10
random_seed = 123
n_trials = 1000
```

Thus the latent process is a pure Gaussian random walk rather than a mean-reverting process. Volatility is manipulated through the process variance Q, while observation noise is fixed.

The 1000 trials are divided into five 200-trial blocks:

```text
S1: trials   0-199
V1: trials 200-399
S2: trials 400-599
V2: trials 600-799
S3: trials 800-999
```

Three current environment configurations are provided.

| Environment | Stream A in V1/V2 | Stream B in V1/V2 |
|---|---|---|
| `local_A` | volatile | stable |
| `local_B` | stable | volatile |
| `global` | volatile | volatile |

Configuration files:

```text
config/hgf_random_walk_local_A.yaml
config/hgf_random_walk_local_B.yaml
config/hgf_random_walk_global.yaml
```

In the `global` environment, the two streams share the same process-variance schedule but retain independent Gaussian innovations. The environment therefore shares volatility structure without forcing the latent trajectories to move in the same direction.

### Important generative/inference distinction

The environment uses `lambda_x = 1.0` for the generative random walk.

The inference models currently use a value-state autoconnection strength of `0.95`.

This mismatch is intentional. The environment is not generated from the inference models themselves, and the inference models are not assumed to know the exact generative dynamics.

---

## 2. Candidate HGF models

Three candidate HGF architectures are implemented in `src/models/`.

### Local-only model

```text
input_A <- value_A <- volatility_A
input_B <- value_B <- volatility_B
```

Each stream has its own volatility estimate. There is no shared/global volatility node.

File:

```text
src/models/local_only.py
```

Expected qualitative fingerprint: if only stream A becomes volatile, the learning rate for A should increase while B should not systematically inherit that increase.

### Global-only model

```text
input_A <- value_A <-
                      shared_volatility
input_B <- value_B <-
```

The two value states share a single volatility parent.

File:

```text
src/models/global_only.py
```

Expected qualitative fingerprint: when one stream becomes volatile, the shared volatility estimate can alter learning in both streams.

### Hybrid / shared-volatility model

The code-level model name is `shared_volatility`. Structurally, it is a hybrid architecture:

```text
input_A <- value_A <- local_volatility_A <-
                                           shared higher parent
input_B <- value_B <- local_volatility_B <-
```

Each stream retains a local volatility state, but the two local volatility states are coupled through a higher shared value parent.

File:

```text
src/models/shared_volatility.py
```

This architecture can represent both stream-specific and partially shared volatility.

### Current default model parameters

| Parameter | Local-only | Global-only | Hybrid/shared |
|---|---:|---:|---:|
| input precision | 100 | 100 | 100 |
| value tonic volatility | -7.824 | -7.824 | -7.824 |
| value autoconnection | 0.95 | 0.95 | 0.95 |
| value coupling to input | 1.0 | 1.0 | 1.0 |
| local volatility tonic volatility | -6.0 | — | -6.0 |
| local volatility coupling | 1.0 | — | 0.7 |
| local volatility autoconnection | — | — | 0.7 |
| shared volatility tonic volatility | — | -6.0 | — |
| shared volatility coupling | — | 1.0 | — |
| shared higher-parent tonic volatility | — | — | -4.0 |
| shared higher-parent coupling | — | — | 0.75 |

The hybrid values `0.7`, `0.75`, and `-4.0` are currently provisional modelling choices introduced while stabilizing and exploring the hybrid architecture. They should not be treated as final paper-derived parameters.

---

## 3. HGF learning-rate analysis

The main analysis is implemented in:

```text
src/analysis/learning_rates.py
src/analysis/plot_learning_rates.py
scripts/analyze_learning_rates.py
```

For all three current models, the first four node IDs are aligned:

```text
0 = input_A
1 = input_B
2 = value_A
3 = value_B
```

The trial-wise learning-rate measure used in this project is the precision ratio:

```text
stream A:
x_0_expected_precision / x_2_expected_precision

stream B:
x_1_expected_precision / x_3_expected_precision
```

Trial 0 is excluded from the default summary analysis because it is treated as an initialization trial.

### Block-wise volatility effects

For stream A:

\[
D_{A,1}
=
arlpha_A(V1)
-
rac{arlpha_A(S1)+arlpha_A(S2)}{2}
\]

\[
D_{A,2}
=
arlpha_A(V2)
-
rac{arlpha_A(S2)+arlpha_A(S3)}{2}
\]

\[
D_A
=
rac{D_{A,1}+D_{A,2}}{2}.
\]

The same calculation is performed for stream B to obtain \(D_B\).

### Cross-transfer index

For `local_A`:

\[
\mathrm{CTI}_{Aightarrow B}
=
rac{D_B}{D_A}.
\]

For `local_B`:

\[
\mathrm{CTI}_{Bightarrow A}
=
rac{D_A}{D_B}.
\]

The CTI is only treated as interpretable when the directly manipulated stream shows a positive learning-rate effect.

For the `global` environment, CTI is not defined because both streams are objectively volatile. In that condition, `D_A` and `D_B` are both direct volatility effects.

The block-average CTI is only one summary. Trial-wise learning-rate trajectories should also be inspected because transfer may be delayed or transient, particularly in the hybrid architecture.

---

## 4. Continuous response model

Synthetic participant responses can be generated from HGF predictions using:

```text
scripts/simulate_continuous_predictions.py
src/response_models/continuous_prediction.py
```

For each stream,

\[
y_s(t)
\sim
\mathcal N\left(
\hat{\mu}_s(t),
\sigma_{\mathrm{resp},s}^2
ight).
\]

The current model prediction is taken from:

```text
stream A: x_2_expected_mean
stream B: x_3_expected_mean
```

These are interpreted as the model's pre-outcome predictions for the two streams.

The default response-noise standard deviation is `0.10`. Separate response-noise values for A and B are supported. Trial 0 is excluded from the response likelihood by default.

The response-model code also computes a Nassar-style behavioral quantity based on update divided by prediction error. That diagnostic should not be confused with the HGF precision-ratio learning-rate measure used in the main latent-state analysis.

---

## 5. Repository structure

```text
.
├── README.md
├── .gitignore
├── requirements.txt
├── requirements-dev.txt
├── requirements-lock.txt
│
├── config/
│   ├── hgf_random_walk_local_A.yaml
│   ├── hgf_random_walk_local_B.yaml
│   └── hgf_random_walk_global.yaml
│
├── src/
│   ├── environments/
│   │   └── two_stream_hgf_random_walk.py
│   ├── models/
│   │   ├── local_only.py
│   │   ├── global_only.py
│   │   └── shared_volatility.py
│   ├── pipeline/
│   │   └── model_runner.py
│   ├── analysis/
│   │   ├── learning_rates.py
│   │   └── plot_learning_rates.py
│   ├── response_models/
│   │   └── continuous_prediction.py
│   └── utils/
│       └── paths.py
│
├── scripts/
│   ├── run_models.py
│   ├── run_local_only_model.py
│   ├── run_global_only_model.py
│   ├── run_shared_volatility_model.py
│   ├── analyze_learning_rates.py
│   ├── simulate_continuous_predictions.py
│   └── plotting / parameter-table utilities
│
├── experiments/
│   ├── model_recovery/
│   └── parameter_recovery/
│       └── smoke_tests/
│
├── tests/
│   ├── test_environment.py
│   ├── test_learning_rates.py
│   └── test_model_runner.py
│
├── archive/
│   ├── changepoint_pipeline/
│   └── legacy_scripts/
│
└── outputs/
    └── generated locally; not tracked by Git
```

`config/` contains the current environment definitions.

`src/` contains reusable implementation code.

`scripts/` contains the normal command-line entry points for running the current pipeline.

`experiments/` contains parameter-recovery, model-recovery, and identifiability experiments. These are not required for a standard environment -> model -> learning-rate run.

`tests/` contains automated tests for the current environment design, model-input preparation, learning-rate equations, volatility effects, and CTI logic.

`archive/` contains superseded code retained for provenance. The changepoint environment under this directory is not part of the current Gaussian-random-walk pipeline.

`outputs/` contains generated data, figures, model trajectories, synthetic responses, and recovery results. It is intentionally excluded from Git.

---

## 6. Python environment

The reference development environment uses Python 3.12.

Create and activate a virtual environment from the repository root:

```bash
python3.12 -m venv .myenv
source .myenv/bin/activate
python -m pip install --upgrade pip
```

Install the main project dependencies:

```bash
python -m pip install -r requirements.txt
```

For development and testing:

```bash
python -m pip install -r requirements-dev.txt
```

`requirements.txt` contains the direct project dependencies. `requirements-lock.txt` is a complete frozen snapshot of a known-working environment and can be used when exact environment reproduction is required:

```bash
python -m pip install -r requirements-lock.txt
```

Important current package versions include:

```text
pyhgf 0.2.12
pymc 5.28.5
pytensor 2.38.3
jax 0.4.31
jaxlib 0.4.31
numpy 2.4.6
pandas 3.0.3
arviz 0.23.4
```

---

## 7. Quick start

All commands below assume the repository root is the current working directory and the virtual environment is active.

### Step 1: generate an environment

Generate the default `local_A` environment:

```bash
python src/environments/two_stream_hgf_random_walk.py     --config config/hgf_random_walk_local_A.yaml     --output-dir outputs/hgf_walk_outputs/local_A
```

This creates:

```text
outputs/hgf_walk_outputs/local_A/
├── environment_long.csv
├── environment_wide.csv
├── environment_summary.csv
└── environment_metadata.json
```

### Step 2: run the three HGF models

```bash
python scripts/run_models.py --environment-name local_A
```

By default this runs:

```text
local_only
global_only
shared_volatility
```

and writes them under:

```text
outputs/hgf_walk_outputs/local_A/model_runs/
```

### Step 3: analyze learning rates

```bash
python scripts/analyze_learning_rates.py     --model-runs-dir outputs/hgf_walk_outputs/local_A/model_runs     --output-dir outputs/hgf_walk_outputs/local_A/learning_rates     --figure-dir outputs/hgf_walk_outputs/local_A/figures/learning_rates
```

The main CSV outputs are:

```text
trialwise_learning_rates_raw_including_initial_trial.csv
trialwise_learning_rates.csv
block_learning_rate_summary.csv
volatility_effect_summary.csv
```

### Step 4: simulate continuous responses

For example, generate responses from the hybrid/shared-volatility model:

```bash
python scripts/simulate_continuous_predictions.py     --model-runs-dir outputs/hgf_walk_outputs/local_A/model_runs     --model-name shared_volatility     --output-dir outputs/hgf_walk_outputs/local_A/synthetic_responses/shared_volatility     --response-sd 0.10
```

The main outputs are:

```text
synthetic_continuous_predictions.csv
response_function_inputs.npy
response_model_metadata.json
```

---

## 8. Running the other environments

### Local-B

```bash
python src/environments/two_stream_hgf_random_walk.py     --config config/hgf_random_walk_local_B.yaml     --output-dir outputs/hgf_walk_outputs/local_B

python scripts/run_models.py --environment-name local_B

python scripts/analyze_learning_rates.py     --model-runs-dir outputs/hgf_walk_outputs/local_B/model_runs     --output-dir outputs/hgf_walk_outputs/local_B/learning_rates     --figure-dir outputs/hgf_walk_outputs/local_B/figures/learning_rates
```

### Global

```bash
python src/environments/two_stream_hgf_random_walk.py     --config config/hgf_random_walk_global.yaml     --output-dir outputs/hgf_walk_outputs/global

python scripts/run_models.py --environment-name global

python scripts/analyze_learning_rates.py     --model-runs-dir outputs/hgf_walk_outputs/global/model_runs     --output-dir outputs/hgf_walk_outputs/global/learning_rates     --figure-dir outputs/hgf_walk_outputs/global/figures/learning_rates
```

---

## 9. Running a subset of models

Run only selected models:

```bash
python scripts/run_models.py     --environment-name local_A     --models local_only global_only
```

The dedicated model runner scripts remain available:

```text
scripts/run_local_only_model.py
scripts/run_global_only_model.py
scripts/run_shared_volatility_model.py
```

The shared-volatility runner exposes additional hybrid-specific parameter overrides and is useful when debugging or exploring that architecture.

---

## 10. Output layout

```text
outputs/hgf_walk_outputs/<environment_name>/
├── environment_long.csv
├── environment_wide.csv
├── environment_summary.csv
├── environment_metadata.json
├── model_runs/
│   ├── model_input_data.npy
│   ├── model_input_data.csv
│   ├── model_labels.csv
│   ├── model_input_metadata.json
│   ├── local_only/
│   ├── global_only/
│   └── shared_volatility/
├── learning_rates/
├── synthetic_responses/
└── figures/
```

Generated outputs are intentionally excluded from Git. Large Bayesian posterior files such as NetCDF (`*.nc`) should also not be committed.

---

## 11. Recovery experiments

Recovery work is separated from the normal pipeline.

### Model recovery

```text
experiments/model_recovery/
```

contains:

```text
fixed_parameter_model_recovery.py
bayesian_model_recovery.py
bayesian_repeated_model_recovery.py
```

Fixed-parameter model recovery is a diagnostic. Fair model recovery requires fitting each candidate model to the generated responses before comparing models.

Bayesian model comparison uses PyMC-based inference and PSIS-LOO through ArviZ.

### Parameter-recovery smoke tests

```text
experiments/parameter_recovery/smoke_tests/
```

contains the sequence of experiments used to test recoverability of different parameter combinations.

These scripts are retained because they document the development and identifiability history of the modelling pipeline. They are not required to run the standard simulation pipeline.

---

## 12. Tests

Run all tests from the repository root:

```bash
python -m pytest -q
```

The current tests protect:

- environment block structure and regime schedules;
- fixed-seed reproducibility;
- the distinction between shared variance schedules and independent innovations;
- the requirement that the generative environment uses `lambda_x = 1.0`;
- A/B ordering in model inputs;
- missing/invalid model-input handling;
- the HGF precision-ratio learning-rate equations;
- block-wise volatility-effect calculations;
- environment classification;
- local-A and local-B CTI definitions;
- the absence of CTI in the global condition.

Tests should be run before committing scientific or structural changes to the pipeline.

---

## 13. Reproducibility notes

The random-walk environment configurations currently use a fixed seed of `123`.

For two-stream generation, stream A and stream B use separate random-number generators. This preserves reproducibility while ensuring the two streams do not share identical latent or observation innovations.

Synthetic response generation uses a separate response-noise seed, currently `1234` by default.

The repository deliberately separates `requirements.txt` for direct project dependencies from `requirements-lock.txt` for the complete frozen environment used during development.

---

## 14. Current project status and limitations

This repository is an active modelling project rather than a finalized analysis package.

The current engineering pipeline is functional and tested for:

```text
environment generation
        ->
HGF model execution
        ->
learning-rate / CTI analysis
        ->
synthetic response generation
        ->
recovery experiments
```

Several scientific choices are still under development:

- the current model parameters are not yet the final paper-derived parameterization;
- the hybrid/shared-volatility parameters are provisional;
- the current environment set contains `local_A`, `local_B`, and `global`; an additional hybrid-like generative environment may be added later;
- block-average CTI is a useful summary but can miss delayed or transient transfer effects, so trial-wise learning-rate trajectories remain important;
- recovery experiments are still being refined, and fixed-parameter model recovery should not be treated as the final model-comparison procedure;
- no human behavioral dataset is part of the current repository.

Scientific changes should preserve a distinction between tested pipeline infrastructure and still-evolving modelling assumptions.

---

## 15. Recommended workflow for new development

When changing the project:

1. make the scientific or engineering change in a dedicated Git branch;
2. run `python -m pytest -q`;
3. run a small end-to-end simulation;
4. inspect the generated trajectories and learning-rate outputs;
5. update configuration or documentation when defaults change;
6. commit generated scientific summaries only when there is a clear reason to version them.

Avoid committing virtual environments, routine generated outputs, large posterior files, cache directories, or machine-specific files.

---

## License and citation

A license and formal repository citation have not yet been added. These should be defined before public archival or publication if the repository is intended for reuse outside the immediate project.
