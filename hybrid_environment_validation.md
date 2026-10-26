# Validation of the delivered Hybrid generator

Validation date: 2026-10-05.

## Scope

A temporary repository was assembled using the four files uploaded in this
conversation: the existing Gaussian random-walk generator, Local-A YAML,
model_runner.py, and paths.py. The new module/config/tests and minimal paths
update were installed there. The complete original repository and inference
model modules were not supplied.

## Results actually checked

- `python -m pytest -q tests/test_hybrid_environment.py`: **19 passed**.
- Ran the command-line generator with seed **123**, then every environment seed
  **0 through 99**, without seed substitution.
- The default long dataset has **2000 rows**; the wide dataset has **1000 rows**.
- The supplied loader and input conversion accepted the saved CSV and returned a
  finite **(1000,2)** observation matrix in A/B order.
- Regenerated Local-A with its unchanged config after the paths update: its
  serialized CSV **matched exactly** the baseline captured before the update.
- Original generator, original model_runner.py and Local-A configuration were
  **unchanged byte-for-byte**. The only functional path change adds Hybrid.
- Resolved YAML reload, repeatability, Eq. 65 conversion, pulse timing, the
  within-trial equations, independent noise sources, output protection, and
  numerical-failure recording were tested.
- A deliberately overflowing configuration was rejected and its test seeds were
  recorded as failures; no clipping or replacement seed was applied.

These are the tests for this addition, not the full original project test suite.
No pyHGF filtering, response simulation, fitting, model comparison, or recovery
was run.

## The 100-seed check is not an adequacy certificate

All **100 seeds** had finite outputs, giving 200 seed/stream diagnostic records
and **zero numerical failures**. Their ranges nevertheless varied widely:

| Diagnostic across the checked runs | Result |
|---|---:|
| Smallest trial-wise process variance | 4.3431936e-05 |
| Largest trial-wise process variance | 142930.25 |
| Median of the per-seed, per-stream maximum process variances | 68.469293 |
| 95th percentile of those maximum process variances | 2523.9218 |
| Largest absolute latent value | 5233.6721 |

This confirms the substantial wandering anticipated with lambda_v=1 and
omega_v=-3 over 1000 trials. Extreme-but-finite outcomes were not removed.
The supplied --check-seeds 100 command regenerates these summaries using seeds
0 through 99. The delivery archive contains code and documentation, not all
of the test-run datasets.

Before recovery, inspect whether this distribution of environments is appropriate
for the study. Zero numerical failures is not proof that the design, inference
defaults, or recoverability are satisfactory.

## Runtime

NumPy 2.3.5, pandas 2.2.3, PyYAML 6.0.3, pytest 9.0.2. Generated metadata records
the RNG roles and library versions. Generation needs no dependency beyond the
existing generator's scientific libraries.
