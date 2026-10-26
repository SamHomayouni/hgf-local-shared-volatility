# Environment plotting update

## Install

Merge these files into the current repository, keeping their paths:

- **Replace** `scripts/plot_environment_hgf_random_walk.py`.
- **Add** `src/analysis/plot_environments.py`.
- **Add** `tests/test_environment_plotting.py`.
- **Add** this document as `docs/environment_plotting.md`.

The replacement script and the new plotting module are BOTH required. This is
one plotting command, not separate Local/Global/Hybrid plotting pipelines.
The script is now a thin command-line entry point, with reusable functions in
src/analysis, following the repository placement guide.

Do not replace or edit either environment generator, environment configurations,
paths.py, model_runner.py, or the inference/response models for this update.
No new environment generation is required. Existing CSV files are sufficient.
No package beyond the existing NumPy, pandas and Matplotlib is required for
plotting. Tests additionally use pytest.

## Run

From the repository root:

```sh
python -m pytest -q tests/test_environment_plotting.py
```

The user's existing command still works:

```sh
python scripts/plot_environment_hgf_random_walk.py --environment outputs/hgf_walk_outputs/hybrid/environment_long.csv --output-dir outputs/hgf_walk_outputs/hybrid/figures/environment/
```

Likewise for Local-A, Local-B, and Global. For example:

```sh
python scripts/plot_environment_hgf_random_walk.py --environment outputs/hgf_walk_outputs/local_A/environment_long.csv --output-dir outputs/hgf_walk_outputs/local_A/figures/environment/
python scripts/plot_environment_hgf_random_walk.py --environment outputs/hgf_walk_outputs/global/environment_long.csv --output-dir outputs/hgf_walk_outputs/global/figures/environment/
```

The output directory can now be omitted. It defaults to `figures/environment/`
inside the folder containing the input CSV. Thus specifying a Hybrid input
without --output-dir no longer sends figures to the Local-A output directory.
With no arguments at all, the existing default remains Local-A.

To write ONLY the new volatility plots, leaving the seven familiar PNGs alone:

```sh
python scripts/plot_environment_hgf_random_walk.py --environment outputs/hgf_walk_outputs/hybrid/environment_long.csv --volatility-only
```

No display/GUI is needed. The CLI uses Matplotlib's Agg file-output backend.

## Which quantities really exist in each environment?

The supplied `two_stream_hgf_random_walk.py` directly chooses Q_low or Q_high
from the block regime. It does not simulate local/shared HGF volatility-state
nodes. Its process variance Q(t) IS the environmental volatility schedule.
A Local/Global HGF learner's inferred volatility beliefs are different objects,
stored in model trajectories, and are not plotted by this environment command.

Consequently, the new plot for these baseline environments is **ln(Q(t))**:

- Local-A/Local-B: separate stream A and B log-process-variance curves.
- Global: one common curve when the saved Q_A and Q_B values match exactly.

This common curve is a shared prescribed schedule, not a newly simulated parent.
No omega or kappa from the Hybrid model is borrowed to reconstruct a fictional
Local/Global hidden state. ln(Q) can be negative while Q remains positive.

The Hybrid CSV does contain actual `local_volatility` and `shared_state`
columns, so these are plotted directly. Hybrid ln(Q) and v are related by the
generator's coupling equation but they are not labelled as the same quantity.

## Output plots

The original seven filenames are preserved:

1. `hgf_random_walk_stream_A.png`
2. `hgf_random_walk_stream_B.png`
3. `hgf_random_walk_latent_means_both_streams.png`
4. `hgf_random_walk_process_variance.png`
5. `hgf_random_walk_process_sd.png`
6. `hgf_random_walk_latent_innovations.png`
7. `hgf_random_walk_observation_noise.png`

Their numeric curve data are unchanged. Titles and legends now identify the
environment, use readable labels, and keep legends outside the plotting area.

Every environment also gets:

- `hgf_environment_log_process_variance.png`: the natural log of saved Q.
  This makes volatility changes visible without inventing hidden states.

The delivered Hybrid generator supplies the columns for five further plots:

- `hgf_environment_shared_state.png`: one realized x_c trajectory and its
  prescribed mean. The duplicated A/B copies are checked for agreement first.
- `hgf_environment_local_volatility.png`: actual local-volatility states A/B.
- `hgf_environment_local_volatility_reference.png`: the same actual states
  against their saved no-noise reference trajectories. The reference is not a
  fitted curve or a desired behavioural result.
- `hgf_environment_volatility_random_component.png`: actual v minus its saved
  no-noise reference. In this generator it contains accumulated LOCAL innovations
  AND the smaller accumulated SHARED-signal jitter. It is not purely local noise.
- `hgf_environment_local_volatility_innovations.png`: local-volatility random
  steps, NOT the value-state innovations in the original innovation plot.

Default totals with the supplied formats: **8 PNGs** for Local-A, Local-B or
Global; **13 PNGs** for Hybrid. With --volatility-only: 1 or 6 PNGs respectively.
Additional state plots are conditional on actual saved columns; missing states
are not fabricated. If a deterministic reference is absent, plots requiring that
reference are skipped rather than reconstructing one with assumed parameters.

## Labels and time intervals

For Hybrid, shaded V blocks are labelled **Nominal volatile block (A + B)**.
They represent the planned manipulation, not a threshold applied to realized Q.
Hatched areas mark `is_transition` intervals, including the downward pulses at
the beginning of nominal S blocks. These intervals come from the CSV; the
plotter does not hard-code 200, 400, 600, 800, a pulse size, or a pulse duration.

The x axis is the saved zero-based `trial_index`. No shift, smoothing, clipping,
noise removal, resetting, or normalization is applied to any actual trajectory.

## Safety and provenance

Inputs must be long-format, with one A and one B row per trial. Duplicate rows,
missing stream/trial rows, non-finite required data, non-positive Q, conflicting
shared copies, inconsistent process SDs, and inconsistent saved ln(Q) values
produce explicit errors instead of being silently repaired.

PNG files with the same names are regenerated and overwritten, as in the old
plotter. Environment CSV/YAML/metadata files are NOT modified. Unrelated existing
figures are not deleted. `environment_plot_manifest.json` records exactly which
plots were written in the current invocation and the input CSV's SHA-256 hash.
In --volatility-only mode the manifest lists only that invocation's new plots.

The command does not run filtering, responses, or recovery; it does not assess
recoverability or select environment seeds.

## Validation performed for this delivery

A temporary repository was assembled from the user's uploaded generator,
Local-A configuration, paths utility and model runner, plus the previously
delivered Hybrid generator/configuration/tests. The full original repository
was not available.

- **27 new plotting tests passed**.
- The **19 existing Hybrid-generator tests also passed**, for 46 combined.
- The CLI rendered complete 1000-trial Local-A and Hybrid test datasets using
  the supplied generators. Local-B and Global tests reused the old generator
  with the documented mirrored/shared regime schedules in temporary configs;
  they were not reads of the user's unprovided Local-B/Global CSV files.
- Figure totals were 8, 8, 8 and 13 for Local-A, Local-B, Global and Hybrid.
- The input CSV bytes were unchanged by plotting.
- Numeric x/y arrays in all seven legacy plots matched the original uploaded
  plotting functions exactly for all four test conditions. Image appearance
  changed intentionally because of new titles, legends and nominal shading.
- New figures were visually inspected for correct labels and readable layout.

These are plotting/regression checks, not a recovery result or a run of the
full original repository test suite.
