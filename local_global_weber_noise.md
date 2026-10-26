# Local/Global update: keep Q, use Weber observation noise

## Scope

The user explicitly chose to retain q_low=0.0004 and q_high=0.01.
The only parsed configuration field changed in these three variants is
`environment.latent_dynamics.observation_noise_sd`, from 0.10 to
1.6487212707001282. Seed 123, 1000 trials, all block schedules, simulation
names, lambda_x=1.0, rho_x=0.0, and both initial latent values=0 are unchanged.
There are no Python replacements or new Python entry points in this package.
No Hybrid environment, inference model, or response-model parameter is changed.

The `_weber_noise` filenames are deliberate: these are existing blockwise-Q
processes with Weber-based observation noise, NOT full replications of Weber's
hierarchical generative process. These generators do not contain volatility
nodes, so alpha, kappa, local-volatility omega, and local-volatility initial
states are not additional settings to add to their YAMLs. Q is set directly.

## Provenance

Source: the uploaded *Weber et al., The generalized Hierarchical Gaussian
Filter, updated manuscript*. Table 1, Example 1 PROCESS column (p. 27) gives
epsilon_u=1, initial value states=0, all drifts=0, and autoconnections=1.
Appendix Eq. 65 (p. 41) defines observation precision as 1/exp(epsilon_u).
Under the user-approved Appendix convention:

    observation log variance = 1
    observation variance = exp(1) = 2.718281828459045
    observation SD = exp(0.5) = 1.6487212707001282
    matching input precision = exp(-1) = 0.36787944117144233

Eq. 9 elsewhere in the manuscript places epsilon directly in the variance
position. We retain the agreed Appendix convention; this is not a verification
of the authors' original simulation code.

The retained Q levels, block schedule, trial count, and seed are project
choices, not parameters extracted from that table.

## Install

Merge the package's `config/`, `outputs/`, and `docs/` directories into your
repository root. Do not replace any Python files or remove the original YAMLs.
The archive contains the regenerated seed-123 data as well as the configs.

New configs:

    config/hgf_random_walk_local_A_weber_noise.yaml
    config/hgf_random_walk_local_B_weber_noise.yaml
    config/hgf_random_walk_global_weber_noise.yaml

Each generated output is in a new variant subfolder:

    outputs/hgf_walk_outputs/local_A/weber_noise/
    outputs/hgf_walk_outputs/local_B/weber_noise/
    outputs/hgf_walk_outputs/global/weber_noise/

Each contains the original generator's four files:
`environment_long.csv`, `environment_wide.csv`, `environment_summary.csv`,
and `environment_metadata.json`. Existing parent-folder test data remain intact.
Do not overwrite any newer files if a `weber_noise` folder already exists locally.

## Regenerate locally

The data in the archive are already generated. The following commands reproduce
the variants from the repository root using your existing generator:

```sh
python src/environments/two_stream_hgf_random_walk.py --config config/hgf_random_walk_local_A_weber_noise.yaml --output-dir outputs/hgf_walk_outputs/local_A/weber_noise
python src/environments/two_stream_hgf_random_walk.py --config config/hgf_random_walk_local_B_weber_noise.yaml --output-dir outputs/hgf_walk_outputs/local_B/weber_noise
python src/environments/two_stream_hgf_random_walk.py --config config/hgf_random_walk_global_weber_noise.yaml --output-dir outputs/hgf_walk_outputs/global/weber_noise
```

The generator overwrites same-named files in the selected output directory.
These commands leave the old parent-folder outputs alone, but overwrite the
new variant data if those data have already been extracted or generated.
Choose another variant folder for a new run that must be preserved separately.

## Plot

Use the existing plotting command on the NEW data path. For example:

```sh
python scripts/plot_environment_hgf_random_walk.py --environment outputs/hgf_walk_outputs/local_A/weber_noise/environment_long.csv --output-dir outputs/hgf_walk_outputs/local_A/weber_noise/figures/environment
```

Substitute `local_B` or `global` for the other conditions. The plotting script
and environment generators do not require modification for this update.
No figures are bundled, and no new plotting behavior was tested in this update.

## What changes in the data

For the supplied generator, retaining the seed and all latent settings preserves
the latent values, innovations, regimes, and Q trajectories exactly. The same
standard-normal observation-noise draws are multiplied by a new SD. Their
amplitudes are larger by exp(0.5)/0.10, approximately 16.4872. Observations change;
latent states do not. Observation variance is larger by approximately 271.828.

This changes the process-to-observation-noise ratio. It does not establish that
recovery will succeed or fail. The unchanged Q values are the user's choice and
have not been adjusted to compensate for the increased noise. The Hybrid still
has a different process-variance scale and process, despite matching observation
noise. This update does not make the environments identical in other respects.

## Later inference

Existing convenience-command defaults still point to the old parent-folder
CSV. To use these variants later, pass their explicit environment paths and a
new model-output folder rather than assuming the default changed. No new
environment-name registry entries are needed for generation or plotting with
explicit paths.

The supplied model runner still defaults to input_precision=100. For a learner
whose observation-noise assumption is meant to match these stimuli, the correct
value is exp(-1), approximately 0.36787944117144233. Metadata records that value,
but the runner does not automatically load it. Do not mistake the environment
noise update for a change to the perceptual-model defaults. Other perceptual
parameters, priors, and response noise remain separate decisions.

## Validation actually performed

Using exact copies of the supplied generator, paths module, and model-input
loader in a temporary repository, all three variants passed the following:

- Original and new YAML dictionaries differ only in observation_noise_sd.
- Both Q values, all regimes/blocks, the seed, and initial values are unchanged.
- Repeated simulations with the same seed produce exactly identical tables.
- All latent quantities and labels match the old simulation exactly in memory.
- Observation-noise samples equal the old samples times the SD ratio, to
  floating-point tolerance; observations equal latent value plus new noise.
- The actual generator command succeeds and saves its four standard outputs.
- Saved data contain 2000 long rows and 1000 wide rows, all numerical data finite.
- The supplied model-input loader returns a finite (1000, 2) matrix in A/B order.
- Metadata reports retained Q values, the new observation SD, and exp(-1)
  as corresponding input precision.
- Global Q schedules remain identical between streams, with distinct innovations.
- Hash checks confirm no uploaded YAML or Python source was modified.

These are focused configuration/regeneration checks, not a run of the full
project test suite. No HGF filtering, response simulation, fitting, parameter
recovery, or model recovery was performed. No seed selection was used.

Validation runtime: Python 3.13.5, NumPy 2.3.5, pandas 2.2.3, PyYAML 6.0.3.
