# Hybrid environment: install and use

This addition implements the design agreed in the modelling chat. It generates
observations and ground-truth environment states only. It does not run HGF
inference, simulate participant responses, or fit parameters.

## Installation

Merge the archive's files into your existing repository root:

- `src/environments/two_stream_hgf_hybrid.py` (new generator)
- `config/hgf_hybrid_weber.yaml` (new configuration)
- `tests/test_hybrid_environment.py` (new tests)
- `src/utils/paths.py` (small update)
- `docs/hybrid_environment.md` and `docs/hybrid_environment_validation.md`

The supplied paths.py is your uploaded file with only two additions: the Hybrid
output directory in its docstring and `"hybrid",` in ENVIRONMENT_NAMES. The
default environment remains local_A. If you have edited paths.py since uploading
it, add the tuple entry yourself instead of replacing your newer file.

Keep your existing `src/environments/two_stream_hgf_random_walk.py` and
`src/pipeline/model_runner.py`. The new generator imports existing block/config
and table-format helpers. It does not duplicate or replace your Local/Global
generator. No candidate-model file or parameter default is changed.

Generation uses the same NumPy, pandas and PyYAML dependencies as your supplied
generator. Tests additionally require pytest. No pyHGF installation is needed to
generate the environment or run these tests.

## Commands

Run from the repository root, in your existing Python environment.

Test the addition:

```sh
python -m pytest -q tests/test_hybrid_environment.py
```

Generate the default dataset (seed 123), and also check every environment seed
0 through 99, saving summaries rather than 100 additional full datasets:

```sh
python src/environments/two_stream_hgf_hybrid.py --check-seeds 100
```

The default configuration is `config/hgf_hybrid_weber.yaml`; the default output
is `outputs/hgf_walk_outputs/hybrid/`. Omit --check-seeds to generate just the
one dataset. An explicit version of the same command is:

```sh
python src/environments/two_stream_hgf_hybrid.py \
  --config config/hgf_hybrid_weber.yaml \
  --output-dir outputs/hgf_walk_outputs/hybrid \
  --seed 123 \
  --check-seeds 100 \
  --check-start-seed 0
```

Existing output files are never overwritten. To create another run, select a new
output directory:

```sh
python src/environments/two_stream_hgf_hybrid.py \
  --seed 124 \
  --output-dir outputs/hgf_walk_outputs/hybrid/seed_124
```

Do not run a second generation command into the same output directory. The
module refuses to write inside the canonical Local-A, Local-B and Global output
directories. It never deletes existing results.

## Output files

The familiar four files are preserved:

- `environment_long.csv`: two rows per trial, A then B.
- `environment_wide.csv`: one row per trial with explicit A/B columns.
- `environment_summary.csv`: descriptive block summaries for both streams.
- `environment_metadata.json`: parameters, source distinctions, Eq. 65 noise
  conversion, seed/RNG details, initialization, and numerical ranges.

Two additional files document the Hybrid manipulation:

- `environment_pulse_summary.csv`: expected and realized shared pulse areas,
  local-volatility changes and log-process-variance changes. These are
  environmental quantities, not learning rates or cross-transfer indices.
- `environment_config_resolved.yaml`: reloadable configuration, including the
  effective seed after any command-line override.

With --check-seeds, `seed_checks/` contains seed_diagnostics.csv,
block_summaries.csv and summary.json. All specified seeds are included in the
record. Numerical failures are recorded, not replaced with new seeds. Finite
runs with unexpected environmental patterns are not removed. Full additional
trajectories can be regenerated from the recorded seed and resolved config.

## Equations and timing

For zero-based trial_index t, the prescribed shared-state mean is +0.1 on
[200,220) and [600,620), -0.1 on [400,420) and [800,820), and zero otherwise.
Intervals are half-open: the start is included and the stop is excluded.

```
c[t] = m[t] + 0.005*z_c[t]
v_s[t] = v_s[t-1] + 2*c[t] + exp(-3/2)*z_v_s[t]
log_q_s[t] = -2 + 0.5*v_s[t]
q_s[t] = exp(log_q_s[t])
x_s[t] = x_s[t-1] + exp(log_q_s[t]/2)*z_x_s[t]
u_s[t] = x_s[t] + exp(1/2)*z_u_s[t]
```

All z terms are standard-normal innovations. There is ONE shared z_c series.
Local-volatility, value, and observation innovations use separate streams for A
and B: seven PCG64 generators derived from SeedSequence(random_seed). The
resulting local-volatility states are related through their shared drift; their
local innovations are independent.

The shared signal is independent Gaussian jitter around a prescribed mean, NOT
a random walk or autoregressive process. Each burst is a noisy 20-trial plateau.
There is no additional smoothing. +/-0.1 are means, not hard peak bounds. Noise
continues near zero outside the pulses. Realized noise is not demeaned to force
zero net pulse area.

The initial states are 6 and 4 for local volatility and 0 and 0 for values.
Your original generator updates a state before writing row 0. This addition
uses that same convention: these initial values are BEFORE the first saved
trial, or v_A[-1]=6, v_B[-1]=4, x_A[-1]=x_B[-1]=0 with zero-based row indexing.
Row 0 contains the first simulated local-volatility and value updates. In notes
that call the initial state v(0), the first simulated state is simply v(1).
Within each trial, current c is used to update current v, current v determines q,
and current q is used to update x before observation noise is added.

## Parameters and source attribution

The uploaded Weber et al. updated manuscript supplies the architecture in
Figure 6 (p. 23) and the process parameters in Table 1, Example 1, PROCESS column
(p. 27): alpha=2, kappa=0.5, value tonic volatility=-2, local-volatility tonic
volatility=-3, lambda=1, rho=0, and the initial values above. The generator's
validation protects the chosen lambda=1, rho=0 design; it does not modify the
inference models' autoconnections.

The 1000-trial schedule, pulse means +/-0.1, durations 20, and shared-signal noise
SD 0.005 are project choices. Weber pre-specified c and the paper's table does not
provide its exact vector. This is a Weber-inspired extension, not an exact
replication of Figure 6 or a claim that every parameter is paper-reported.

### Observation noise: user-approved Appendix Equation 65

```
observation_log_variance = epsilon = 1
observation_variance = exp(epsilon) = 2.718281828459045
observation_sd = exp(epsilon/2) = 1.6487212707001282
corresponding_input_precision = exp(-epsilon) = 0.36787944117144233
```

The Gaussian observation noise is therefore standard-normal noise multiplied
by exp(0.5), not by 1 or exp(1). The manuscript's Eq. 9 writes epsilon directly
in the variance position, whereas Appendix Eq. 65 explicitly uses exp(epsilon).
This addition follows the Appendix convention by agreement, not a verified
reconstruction of the original Figure 6 source code.

### Pulse sanity check

Each positive prescribed pulse has area 20*0.1=2, giving delta_v=4 and
delta_log_q=2: a variance multiplier exp(2), approximately 7.389, when random
innovations are omitted. The prescribed negative pulse cancels that shift.
Actual noisy pulse areas and local-volatility changes need not cancel.

The deterministic reference columns omit shared jitter and local innovations.
They are references for the intended drift, not evidence of a forced response.
In particular, they are NOT a claim that mean process variance equals the
exponential of mean local volatility.

## Compatibility and interpretation limits

Existing required columns remain: trial_index, trial, stream, block, regime,
latent_mean, observation, and changepoint. changepoint=False means no value
resets; pulse transitions are marked separately with is_transition and
pulse_name. The new truth columns include shared_mean, shared_state,
shared_noise, shared_volatility_drift, local_volatility,
local_volatility_innovation, log_process_variance, and deterministic references.
The wide file keeps shared quantities once and local quantities separately for
both streams.

Your supplied model_runner.load_environment_long and prepare_model_input were
tested directly and yield a finite (1000,2) array, A then B. Only observations
enter that input array. The unchanged make_labels_table drops the additional
Hybrid truth columns; they remain in environment files and can be joined by
trial_index during a future Hybrid-specific analysis.

S1/V1/S2/V2/S3 and regime describe the NOMINAL shared manipulation. V1 and V2
include upward transitions. S2 and S3 initially include downward transitions.
There is no q_low or q_high assignment, no guarantee of the intended ordering
in every seed, and no requirement that A stays more volatile than B. Actual
process variances must be inspected, not inferred from labels. The files mark
regime_is_nominal=True and environment_type="hybrid" explicitly.

Do not interpret the usual Local-A cross-transfer index here: both streams
receive the shared manipulation. The existing learning-rate-analysis pipeline
has NOT been validated for this new environment type.

**Data-format compatibility does not mean calibrated inference.** Your uploaded
runner defaults to input_precision=100 and does not automatically load this
new metadata. The corresponding precision for this environment's observations
is exp(-1), and its process-variance scale is also different. Candidate-model
settings must be reviewed before response simulation and recovery. This addition
does not change any model or response-model parameter.

Recovery from synthetic participant responses concerns the RESPONSE-GENERATING
PERCEPTUAL MODEL, conditional on the observation sequence, not automatically
the environment's process parameters. In particular, the environment's alpha=2
is not automatically the true parameter of a participant model; Weber's Process
and Model columns differ.

Returning c to near zero removes its mean drift, not the local random walks.
The local-noise contribution to each v has SD sqrt(1000*exp(-3)), approximately
7.056, after 1000 updates; shared jitter adds a smaller common contribution.
Thus variances can become large without numerical overflow.

The code does not clip q/v, reset states, normalize observations, rescale
parameters, select seeds, condition on behaviour, or retry failed seeds. It
stops on numerical underflow/overflow or nonfinite states. 'Finite' is NOT a
scientific-adequacy verdict. Inspect distributions and trajectories across the
predeclared seeds before launching recovery. Actual delivery checks are in
hybrid_environment_validation.md.
