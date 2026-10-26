"""Generate the two-stream, Weber-inspired Hybrid environment (not HGF beliefs).

For stream s and zero-based trial t, with independent standard-normal innovations:
    c[t] = prescribed_mean[t] + shared_noise_sd * z_c[t]
    v_s[t] = v_s[t-1] + alpha*c[t] + exp(omega_v/2)*z_v_s[t]
    log_q_s[t] = omega_x + kappa*v_s[t]
    x_s[t] = x_s[t-1] + exp(log_q_s[t]/2)*z_x_s[t]
    u_s[t] = x_s[t] + exp(observation_log_variance/2)*z_u_s[t]

The same realized c[t] drives both local volatility states. Initial states are
PRE-TRIAL states, before the update at trial_index=0, as in the existing generator.
There is no clipping, resetting, noise centering, or seed selection.

Sources: uploaded Weber et al. updated manuscript, Figure 6 / Table 1 (Example 1,
Process column), pp. 23 and 27; observation log-variance convention from Appendix
Eq. 65, p. 41. The 1000-trial pulse schedule and c jitter are project choices, not
an exact reconstruction of the authors' simulation. Block/regime labels describe
the intended shared manipulation; actual process variances remain stochastic.

Run from the repository root:
    python src/environments/two_stream_hgf_hybrid.py --check-seeds 100
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = next(
    (p for p in Path(__file__).resolve().parents
     if (p / "src").is_dir() and (p / "config").is_dir()), None
)
if PROJECT_ROOT is None:
    raise RuntimeError("Place this module in your repository's src/environments/ directory.")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.environments.two_stream_hgf_random_walk import (
    Block, get_block_and_regime_for_index, load_yaml_config,
    make_wide_environment, parse_blocks, summarize_environment,
)
from src.utils.paths import CONFIG_DIR, get_environment_output_dir

ENVIRONMENT_NAME = "hybrid"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "hgf_hybrid_weber.yaml"
RNG_ROLES = (
    "shared_state", "local_volatility_A", "local_volatility_B",
    "value_A", "value_B", "observation_A", "observation_B",
)
SHARED_COLUMNS = (
    "shared_mean", "shared_state", "shared_noise", "shared_volatility_drift",
    "pulse_name", "is_transition", "regime_is_nominal", "environment_type",
)
STREAM_COLUMNS = (
    "local_volatility", "local_volatility_innovation",
    "deterministic_local_volatility", "log_process_variance",
    "deterministic_log_process_variance",
)
OUTPUT_NAMES = (
    "environment_long.csv", "environment_wide.csv", "environment_summary.csv",
    "environment_metadata.json", "environment_pulse_summary.csv",
    "environment_config_resolved.yaml",
)


@dataclass(frozen=True)
class Pulse:
    """A constant-mean pulse on the half-open interval [start_index, stop_index)."""
    name: str
    start_index: int
    stop_index: int
    mean: float


@dataclass(frozen=True)
class SharedSignal:
    baseline_mean: float
    noise_sd: float
    pulses: tuple[Pulse, ...]


@dataclass(frozen=True)
class Dynamics:
    """Common dynamics for A and B; only their initial volatilities differ."""
    lambda_x: float
    rho_x: float
    value_tonic_volatility: float
    volatility_coupling: float
    lambda_v: float
    rho_v: float
    local_volatility_tonic_volatility: float
    shared_value_coupling: float
    observation_log_variance: float


@dataclass(frozen=True)
class InitialState:
    initial_mean: float
    initial_volatility: float


@dataclass(frozen=True)
class HybridConfig:
    name: str
    random_seed: int
    n_trials: int
    blocks: tuple[Block, ...]
    shared_signal: SharedSignal
    dynamics: Dynamics
    stream_a: InitialState
    stream_b: InitialState


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


def parse_environment_config(config_path: str | Path) -> HybridConfig:
    """Read the project YAML format without changing the old environment parser."""
    raw = load_yaml_config(config_path)
    sim = raw["simulation"]
    env = raw["environment"]
    signal = env["shared_signal"]
    x = env["latent_dynamics"]
    v = env["local_volatility_dynamics"]
    pulses = tuple(
        Pulse(str(p["name"]), _integer(p["start_index"], "pulse start_index"),
              _integer(p["stop_index"], "pulse stop_index", 1), float(p["mean"]))
        for p in signal["pulses"]
    )
    # The existing block parser uses int(); first reject fractional indices
    # instead of silently truncating a YAML value.
    for block in raw["blocks"]:
        _integer(block["start_index"], "block start_index")
        _integer(block["stop_index"], "block stop_index", 1)
    config = HybridConfig(
        name=str(sim["name"]),
        random_seed=_integer(sim["random_seed"], "random_seed"),
        n_trials=_integer(sim["n_trials"], "n_trials", 1),
        blocks=tuple(parse_blocks(raw)),
        shared_signal=SharedSignal(float(signal["baseline_mean"]),
                                   float(signal["noise_sd"]), pulses),
        dynamics=Dynamics(
            lambda_x=float(x["lambda_x"]), rho_x=float(x["rho_x"]),
            value_tonic_volatility=float(x["value_tonic_volatility"]),
            volatility_coupling=float(x["volatility_coupling"]),
            lambda_v=float(v["lambda_v"]), rho_v=float(v["rho_v"]),
            local_volatility_tonic_volatility=float(v["tonic_volatility"]),
            shared_value_coupling=float(v["shared_value_coupling"]),
            observation_log_variance=float(env["observation"]["log_variance"]),
        ),
        stream_a=InitialState(**{k: float(env["stream_a"][k]) for k in
                                 ("initial_mean", "initial_volatility")}),
        stream_b=InitialState(**{k: float(env["stream_b"][k]) for k in
                                 ("initial_mean", "initial_volatility")}),
    )
    validate_environment_config(config)
    return config


def validate_environment_config(config: HybridConfig) -> None:
    _integer(config.n_trials, "n_trials", 1)
    _integer(config.random_seed, "random_seed")
    if not config.name.strip():
        raise ValueError("simulation.name must not be empty.")
    d, shared = config.dynamics, config.shared_signal
    numbers = list(asdict(d).values()) + [shared.baseline_mean, shared.noise_sd]
    numbers += list(asdict(config.stream_a).values()) + list(asdict(config.stream_b).values())
    if not np.isfinite(numbers).all() or shared.noise_sd < 0:
        raise ValueError("All parameters must be finite; shared noise_sd must be >= 0.")
    if (d.lambda_x, d.lambda_v, d.rho_x, d.rho_v) != (1.0, 1.0, 0.0, 0.0):
        raise ValueError("This Weber random-walk design requires lambda_x=lambda_v=1 and rho_x=rho_v=0.")
    if d.volatility_coupling < 0 or d.shared_value_coupling < 0:
        raise ValueError("Couplings must be non-negative in this design.")
    with np.errstate(over="raise", under="raise", invalid="raise"):
        try:
            np.exp([d.local_volatility_tonic_volatility,
                    d.observation_log_variance, -d.observation_log_variance])
        except FloatingPointError as exc:
            raise ValueError("Configured variance/precision is not safely representable in float64.") from exc
    end, names = 0, set()
    for block in config.blocks:
        _integer(block.start_index, "block start_index")
        _integer(block.stop_index, "block stop_index", 1)
        if block.start_index != end or block.stop_index <= block.start_index:
            raise ValueError("Blocks must be contiguous, ordered, nonempty, and start at 0.")
        if not block.name.strip() or block.name in names:
            raise ValueError("Block names must be nonempty and unique.")
        if block.stream_a_regime not in {"stable", "volatile"}:
            raise ValueError("Nominal block regimes must be stable or volatile.")
        if block.stream_a_regime != block.stream_b_regime:
            raise ValueError("This shared-drive design requires the same nominal regime labels for A and B.")
        end = block.stop_index
        names.add(block.name)
    if end != config.n_trials:
        raise ValueError("Blocks must cover exactly n_trials.")
    end, names = 0, set()
    for pulse in shared.pulses:
        _integer(pulse.start_index, "pulse start_index")
        _integer(pulse.stop_index, "pulse stop_index", 1)
        if not (end <= pulse.start_index < pulse.stop_index <= config.n_trials):
            raise ValueError("Pulses must be ordered, non-overlapping, and within n_trials.")
        if not pulse.name.strip() or pulse.name in names or pulse.name == "none":
            raise ValueError("Pulse names must be unique, nonempty, and different from 'none'.")
        if not np.isfinite(pulse.mean):
            raise ValueError("Pulse means must be finite.")
        end = pulse.stop_index
        names.add(pulse.name)


def build_shared_signal(config: HybridConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Independent Gaussian jitter about the prescribed mean, NOT a random walk."""
    n = config.n_trials
    mean = np.full(n, config.shared_signal.baseline_mean, dtype=float)
    pulse_name = np.full(n, "none", dtype=object)
    is_transition = np.zeros(n, dtype=bool)
    for pulse in config.shared_signal.pulses:
        sl = slice(pulse.start_index, pulse.stop_index)
        mean[sl] = pulse.mean
        pulse_name[sl] = pulse.name
        is_transition[sl] = True
    noise = config.shared_signal.noise_sd * rng.standard_normal(n)
    return pd.DataFrame({
        "shared_mean": mean, "shared_state": mean + noise, "shared_noise": noise,
        "shared_volatility_drift": config.dynamics.shared_value_coupling * (mean + noise),
        "pulse_name": pulse_name, "is_transition": is_transition,
        "regime_is_nominal": True, "environment_type": ENVIRONMENT_NAME,
    })


def simulate_two_stream_environment(config: HybridConfig) -> pd.DataFrame:
    """Generate a long table compatible with model_runner.prepare_model_input."""
    validate_environment_config(config)
    children = np.random.SeedSequence(config.random_seed).spawn(len(RNG_ROLES))
    rng = {role: np.random.Generator(np.random.PCG64(child))
           for role, child in zip(RNG_ROLES, children)}
    shared = build_shared_signal(config, rng["shared_state"])
    d, n = config.dynamics, config.n_trials
    local_sd = float(np.exp(d.local_volatility_tonic_volatility / 2))
    observation_sd = float(np.exp(d.observation_log_variance / 2))
    tables = []
    for stream, initial in (("A", config.stream_a), ("B", config.stream_b)):
        local_innovation = local_sd * rng[f"local_volatility_{stream}"].standard_normal(n)
        z_value = rng[f"value_{stream}"].standard_normal(n)
        observation_noise = observation_sd * rng[f"observation_{stream}"].standard_normal(n)
        volatility = np.empty(n)
        reference_volatility = np.empty(n)
        log_q, q, step_sd = np.empty(n), np.empty(n), np.empty(n)
        value, value_innovation = np.empty(n), np.empty(n)
        previous_v, previous_x = initial.initial_volatility, initial.initial_mean
        previous_reference = initial.initial_volatility
        shared_values = shared["shared_state"].to_numpy()
        shared_means = shared["shared_mean"].to_numpy()
        for t in range(n):
            try:
                with np.errstate(over="raise", under="raise", invalid="raise"):
                    volatility[t] = previous_v + d.shared_value_coupling * shared_values[t] + local_innovation[t]
                    reference_volatility[t] = previous_reference + d.shared_value_coupling * shared_means[t]
                    log_q[t] = d.value_tonic_volatility + d.volatility_coupling * volatility[t]
                    q[t] = np.exp(log_q[t])
                    step_sd[t] = np.exp(log_q[t] / 2)
                    value_innovation[t] = step_sd[t] * z_value[t]
                    value[t] = previous_x + value_innovation[t]
                if not np.isfinite([volatility[t], reference_volatility[t], log_q[t], q[t],
                                    step_sd[t], value_innovation[t], value[t]]).all():
                    raise FloatingPointError("A generated quantity was not finite.")
            except FloatingPointError as exc:
                raise FloatingPointError(
                    f"Seed {config.random_seed}, stream {stream}, trial_index {t}: "
                    "a value/variance was not safely representable. No clipping or retry was applied."
                ) from exc
            previous_v, previous_x = volatility[t], value[t]
            previous_reference = reference_volatility[t]
        block_labels = [get_block_and_regime_for_index(t, config.blocks, stream) for t in range(n)]
        observation = value + observation_noise
        if not np.isfinite(observation).all():
            raise FloatingPointError(f"Non-finite observations, seed={config.random_seed}, stream={stream}.")
        table = pd.DataFrame({
            "trial_index": np.arange(n), "trial": np.arange(1, n + 1),
            "stream": stream, "block": [b.name for b, _ in block_labels],
            "regime": [regime for _, regime in block_labels],
            "latent_mean": value, "observation": observation,
            "changepoint": False,  # No value resets; pulse transitions are marked separately.
            "process_variance": q, "process_sd": step_sd,
            "latent_innovation": value_innovation, "observation_noise": observation_noise,
            "is_a_volatile_block": [b.stream_a_regime == "volatile" for b, _ in block_labels],
            "local_volatility": volatility,
            "local_volatility_innovation": local_innovation,
            "deterministic_local_volatility": reference_volatility,
            "log_process_variance": log_q,
            "deterministic_log_process_variance": (
                d.value_tonic_volatility + d.volatility_coupling * reference_volatility),
        })
        tables.append(pd.concat([table, shared], axis=1))
    return pd.concat(tables, ignore_index=True).sort_values(
        ["trial_index", "stream"]).reset_index(drop=True)


def make_wide_hybrid_environment(df: pd.DataFrame) -> pd.DataFrame:
    """Reuse the old wide-table schema, then retain the extra Hybrid truth columns."""
    wide = make_wide_environment(df).set_index("trial_index")
    shared = df.loc[df["stream"] == "A"].set_index("trial_index")[list(SHARED_COLUMNS)]
    parts = [wide, shared]
    for column in STREAM_COLUMNS:
        part = df.pivot(index="trial_index", columns="stream", values=column)[["A", "B"]]
        part.columns = [f"{column}_A", f"{column}_B"]
        parts.append(part)
    return pd.concat(parts, axis=1).sort_index().reset_index()


def summarize_hybrid_environment(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive block summaries; they are not pass/fail manipulation criteria."""
    keys = ["stream", "block", "regime"]
    extra = df.groupby(keys, as_index=False).agg(
        mean_local_volatility=("local_volatility", "mean"),
        min_local_volatility=("local_volatility", "min"),
        max_local_volatility=("local_volatility", "max"),
        mean_log_process_variance=("log_process_variance", "mean"),
        min_log_process_variance=("log_process_variance", "min"),
        max_log_process_variance=("log_process_variance", "max"),
        median_process_variance=("process_variance", "median"),
        min_process_variance=("process_variance", "min"),
        max_process_variance=("process_variance", "max"),
        n_transition_trials=("is_transition", "sum"),
    )
    summary = summarize_environment(df).merge(extra, on=keys, validate="one_to_one")
    summary["regime_is_nominal"] = True
    return summary.sort_values(["stream", "first_index"]).reset_index(drop=True)


def summarize_pulses(df: pd.DataFrame, config: HybridConfig) -> pd.DataFrame:
    """Report intended shared effects separately from realized noisy effects."""
    shared = df[df["stream"] == "A"].set_index("trial_index")
    d, rows = config.dynamics, []
    for pulse in config.shared_signal.pulses:
        sl = slice(pulse.start_index, pulse.stop_index - 1)  # pandas .loc is inclusive.
        expected_area = float(shared.loc[sl, "shared_mean"].sum())
        actual_area = float(shared.loc[sl, "shared_state"].sum())
        for stream in ("A", "B"):
            local_sum = float(df.loc[
                (df["stream"] == stream) & df["trial_index"].between(
                    pulse.start_index, pulse.stop_index - 1),
                "local_volatility_innovation"].sum())
            rows.append({
                "pulse": pulse.name, "stream": stream,
                "start_index": pulse.start_index, "stop_index": pulse.stop_index,
                "n_trials": pulse.stop_index - pulse.start_index,
                "expected_shared_area": expected_area, "realized_shared_area": actual_area,
                "deterministic_delta_local_volatility": d.shared_value_coupling * expected_area,
                "realized_delta_local_volatility": d.shared_value_coupling * actual_area + local_sum,
                "deterministic_delta_log_q": d.volatility_coupling * d.shared_value_coupling * expected_area,
                "realized_delta_log_q": d.volatility_coupling * (d.shared_value_coupling * actual_area + local_sum),
            })
    return pd.DataFrame(rows, columns=[
        "pulse", "stream", "start_index", "stop_index", "n_trials",
        "expected_shared_area", "realized_shared_area",
        "deterministic_delta_local_volatility", "realized_delta_local_volatility",
        "deterministic_delta_log_q", "realized_delta_log_q",
    ])


def diagnostic_records(df: pd.DataFrame, config: HybridConfig) -> list[dict[str, Any]]:
    """Numeric ranges for every seed, including counter-to-schedule trajectories."""
    records = []
    for stream, part in df.groupby("stream", sort=True):
        records.append({
            "random_seed": config.random_seed, "stream": stream, "status": "finite",
            "error": "", "n_trials": len(part),
            "min_local_volatility": float(part.local_volatility.min()),
            "max_local_volatility": float(part.local_volatility.max()),
            "final_local_volatility": float(part.local_volatility.iloc[-1]),
            "min_log_process_variance": float(part.log_process_variance.min()),
            "max_log_process_variance": float(part.log_process_variance.max()),
            "min_process_variance": float(part.process_variance.min()),
            "max_process_variance": float(part.process_variance.max()),
            "max_abs_latent": float(part.latent_mean.abs().max()),
            "observation_noise_sd_empirical": float(part.observation_noise.std(ddof=1)) if len(part) > 1 else None,
            "shared_noise_sd_empirical": float(part.shared_noise.std(ddof=1)) if len(part) > 1 else None,
            "expected_total_shared_area": float(part.shared_mean.sum()),
            "realized_total_shared_area": float(part.shared_state.sum()),
        })
    return records


def _resolved_config(config: HybridConfig) -> dict[str, Any]:
    """Return a loadable YAML mapping containing the effective seed and values."""
    d = config.dynamics
    return {
        "simulation": {"name": config.name, "random_seed": config.random_seed, "n_trials": config.n_trials},
        "blocks": [asdict(b) for b in config.blocks],
        "environment": {
            "shared_signal": {"baseline_mean": config.shared_signal.baseline_mean,
                              "noise_sd": config.shared_signal.noise_sd,
                              "pulses": [asdict(p) for p in config.shared_signal.pulses]},
            "latent_dynamics": {k: getattr(d, k) for k in (
                "lambda_x", "rho_x", "value_tonic_volatility", "volatility_coupling")},
            "local_volatility_dynamics": {"lambda_v": d.lambda_v, "rho_v": d.rho_v,
                "tonic_volatility": d.local_volatility_tonic_volatility,
                "shared_value_coupling": d.shared_value_coupling},
            "observation": {"log_variance": d.observation_log_variance},
            "stream_a": asdict(config.stream_a), "stream_b": asdict(config.stream_b),
        },
    }


def _check_output_dir(output_dir: str | Path) -> Path:
    out = Path(output_dir).resolve()
    for old in ("local_A", "local_B", "global"):
        protected = get_environment_output_dir(old).resolve()
        if out == protected or protected in out.parents:
            raise ValueError(f"Do not save Hybrid output inside the existing {old} environment directory.")
    existing = [name for name in OUTPUT_NAMES if (out / name).exists()]
    if existing or (out / "seed_checks").exists():
        raise FileExistsError(f"Output already exists at {out}. Use a NEW --output-dir; nothing was overwritten.")
    return out


def save_environment(*, environment_df: pd.DataFrame, config: HybridConfig,
                     output_dir: str | Path) -> None:
    """Write pipeline-format data plus Hybrid metadata; never overwrite a run."""
    out = _check_output_dir(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    d = config.dynamics
    wide = make_wide_hybrid_environment(environment_df)
    summary = summarize_hybrid_environment(environment_df)
    pulse_summary = summarize_pulses(environment_df, config)
    metadata = {
        "name": config.name, "environment_type": ENVIRONMENT_NAME,
        "random_seed": config.random_seed, "n_trials": config.n_trials,
        "generator": "two_stream_hgf_hybrid.py", "schema_version": 1,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "indexing": "zero-based trial_index; half-open blocks and pulses; trial=trial_index+1",
        "initialization": "Initial states are before trial_index=0; row 0 follows the first v and x updates.",
        "within_trial_order": "draw shared c; update local v; compute q from current v; update x; draw u",
        "source": {"paper": "Weber et al., The generalized Hierarchical Gaussian Filter, uploaded updated manuscript",
                   "parameters": "Table 1, Example 1, Process column (p. 27)",
                   "architecture": "Figure 6 (p. 23)",
                   "observation_convention": "Appendix Eq. 65 (p. 41): log variance epsilon; not literal Eq. 9",
                   "project_choices": "Pulse schedule and shared jitter are project choices; see resolved_parameters for this run. The supplied default has 1000 trials, four 20-trial pulses with means +/-0.1 and shared jitter SD 0.005."},
        "equations": {
            "shared": "c[t] = m[t] + shared_noise_sd * z_c[t]",
            "local_volatility": "v_s[t] = v_s[t-1] + alpha*c[t] + exp(omega_v/2)*z_v_s[t]",
            "process_variance": "q_s[t] = exp(omega_x + kappa*v_s[t])",
            "value": "x_s[t] = x_s[t-1] + sqrt(q_s[t])*z_x_s[t]",
            "observation": "u_s[t] = x_s[t] + exp(observation_log_variance/2)*z_u_s[t]",
        },
        "resolved_parameters": _resolved_config(config),
        "derived_parameters": {
            "local_volatility_innovation_variance": float(np.exp(d.local_volatility_tonic_volatility)),
            "local_volatility_innovation_sd": float(np.exp(d.local_volatility_tonic_volatility / 2)),
            "observation_noise_variance": float(np.exp(d.observation_log_variance)),
            "observation_noise_sd": float(np.exp(d.observation_log_variance / 2)),
            "corresponding_input_precision": float(np.exp(-d.observation_log_variance)),
        },
        "random_generation": {"bit_generator": "PCG64", "child_streams": list(RNG_ROLES),
                              "method": "SeedSequence(random_seed).spawn(7)", "numpy_version": np.__version__,
                              "pandas_version": pd.__version__, "pyyaml_version": yaml.__version__},
        "interpretation": [
            "regime and is_a_volatile_block are NOMINAL schedule labels, not guarantees about actual q.",
            "Both local volatility states wander even outside pulses; either may exceed the other.",
            "Pulse windows are transitions, not instantaneous switches to a fixed q_high/q_low.",
            "The default prescribed positive/negative pulse areas balance; realized noisy areas need not.",
            "deterministic_* columns omit shared jitter and local innovations; they are not mean q.",
            "No value changepoints, clipping, resets, detrending, noise centering, or seed rejection.",
            "Noise in c is sampled independently each trial; +/-0.1 are means, not hard bounds.",
            "Extra ground-truth columns stay in environment files; the unchanged model runner drops them from labels.",
            "Compatible model input does not imply calibrated HGF parameters or validated model recovery.",
        ],
        "diagnostics": diagnostic_records(environment_df, config),
        "columns_long": list(environment_df.columns), "columns_wide": list(wide.columns),
        "columns_summary": list(summary.columns),
    }
    metadata_text = json.dumps(metadata, indent=2, allow_nan=False)
    environment_df.to_csv(out / "environment_long.csv", index=False)
    wide.to_csv(out / "environment_wide.csv", index=False)
    summary.to_csv(out / "environment_summary.csv", index=False)
    pulse_summary.to_csv(out / "environment_pulse_summary.csv", index=False)
    (out / "environment_metadata.json").write_text(metadata_text, encoding="utf-8")
    (out / "environment_config_resolved.yaml").write_text(
        yaml.safe_dump(_resolved_config(config), sort_keys=False), encoding="utf-8")


def check_environment_seeds(config: HybridConfig, seeds: range,
                            output_dir: str | Path) -> tuple[int, int]:
    """Check every specified seed; save ranges, block summaries, AND failures.

    'finite' means numerically representable only, not scientifically adequate.
    Full trajectories for these additional seeds are not saved.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    records, summaries, n_fail = [], [], 0
    for seed in seeds:
        current = replace(config, random_seed=seed)
        try:
            df = simulate_two_stream_environment(current)
            records.extend(diagnostic_records(df, current))
            summary = summarize_hybrid_environment(df)
            summary.insert(0, "random_seed", seed)
            summaries.append(summary)
        except FloatingPointError as exc:
            n_fail += 1
            records.append({"random_seed": seed, "stream": "unavailable",
                            "status": "numerical_failure", "error": str(exc)})
    pd.DataFrame(records).to_csv(out / "seed_diagnostics.csv", index=False)
    if summaries:
        pd.concat(summaries, ignore_index=True).to_csv(out / "block_summaries.csv", index=False)
    report = {"seeds": list(seeds), "n_attempted": len(seeds), "n_failed": n_fail,
              "n_finite": len(seeds) - n_fail,
              "note": "All seeds were retained in the record. Finite does not mean suitable for recovery."}
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return len(seeds) - n_fail, n_fail


def run_from_config(*, config_path: str | Path, output_dir: str | Path | None = None,
                    random_seed: int | None = None) -> pd.DataFrame:
    config = parse_environment_config(config_path)
    if random_seed is not None:
        config = replace(config, random_seed=random_seed)
    if output_dir is not None:
        _check_output_dir(output_dir)
    df = simulate_two_stream_environment(config)
    if output_dir is not None:
        save_environment(environment_df=df, config=config, output_dir=output_dir)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: canonical outputs/hgf_walk_outputs/hybrid. Existing runs are not overwritten.")
    parser.add_argument("--seed", type=int, default=None, help="Override YAML random_seed; saved in metadata.")
    parser.add_argument("--check-seeds", type=int, default=0,
                        help="Also check this many consecutive environment seeds, saving summaries only.")
    parser.add_argument("--check-start-seed", type=int, default=0)
    args = parser.parse_args()
    try:
        _integer(args.check_seeds, "check-seeds")
        _integer(args.check_start_seed, "check-start-seed")
        config = parse_environment_config(args.config)
        if args.seed is not None:
            config = replace(config, random_seed=args.seed)
        out = _check_output_dir(args.output_dir or get_environment_output_dir(ENVIRONMENT_NAME))
        df = simulate_two_stream_environment(config)
        save_environment(environment_df=df, config=config, output_dir=out)
        print(f"Saved Hybrid environment: {out}\nRows: {len(df)} ({config.n_trials} trials x 2 streams)")
        print(f"Observation SD = {np.exp(config.dynamics.observation_log_variance / 2):.6f}; "
              f"corresponding input precision = {np.exp(-config.dynamics.observation_log_variance):.6f}")
        cols = ["stream", "block", "mean_local_volatility", "min_log_process_variance", "max_log_process_variance"]
        print(summarize_hybrid_environment(df)[cols].to_string(index=False))
        print("NOTE: S/V labels are nominal. No inference model has been run or calibrated.")
        if args.check_seeds:
            seeds = range(args.check_start_seed, args.check_start_seed + args.check_seeds)
            n_finite, n_fail = check_environment_seeds(config, seeds, out / "seed_checks")
            print(f"Environment checks: {n_finite} finite, {n_fail} numerical failures. "
                  "Inspect seed_checks/ before drawing scientific conclusions.")
            if n_fail:
                raise SystemExit(2)
    except (KeyError, TypeError, ValueError, OSError, FloatingPointError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
