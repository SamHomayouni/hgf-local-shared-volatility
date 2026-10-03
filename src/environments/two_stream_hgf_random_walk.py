"""
Two-stream HGF-style random-walk environment.

This module generates a second synthetic environment for the project.

The environment is designed to test whether an inference model with a
global/shared volatility estimate shows cross-stream transfer of learning rate.

The manipulation is:
    - Stream A has higher latent process variance in two volatile blocks.
    - Stream B remains objectively stable throughout.
    - Observation noise is fixed across streams and blocks.

This differs from src/environments/two_stream_changepoint.py:
    - There are no changepoints.
    - The latent value evolves as a Gaussian random walk.
    - Volatility is implemented as blockwise process variance Q.

Core equations
--------------
For stream s in {A, B}:

    x_s(k) = lambda_x * x_s(k-1) + rho_x + sqrt(Q_s(k)) * epsilon_s(k)

    u_s(k) = x_s(k) + observation_noise_sd * eta_s(k)

where:

    epsilon_s(k), eta_s(k) ~ Normal(0, 1)

For the current design:

    Q_A(k) = q_low   during stable blocks
    Q_A(k) = q_high  during A-volatile blocks
    Q_B(k) = q_low   throughout

This module only generates the environment.
It does not fit any model.
It does not compute learning rates.
It does not compute the cross-transfer index.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import argparse
import json
import sys

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = next(
    p
    for p in Path(__file__).resolve().parents
    if (p / "src").is_dir() and (p / "config").is_dir()
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.utils.paths import (
    CONFIG_DIR,
    DEFAULT_ENVIRONMENT_DIR,
)

DEFAULT_CONFIG_PATH = (
    CONFIG_DIR / "hgf_random_walk_local_A.yaml"
)

DEFAULT_OUTPUT_DIR = DEFAULT_ENVIRONMENT_DIR


@dataclass(frozen=True)
class LatentDynamicsParameters:
    """
    Shared latent-dynamics parameters for both streams.

    Parameters
    ----------
    lambda_x:
        Autoconnection / autoregressive coefficient for latent value states.
        lambda_x = 1 gives a pure random walk.
        lambda_x < 1 gives mean-reverting dynamics.

    rho_x:
        Constant drift term. Kept at 0.0 for the first HGF-style environment.

    q_low:
        Process variance during stable periods.

    q_high:
        Process variance during volatile periods.

    observation_noise_sd:
        Standard deviation of observation noise around the latent value.
    """

    lambda_x: float
    rho_x: float
    q_low: float
    q_high: float
    observation_noise_sd: float

    def process_variance(self, regime: str) -> float:
        """
        Return process variance for a named regime.

        Parameters
        ----------
        regime:
            Either "stable" or "volatile".
        """
        if regime == "stable":
            return self.q_low

        if regime == "volatile":
            return self.q_high

        raise ValueError(
            f"Unknown regime {regime!r}. Expected 'stable' or 'volatile'."
        )


@dataclass(frozen=True)
class StreamInitialState:
    """
    Initial latent value for one stream.
    """

    initial_mean: float


@dataclass(frozen=True)
class Block:
    """
    Zero-based half-open block definition.

    Example
    -------
    Block(
        name="S1",
        start_index=0,
        stop_index=200,
        ...
    )

    contains trial indices:
        0, 1, 2, ..., 199
    """

    name: str
    start_index: int
    stop_index: int
    stream_a_regime: str
    stream_b_regime: str

    def contains_index(self, trial_index: int) -> bool:
        """Return True if a zero-based trial index belongs to this block."""
        return self.start_index <= trial_index < self.stop_index


@dataclass(frozen=True)
class EnvironmentConfig:
    """
    Parsed environment configuration.
    """

    name: str
    random_seed: int
    n_trials: int
    blocks: list[Block]
    latent_dynamics: LatentDynamicsParameters
    stream_a: StreamInitialState
    stream_b: StreamInitialState


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load a YAML configuration file.
    """

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Config file did not produce a dictionary.")

    return config


def parse_latent_dynamics(raw_config: dict[str, Any]) -> LatentDynamicsParameters:
    """
    Extract shared latent-dynamics parameters from the config.
    """

    try:
        dynamics_config = raw_config["environment"]["latent_dynamics"]
    except KeyError as error:
        raise KeyError(
            "Missing environment.latent_dynamics section in config."
        ) from error

    return LatentDynamicsParameters(
        lambda_x=float(dynamics_config["lambda_x"]),
        rho_x=float(dynamics_config["rho_x"]),
        q_low=float(dynamics_config["q_low"]),
        q_high=float(dynamics_config["q_high"]),
        observation_noise_sd=float(dynamics_config["observation_noise_sd"]),
    )


def parse_stream_initial_state(raw_config: dict[str, Any], stream_name: str) -> StreamInitialState:
    """
    Extract initial state for stream_a or stream_b.
    """

    try:
        stream_config = raw_config["environment"][stream_name]
    except KeyError as error:
        raise KeyError(f"Missing environment.{stream_name} in config.") from error

    return StreamInitialState(
        initial_mean=float(stream_config["initial_mean"]),
    )


def parse_blocks(raw_config: dict[str, Any]) -> list[Block]:
    """
    Extract zero-based block definitions from the config.
    """

    blocks_config = raw_config.get("blocks")

    if not isinstance(blocks_config, list) or len(blocks_config) == 0:
        raise ValueError("Config must contain a non-empty 'blocks' list.")

    blocks: list[Block] = []

    for block_config in blocks_config:
        block = Block(
            name=str(block_config["name"]),
            start_index=int(block_config["start_index"]),
            stop_index=int(block_config["stop_index"]),
            stream_a_regime=str(block_config["stream_a_regime"]),
            stream_b_regime=str(block_config["stream_b_regime"]),
        )

        blocks.append(block)

    return blocks


def parse_environment_config(config_path: str | Path) -> EnvironmentConfig:
    """
    Load and parse the full YAML config into an EnvironmentConfig object.
    """

    raw_config = load_yaml_config(config_path)

    try:
        simulation_config = raw_config["simulation"]
    except KeyError as error:
        raise KeyError("Missing 'simulation' section in config.") from error

    environment_config = EnvironmentConfig(
        name=str(simulation_config["name"]),
        random_seed=int(simulation_config["random_seed"]),
        n_trials=int(simulation_config["n_trials"]),
        blocks=parse_blocks(raw_config),
        latent_dynamics=parse_latent_dynamics(raw_config),
        stream_a=parse_stream_initial_state(raw_config, "stream_a"),
        stream_b=parse_stream_initial_state(raw_config, "stream_b"),
    )

    validate_environment_config(environment_config)

    return environment_config


def validate_environment_config(config: EnvironmentConfig) -> None:
    """
    Validate the parsed environment configuration.

    This catches common mistakes early, for example:
        - invalid number of trials
        - first block not starting at 0
        - final block not stopping at n_trials
        - gaps between blocks
        - overlaps between blocks
        - invalid regime labels
        - invalid process variances
    """

    if config.n_trials <= 0:
        raise ValueError("n_trials must be positive.")

    if len(config.blocks) == 0:
        raise ValueError("At least one block is required.")

    expected_start_index = 0

    for block in config.blocks:
        if block.start_index != expected_start_index:
            raise ValueError(
                f"Block {block.name!r} starts at index {block.start_index}, "
                f"but expected index {expected_start_index}. "
                "Check for gaps or overlaps."
            )

        if block.stop_index <= block.start_index:
            raise ValueError(
                f"Block {block.name!r} has stop_index <= start_index."
            )

        for regime in (block.stream_a_regime, block.stream_b_regime):
            if regime not in {"stable", "volatile"}:
                raise ValueError(
                    f"Invalid regime {regime!r} in block {block.name!r}. "
                    "Expected 'stable' or 'volatile'."
                )

        expected_start_index = block.stop_index

    final_stop_index = config.blocks[-1].stop_index

    if final_stop_index != config.n_trials:
        raise ValueError(
            f"Final block stops at index {final_stop_index}, "
            f"but n_trials is {config.n_trials}."
        )

    dynamics = config.latent_dynamics

    if not np.isclose(dynamics.lambda_x, 1.0):
        raise ValueError("lambda_x must be 1.0 for the Gaussian random-walk environments.")

    if dynamics.q_low <= 0.0:
        raise ValueError("q_low must be positive.")

    if dynamics.q_high <= 0.0:
        raise ValueError("q_high must be positive.")

    if dynamics.q_high <= dynamics.q_low:
        raise ValueError(
            "q_high should be greater than q_low for this volatility manipulation."
        )

    if dynamics.observation_noise_sd < 0.0:
        raise ValueError("observation_noise_sd must be non-negative.")


def get_block_and_regime_for_index(trial_index: int, blocks: list[Block], stream: str) -> tuple[Block, str]:
    """
    Return the block object and regime for a zero-based trial index.

    Parameters
    ----------
    trial_index:
        Zero-based trial index.

    blocks:
        List of block definitions.

    stream:
        Either "A" or "B".

    Returns
    -------
    tuple[Block, str]
        Example: (Block(...), "volatile")
    """

    if stream not in {"A", "B"}:
        raise ValueError("stream must be either 'A' or 'B'.")

    for block in blocks:
        if block.contains_index(trial_index):
            if stream == "A":
                return block, block.stream_a_regime

            return block, block.stream_b_regime

    raise ValueError(f"Trial index {trial_index} is not covered by any block.")


def simulate_one_stream(
    *,
    n_trials: int,
    initial_state: StreamInitialState,
    latent_dynamics: LatentDynamicsParameters,
    blocks: list[Block],
    stream: str,
    rng: np.random.Generator) -> pd.DataFrame:
    """
    Simulate one continuous stream using HGF-style latent dynamics.

    Generative process
    ------------------
    For each trial:
        1. Determine the stream's current regime.
        2. Choose process variance Q from the regime.
        3. Generate a latent innovation:
               sqrt(Q) * epsilon
        4. Update the latent value:
               x(k) = x(k-1) + rho_x + innovation
        5. Generate an observation:
               u(k) = x(k) + observation_noise_sd * eta
    """

    if n_trials <= 0:
        raise ValueError("n_trials must be positive.")

    latent_mean = np.empty(n_trials, dtype=float)
    observation = np.empty(n_trials, dtype=float)
    process_variance = np.empty(n_trials, dtype=float)
    process_sd = np.empty(n_trials, dtype=float)
    latent_innovation = np.empty(n_trials, dtype=float)
    observation_noise = np.empty(n_trials, dtype=float)

    changepoint = np.zeros(n_trials, dtype=bool)

    block_names: list[str] = []
    regimes: list[str] = []
    is_a_volatile_block: list[bool] = []

    current_latent = initial_state.initial_mean

    for trial_index in range(n_trials):
        block, regime = get_block_and_regime_for_index(
            trial_index=trial_index,
            blocks=blocks,
            stream=stream,
        )

        q_current = latent_dynamics.process_variance(regime)
        process_sd_current = float(np.sqrt(q_current))

        epsilon = rng.normal(loc=0.0, scale=1.0)
        innovation_current = process_sd_current * epsilon

        current_latent = (latent_dynamics.lambda_x * current_latent + latent_dynamics.rho_x + innovation_current)

        eta = rng.normal(loc=0.0, scale=1.0)
        observation_noise_current = latent_dynamics.observation_noise_sd * eta
        observation_current = current_latent + observation_noise_current

        latent_mean[trial_index] = current_latent
        observation[trial_index] = observation_current
        process_variance[trial_index] = q_current
        process_sd[trial_index] = process_sd_current
        latent_innovation[trial_index] = innovation_current
        observation_noise[trial_index] = observation_noise_current

        block_names.append(block.name)
        regimes.append(regime)
        is_a_volatile_block.append(block.stream_a_regime == "volatile")

    return pd.DataFrame(
        {
            # Computational index: use this for slicing and modelling.
            "trial_index": np.arange(n_trials),

            # Human-readable trial number: useful for plotting.
            "trial": np.arange(1, n_trials + 1),
            "stream": stream,
            "block": block_names,
            "regime": regimes,
            "latent_mean": latent_mean,
            "observation": observation,

            # Kept for compatibility with current model runners.
            # This environment has no changepoints.
            "changepoint": changepoint,

            # HGF-style random-walk specific quantities.
            "process_variance": process_variance,
            "process_sd": process_sd,
            "latent_innovation": latent_innovation,
            "observation_noise": observation_noise,

            # Same value for A and B rows; useful for plotting block shading.
            "is_a_volatile_block": is_a_volatile_block,
        }
    )


def simulate_two_stream_environment(config: EnvironmentConfig) -> pd.DataFrame:
    """
    Simulate the full two-stream environment.

    Returns
    -------
    pandas.DataFrame
        Long-format dataframe.

        Each trial has two rows:
            - one row for stream A
            - one row for stream B
    """

    rng_a = np.random.default_rng(config.random_seed)
    rng_b = np.random.default_rng(config.random_seed + 1)

    stream_a_df = simulate_one_stream(
        n_trials=config.n_trials,
        initial_state=config.stream_a,
        latent_dynamics=config.latent_dynamics,
        blocks=config.blocks,
        stream="A",
        rng=rng_a,
    )

    stream_b_df = simulate_one_stream(
        n_trials=config.n_trials,
        initial_state=config.stream_b,
        latent_dynamics=config.latent_dynamics,
        blocks=config.blocks,
        stream="B",
        rng=rng_b,
    )

    environment_df = pd.concat([stream_a_df, stream_b_df], ignore_index=True)

    environment_df = (
        environment_df.sort_values(
            by=["trial_index", "stream"],
            ascending=[True, True],
        )
        .reset_index(drop=True)
    )

    return environment_df


def make_wide_environment(environment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the long-format environment dataframe to wide format.

    Long format is better for plotting and summaries.
    Wide format is often better for model input and sanity checks.

    Returns
    -------
    pandas.DataFrame
        One row per trial, with columns such as:
            observation_A
            observation_B
            latent_mean_A
            latent_mean_B
            process_variance_A
            process_variance_B
    """

    required_columns = {
        "trial_index",
        "trial",
        "stream",
        "block",
        "regime",
        "latent_mean",
        "observation",
        "changepoint",
        "process_variance",
        "process_sd",
        "latent_innovation",
        "observation_noise",
        "is_a_volatile_block",
    }

    missing_columns = required_columns - set(environment_df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    base = (
        environment_df[environment_df["stream"] == "A"]
        .set_index("trial_index")
        .loc[:, ["trial", "block", "is_a_volatile_block"]]
        .sort_index()
    )

    wide_parts = [base]

    value_columns = [
        "latent_mean",
        "observation",
        "changepoint",
        "regime",
        "process_variance",
        "process_sd",
        "latent_innovation",
        "observation_noise",
    ]

    for value_column in value_columns:
        pivoted = environment_df.pivot(
            index="trial_index",
            columns="stream",
            values=value_column,
        ).sort_index()

        pivoted = pivoted.loc[:, ["A", "B"]]
        pivoted.columns = [
            f"{value_column}_{stream_label}"
            for stream_label in pivoted.columns
        ]

        wide_parts.append(pivoted)

    wide_df = pd.concat(wide_parts, axis=1)
    wide_df = wide_df.reset_index()

    return wide_df


def summarize_environment(environment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary table for quick sanity checks.

    Important checks:
        - Stream A should have higher process variance in V1 and V2.
        - Stream B should have low process variance in all blocks.
        - changepoint columns should remain zero because this environment
          contains no changepoints.
    """

    summary = (
        environment_df.groupby(["stream", "block", "regime"], as_index=False)
        .agg(
            n_trials=("trial_index", "count"),
            first_index=("trial_index", "min"),
            last_index=("trial_index", "max"),
            n_changepoints=("changepoint", "sum"),
            mean_observation=("observation", "mean"),
            sd_observation=("observation", "std"),
            mean_latent=("latent_mean", "mean"),
            sd_latent=("latent_mean", "std"),
            mean_process_variance=("process_variance", "mean"),
            mean_process_sd=("process_sd", "mean"),
            sd_latent_innovation=("latent_innovation", "std"),
            sd_observation_noise=("observation_noise", "std"),
        )
        .sort_values(["stream", "first_index"])
        .reset_index(drop=True)
    )

    summary["changepoint_rate_observed"] = (
        summary["n_changepoints"] / summary["n_trials"]
    )

    return summary


def save_environment(*, environment_df: pd.DataFrame, config: EnvironmentConfig, output_dir: str | Path) -> None:
    """
    Save the simulated environment and metadata.

    This writes:
        - environment_long.csv
        - environment_wide.csv
        - environment_metadata.json
        - environment_summary.csv
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    long_path = output_dir / "environment_long.csv"
    wide_path = output_dir / "environment_wide.csv"
    metadata_path = output_dir / "environment_metadata.json"
    summary_path = output_dir / "environment_summary.csv"

    wide_df = make_wide_environment(environment_df)
    summary_df = summarize_environment(environment_df)

    environment_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    dynamics = config.latent_dynamics

    metadata = {
        "name": config.name,
        "random_seed": config.random_seed,
        "n_trials": config.n_trials,
        "generator": "two_stream_hgf_random_walk.py",
        "indexing": "zero-based trial_index with half-open block intervals",
        "generative_equations": {
            "latent_state": (
                "x_s(k) = lambda_x * x_s(k-1) + rho_x "
                "+ sqrt(Q_s(k)) * epsilon_s(k)"
            ),
            "observation": (
                "u_s(k) = x_s(k) + observation_noise_sd * eta_s(k)"
            ),
            "random_terms": "epsilon_s(k), eta_s(k) ~ Normal(0, 1)",
            "process_variance": (
                "Q_s(k) = q_low in stable regimes; "
                "Q_s(k) = q_high in volatile regimes"
            ),
        },
        "latent_dynamics": {
            "lambda_x": dynamics.lambda_x,
            "rho_x": dynamics.rho_x,
            "q_low": dynamics.q_low,
            "q_high": dynamics.q_high,
            "sqrt_q_low": float(np.sqrt(dynamics.q_low)),
            "sqrt_q_high": float(np.sqrt(dynamics.q_high)),
            "q_high_to_q_low_ratio": dynamics.q_high / dynamics.q_low,
            "observation_noise_sd": dynamics.observation_noise_sd,
            "corresponding_input_precision": (
                None
                if dynamics.observation_noise_sd == 0.0
                else 1.0 / dynamics.observation_noise_sd**2
            ),
        },
        "stream_initial_states": {
            "A": config.stream_a.initial_mean,
            "B": config.stream_b.initial_mean,
        },
        "blocks": [
            {
                "name": block.name,
                "start_index": block.start_index,
                "stop_index": block.stop_index,
                "stream_a_regime": block.stream_a_regime,
                "stream_b_regime": block.stream_b_regime,
            }
            for block in config.blocks
        ],
        "columns_long": list(environment_df.columns),
        "columns_wide": list(wide_df.columns),
        "columns_summary": list(summary_df.columns),
        "compatibility_note": (
            "The changepoint column is kept for compatibility with existing "
            "model runners. It is False for every row because this environment "
            "contains no changepoints."
        ),
    }

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"Saved long environment:    {long_path}")
    print(f"Saved wide environment:    {wide_path}")
    print(f"Saved summary table:       {summary_path}")
    print(f"Saved environment metadata:{metadata_path}")


def run_from_config(*, config_path: str | Path, output_dir: str | Path | None = None) -> pd.DataFrame:
    """
    Main programmatic entry point.

    This is the function that future scripts can import and call.
    """

    config = parse_environment_config(config_path)
    environment_df = simulate_two_stream_environment(config)

    if output_dir is not None:
        save_environment(
            environment_df=environment_df,
            config=config,
            output_dir=output_dir,
        )

    return environment_df


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build command-line argument parser.

    This lets you run the file directly from the terminal.
    """

    parser = argparse.ArgumentParser(
        description="Simulate the two-stream HGF-style random-walk environment."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to an HGF random-walk environment YAML config.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where environment CSV files should be saved.",
    )

    return parser


def main() -> None:
    """
    Command-line entry point.

    Example
    -------
    From the project root:

        python src/environments/two_stream_hgf_random_walk.py

    or explicitly:

        python src/environments/two_stream_hgf_random_walk.py \\
            --config config/hgf_random_walk_local_A.yaml \\
            --output-dir outputs/hgf_walk_outputs/local_A
    """

    parser = build_arg_parser()
    args = parser.parse_args()

    environment_df = run_from_config(
        config_path=args.config,
        output_dir=args.output_dir,
    )

    summary = summarize_environment(environment_df)

    print("\nHGF-style random-walk environment generated successfully.")
    print("\nSummary:")
    print(summary.to_string(index=False))

    print(f"\nSaved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()