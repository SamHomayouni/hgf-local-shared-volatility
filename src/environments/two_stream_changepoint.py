"""
Two-stream change-point environment.

This module generates the first synthetic environment for the project.

The environment is designed to test whether an inference model with a
global/shared volatility estimate shows cross-stream transfer of learning rate.

The manipulation is intentionally simple:
    - Stream A becomes volatile in two blocks.
    - Stream B remains objectively stable throughout.

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
import numpy as np
import pandas as pd
import yaml

@dataclass(frozen= True)
class StreamParameters:
    """
    Parameters controlling one continuous observation stream.

    Each stream has:
        - an initial latent mean
        - a stable change-point probability
        - a volatile change-point probability
        - stable and volatile jump sizes
        - small continuous latent drift
        - observation noise

    In this first simulation, stream A actually uses both stable and volatile
    regimes. Stream B is set to stable for all blocks.
    """

    initial_mean: float
    changepoint_probability_stable: float
    changepoint_probability_volatile: float
    changepoint_jump_sd_stable: float
    changepoint_jump_sd_volatile: float
    latent_drift_sd: float
    observation_noise_sd: float

    def changepoint_probability(self, regime: str) -> float:
        """
        Return the change-point probability for a named regime.

        Parameters
        ----------
        regime:
            Either "stable" or "volatile".
        """
        if regime == "stable":
            return self.changepoint_probability_stable
        if regime == "volatile":
            return self.changepoint_probability_volatile
        
        raise ValueError(
            f"Unknown regime {regime!r}. Expected 'stable' or 'volatile'."
        )
    

    def changepoint_jump_sd(self, regime: str) -> float:
        """
        Return the change-point jump standard deviation for a named regime.

        Parameters
        ----------
        regime:
            Either "stable" or "volatile".
        """
        if regime == "stable":
            return self.changepoint_jump_sd_stable
        if regime == "volatile":
            return self.changepoint_jump_sd_volatile
        
        raise ValueError(
            f"Unknown regime {regime!r}. Expected 'stable' or 'volatile'."
        )
    
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

    This is a cleaner Python representation of the YAML file.
    """
    name: str
    random_seed: int
    n_trials: int
    blocks: list[Block]
    stream_a: StreamParameters
    stream_b: StreamParameters

def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    config_path:
        Path to the YAML file.

    Returns
    -------
    dict
        Raw configuration dictionary.
    """

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Config file did not produce  dictionary.")
    return config

def parse_stream_parameters(raw_config: dict[str, Any], stream_name:str) -> StreamParameters:
    """
    Extract StreamParameters for stream_a or stream_b from the config.

    Parameters
    ----------
    raw_config:
        Raw dictionary loaded from YAML.
    stream_name:
        Either "stream_a" or "stream_b".
    """
    try:
        stream_config = raw_config["environment"][stream_name]
    except KeyError as error:
        raise KeyError(f"Missing environment.{stream_name} in config") from error
    
    return StreamParameters(
        initial_mean= float(stream_config["initial_mean"]),
        changepoint_probability_stable= float(stream_config["changepoint_probability_stable"]),
        changepoint_probability_volatile= float(stream_config["changepoint_probability_volatile"]),
        changepoint_jump_sd_stable= float(stream_config["changepoint_jump_sd_stable"]),
        changepoint_jump_sd_volatile= float(stream_config["changepoint_jump_sd_volatile"]),
        latent_drift_sd= float(stream_config["latent_drift_sd"]),
        observation_noise_sd= float(stream_config["observation_noise_sd"])
    )

def parse_blocks(raw_config: dict[str, Any]) -> list[Block]:
    """
    Extract zero-based block definitions from the config.

    Expected YAML format
    --------------------
    blocks:
      - name: "S1"
        start_index: 0
        stop_index: 200
        stream_a_regime: "stable"
        stream_b_regime: "stable"

    Returns
    -------
    list[Block]
        Ordered list of experimental blocks.
    """

    blocks_config = raw_config.get("blocks")
    if not isinstance(blocks_config, list) or len(blocks_config) == 0:
        raise ValueError("Config must contain a non-empty 'blocks' list.")
    blocks: list[Block] = []

    for block_config in blocks_config:
        block = Block(
            name= str(block_config["name"]),
            start_index= int(block_config["start_index"]),
            stop_index= int(block_config["stop_index"]),
            stream_a_regime= str(block_config["stream_a_regime"]),
            stream_b_regime= str(block_config["stream_b_regime"]),
        )
        blocks.append(block)
    return blocks





def parse_environment_config(config_path: str | Path) -> EnvironmentConfig:
    """
    Load and parse the full YAML config into an EnvironmentConfig object
    """
    raw_config = load_yaml_config(config_path)
    try:
        simulation_config = raw_config["simulation"]
    except KeyError as error:
        raise KeyError("Missing 'simulation' section in config.") from error
    
    environment_config = EnvironmentConfig(
        name= str(simulation_config["name"]),
        random_seed= int(simulation_config["random_seed"]),
        n_trials= int(simulation_config["n_trials"]),
        blocks= parse_blocks(raw_config),
        stream_a= parse_stream_parameters(raw_config, "stream_a"),
        stream_b= parse_stream_parameters(raw_config, "stream_b"),
    )

    validate_environment_config(environment_config)

    return environment_config

def validate_environment_config(config: EnvironmentConfig) -> None:
    """
    Validate the parsed environment configuration.

    This catches common mistakes early, for example:
        - first block not starting at 0
        - final block not stopping at n_trials
        - gaps between blocks
        - overlaps between blocks
        - invalid regime labels
    """
    
    if config.n_trials <= 0:
        raise ValueError("n_trials must be positive")
    if len(config.blocks) == 0:
        raise ValueError("At least one block is required")
    
    expected_start_index = 0

    for block in config.blocks:
        if block.start_index != expected_start_index:
            raise ValueError(
                f"Block {block.name!r} starts at index {block.start_index}, but expected index {expected_start_index}. Check for gaps or overlaps."
            )
        
        if block.stop_index <= block.start_index:
            raise ValueError(f"Block {block.name!r} has stop_index <= start_index")
        for regime in (block.stream_a_regime, block.stream_b_regime):
            if regime not in {"stable", "volatile"}:
                raise ValueError(f"Invalid regime {regime!r} in block {block.name!r}. Expected 'stable' or 'volatile'.")
        expected_start_index = block.stop_index
    final_stop_index = config.blocks[-1].stop_index

    if final_stop_index != config.n_trials:
        raise ValueError(f"Final block stops at index {final_stop_index}, but n_trials is {config.n_trials}.")
    
def get_block_and_regime_for_index(trial_index: int, blocks: list[Block], stream: str) -> tuple[str, str]:
    """
    Return the block name and regime for a zero-based trial index.

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
    tuple[str, str]
        Example: ("V1", "volatile")
    """

    if stream not in {"A", "B"}:
        raise ValueError("stream must be either 'A' or 'B'.")
    
    for block in blocks:
        if block.contains_index(trial_index):
            if stream == "A":
                return block.name, block.stream_a_regime
            return block.name, block.stream_b_regime
    raise ValueError(f"Trial index {trial_index} is not covered by any block.")

def simulate_one_stream(*, n_trials: int, stream_parameters: StreamParameters, blocks: list[Block], stream: str, rng: np.random.Generator) -> pd.DataFrame:
    """
    Simulate one continuous stream with change points.

    Generative process
    ------------------
    For each trial:
        1. Determine whether the stream is currently stable or volatile.
        2. Sample whether a change point occurs.
        3. If change point occurs:
              latent mean receives a larger jump.
           Else:
              latent mean receives only small drift.
        4. Observation is sampled around the latent mean.

    The latent process is not meant to be a perfect gHGF generative model.
    It is a transparent first environment for debugging the pipeline.
    """
    if n_trials <= 0:
        raise ValueError("n_trials must be positive.")
    latent_mean = np.empty(n_trials, dtype= float)
    observation = np.empty(n_trials, dtype= float)
    changepoint = np.zeros(n_trials, dtype= bool)

    block_names: list[str] = []
    regimes: list[str] = []

    current_mean = stream_parameters.initial_mean

    for trial_index in range(n_trials):
        block_name, regime = get_block_and_regime_for_index(trial_index, blocks, stream)
        cp_prob = stream_parameters.changepoint_probability(regime)
        cp_jump_sd = stream_parameters.changepoint_jump_sd(regime)
        has_changepoint = rng.random() < cp_prob

        if has_changepoint:
            #A change point resets the latent mean to a new value around the stream's baseline. This avoids unbounded drift.
            current_mean = rng.normal(loc=stream_parameters.initial_mean, scale= cp_jump_sd)
        else: 
            #Without a change point, allow only tiny random-walk drift.
            current_mean = current_mean + rng.normal(loc= 0.0, scale= stream_parameters.latent_drift_sd)
        current_observation = rng.normal(loc= current_mean, scale= stream_parameters.observation_noise_sd)
        latent_mean[trial_index] = current_mean
        observation[trial_index] = current_observation
        changepoint[trial_index] = has_changepoint
        block_names.append(block_name)
        regimes.append(regime)

    return pd.DataFrame(
        {
            #computational index: use this for slicing and modeling.
            "trial_index": np.arange(n_trials),

            #human readable trial number usefull for plotting
            "trial": np.arange(1, n_trials + 1),
            "stream": stream,
            "block": block_names,
            "regime": regimes,
            "latent_mean": latent_mean,
            "observation": observation,
            "changepoint": changepoint,
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

        Important columns:
            trial_index
            trial
            stream
            block
            regime
            latent_mean
            observation
            changepoint
    """

    rng_a = np.random.default_rng(config.random_seed)
    rng_b = np.random.default_rng(config.random_seed + 1)
    stream_a_df = simulate_one_stream(n_trials= config.n_trials, stream_parameters= config.stream_a, blocks= config.blocks, stream= "A", rng= rng_a)
    stream_b_df = simulate_one_stream(n_trials= config.n_trials, stream_parameters= config.stream_b, blocks= config.blocks, stream= "B", rng= rng_b)
    environment_df = pd.concat([stream_a_df, stream_b_df], ignore_index=True)
    environment_df = environment_df.sort_values(by=["trial_index", "stream"], ascending=[True, True]).reset_index(drop=True)
    return environment_df

def make_wide_environment(environment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the long-format environment dataframe to wide format.

    Long format is better for plotting and summaries.
    Wide format is often better for model input.

    Returns
    -------
    pandas.DataFrame
        One row per trial, with columns such as:
            observation_A
            observation_B
            latent_mean_A
            latent_mean_B
            changepoint_A
            changepoint_B
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
    }

    missing_columns = required_columns - set(environment_df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    
    wide_parts = []
    for value_column in ["latent_mean", "observation", "changepoint", "regime"]:
        pivoted = environment_df.pivot(index="trial_index", columns="stream", values= value_column)
        pivoted.columns = [f"{value_column}_{stream_label}" for stream_label in pivoted.columns]
        wide_parts.append(pivoted)

    # Trial number and block are the same for A and B in this design.
    trial_by_index = (environment_df[environment_df["stream"] == "A"].set_index("trial_index")["trial"].rename("block"))
    block_by_index = (environment_df[environment_df["stream"] == "A"].set_index("trial_index")["block"].rename("block"))

    wide_df = pd.concat([trial_by_index, block_by_index, *wide_parts], axis=1)
    wide_df = wide_df.reset_index()

    return wide_df

def save_environment(*, environment_df: pd.DataFrame, config: EnvironmentConfig, output_dir: str | Path) -> None:
    """
    Save the simulated environment and metadata.

    This writes:
        - environment_long.csv
        - environment_wide.csv
        - environment_metadata.json
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    long_path = output_dir / "environment_long.csv"
    wide_path = output_dir / "environment_wide.csv"
    metadata_path = output_dir / "environment_metadata.json"
    wide_df = make_wide_environment(environment_df)
    environment_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)

    metadata = {
        "name": config.name,
        "random_seed": config.random_seed,
        "n_trials": config.n_trials,
        "indexing": "zero-based trial_index with half-open block intervals", 
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
    }

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

def summarize_environment(environment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a simple summary table for quick sanity checks.

    The most important sanity check:
        Stream A should have more change points in volatile blocks.
        Stream B should not.
    """

    summary = (
        environment_df.groupby(["stream", "block", "regime"], as_index=False).agg(
            n_trials=("trial_index", "count"),
            first_index=("trial_index", "min"),
            last_index=("trial_index", "max"),
            n_changepoints=("changepoint", "sum"),
            mean_observation=("observation", "mean"),
            sd_observation=("observation", "std"),
            mean_latent=("latent_mean", "mean"),
            sd_latent=("latent_mean", "std"),
        )
    )
    summary["changepoint_rate_observed"] = (summary["n_changepoints"] / summary["n_trials"])
    return summary

def run_from_config(*, config_path: str | Path, output_dir: str | Path | None = None) -> pd.DataFrame:
    """
    Main programmatic entry point.

    This is the function that future scripts can import and call.
    """
    config = parse_environment_config(config_path)
    environment_df = simulate_two_stream_environment(config)
    if output_dir is not None:
        save_environment(environment_df= environment_df, config= config, output_dir= output_dir)
        summary = summarize_environment(environment_df)
        summary_path = Path(output_dir) / "environment_summary.csv"
        summary.to_csv(summary_path, index=False)
    return environment_df

def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build command-line argument parser.

    This lets you run the file directly from the terminal.
    """
    parser = argparse.ArgumentParser(
        description="Simulate the first two-stream change-point environment."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to first_simulation.yaml.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where environment CSV files should be saved.",
    )

    return parser

def main() -> None:
    """
    Command-line entry point.

    Example
    -------
    From the project root, on Windows:

        python src/environments/two_stream_changepoint.py ^
            --config config/first_simulation.yaml ^
            --output-dir outputs/data

    On Mac/Linux:

        python src/environments/two_stream_changepoint.py \\
            --config config/first_simulation.yaml \\
            --output-dir outputs/data
    """

    parser = build_arg_parser()
    args = parser.parse_args()
    environment_df = run_from_config(config_path = args.config, output_dir = args.output_dir)
    summary = summarize_environment(environment_df)
    print("\nEnvironment generated successfully.")
    print("\nSummary:")
    print(summary.to_string(index=False))

    if args.output_dir is not None:
        print(f"\nSaved outputs to: {args.output_dir}")

if __name__ == "__main__":
    main()