"""
Export environment parameter tables for the HGF-style random-walk environment.

This script reads the YAML config used to generate the two-stream HGF-style
random-walk environment and saves readable tables for:

    1. simulation settings
    2. latent dynamics
    3. stream initial states
    4. derived environment parameters
    5. block schedule / volatility manipulation

Expected input:
    config/hgf_random_walk_local_A.yaml

Expected outputs:
    outputs/hgf_walk_outputs/tables/environment_parameters/
        simulation_settings.csv
        simulation_settings.png
        latent_dynamics.csv
        latent_dynamics.png
        stream_initial_states.csv
        stream_initial_states.png
        derived_environment_parameters.csv
        derived_environment_parameters.png
        block_schedule.csv
        block_schedule.png

Run from project root:

    python scripts/export_environment_parameter_tables_hgf_random_walk.py
"""

from __future__ import annotations

from pathlib import Path
import math
import sys

import matplotlib.pyplot as plt
import pandas as pd
import yaml


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir() and (p / "config").is_dir())

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "hgf_random_walk_local_A.yaml"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "hgf_walk_outputs"
    / "tables"
    / "environment_parameters"
)


PARAMETER_DESCRIPTIONS = {
    # Simulation
    "name": "Name of this simulation run.",
    "random_seed": "Random seed used for reproducible environment generation.",
    "n_trials": "Total number of trials in the simulated environment.",

    # Latent dynamics
    "lambda_x": (
        "Autoconnection / autoregressive coefficient for the latent value "
        "states. lambda_x = 1 is a pure random walk; lambda_x < 1 is "
        "mean-reverting."
    ),
    "rho_x": (
        "Constant drift term in the latent value equation. Kept at 0.0 in "
        "the current environment."
    ),
    "q_low": "Stable-block latent process variance.",
    "q_high": "Volatile-block latent process variance for stream A.",
    "observation_noise_sd": (
        "Observation noise standard deviation around the latent value. This "
        "is fixed across streams and blocks."
    ),

    # Stream initial states
    "initial_mean": "Initial latent value of the stream.",

    # Derived quantities
    "sqrt_q_low": "Stable-block latent innovation standard deviation.",
    "sqrt_q_high": "Volatile-block latent innovation standard deviation.",
    "q_high_to_q_low_ratio": (
        "Ratio between volatile and stable latent process variance."
    ),
    "log_q_low": (
        "Natural log of q_low. This is the value-level tonic volatility that "
        "matches the stable process variance when the volatility parent starts "
        "near zero and coupling is one."
    ),
    "log_q_high": (
        "Natural log of q_high. This is the approximate log-process variance "
        "needed during volatile blocks."
    ),
    "log_q_difference": (
        "Difference log(q_high) - log(q_low). Under coupling of one, this is "
        "roughly the volatility-parent increase needed to explain volatile "
        "blocks."
    ),
    "corresponding_input_precision": (
        "Observation precision corresponding to observation_noise_sd: "
        "1 / observation_noise_sd**2."
    ),

    # Blocks
    "start_index": "First trial index included in this block.",
    "stop_index": (
        "First trial index after this block; Python-style half-open interval."
    ),
    "stream_a_regime": "Regime assigned to stream A in this block.",
    "stream_b_regime": "Regime assigned to stream B in this block.",
}


def load_yaml_config(config_path: Path) -> dict:
    """
    Load the YAML environment config.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find config file: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Config file did not load as a dictionary.")

    return config


def format_value(value: object) -> str:
    """
    Format values for readable tables.
    """
    if isinstance(value, float):
        if value == 0:
            return "0.0"

        if abs(value) >= 1e5 or abs(value) < 1e-4:
            return f"{value:.1e}"

        return f"{value:g}"

    return str(value)


def make_simulation_settings_table(config: dict) -> pd.DataFrame:
    """
    Make table for top-level simulation settings.
    """
    simulation = config.get("simulation", {})

    rows = []

    for parameter, value in simulation.items():
        rows.append(
            {
                "Parameter": parameter,
                "Value": format_value(value),
                "Description": PARAMETER_DESCRIPTIONS.get(
                    parameter,
                    "No description added yet.",
                ),
            }
        )

    return pd.DataFrame(rows)


def make_latent_dynamics_table(config: dict) -> pd.DataFrame:
    """
    Make table for latent-dynamics parameters.
    """
    latent_dynamics = (
        config.get("environment", {})
        .get("latent_dynamics", {})
    )

    rows = []

    for parameter, value in latent_dynamics.items():
        rows.append(
            {
                "Parameter": parameter,
                "Value": format_value(value),
                "Description": PARAMETER_DESCRIPTIONS.get(
                    parameter,
                    "No description added yet.",
                ),
            }
        )

    return pd.DataFrame(rows)


def make_stream_initial_states_table(config: dict) -> pd.DataFrame:
    """
    Make table comparing initial states for stream A and stream B.
    """
    environment = config.get("environment", {})

    stream_a = environment.get("stream_a", {})
    stream_b = environment.get("stream_b", {})

    all_parameters = sorted(set(stream_a.keys()) | set(stream_b.keys()))

    rows = []

    for parameter in all_parameters:
        rows.append(
            {
                "Parameter": parameter,
                "Stream A": format_value(stream_a.get(parameter, "")),
                "Stream B": format_value(stream_b.get(parameter, "")),
                "Description": PARAMETER_DESCRIPTIONS.get(
                    parameter,
                    "No description added yet.",
                ),
            }
        )

    return pd.DataFrame(rows)


def make_derived_environment_parameters_table(config: dict) -> pd.DataFrame:
    """
    Make table for derived quantities from the HGF random-walk config.

    These are not separately specified in the YAML file, but they are useful
    for checking model/environment alignment.
    """
    latent_dynamics = (
        config.get("environment", {})
        .get("latent_dynamics", {})
    )

    q_low = float(latent_dynamics["q_low"])
    q_high = float(latent_dynamics["q_high"])
    observation_noise_sd = float(latent_dynamics["observation_noise_sd"])

    derived = {
        "sqrt_q_low": math.sqrt(q_low),
        "sqrt_q_high": math.sqrt(q_high),
        "q_high_to_q_low_ratio": q_high / q_low,
        "log_q_low": math.log(q_low),
        "log_q_high": math.log(q_high),
        "log_q_difference": math.log(q_high) - math.log(q_low),
        "corresponding_input_precision": (
            math.inf
            if observation_noise_sd == 0
            else 1.0 / observation_noise_sd**2
        ),
    }

    rows = []

    for parameter, value in derived.items():
        rows.append(
            {
                "Parameter": parameter,
                "Value": format_value(value),
                "Description": PARAMETER_DESCRIPTIONS.get(
                    parameter,
                    "No description added yet.",
                ),
            }
        )

    return pd.DataFrame(rows)


def make_block_schedule_table(config: dict) -> pd.DataFrame:
    """
    Make table for the block schedule.
    """
    blocks = config.get("blocks", [])

    rows = []

    for block in blocks:
        start_index = block.get("start_index")
        stop_index = block.get("stop_index")

        if start_index is not None and stop_index is not None:
            n_trials = int(stop_index) - int(start_index)
            trial_range = f"{start_index}-{stop_index - 1}"
        else:
            n_trials = ""
            trial_range = ""

        rows.append(
            {
                "Block": block.get("name", ""),
                "Trial indices": trial_range,
                "Start index": start_index,
                "Stop index": stop_index,
                "N trials": n_trials,
                "Stream A regime": block.get("stream_a_regime", ""),
                "Stream B regime": block.get("stream_b_regime", ""),
            }
        )

    return pd.DataFrame(rows)


def save_png_table(
    dataframe: pd.DataFrame,
    output_path: Path,
    title: str,
    figure_width: float = 15,
) -> None:
    """
    Save dataframe as a PNG table.
    """
    n_rows = len(dataframe)
    n_columns = len(dataframe.columns)

    figure_height = max(2.8, 0.5 * n_rows + 1.5)

    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    ax.axis("off")

    ax.set_title(
        title,
        fontsize=16,
        fontweight="bold",
        pad=18,
    )

    table = ax.table(
        cellText=dataframe.values,
        colLabels=dataframe.columns,
        cellLoc="left",
        colLoc="left",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.45)

    for column_idx in range(n_columns):
        header_cell = table[(0, column_idx)]
        header_cell.set_text_props(weight="bold")
        header_cell.set_facecolor("#f2f2f2")

    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_table_outputs(
    *,
    dataframe: pd.DataFrame,
    output_stem: str,
    title: str,
    output_dir: Path,
    figure_width: float = 15,
) -> None:
    """
    Save one table as CSV and PNG.
    """
    csv_path = output_dir / f"{output_stem}.csv"
    png_path = output_dir / f"{output_stem}.png"

    dataframe.to_csv(csv_path, index=False)

    save_png_table(
        dataframe=dataframe,
        output_path=png_path,
        title=title,
        figure_width=figure_width,
    )

    print(f"Saved CSV: {csv_path}")
    print(f"Saved PNG: {png_path}")


def main() -> None:
    """
    Export all HGF-random-walk environment parameter tables.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = load_yaml_config(DEFAULT_CONFIG_PATH)

    simulation_table = make_simulation_settings_table(config)
    latent_dynamics_table = make_latent_dynamics_table(config)
    stream_initial_states_table = make_stream_initial_states_table(config)
    derived_table = make_derived_environment_parameters_table(config)
    block_table = make_block_schedule_table(config)

    save_table_outputs(
        dataframe=simulation_table,
        output_stem="simulation_settings",
        title="HGF random-walk simulation settings",
        output_dir=OUTPUT_DIR,
        figure_width=12,
    )

    save_table_outputs(
        dataframe=latent_dynamics_table,
        output_stem="latent_dynamics",
        title="HGF random-walk latent dynamics",
        output_dir=OUTPUT_DIR,
        figure_width=16,
    )

    save_table_outputs(
        dataframe=stream_initial_states_table,
        output_stem="stream_initial_states",
        title="HGF random-walk stream initial states",
        output_dir=OUTPUT_DIR,
        figure_width=14,
    )

    save_table_outputs(
        dataframe=derived_table,
        output_stem="derived_environment_parameters",
        title="Derived HGF random-walk environment parameters",
        output_dir=OUTPUT_DIR,
        figure_width=17,
    )

    save_table_outputs(
        dataframe=block_table,
        output_stem="block_schedule",
        title="HGF random-walk block schedule",
        output_dir=OUTPUT_DIR,
        figure_width=14,
    )

    print("\nFinished exporting HGF-random-walk environment parameter tables.")


if __name__ == "__main__":
    main()