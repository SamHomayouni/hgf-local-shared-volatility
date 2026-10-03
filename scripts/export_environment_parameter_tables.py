"""
Export environment parameter tables.

This script reads the YAML config used to generate the two-stream environment
and saves readable tables for:

    1. simulation settings
    2. stream-level environment parameters
    3. block schedule / volatility manipulation

Expected input:
    config/first_simulation.yaml

Expected outputs:
    outputs/tables/environment_parameters/
        simulation_settings.csv
        simulation_settings.png
        stream_parameters.csv
        stream_parameters.png
        block_schedule.csv
        block_schedule.png

Run from project root:
    python scripts/export_environment_parameter_tables.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import yaml


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "first_simulation.yaml"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables" / "environment_parameters"


PARAMETER_DESCRIPTIONS = {
    "name": "Name of this simulation run.",
    "random_seed": "Random seed used for reproducible environment generation.",
    "n_trials": "Total number of trials in the simulated environment.",

    "initial_mean": "Initial latent mean of the stream.",
    "changepoint_probability_stable": "Probability of a changepoint on each trial during stable blocks.",
    "changepoint_probability_volatile": "Probability of a changepoint on each trial during volatile blocks.",
    "changepoint_jump_sd_stable": "Standard deviation used when resampling latent mean after stable-block changepoints.",
    "changepoint_jump_sd_volatile": "Standard deviation used when resampling latent mean after volatile-block changepoints.",
    "latent_drift_sd": "Small random-walk drift standard deviation between changepoints.",
    "observation_noise_sd": "Observation noise standard deviation around the latent mean.",

    "start_index": "First trial index included in this block.",
    "stop_index": "First trial index after this block; Python-style half-open interval.",
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


def make_stream_parameters_table(config: dict) -> pd.DataFrame:
    """
    Make table comparing stream A and stream B parameters.
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
            trial_range = f"{start_index}–{stop_index - 1}"
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

    # Header styling
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
    Export all environment parameter tables.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = load_yaml_config(DEFAULT_CONFIG_PATH)

    simulation_table = make_simulation_settings_table(config)
    stream_table = make_stream_parameters_table(config)
    block_table = make_block_schedule_table(config)

    save_table_outputs(
        dataframe=simulation_table,
        output_stem="simulation_settings",
        title="Simulation settings",
        output_dir=OUTPUT_DIR,
        figure_width=12,
    )

    save_table_outputs(
        dataframe=stream_table,
        output_stem="stream_parameters",
        title="Environment stream parameters",
        output_dir=OUTPUT_DIR,
        figure_width=16,
    )

    save_table_outputs(
        dataframe=block_table,
        output_stem="block_schedule",
        title="Block schedule",
        output_dir=OUTPUT_DIR,
        figure_width=14,
    )

    print("\nFinished exporting environment parameter tables.")


if __name__ == "__main__":
    main()