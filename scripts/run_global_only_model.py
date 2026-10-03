"""
Run the global-only pyhgf model on the generated two-stream environment.

Run from project root:
    python scripts/run_global_only_model.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir() and (p / "config").is_dir())

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import (
    DEFAULT_ENVIRONMENT_PATH,
    DEFAULT_MODEL_RUNS_DIR,
)


from src.models.global_only import (  # noqa: E402
    MODEL_NAME,
    build_model,
    get_node_ids,
)


@dataclass(frozen=True)
class PreparedModelInput:
    input_data: np.ndarray
    labels: pd.DataFrame
    stream_names: tuple[str, str]


def load_environment_long(environment_path: str | Path) -> pd.DataFrame:
    environment_path = Path(environment_path)

    if not environment_path.exists():
        raise FileNotFoundError(f"Could not find environment file: {environment_path}")

    environment_df = pd.read_csv(environment_path)

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
        raise ValueError(
            f"Environment file is missing required columns: {sorted(missing_columns)}"
        )

    return environment_df


def make_labels_table(environment_df: pd.DataFrame) -> pd.DataFrame:
    base = (
        environment_df[environment_df["stream"] == "A"]
        .loc[:, ["trial_index", "trial", "block"]]
        .sort_values("trial_index")
        .reset_index(drop=True)
    )

    label_parts = [base]

    for value_column in ["regime", "observation", "latent_mean", "changepoint"]:
        table = environment_df.pivot(
            index="trial_index",
            columns="stream",
            values=value_column,
        ).sort_index()

        table = table.loc[:, ["A", "B"]]
        table.columns = [f"{value_column}_A", f"{value_column}_B"]
        table = table.reset_index(drop=True)

        label_parts.append(table)

    return pd.concat(label_parts, axis=1)


def prepare_model_input(environment_df: pd.DataFrame) -> PreparedModelInput:
    expected_streams = ("A", "B")
    actual_streams = tuple(sorted(environment_df["stream"].unique()))

    if actual_streams != expected_streams:
        raise ValueError(
            f"Expected streams {expected_streams}, but found {actual_streams}."
        )

    observation_table = environment_df.pivot(
        index="trial_index",
        columns="stream",
        values="observation",
    ).sort_index()

    observation_table = observation_table.loc[:, list(expected_streams)]

    input_data = observation_table.to_numpy(dtype=float)

    if input_data.ndim != 2:
        raise ValueError(f"input_data should be 2D, got shape {input_data.shape}.")

    if input_data.shape[1] != 2:
        raise ValueError(
            f"input_data should have 2 columns, got shape {input_data.shape}."
        )

    if np.isnan(input_data).any():
        raise ValueError("input_data contains NaN values.")

    labels = make_labels_table(environment_df)

    if len(labels) != input_data.shape[0]:
        raise ValueError(
            f"labels has {len(labels)} rows, but input_data has "
            f"{input_data.shape[0]} rows."
        )

    return PreparedModelInput(
        input_data=input_data,
        labels=labels,
        stream_names=expected_streams,
    )


def flatten_dataframe_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()

    if isinstance(dataframe.columns, pd.MultiIndex):
        dataframe.columns = [
            "__".join(str(part) for part in column if str(part) != "")
            for column in dataframe.columns
        ]
    else:
        dataframe.columns = [str(column) for column in dataframe.columns]

    return dataframe


def save_prepared_input(prepared: PreparedModelInput, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    input_matrix_path = output_dir / "model_input_data.npy"
    input_csv_path = output_dir / "model_input_data.csv"
    labels_path = output_dir / "model_labels.csv"
    metadata_path = output_dir / "model_input_metadata.json"

    np.save(input_matrix_path, prepared.input_data)

    input_df = pd.DataFrame(
        prepared.input_data,
        columns=[f"observation_{name}" for name in prepared.stream_names],
    )
    input_df.insert(0, "trial_index", np.arange(prepared.input_data.shape[0]))

    input_df.to_csv(input_csv_path, index=False)
    prepared.labels.to_csv(labels_path, index=False)

    metadata = {
        "input_shape": list(prepared.input_data.shape),
        "stream_names": list(prepared.stream_names),
        "column_mapping": {
            "input_data[:, 0]": "stream A observation",
            "input_data[:, 1]": "stream B observation",
        },
        "important_note": (
            "Only observations are fed into pyhgf. Labels are saved for later "
            "analysis only."
        ),
    }

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"Saved model input matrix: {input_matrix_path}")
    print(f"Saved model input CSV:    {input_csv_path}")
    print(f"Saved labels:             {labels_path}")


def run_model(
    *,
    input_data: np.ndarray,
    output_dir: Path,
    model_kwargs: dict,
) -> None:
    model_dir = output_dir / MODEL_NAME
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning model: {MODEL_NAME}")

    model = build_model(**model_kwargs)
    maybe_updated_model = model.input_data(input_data=input_data)

    if maybe_updated_model is not None:
        model = maybe_updated_model

    trajectories = flatten_dataframe_columns(model.to_pandas())

    trajectories_path = model_dir / "node_trajectories.csv"
    trajectories.to_csv(trajectories_path, index=False)

    node_ids = get_node_ids()
    node_ids_path = model_dir / "node_ids.json"

    with node_ids_path.open("w", encoding="utf-8") as file:
        json.dump(node_ids, file, indent=2)

    metadata = {
        "model_name": MODEL_NAME,
        "node_ids": node_ids,
        "input_shape": list(input_data.shape),
        "model_kwargs": model_kwargs,
        "saved_files": {
            "node_trajectories": str(trajectories_path),
            "node_ids": str(node_ids_path),
        },
    }

    metadata_path = model_dir / "run_metadata.json"

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"Saved trajectories: {trajectories_path}")
    print(f"Saved node IDs:     {node_ids_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run global-only pyhgf model.")

    parser.add_argument(
        "--environment",
        type=str,
        default=str(DEFAULT_ENVIRONMENT_PATH),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_MODEL_RUNS_DIR),
    )

    parser.add_argument("--input-precision", type=float, default=100.0)

    parser.add_argument(
        "--update-type",
        type=str,
        default="eHGF",
        choices=["eHGF", "standard", "unbounded"],
    )

    parser.add_argument("--max-posterior-precision", type=float, default=1e10)

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    environment_df = load_environment_long(args.environment)
    prepared = prepare_model_input(environment_df)

    print("\nPrepared pyhgf input.")
    print(f"input_data shape: {prepared.input_data.shape}")
    print("input_data[:, 0] = stream A observations")
    print("input_data[:, 1] = stream B observations\n")

    output_dir = Path(args.output_dir)

    save_prepared_input(prepared, output_dir)

    model_kwargs = {
        "input_precision": args.input_precision,
        "update_type": args.update_type,
        "max_posterior_precision": args.max_posterior_precision,
    }

    run_model(
        input_data=prepared.input_data,
        output_dir=output_dir,
        model_kwargs=model_kwargs,
    )

    print("\nFinished global-only model run.")


if __name__ == "__main__":
    main()