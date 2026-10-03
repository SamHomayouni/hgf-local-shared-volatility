"""
Simulate continuous participant predictions from a fitted/run pyhgf model.

This script uses the first response model for the project:

    y_A(t) ~ Normal(mu_hat_A(t), sigma_resp_A^2)
    y_B(t) ~ Normal(mu_hat_B(t), sigma_resp_B^2)

where:
    mu_hat_A(t) = x_2_expected_mean from pyhgf trajectories
    mu_hat_B(t) = x_3_expected_mean from pyhgf trajectories

Inputs
------
A model run directory containing:

    model_labels.csv
    <model_name>/node_trajectories.csv

For example:

    outputs/hgf_walk_outputs/model_runs/shared_volatility/node_trajectories.csv

Outputs
-------
    synthetic_continuous_predictions.csv
    response_function_inputs.npy
    response_model_metadata.json

Run from project root
---------------------
Example:

    python scripts/simulate_continuous_predictions.py \\
        --model-runs-dir outputs/hgf_walk_outputs/model_runs \\
        --model-name shared_volatility \\
        --output-dir outputs/hgf_walk_outputs/synthetic_responses/shared_volatility \\
        --response-sd 0.10
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.response_models.continuous_prediction import (  # noqa: E402
    ContinuousPredictionResponseParameters,
    add_behavioral_learning_rates,
    extract_model_predictions_from_trajectories,
    gaussian_negative_log_likelihood_numpy,
    make_response_function_inputs,
    simulate_continuous_prediction_responses,
)


MODEL_NAMES = [
    "local_only",
    "global_only",
    "shared_volatility",
]


DEFAULT_MODEL_RUNS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "hgf_walk_outputs"
    / "model_runs"
)


DEFAULT_OUTPUT_BASE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "hgf_walk_outputs"
    / "synthetic_responses"
)


def load_labels(model_runs_dir: str | Path) -> pd.DataFrame:
    """
    Load model_labels.csv from a model-run directory.
    """

    labels_path = Path(model_runs_dir) / "model_labels.csv"

    if not labels_path.exists():
        raise FileNotFoundError(f"Could not find labels file: {labels_path}")

    labels = pd.read_csv(labels_path)

    required_columns = {
        "trial_index",
        "trial",
        "block",
        "regime_A",
        "regime_B",
        "observation_A",
        "observation_B",
    }

    missing_columns = required_columns - set(labels.columns)

    if missing_columns:
        raise ValueError(
            f"model_labels.csv is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    return labels


def load_trajectories(
    *,
    model_runs_dir: str | Path,
    model_name: str,
) -> pd.DataFrame:
    """
    Load node_trajectories.csv for one model.
    """

    trajectories_path = (
        Path(model_runs_dir)
        / model_name
        / "node_trajectories.csv"
    )

    if not trajectories_path.exists():
        raise FileNotFoundError(
            f"Could not find node trajectories: {trajectories_path}"
        )

    return pd.read_csv(trajectories_path)


def build_response_dataframe(
    *,
    labels: pd.DataFrame,
    predictions: pd.DataFrame,
    simulated_responses: pd.DataFrame,
    include_min_trial_index: int,
) -> pd.DataFrame:
    """
    Combine labels, model predictions, and simulated participant predictions.
    """

    if len(labels) != len(predictions):
        raise ValueError(
            f"labels has {len(labels)} rows, predictions has "
            f"{len(predictions)} rows."
        )

    if len(labels) != len(simulated_responses):
        raise ValueError(
            f"labels has {len(labels)} rows, simulated_responses has "
            f"{len(simulated_responses)} rows."
        )

    columns_from_labels = [
        "trial_index",
        "trial",
        "block",
        "regime_A",
        "regime_B",
        "observation_A",
        "observation_B",
    ]

    optional_label_columns = [
        "latent_mean_A",
        "latent_mean_B",
        "changepoint_A",
        "changepoint_B",
    ]

    for column in optional_label_columns:
        if column in labels.columns:
            columns_from_labels.append(column)

    response_df = pd.concat(
        [
            labels.loc[:, columns_from_labels].reset_index(drop=True),
            predictions.reset_index(drop=True),
            simulated_responses.reset_index(drop=True),
        ],
        axis=1,
    )

    response_df["include_in_response_likelihood"] = (
        response_df["trial_index"] >= include_min_trial_index
    )

    response_df = add_behavioral_learning_rates(response_df)

    return response_df


def save_outputs(
    *,
    response_df: pd.DataFrame,
    response_function_inputs: np.ndarray,
    metadata: dict,
    output_dir: str | Path,
) -> None:
    """
    Save synthetic responses and metadata.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    response_path = output_dir / "synthetic_continuous_predictions.csv"
    response_inputs_path = output_dir / "response_function_inputs.npy"
    metadata_path = output_dir / "response_model_metadata.json"

    response_df.to_csv(response_path, index=False)
    np.save(response_inputs_path, response_function_inputs)

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"Saved synthetic responses:      {response_path}")
    print(f"Saved response-function inputs: {response_inputs_path}")
    print(f"Saved metadata:                 {metadata_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build command-line interface.
    """

    parser = argparse.ArgumentParser(
        description="Simulate continuous prediction responses from pyhgf trajectories."
    )

    parser.add_argument(
        "--model-runs-dir",
        type=str,
        default=str(DEFAULT_MODEL_RUNS_DIR),
        help="Directory containing model_labels.csv and model subdirectories.",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="shared_volatility",
        choices=MODEL_NAMES,
        help="Which perceptual model should generate synthetic responses.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory where synthetic response files should be saved. "
            "If omitted, a directory under outputs/hgf_walk_outputs/"
            "synthetic_responses/ is used."
        ),
    )

    parser.add_argument(
        "--response-sd",
        type=float,
        default=0.10,
        help=(
            "Shared response-noise standard deviation for streams A and B. "
            "Ignored for B if --response-sd-B is provided."
        ),
    )

    parser.add_argument(
        "--response-sd-B",
        type=float,
        default=None,
        help=(
            "Optional stream-B response-noise standard deviation. "
            "If omitted, --response-sd is used for both streams."
        ),
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=1234,
        help="Random seed for synthetic response noise.",
    )

    parser.add_argument(
        "--prediction-column",
        type=str,
        default="expected_mean",
        choices=["expected_mean", "mean"],
        help=(
            "Which pyhgf trajectory quantity to use as the response-model "
            "prediction. Default is expected_mean."
        ),
    )

    parser.add_argument(
        "--include-min-trial-index",
        type=int,
        default=1,
        help=(
            "First trial_index included in the response likelihood. "
            "Default 1 excludes trial 0."
        ),
    )

    return parser


def main() -> None:
    """
    Simulate synthetic continuous predictions from one perceptual model run.
    """

    parser = build_arg_parser()
    args = parser.parse_args()

    model_runs_dir = Path(args.model_runs_dir)

    if args.output_dir is None:
        output_dir = (
            DEFAULT_OUTPUT_BASE_DIR
            / f"{args.model_name}_sigma{args.response_sd:g}"
        )
    else:
        output_dir = Path(args.output_dir)

    response_sd_A = float(args.response_sd)
    response_sd_B = (
        float(args.response_sd_B)
        if args.response_sd_B is not None
        else float(args.response_sd)
    )

    response_parameters = ContinuousPredictionResponseParameters(
        response_sd_A=response_sd_A,
        response_sd_B=response_sd_B,
    )

    labels = load_labels(model_runs_dir)
    trajectories = load_trajectories(
        model_runs_dir=model_runs_dir,
        model_name=args.model_name,
    )

    predictions = extract_model_predictions_from_trajectories(
        trajectories,
        prediction_column=args.prediction_column,
    )

    rng = np.random.default_rng(args.random_seed)

    simulated_responses = simulate_continuous_prediction_responses(
        predictions=predictions,
        response_parameters=response_parameters,
        rng=rng,
    )

    response_df = build_response_dataframe(
        labels=labels,
        predictions=predictions,
        simulated_responses=simulated_responses,
        include_min_trial_index=args.include_min_trial_index,
    )

    response_function_inputs = make_response_function_inputs(response_df)

    responses_array = response_df[
        [
            "participant_prediction_A",
            "participant_prediction_B",
        ]
    ].to_numpy(dtype=float)

    predictions_array = response_df[
        [
            "model_prediction_A",
            "model_prediction_B",
        ]
    ].to_numpy(dtype=float)

    include_mask = response_df[
        "include_in_response_likelihood"
    ].to_numpy(dtype=bool)

    negative_log_likelihood = gaussian_negative_log_likelihood_numpy(
        responses=responses_array,
        predictions=predictions_array,
        response_sd_A=response_sd_A,
        response_sd_B=response_sd_B,
        include_mask=include_mask,
    )

    metadata = {
        "generating_perceptual_model": args.model_name,
        "model_runs_dir": str(model_runs_dir),
        "prediction_column": args.prediction_column,
        "value_node_A": 2,
        "value_node_B": 3,
        "response_model": (
            "participant_prediction_s ~ Normal("
            "model_prediction_s, response_sd_s**2)"
        ),
        "response_sd_A": response_sd_A,
        "response_sd_B": response_sd_B,
        "random_seed": args.random_seed,
        "include_min_trial_index": args.include_min_trial_index,
        "n_trials_total": int(len(response_df)),
        "n_trials_in_likelihood": int(include_mask.sum()),
        "negative_log_likelihood_under_generating_predictions": (
            negative_log_likelihood
        ),
        "response_function_inputs_columns": [
            "participant_prediction_A",
            "participant_prediction_B",
            "include_in_response_likelihood",
        ],
        "important_timing_note": (
            "Responses are generated around x_2_expected_mean and "
            "x_3_expected_mean, interpreted as pre-outcome predictions for "
            "streams A and B."
        ),
    }

    save_outputs(
        response_df=response_df,
        response_function_inputs=response_function_inputs,
        metadata=metadata,
        output_dir=output_dir,
    )

    print("\nContinuous prediction responses generated successfully.")
    print(f"Generating perceptual model: {args.model_name}")
    print(f"Response SD A: {response_sd_A}")
    print(f"Response SD B: {response_sd_B}")
    print(
        "Negative log-likelihood under generating predictions: "
        f"{negative_log_likelihood:.3f}"
    )


if __name__ == "__main__":
    main()