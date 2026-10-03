"""
Fixed-parameter model-recovery smoke test.

Purpose
-------
For one environment, synthetic responses have already been generated from:

    local_only
    global_only
    shared_volatility

This script evaluates each synthetic response dataset under the predictions
of all three candidate perceptual models, WITHOUT fitting any parameters.

For every generating model:

    generating responses
        -> candidate local-only predictions
        -> candidate global-only predictions
        -> candidate shared-volatility predictions

The response likelihood is:

    y_A(t) ~ Normal(mu_A(t), sigma_A^2)
    y_B(t) ~ Normal(mu_B(t), sigma_B^2)

where candidate predictions are:

    mu_A(t) = x_2_expected_mean
    mu_B(t) = x_3_expected_mean

The response SD is taken from the metadata of the generating synthetic
dataset and is held identical across candidate models within that row.

This is NOT final model recovery because perceptual-model parameters are
not yet optimized. It is a diagnostic test of architectural separability.

Expected directory structure
----------------------------

<environment_dir>/
    model_runs/
        local_only/
            node_trajectories.csv
        global_only/
            node_trajectories.csv
        shared_volatility/
            node_trajectories.csv

    synthetic_responses/
        local_only/
            synthetic_continuous_predictions.csv
            response_model_metadata.json
        global_only/
            synthetic_continuous_predictions.csv
            response_model_metadata.json
        shared_volatility/
            synthetic_continuous_predictions.csv
            response_model_metadata.json

Outputs
-------
<output_dir>/
    fixed_parameter_recovery_long.csv
    fixed_parameter_recovery_nll_matrix.csv
    fixed_parameter_recovery_delta_nll_matrix.csv
    fixed_parameter_recovery_summary.csv
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir() and (p / "config").is_dir())

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.response_models.continuous_prediction import (  # noqa: E402
    extract_model_predictions_from_trajectories,
    gaussian_negative_log_likelihood_numpy,
)


MODEL_NAMES = [
    "local_only",
    "global_only",
    "shared_volatility",
]


DEFAULT_ENVIRONMENT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "hgf_walk_outputs"
    / "local_A"
)


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def load_candidate_predictions(
    *,
    environment_dir: Path,
    candidate_model: str,
) -> pd.DataFrame:
    """
    Load one candidate model's pyhgf trajectories and extract its
    pre-outcome predictions.
    """

    trajectories_path = (
        environment_dir
        / "model_runs"
        / candidate_model
        / "node_trajectories.csv"
    )

    if not trajectories_path.exists():
        raise FileNotFoundError(
            f"Could not find candidate trajectories: {trajectories_path}"
        )

    trajectories = pd.read_csv(trajectories_path)

    predictions = extract_model_predictions_from_trajectories(
        trajectories,
        value_node_A=2,
        value_node_B=3,
        prediction_column="expected_mean",
    )

    return predictions


def load_generating_responses(
    *,
    environment_dir: Path,
    generating_model: str,
) -> tuple[pd.DataFrame, dict]:
    """
    Load synthetic participant responses and response-model metadata
    for one generating perceptual model.
    """

    response_dir = (
        environment_dir
        / "synthetic_responses"
        / generating_model
    )

    response_path = (
        response_dir
        / "synthetic_continuous_predictions.csv"
    )

    metadata_path = (
        response_dir
        / "response_model_metadata.json"
    )

    if not response_path.exists():
        raise FileNotFoundError(
            f"Could not find synthetic responses: {response_path}"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Could not find response metadata: {metadata_path}"
        )

    responses = pd.read_csv(response_path)

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    required_columns = {
        "participant_prediction_A",
        "participant_prediction_B",
        "include_in_response_likelihood",
    }

    missing_columns = required_columns - set(responses.columns)

    if missing_columns:
        raise ValueError(
            f"Synthetic responses for {generating_model} are missing "
            f"required columns: {sorted(missing_columns)}"
        )

    metadata_generating_model = metadata.get(
        "generating_perceptual_model"
    )

    if metadata_generating_model != generating_model:
        raise ValueError(
            "Generating-model mismatch.\n"
            f"Directory says: {generating_model}\n"
            f"Metadata says:  {metadata_generating_model}"
        )

    return responses, metadata


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def convert_include_mask(series: pd.Series) -> np.ndarray:
    """
    Convert the saved likelihood-mask column robustly to boolean.
    """

    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=bool)

    if pd.api.types.is_numeric_dtype(series):
        return series.to_numpy(dtype=float) > 0.0

    normalized = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )

    valid_true = {
        "true",
        "1",
        "yes",
    }

    valid_false = {
        "false",
        "0",
        "no",
    }

    unexpected = set(normalized.unique()) - valid_true - valid_false

    if unexpected:
        raise ValueError(
            "Could not interpret include_in_response_likelihood. "
            f"Unexpected values: {sorted(unexpected)}"
        )

    return normalized.isin(valid_true).to_numpy(dtype=bool)


def extract_response_array(
    response_df: pd.DataFrame,
) -> np.ndarray:
    """
    Extract participant predictions as an (n_trials, 2) array.
    """

    responses = response_df[
        [
            "participant_prediction_A",
            "participant_prediction_B",
        ]
    ].to_numpy(dtype=float)

    if np.isnan(responses).any():
        raise ValueError(
            "Synthetic participant predictions contain NaN values."
        )

    return responses


def extract_prediction_array(
    predictions: pd.DataFrame,
) -> np.ndarray:
    """
    Extract candidate predictions as an (n_trials, 2) array.
    """

    prediction_array = predictions[
        [
            "model_prediction_A",
            "model_prediction_B",
        ]
    ].to_numpy(dtype=float)

    if np.isnan(prediction_array).any():
        raise ValueError(
            "Candidate model predictions contain NaN values."
        )

    return prediction_array


# ---------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------

def score_candidate(
    *,
    generating_model: str,
    candidate_model: str,
    response_df: pd.DataFrame,
    response_metadata: dict,
    candidate_predictions: pd.DataFrame,
) -> dict:
    """
    Score one candidate model against one synthetic response dataset.
    """

    if len(response_df) != len(candidate_predictions):
        raise ValueError(
            f"Length mismatch for generating={generating_model}, "
            f"candidate={candidate_model}: "
            f"{len(response_df)} responses vs "
            f"{len(candidate_predictions)} predictions."
        )

    responses = extract_response_array(
        response_df
    )

    predictions = extract_prediction_array(
        candidate_predictions
    )

    include_mask = convert_include_mask(
        response_df[
            "include_in_response_likelihood"
        ]
    )

    try:
        response_sd_A = float(
            response_metadata["response_sd_A"]
        )

        response_sd_B = float(
            response_metadata["response_sd_B"]
        )

    except KeyError as error:
        raise KeyError(
            "Response metadata must contain response_sd_A "
            "and response_sd_B."
        ) from error

    nll = gaussian_negative_log_likelihood_numpy(
        responses=responses,
        predictions=predictions,
        response_sd_A=response_sd_A,
        response_sd_B=response_sd_B,
        include_mask=include_mask,
    )

    residual_A = (
        responses[:, 0]
        - predictions[:, 0]
    )

    residual_B = (
        responses[:, 1]
        - predictions[:, 1]
    )

    included_A = residual_A[
        include_mask
    ]

    included_B = residual_B[
        include_mask
    ]

    rmse_A = float(
        np.sqrt(
            np.mean(
                included_A**2
            )
        )
    )

    rmse_B = float(
        np.sqrt(
            np.mean(
                included_B**2
            )
        )
    )

    combined_squared_errors = np.concatenate(
        [
            included_A**2,
            included_B**2,
        ]
    )

    rmse_combined = float(
        np.sqrt(
            np.mean(
                combined_squared_errors
            )
        )
    )

    return {
        "generating_model": generating_model,
        "candidate_model": candidate_model,
        "response_sd_A": response_sd_A,
        "response_sd_B": response_sd_B,
        "n_trials_total": int(len(response_df)),
        "n_trials_in_likelihood": int(
            include_mask.sum()
        ),
        "negative_log_likelihood": float(nll),
        "rmse_A": rmse_A,
        "rmse_B": rmse_B,
        "rmse_combined": rmse_combined,
        "is_generating_architecture": (
            generating_model == candidate_model
        ),
    }


# ---------------------------------------------------------------------
# Recovery summaries
# ---------------------------------------------------------------------

def build_nll_matrix(
    recovery_long: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create generating-model x candidate-model NLL matrix.
    """

    matrix = recovery_long.pivot(
        index="generating_model",
        columns="candidate_model",
        values="negative_log_likelihood",
    )

    matrix = matrix.reindex(
        index=MODEL_NAMES,
        columns=MODEL_NAMES,
    )

    matrix.index.name = "generating_model"

    return matrix


def build_delta_nll_matrix(
    nll_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert each row to delta NLL relative to the best candidate.

    Best candidate in each row therefore has delta NLL = 0.
    """

    delta = nll_matrix.sub(
        nll_matrix.min(
            axis=1
        ),
        axis=0,
    )

    delta.index.name = "generating_model"

    return delta


def build_recovery_summary(
    nll_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identify the winning candidate for each generating model.
    """

    rows = []

    for generating_model in nll_matrix.index:

        row = nll_matrix.loc[
            generating_model
        ].dropna()

        if row.empty:
            raise ValueError(
                f"No valid candidate scores for {generating_model}."
            )

        sorted_scores = row.sort_values()

        winning_model = str(
            sorted_scores.index[0]
        )

        best_nll = float(
            sorted_scores.iloc[0]
        )

        if len(sorted_scores) >= 2:
            second_best_nll = float(
                sorted_scores.iloc[1]
            )

            nll_margin = (
                second_best_nll
                - best_nll
            )
        else:
            second_best_nll = np.nan
            nll_margin = np.nan

        generating_model_nll = float(
            row.loc[
                generating_model
            ]
        )

        rows.append(
            {
                "generating_model": generating_model,
                "winning_candidate": winning_model,
                "correct_recovery": (
                    winning_model
                    == generating_model
                ),
                "best_nll": best_nll,
                "second_best_nll": second_best_nll,
                "winner_nll_margin": nll_margin,
                "generating_model_nll": generating_model_nll,
                "generating_model_delta_from_best": (
                    generating_model_nll
                    - best_nll
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ---------------------------------------------------------------------
# Main recovery procedure
# ---------------------------------------------------------------------

def run_fixed_parameter_recovery(
    *,
    environment_dir: Path,
    model_names: list[str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Run the full 3 x 3 fixed-parameter recovery test.
    """

    # Load candidate predictions once because the same candidate
    # predictions are used against all three generating datasets.
    candidate_predictions = {}

    for candidate_model in model_names:
        print(
            f"Loading candidate predictions: "
            f"{candidate_model}"
        )

        candidate_predictions[
            candidate_model
        ] = load_candidate_predictions(
            environment_dir=environment_dir,
            candidate_model=candidate_model,
        )

    recovery_rows = []

    for generating_model in model_names:

        print(
            f"\nGenerating model: "
            f"{generating_model}"
        )

        response_df, metadata = (
            load_generating_responses(
                environment_dir=environment_dir,
                generating_model=generating_model,
            )
        )

        for candidate_model in model_names:

            result = score_candidate(
                generating_model=generating_model,
                candidate_model=candidate_model,
                response_df=response_df,
                response_metadata=metadata,
                candidate_predictions=(
                    candidate_predictions[
                        candidate_model
                    ]
                ),
            )

            recovery_rows.append(
                result
            )

            print(
                f"    Candidate "
                f"{candidate_model:<20} "
                f"NLL = "
                f"{result['negative_log_likelihood']:.3f}"
            )

    recovery_long = pd.DataFrame(
        recovery_rows
    )

    nll_matrix = build_nll_matrix(
        recovery_long
    )

    delta_nll_matrix = build_delta_nll_matrix(
        nll_matrix
    )

    recovery_summary = build_recovery_summary(
        nll_matrix
    )

    return (
        recovery_long,
        nll_matrix,
        delta_nll_matrix,
        recovery_summary,
    )


# ---------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------

def save_outputs(
    *,
    recovery_long: pd.DataFrame,
    nll_matrix: pd.DataFrame,
    delta_nll_matrix: pd.DataFrame,
    recovery_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Save recovery results.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    long_path = (
        output_dir
        / "fixed_parameter_recovery_long.csv"
    )

    nll_matrix_path = (
        output_dir
        / "fixed_parameter_recovery_nll_matrix.csv"
    )

    delta_matrix_path = (
        output_dir
        / "fixed_parameter_recovery_delta_nll_matrix.csv"
    )

    summary_path = (
        output_dir
        / "fixed_parameter_recovery_summary.csv"
    )

    recovery_long.to_csv(
        long_path,
        index=False,
    )

    nll_matrix.to_csv(
        nll_matrix_path
    )

    delta_nll_matrix.to_csv(
        delta_matrix_path
    )

    recovery_summary.to_csv(
        summary_path,
        index=False,
    )

    print(
        f"\nSaved long recovery results: {long_path}"
    )

    print(
        f"Saved NLL matrix:             {nll_matrix_path}"
    )

    print(
        f"Saved delta-NLL matrix:       {delta_matrix_path}"
    )

    print(
        f"Saved recovery summary:       {summary_path}"
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build command-line interface.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run fixed-parameter model-recovery smoke test."
        )
    )

    parser.add_argument(
        "--environment-dir",
        type=str,
        default=str(
            DEFAULT_ENVIRONMENT_DIR
        ),
        help=(
            "Environment directory containing model_runs/ "
            "and synthetic_responses/."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output directory. If omitted, results are saved under "
            "<environment-dir>/model_recovery/fixed_parameter/."
        ),
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=MODEL_NAMES,
        help="Models to include in recovery.",
    )

    return parser


def main() -> None:
    """
    Run fixed-parameter model recovery.
    """

    parser = build_arg_parser()
    args = parser.parse_args()

    environment_dir = Path(
        args.environment_dir
    )

    if args.output_dir is None:
        output_dir = (
            environment_dir
            / "model_recovery"
            / "fixed_parameter"
        )
    else:
        output_dir = Path(
            args.output_dir
        )

    print(
        "\nFixed-parameter model recovery"
    )

    print(
        f"Environment directory: "
        f"{environment_dir}"
    )

    (
        recovery_long,
        nll_matrix,
        delta_nll_matrix,
        recovery_summary,
    ) = run_fixed_parameter_recovery(
        environment_dir=environment_dir,
        model_names=list(args.models),
    )

    save_outputs(
        recovery_long=recovery_long,
        nll_matrix=nll_matrix,
        delta_nll_matrix=delta_nll_matrix,
        recovery_summary=recovery_summary,
        output_dir=output_dir,
    )

    print(
        "\nNLL matrix"
    )

    print(
        nll_matrix.to_string()
    )

    print(
        "\nDelta-NLL matrix"
    )

    print(
        delta_nll_matrix.to_string()
    )

    print(
        "\nRecovery summary"
    )

    print(
        recovery_summary.to_string(
            index=False
        )
    )

    n_correct = int(
        recovery_summary[
            "correct_recovery"
        ].sum()
    )

    n_total = int(
        len(recovery_summary)
    )

    print(
        f"\nCorrect recoveries: "
        f"{n_correct}/{n_total}"
    )

    print(
        "\nFinished fixed-parameter recovery."
    )


if __name__ == "__main__":
    main()