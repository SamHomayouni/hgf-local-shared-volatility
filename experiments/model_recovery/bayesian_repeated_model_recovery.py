"""
Repeated Bayesian model recovery across response-noise seeds.

This script reuses the already validated fitting code in:

    experiments/model_recovery/bayesian_model_recovery.py

For every supplied environment directory, generating model, and response
seed it:

1. loads the generating model's trial-wise predictions;
2. generates a new synthetic participant response dataset by adding
   Gaussian response noise;
3. fits all three candidate architectures to exactly the same responses;
4. computes PSIS-LOO for each candidate;
5. stores winner, ELPD values, Pareto-k diagnostics, posterior sampling
   diagnostics, and pairwise Delta-ELPD values;
6. aggregates all completed datasets into recovery/confusion matrices and
   Delta-ELPD summaries/plots.

Important design choice
-----------------------
For a given environment and response seed, the same standardized response
noise realization is used for all three generating models. This gives a
controlled comparison across generators while retaining independent noise
for streams A and B.

This stage varies response noise while holding each supplied environment
realization fixed. Later, the same script can aggregate across multiple
independently generated environment directories as long as each directory
contains the standard model_runs/ and synthetic_responses/ structure.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import gc
import importlib.util
import json
import sys
import traceback
from typing import Iterable

import arviz as az
import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =====================================================================
# Project setup
# =====================================================================

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir() and (p / "config").is_dir())

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MODEL_NAMES = (
    "local_only",
    "global_only",
    "shared_volatility",
)

MODEL_LABELS = {
    "local_only": "Local",
    "global_only": "Global",
    "shared_volatility": "Hybrid",
}

PAIR_DEFINITIONS = (
    ("local_only", "global_only"),
    ("local_only", "shared_volatility"),
    ("global_only", "shared_volatility"),
)


# =====================================================================
# Load the already-tested single-dataset recovery module
# =====================================================================


def load_recovery_module():
    """Load experiments/model_recovery/bayesian_model_recovery.py without requiring scripts
    to be an importable Python package.
    """

    module_path = (
        PROJECT_ROOT
        / "experiments"
        / "model_recovery"
        / "bayesian_model_recovery.py"
    )

    if not module_path.exists():
        raise FileNotFoundError(
            "Could not find the existing recovery script: "
            f"{module_path}\n"
            "Create experiments/model_recovery/bayesian_model_recovery.py first."
        )

    spec = importlib.util.spec_from_file_location(
        "bayesian_model_recovery",
        module_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load module specification for {module_path}."
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RECOVERY = load_recovery_module()


# =====================================================================
# Small utilities
# =====================================================================


def safe_name(value: str) -> str:
    """Filesystem-safe identifier."""

    return (
        value
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )



def json_dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)



def to_builtin(value):
    """Convert NumPy/Pandas scalars to JSON-safe builtins."""

    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


# =====================================================================
# Generating predictions and response-noise metadata
# =====================================================================


def load_generating_predictions(
    *,
    environment_dir: Path,
    generating_model: str,
    response_sd_override: float | None,
) -> tuple[np.ndarray, float, float, dict]:
    """Load the pre-outcome predictions that were already exported for a
    generating model.

    We deliberately use the saved model_prediction_A/B columns rather than
    the original participant_prediction columns, because each batch seed
    must receive a fresh response-noise realization.
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
            "Could not find generating predictions: "
            f"{response_path}"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            "Could not find response metadata: "
            f"{metadata_path}"
        )

    response_df = pd.read_csv(response_path)

    required_columns = {
        "model_prediction_A",
        "model_prediction_B",
    }

    missing = required_columns - set(response_df.columns)

    if missing:
        raise ValueError(
            f"{response_path} is missing columns: {sorted(missing)}"
        )

    predictions = response_df[
        [
            "model_prediction_A",
            "model_prediction_B",
        ]
    ].to_numpy(dtype=float)

    if not np.isfinite(predictions).all():
        raise ValueError(
            f"Generating predictions contain non-finite values: "
            f"{response_path}"
        )

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    actual_generator = metadata.get(
        "generating_perceptual_model"
    )

    if actual_generator != generating_model:
        raise ValueError(
            f"Requested generator {generating_model!r}, but "
            f"metadata reports {actual_generator!r}."
        )

    if response_sd_override is not None:
        response_sd_A = float(response_sd_override)
        response_sd_B = float(response_sd_override)
    else:
        if "response_sd_A" not in metadata or "response_sd_B" not in metadata:
            raise KeyError(
                "response_model_metadata.json must contain response_sd_A "
                "and response_sd_B unless --response-sd is supplied."
            )

        response_sd_A = float(metadata["response_sd_A"])
        response_sd_B = float(metadata["response_sd_B"])

    if response_sd_A <= 0.0 or response_sd_B <= 0.0:
        raise ValueError("Response standard deviations must be positive.")

    return (
        predictions,
        response_sd_A,
        response_sd_B,
        metadata,
    )



def make_standardized_response_noise(
    *,
    n_trials: int,
    response_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate standardized A/B response noise.

    The same z_A and z_B arrays are reused across generating models for a
    given environment and response seed.
    """

    rng = np.random.default_rng(response_seed)

    z_A = rng.normal(
        loc=0.0,
        scale=1.0,
        size=n_trials,
    )

    z_B = rng.normal(
        loc=0.0,
        scale=1.0,
        size=n_trials,
    )

    return z_A, z_B



def simulate_responses(
    *,
    predictions: np.ndarray,
    z_A: np.ndarray,
    z_B: np.ndarray,
    response_sd_A: float,
    response_sd_B: float,
) -> np.ndarray:
    """Create one two-stream synthetic participant response matrix."""

    n_trials = len(predictions)

    if len(z_A) != n_trials or len(z_B) != n_trials:
        raise ValueError(
            "Noise arrays and prediction arrays have different lengths."
        )

    responses = np.column_stack(
        [
            predictions[:, 0]
            + response_sd_A * z_A,
            predictions[:, 1]
            + response_sd_B * z_B,
        ]
    )

    if not np.isfinite(responses).all():
        raise ValueError("Simulated responses contain non-finite values.")

    return responses


# =====================================================================
# LOO evaluation
# =====================================================================


def loo_for_models(
    fitted_models: dict[str, az.InferenceData],
) -> tuple[
    dict[str, az.ELPDData],
    pd.DataFrame,
    pd.DataFrame,
]:
    """Compute model-wise PSIS-LOO and the ArviZ comparison table."""

    loo_objects: dict[str, az.ELPDData] = {}
    loo_rows: list[dict] = []

    for model_name, idata in fitted_models.items():
        loo = az.loo(
            idata,
            pointwise=True,
            var_name="response",
        )

        loo_objects[model_name] = loo

        pareto_k = np.asarray(loo.pareto_k, dtype=float)

        loo_rows.append(
            {
                "candidate_model": model_name,
                "elpd_loo": float(loo.elpd_loo),
                "se": float(loo.se),
                "p_loo": float(loo.p_loo),
                "warning": bool(loo.warning),
                "max_pareto_k": float(np.max(pareto_k)),
                "n_pareto_k_gt_0_7": int(np.sum(pareto_k > 0.7)),
                "n_pareto_k_gt_1_0": int(np.sum(pareto_k > 1.0)),
            }
        )

    loo_df = (
        pd.DataFrame(loo_rows)
        .sort_values(
            "elpd_loo",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    comparison = az.compare(
        fitted_models,
        ic="loo",
        method="stacking",
        scale="log",
        var_name="response",
    )

    comparison = (
        comparison
        .reset_index()
        .rename(columns={"index": "candidate_model"})
    )

    return loo_objects, loo_df, comparison



def pairwise_loo_rows(
    loo_objects: dict[str, az.ELPDData],
) -> pd.DataFrame:
    """Compute signed pairwise Delta-ELPD and its standard error.

    Delta-ELPD is always:

        ELPD(first_model) - ELPD(second_model)

    Therefore positive values favor the first model in the pair.
    """

    rows: list[dict] = []

    for first_model, second_model in PAIR_DEFINITIONS:
        first = loo_objects[first_model]
        second = loo_objects[second_model]

        first_i = np.asarray(first.loo_i, dtype=float).reshape(-1)
        second_i = np.asarray(second.loo_i, dtype=float).reshape(-1)

        if first_i.shape != second_i.shape:
            raise ValueError(
                "Pointwise LOO arrays differ in shape for pair "
                f"{first_model} vs {second_model}."
            )

        pointwise_difference = first_i - second_i

        delta_elpd = float(
            first.elpd_loo
            - second.elpd_loo
        )

        if len(pointwise_difference) > 1:
            dse = float(
                np.sqrt(
                    len(pointwise_difference)
                    * np.var(
                        pointwise_difference,
                        ddof=1,
                    )
                )
            )
        else:
            dse = float("nan")

        if np.isfinite(dse) and dse > 0.0:
            z_like = delta_elpd / dse
        else:
            z_like = float("nan")

        rows.append(
            {
                "first_model": first_model,
                "second_model": second_model,
                "pair": f"{first_model}_minus_{second_model}",
                "delta_elpd": delta_elpd,
                "dse": dse,
                "delta_over_dse": z_like,
                "favored_model": (
                    first_model
                    if delta_elpd > 0.0
                    else second_model
                    if delta_elpd < 0.0
                    else "tie"
                ),
            }
        )

    return pd.DataFrame(rows)


# =====================================================================
# Fit one synthetic dataset
# =====================================================================


def fit_one_dataset(
    *,
    environment_dir: Path,
    environment_name: str,
    generating_model: str,
    response_seed: int,
    responses: np.ndarray,
    response_sd_A: float,
    response_sd_B: float,
    dataset_dir: Path,
    input_precision: float,
    update_type: str,
    max_posterior_precision: float,
    draws: int,
    tune: int,
    chains: int,
    cores: int,
    target_accept: float,
    sampler_seed_base: int,
    save_fit_details: bool,
    save_idata: bool,
) -> dict:
    """Fit all three candidate architectures to one response dataset."""

    fitted_models: dict[str, az.InferenceData] = {}
    diagnostics_rows: list[dict] = []

    candidate_root = dataset_dir / "candidates"

    for candidate_index, candidate_name in enumerate(MODEL_NAMES):
        print(
            f"        fitting candidate: {candidate_name}"
        )

        candidate_seed = (
            sampler_seed_base
            + candidate_index
        )

        _, idata = RECOVERY.fit_candidate(
            candidate_name=candidate_name,
            input_data=RECOVERY.load_model_input(
                environment_dir
            ),
            responses=responses,
            input_precision=input_precision,
            update_type=update_type,
            max_posterior_precision=max_posterior_precision,
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            random_seed=candidate_seed,
        )

        fitted_models[candidate_name] = idata

        summary, diagnostics = RECOVERY.candidate_summary(
            candidate_name=candidate_name,
            idata=idata,
        )

        diagnostics_rows.append(
            {
                "environment": environment_name,
                "environment_path": str(environment_dir),
                "generating_model": generating_model,
                "response_seed": int(response_seed),
                **{
                    key: to_builtin(value)
                    for key, value in diagnostics.items()
                },
            }
        )

        if save_fit_details or save_idata:
            candidate_dir = (
                candidate_root
                / candidate_name
            )
            candidate_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            if save_fit_details:
                summary.to_csv(
                    candidate_dir
                    / "posterior_summary.csv"
                )

                json_dump(
                    candidate_dir
                    / "diagnostics.json",
                    diagnostics,
                )

            if save_idata:
                idata.to_netcdf(
                    candidate_dir
                    / "inference_data.nc"
                )

    # -----------------------------------------------------------------
    # LOO and model comparison
    # -----------------------------------------------------------------

    (
        loo_objects,
        loo_df,
        comparison_df,
    ) = loo_for_models(
        fitted_models
    )

    pairwise_df = pairwise_loo_rows(
        loo_objects
    )

    # -----------------------------------------------------------------
    # Add dataset identifiers
    # -----------------------------------------------------------------

    id_columns = {
        "environment": environment_name,
        "environment_path": str(environment_dir),
        "generating_model": generating_model,
        "response_seed": int(response_seed),
    }

    for key, value in reversed(list(id_columns.items())):
        loo_df.insert(0, key, value)
        comparison_df.insert(0, key, value)
        pairwise_df.insert(0, key, value)

    diagnostics_df = pd.DataFrame(
        diagnostics_rows
    )

    # -----------------------------------------------------------------
    # Recovery verdict
    # -----------------------------------------------------------------

    best_model = str(
        comparison_df.iloc[0][
            "candidate_model"
        ]
    )

    recovered_correctly = bool(
        best_model == generating_model
    )

    sorted_elpd = (
        loo_df
        .sort_values(
            "elpd_loo",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    winner_margin = float(
        sorted_elpd.iloc[0]["elpd_loo"]
        - sorted_elpd.iloc[1]["elpd_loo"]
    )

    generator_elpd = float(
        loo_df.loc[
            loo_df["candidate_model"]
            == generating_model,
            "elpd_loo",
        ].iloc[0]
    )

    best_elpd = float(
        sorted_elpd.iloc[0]["elpd_loo"]
    )

    generator_gap_from_best = float(
        best_elpd - generator_elpd
    )

    # Pair of greatest current scientific interest.
    gh_row = pairwise_df.loc[
        pairwise_df["pair"]
        == "global_only_minus_shared_volatility"
    ].iloc[0]

    dataset_result = {
        "environment": environment_name,
        "environment_path": str(environment_dir),
        "generating_model": generating_model,
        "response_seed": int(response_seed),
        "response_sd_A": float(response_sd_A),
        "response_sd_B": float(response_sd_B),
        "best_model": best_model,
        "recovered_correctly": recovered_correctly,
        "winner_margin_elpd": winner_margin,
        "generator_gap_from_best_elpd": generator_gap_from_best,
        "global_minus_hybrid_delta_elpd": float(
            gh_row["delta_elpd"]
        ),
        "global_minus_hybrid_dse": float(
            gh_row["dse"]
        ),
        "global_minus_hybrid_delta_over_dse": float(
            gh_row["delta_over_dse"]
        ),
        "n_trials_total": int(len(responses)),
        "n_trials_in_likelihood": int(len(responses) - 1),
        "sampling": {
            "draws": int(draws),
            "tune": int(tune),
            "chains": int(chains),
            "cores": int(cores),
            "target_accept": float(target_accept),
            "sampler_seed_base": int(sampler_seed_base),
        },
    }

    # -----------------------------------------------------------------
    # Save per-dataset checkpoint
    # -----------------------------------------------------------------

    dataset_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    loo_df.to_csv(
        dataset_dir / "loo.csv",
        index=False,
    )

    comparison_df.to_csv(
        dataset_dir / "comparison.csv",
        index=False,
    )

    pairwise_df.to_csv(
        dataset_dir / "pairwise_delta_elpd.csv",
        index=False,
    )

    diagnostics_df.to_csv(
        dataset_dir / "fit_diagnostics.csv",
        index=False,
    )

    json_dump(
        dataset_dir / "dataset_result.json",
        dataset_result,
    )

    # -----------------------------------------------------------------
    # Release posterior objects before next dataset
    # -----------------------------------------------------------------

    del fitted_models
    del loo_objects
    gc.collect()

    return dataset_result


# =====================================================================
# Aggregation
# =====================================================================


def read_all_csvs(
    output_dir: Path,
    filename: str,
) -> pd.DataFrame:
    paths = sorted(
        output_dir.glob(
            f"runs/**/{filename}"
        )
    )

    if not paths:
        return pd.DataFrame()

    frames = [
        pd.read_csv(path)
        for path in paths
    ]

    return pd.concat(
        frames,
        ignore_index=True,
    )



def read_all_dataset_results(
    output_dir: Path,
) -> pd.DataFrame:
    rows: list[dict] = []

    for path in sorted(
        output_dir.glob(
            "runs/**/dataset_result.json"
        )
    ):
        with path.open("r", encoding="utf-8") as file:
            result = json.load(file)

        # Flatten only the top-level fields needed for aggregate tables.
        rows.append(
            {
                key: value
                for key, value in result.items()
                if key != "sampling"
            }
        )

    return pd.DataFrame(rows)



def make_confusion_tables(
    results_df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Overall counts, overall row-normalized proportions, and long-form
    environment-specific confusion results.
    """

    model_order = list(MODEL_NAMES)

    counts = pd.crosstab(
        results_df["generating_model"],
        results_df["best_model"],
    )

    counts = counts.reindex(
        index=model_order,
        columns=model_order,
        fill_value=0,
    )

    proportions = counts.div(
        counts.sum(axis=1).replace(0, np.nan),
        axis=0,
    )

    env_rows: list[dict] = []

    for environment, env_df in results_df.groupby(
        "environment",
        sort=True,
    ):
        env_counts = pd.crosstab(
            env_df["generating_model"],
            env_df["best_model"],
        ).reindex(
            index=model_order,
            columns=model_order,
            fill_value=0,
        )

        for generator in model_order:
            row_total = int(
                env_counts.loc[generator].sum()
            )

            for winner in model_order:
                count = int(
                    env_counts.loc[
                        generator,
                        winner,
                    ]
                )

                proportion = (
                    count / row_total
                    if row_total > 0
                    else np.nan
                )

                env_rows.append(
                    {
                        "environment": environment,
                        "generating_model": generator,
                        "recovered_model": winner,
                        "count": count,
                        "proportion": proportion,
                    }
                )

    by_environment = pd.DataFrame(
        env_rows
    )

    return (
        counts,
        proportions,
        by_environment,
    )



def make_recovery_summary(
    results_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    groupings: list[tuple[str, Iterable[tuple]]] = [
        (
            "by_environment_and_generator",
            results_df.groupby(
                [
                    "environment",
                    "generating_model",
                ],
                sort=True,
            ),
        ),
        (
            "by_generator",
            results_df.groupby(
                ["generating_model"],
                sort=True,
            ),
        ),
    ]

    for summary_level, grouped in groupings:
        for group_key, frame in grouped:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)

            if summary_level == "by_environment_and_generator":
                environment = group_key[0]
                generating_model = group_key[1]
            else:
                environment = "ALL"
                generating_model = group_key[0]

            rows.append(
                {
                    "summary_level": summary_level,
                    "environment": environment,
                    "generating_model": generating_model,
                    "n_datasets": int(len(frame)),
                    "n_correct": int(
                        frame["recovered_correctly"].sum()
                    ),
                    "recovery_rate": float(
                        frame["recovered_correctly"].mean()
                    ),
                    "mean_winner_margin_elpd": float(
                        frame["winner_margin_elpd"].mean()
                    ),
                    "median_winner_margin_elpd": float(
                        frame["winner_margin_elpd"].median()
                    ),
                    "mean_generator_gap_from_best_elpd": float(
                        frame["generator_gap_from_best_elpd"].mean()
                    ),
                    "median_generator_gap_from_best_elpd": float(
                        frame["generator_gap_from_best_elpd"].median()
                    ),
                }
            )

    rows.append(
        {
            "summary_level": "overall",
            "environment": "ALL",
            "generating_model": "ALL",
            "n_datasets": int(len(results_df)),
            "n_correct": int(
                results_df["recovered_correctly"].sum()
            ),
            "recovery_rate": float(
                results_df["recovered_correctly"].mean()
            ),
            "mean_winner_margin_elpd": float(
                results_df["winner_margin_elpd"].mean()
            ),
            "median_winner_margin_elpd": float(
                results_df["winner_margin_elpd"].median()
            ),
            "mean_generator_gap_from_best_elpd": float(
                results_df["generator_gap_from_best_elpd"].mean()
            ),
            "median_generator_gap_from_best_elpd": float(
                results_df["generator_gap_from_best_elpd"].median()
            ),
        }
    )

    return pd.DataFrame(rows)



def make_pairwise_summary(
    pairwise_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    grouped = pairwise_df.groupby(
        [
            "environment",
            "generating_model",
            "pair",
            "first_model",
            "second_model",
        ],
        sort=True,
    )

    for keys, frame in grouped:
        (
            environment,
            generating_model,
            pair,
            first_model,
            second_model,
        ) = keys

        delta = frame["delta_elpd"].astype(float)
        dse = frame["dse"].astype(float)
        z_like = frame["delta_over_dse"].astype(float)

        rows.append(
            {
                "environment": environment,
                "generating_model": generating_model,
                "pair": pair,
                "first_model": first_model,
                "second_model": second_model,
                "n_datasets": int(len(frame)),
                "mean_delta_elpd": float(delta.mean()),
                "sd_delta_elpd": float(delta.std(ddof=1))
                if len(delta) > 1
                else np.nan,
                "median_delta_elpd": float(delta.median()),
                "q05_delta_elpd": float(delta.quantile(0.05)),
                "q25_delta_elpd": float(delta.quantile(0.25)),
                "q75_delta_elpd": float(delta.quantile(0.75)),
                "q95_delta_elpd": float(delta.quantile(0.95)),
                "proportion_delta_gt_0": float(
                    (delta > 0.0).mean()
                ),
                "mean_dse": float(dse.mean()),
                "mean_abs_delta_over_dse": float(
                    np.abs(z_like).mean()
                ),
                "proportion_abs_delta_over_dse_gt_2": float(
                    (np.abs(z_like) > 2.0).mean()
                ),
            }
        )

    return pd.DataFrame(rows)


# =====================================================================
# Plots
# =====================================================================


def plot_confusion_matrix(
    *,
    proportions: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    """Plot row-normalized recovery probabilities."""

    values = proportions.to_numpy(dtype=float)

    fig, ax = plt.subplots(
        figsize=(7.0, 6.0)
    )

    image = ax.imshow(
        values,
        vmin=0.0,
        vmax=1.0,
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Recovery proportion",
    )

    labels = [
        MODEL_LABELS[name]
        for name in MODEL_NAMES
    ]

    ax.set_xticks(
        np.arange(len(labels)),
        labels=labels,
    )

    ax.set_yticks(
        np.arange(len(labels)),
        labels=labels,
    )

    ax.set_xlabel(
        "Recovered model"
    )

    ax.set_ylabel(
        "Generating model"
    )

    ax.set_title(title)

    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            value = values[
                row_index,
                col_index,
            ]

            if np.isfinite(value):
                ax.text(
                    col_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=180,
    )
    plt.close(fig)



def plot_global_minus_hybrid_distribution(
    *,
    pairwise_df: pd.DataFrame,
    environment: str,
    output_path: Path,
) -> None:
    """Plot the key Global-minus-Hybrid Delta-ELPD distribution.

    Positive values favor Global; negative values favor Hybrid.
    """

    subset = pairwise_df[
        (pairwise_df["environment"] == environment)
        & (
            pairwise_df["pair"]
            == "global_only_minus_shared_volatility"
        )
    ].copy()

    if subset.empty:
        return

    data = []
    labels = []

    for generator in MODEL_NAMES:
        values = subset.loc[
            subset["generating_model"]
            == generator,
            "delta_elpd",
        ].to_numpy(dtype=float)

        if len(values) > 0:
            data.append(values)
            labels.append(
                MODEL_LABELS[generator]
            )

    if not data:
        return

    fig, ax = plt.subplots(
        figsize=(8.0, 5.5)
    )

    ax.boxplot(
        data,
        labels=labels,
        showmeans=True,
    )

    ax.axhline(
        0.0,
        linewidth=1.0,
    )

    ax.set_xlabel(
        "Generating model"
    )

    ax.set_ylabel(
        "Delta ELPD (Global - Hybrid)"
    )

    ax.set_title(
        f"Global vs Hybrid predictive separation: {environment}"
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=180,
    )
    plt.close(fig)



def plot_environment_confusion_matrices(
    *,
    results_df: pd.DataFrame,
    plot_dir: Path,
) -> None:
    plot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_order = list(MODEL_NAMES)

    for environment, frame in results_df.groupby(
        "environment",
        sort=True,
    ):
        counts = pd.crosstab(
            frame["generating_model"],
            frame["best_model"],
        ).reindex(
            index=model_order,
            columns=model_order,
            fill_value=0,
        )

        proportions = counts.div(
            counts.sum(axis=1).replace(0, np.nan),
            axis=0,
        )

        plot_confusion_matrix(
            proportions=proportions,
            output_path=(
                plot_dir
                / f"confusion_matrix_{safe_name(environment)}.png"
            ),
            title=f"Model recovery: {environment}",
        )



def aggregate_outputs(
    output_dir: Path,
) -> None:
    """Regenerate all aggregate CSVs and plots from completed checkpoints."""

    aggregate_dir = (
        output_dir
        / "aggregate"
    )

    plot_dir = (
        aggregate_dir
        / "plots"
    )

    aggregate_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df = read_all_dataset_results(
        output_dir
    )

    loo_df = read_all_csvs(
        output_dir,
        "loo.csv",
    )

    comparison_df = read_all_csvs(
        output_dir,
        "comparison.csv",
    )

    pairwise_df = read_all_csvs(
        output_dir,
        "pairwise_delta_elpd.csv",
    )

    diagnostics_df = read_all_csvs(
        output_dir,
        "fit_diagnostics.csv",
    )

    if results_df.empty:
        print(
            "No completed datasets found; aggregate outputs were not created."
        )
        return

    results_df.to_csv(
        aggregate_dir
        / "dataset_results.csv",
        index=False,
    )

    if not loo_df.empty:
        loo_df.to_csv(
            aggregate_dir
            / "loo_long.csv",
            index=False,
        )

    if not comparison_df.empty:
        comparison_df.to_csv(
            aggregate_dir
            / "model_comparison_long.csv",
            index=False,
        )

    if not pairwise_df.empty:
        pairwise_df.to_csv(
            aggregate_dir
            / "pairwise_delta_elpd_long.csv",
            index=False,
        )

    if not diagnostics_df.empty:
        diagnostics_df.to_csv(
            aggregate_dir
            / "fit_diagnostics_long.csv",
            index=False,
        )

    (
        confusion_counts,
        confusion_proportions,
        confusion_by_environment,
    ) = make_confusion_tables(
        results_df
    )

    confusion_counts.to_csv(
        aggregate_dir
        / "confusion_counts_overall.csv"
    )

    confusion_proportions.to_csv(
        aggregate_dir
        / "confusion_proportions_overall.csv"
    )

    confusion_by_environment.to_csv(
        aggregate_dir
        / "confusion_by_environment.csv",
        index=False,
    )

    recovery_summary = make_recovery_summary(
        results_df
    )

    recovery_summary.to_csv(
        aggregate_dir
        / "recovery_summary.csv",
        index=False,
    )

    if not pairwise_df.empty:
        pairwise_summary = make_pairwise_summary(
            pairwise_df
        )

        pairwise_summary.to_csv(
            aggregate_dir
            / "pairwise_delta_elpd_summary.csv",
            index=False,
        )

    # -----------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------

    plot_confusion_matrix(
        proportions=confusion_proportions,
        output_path=(
            plot_dir
            / "confusion_matrix_overall.png"
        ),
        title="Overall Bayesian model recovery",
    )

    plot_environment_confusion_matrices(
        results_df=results_df,
        plot_dir=plot_dir,
    )

    if not pairwise_df.empty:
        for environment in sorted(
            pairwise_df["environment"].unique()
        ):
            plot_global_minus_hybrid_distribution(
                pairwise_df=pairwise_df,
                environment=str(environment),
                output_path=(
                    plot_dir
                    / (
                        "delta_elpd_global_minus_hybrid_"
                        f"{safe_name(str(environment))}.png"
                    )
                ),
            )

    # -----------------------------------------------------------------
    # Compact terminal summary
    # -----------------------------------------------------------------

    print(
        "\n========================================"
    )
    print("Aggregate recovery summary")
    print("========================================")

    print(
        recovery_summary.to_string(
            index=False
        )
    )

    print(
        "\nOverall confusion proportions"
    )
    print(
        confusion_proportions.round(3).to_string()
    )

    if not pairwise_df.empty:
        key_summary = make_pairwise_summary(
            pairwise_df
        )

        key_summary = key_summary[
            key_summary["pair"]
            == "global_only_minus_shared_volatility"
        ]

        print(
            "\nGlobal - Hybrid Delta-ELPD summary"
        )
        print(
            key_summary.to_string(
                index=False
            )
        )

    print(
        f"\nAggregate outputs: {aggregate_dir}"
    )


# =====================================================================
# CLI
# =====================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Repeated Bayesian model recovery across response-noise seeds."
        )
    )

    parser.add_argument(
        "--environment-dirs",
        nargs="+",
        required=True,
        help=(
            "One or more environment directories, e.g. "
            "outputs/hgf_walk_outputs/local_A "
            "outputs/hgf_walk_outputs/global"
        ),
    )

    parser.add_argument(
        "--generating-models",
        nargs="+",
        default=list(MODEL_NAMES),
        choices=MODEL_NAMES,
    )

    parser.add_argument(
        "--n-response-seeds",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--response-seed-start",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--response-sd",
        type=float,
        default=None,
        help=(
            "Optional response SD override. If omitted, use the "
            "response_sd_A/B values saved in each generator's metadata."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "outputs/hgf_walk_outputs/"
            "repeated_bayesian_model_recovery"
        ),
    )

    parser.add_argument(
        "--input-precision",
        type=float,
        default=100.0,
    )

    parser.add_argument(
        "--update-type",
        type=str,
        default="eHGF",
        choices=[
            "eHGF",
            "standard",
            "unbounded",
        ],
    )

    parser.add_argument(
        "--max-posterior-precision",
        type=float,
        default=1e10,
    )

    parser.add_argument(
        "--draws",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--tune",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--chains",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--cores",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--target-accept",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--sampler-seed-start",
        type=int,
        default=20000,
    )

    parser.add_argument(
        "--save-fit-details",
        action="store_true",
        help=(
            "Save posterior summaries and diagnostics inside every "
            "candidate run directory."
        ),
    )

    parser.add_argument(
        "--save-idata",
        action="store_true",
        help=(
            "Save every candidate InferenceData NetCDF file. This can "
            "consume substantial disk space in large recovery runs."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Refit datasets whose dataset_result.json checkpoint already "
            "exists. By default completed datasets are skipped."
        ),
    )

    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help=(
            "Stop the whole batch at the first failed dataset. By default "
            "the error is checkpointed and the batch continues."
        ),
    )

    return parser


# =====================================================================
# Main
# =====================================================================


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.n_response_seeds <= 0:
        raise ValueError(
            "--n-response-seeds must be at least 1."
        )

    environment_dirs = [
        Path(path)
        for path in args.environment_dirs
    ]

    for environment_dir in environment_dirs:
        if not environment_dir.exists():
            raise FileNotFoundError(
                f"Environment directory does not exist: {environment_dir}"
            )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    response_seeds = [
        args.response_seed_start + index
        for index in range(args.n_response_seeds)
    ]

    configuration = {
        "environment_dirs": [
            str(path)
            for path in environment_dirs
        ],
        "generating_models": list(
            args.generating_models
        ),
        "response_seeds": response_seeds,
        "response_sd_override": args.response_sd,
        "input_precision": args.input_precision,
        "update_type": args.update_type,
        "max_posterior_precision": args.max_posterior_precision,
        "draws": args.draws,
        "tune": args.tune,
        "chains": args.chains,
        "cores": args.cores,
        "target_accept": args.target_accept,
        "sampler_seed_start": args.sampler_seed_start,
        "save_fit_details": bool(args.save_fit_details),
        "save_idata": bool(args.save_idata),
        "note": (
            "This batch varies response-noise seeds within each supplied "
            "fixed environment realization."
        ),
    }

    json_dump(
        output_dir / "batch_configuration.json",
        configuration,
    )

    # -----------------------------------------------------------------
    # Validate all generator prediction files and trial counts first.
    # -----------------------------------------------------------------

    generator_cache: dict[
        tuple[str, str],
        tuple[np.ndarray, float, float, dict],
    ] = {}

    input_cache: dict[str, np.ndarray] = {}

    for environment_dir in environment_dirs:
        environment_name = environment_dir.name

        input_data = RECOVERY.load_model_input(
            environment_dir
        )

        input_cache[str(environment_dir)] = input_data

        for generating_model in args.generating_models:
            values = load_generating_predictions(
                environment_dir=environment_dir,
                generating_model=generating_model,
                response_sd_override=args.response_sd,
            )

            predictions = values[0]

            if len(predictions) != len(input_data):
                raise ValueError(
                    "Trial-count mismatch in environment "
                    f"{environment_name}, generator {generating_model}: "
                    f"predictions={len(predictions)}, "
                    f"input={len(input_data)}"
                )

            generator_cache[
                (
                    str(environment_dir),
                    generating_model,
                )
            ] = values

    total_datasets = (
        len(environment_dirs)
        * len(args.generating_models)
        * len(response_seeds)
    )

    print(
        "\n========================================"
    )
    print("Repeated Bayesian model recovery")
    print("========================================")
    print(
        f"Environment realizations: {len(environment_dirs)}"
    )
    print(
        f"Generating models: {len(args.generating_models)}"
    )
    print(
        f"Response seeds per environment/generator: "
        f"{len(response_seeds)}"
    )
    print(
        f"Total synthetic datasets: {total_datasets}"
    )
    print(
        f"Candidate fits per dataset: {len(MODEL_NAMES)}"
    )
    print(
        f"Total candidate fits if none are already complete: "
        f"{total_datasets * len(MODEL_NAMES)}"
    )
    print(
        f"Output directory: {output_dir}"
    )

    dataset_counter = 0
    completed_this_run = 0
    skipped_this_run = 0
    failed_this_run = 0

    for environment_index, environment_dir in enumerate(
        environment_dirs
    ):
        environment_name = environment_dir.name
        input_data = input_cache[
            str(environment_dir)
        ]

        n_trials = len(input_data)

        for response_index, response_seed in enumerate(
            response_seeds
        ):
            # ---------------------------------------------------------
            # Same standardized response noise across generators for
            # this environment/seed pair.
            # ---------------------------------------------------------

            z_A, z_B = make_standardized_response_noise(
                n_trials=n_trials,
                response_seed=response_seed,
            )

            for generator_index, generating_model in enumerate(
                args.generating_models
            ):
                dataset_counter += 1

                dataset_dir = (
                    output_dir
                    / "runs"
                    / safe_name(environment_name)
                    / generating_model
                    / f"response_seed_{response_seed}"
                )

                checkpoint = (
                    dataset_dir
                    / "dataset_result.json"
                )

                print(
                    "\n----------------------------------------"
                )
                print(
                    f"Dataset {dataset_counter}/{total_datasets}"
                )
                print(
                    f"    environment: {environment_name}"
                )
                print(
                    f"    generator:   {generating_model}"
                )
                print(
                    f"    response seed: {response_seed}"
                )

                if checkpoint.exists() and not args.overwrite:
                    print(
                        "    checkpoint exists -> skipping"
                    )
                    skipped_this_run += 1
                    continue

                (
                    predictions,
                    response_sd_A,
                    response_sd_B,
                    _,
                ) = generator_cache[
                    (
                        str(environment_dir),
                        generating_model,
                    )
                ]

                responses = simulate_responses(
                    predictions=predictions,
                    z_A=z_A,
                    z_B=z_B,
                    response_sd_A=response_sd_A,
                    response_sd_B=response_sd_B,
                )

                sampler_seed_base = (
                    args.sampler_seed_start
                    + environment_index * 1_000_000
                    + response_index * 10_000
                    + generator_index * 100
                )

                try:
                    result = fit_one_dataset(
                        environment_dir=environment_dir,
                        environment_name=environment_name,
                        generating_model=generating_model,
                        response_seed=response_seed,
                        responses=responses,
                        response_sd_A=response_sd_A,
                        response_sd_B=response_sd_B,
                        dataset_dir=dataset_dir,
                        input_precision=args.input_precision,
                        update_type=args.update_type,
                        max_posterior_precision=(
                            args.max_posterior_precision
                        ),
                        draws=args.draws,
                        tune=args.tune,
                        chains=args.chains,
                        cores=args.cores,
                        target_accept=args.target_accept,
                        sampler_seed_base=sampler_seed_base,
                        save_fit_details=bool(
                            args.save_fit_details
                        ),
                        save_idata=bool(
                            args.save_idata
                        ),
                    )

                    completed_this_run += 1

                    print(
                        f"    best model: {result['best_model']}"
                    )
                    print(
                        "    correctly recovered: "
                        f"{result['recovered_correctly']}"
                    )
                    print(
                        "    Global - Hybrid Delta ELPD: "
                        f"{result['global_minus_hybrid_delta_elpd']:.4f}"
                    )
                    print(
                        "    Global - Hybrid dSE: "
                        f"{result['global_minus_hybrid_dse']:.4f}"
                    )

                except Exception as error:
                    failed_this_run += 1

                    failure = {
                        "environment": environment_name,
                        "environment_path": str(environment_dir),
                        "generating_model": generating_model,
                        "response_seed": int(response_seed),
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                        "traceback": traceback.format_exc(),
                    }

                    json_dump(
                        dataset_dir
                        / "failure.json",
                        failure,
                    )

                    print(
                        "    FAILED: "
                        f"{type(error).__name__}: {error}"
                    )
                    print(
                        "    failure checkpoint: "
                        f"{dataset_dir / 'failure.json'}"
                    )

                    if args.stop_on_error:
                        raise

                finally:
                    gc.collect()

    # -----------------------------------------------------------------
    # Aggregate all completed checkpoints, including those from earlier
    # interrupted runs.
    # -----------------------------------------------------------------

    aggregate_outputs(
        output_dir
    )

    print(
        "\n========================================"
    )
    print("Batch status")
    print("========================================")
    print(
        f"Completed this run: {completed_this_run}"
    )
    print(
        f"Skipped existing:  {skipped_this_run}"
    )
    print(
        f"Failed this run:    {failed_this_run}"
    )


if __name__ == "__main__":
    main()
