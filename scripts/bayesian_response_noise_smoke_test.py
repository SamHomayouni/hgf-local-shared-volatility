"""
Bayesian response-noise recovery smoke test.

Purpose
-------
Validate the full Bayesian inference pathway for the current two-stream
continuous-prediction model:

    pyHGF perceptual model
        -> expected_mean predictions
        -> custom Gaussian response function
        -> JAX-compatible log probability
        -> PyMC / NUTS posterior sampling

At this stage ALL perceptual-model parameters are fixed.

Only the shared response-noise parameter is estimated:

    y_A(t) ~ Normal(mu_A(t), sigma_resp^2)
    y_B(t) ~ Normal(mu_B(t), sigma_resp^2)

The fitted parameter is represented in log space:

    log_sigma_resp ~ Normal(log(0.2), 1.0)

and

    sigma_resp = exp(log_sigma_resp)

The first trial is removed from the likelihood because it represents the
initial prior state rather than a parameter-dependent prediction.

This script is deliberately a smoke test. It is not yet formal parameter
or model recovery.

Example
-------
python scripts/bayesian_response_noise_smoke_test.py \
    --environment-dir outputs/hgf_walk_outputs/local_A \
    --generating-model local_only \
    --candidate-model local_only
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable
import argparse
import json
import sys

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pymc as pm
from pytensor import wrap_jax


# ---------------------------------------------------------------------
# JAX configuration
# ---------------------------------------------------------------------

# Match the current pyHGF tutorial configuration and avoid float32/float64
# mismatches between JAX and PyTensor.
jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir() and (p / "config").is_dir())

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.models.local_only import (  # noqa: E402
    build_model as build_local_only_model,
)

from src.models.global_only import (  # noqa: E402
    build_model as build_global_only_model,
)

from src.models.shared_volatility import (  # noqa: E402
    build_model as build_shared_volatility_model,
)

from src.response_models.continuous_prediction import (  # noqa: E402
    two_stream_gaussian_prediction_surprise,
)


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

MODEL_NAMES = [
    "local_only",
    "global_only",
    "shared_volatility",
]


MODEL_BUILDERS: dict[str, Callable] = {
    "local_only": build_local_only_model,
    "global_only": build_global_only_model,
    "shared_volatility": build_shared_volatility_model,
}


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_model_input(
    environment_dir: Path,
) -> np.ndarray:
    """
    Load the two-stream environmental observations.
    """

    input_path = (
        environment_dir
        / "model_runs"
        / "model_input_data.npy"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Could not find model input data: {input_path}"
        )

    input_data = np.load(input_path)

    if input_data.ndim != 2:
        raise ValueError(
            f"Expected 2D model input, got shape {input_data.shape}."
        )

    if input_data.shape[1] != 2:
        raise ValueError(
            f"Expected two streams, got shape {input_data.shape}."
        )

    if not np.isfinite(input_data).all():
        raise ValueError(
            "Model input contains non-finite values."
        )

    return input_data.astype(float)


def load_synthetic_responses(
    *,
    environment_dir: Path,
    generating_model: str,
) -> tuple[np.ndarray, dict]:
    """
    Load synthetic responses and their metadata.

    Returns
    -------
    responses:
        Array shape (n_trials, 2).

    metadata:
        Saved response-model metadata.
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
            f"Could not find responses: {response_path}"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Could not find metadata: {metadata_path}"
        )

    response_df = pd.read_csv(
        response_path
    )

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    required_columns = {
        "participant_prediction_A",
        "participant_prediction_B",
    }

    missing_columns = (
        required_columns
        - set(response_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Synthetic response file is missing columns: "
            f"{sorted(missing_columns)}"
        )

    metadata_generator = metadata.get(
        "generating_perceptual_model"
    )

    if metadata_generator != generating_model:
        raise ValueError(
            "Generating-model mismatch:\n"
            f"Requested: {generating_model}\n"
            f"Metadata:  {metadata_generator}"
        )

    responses = response_df[
        [
            "participant_prediction_A",
            "participant_prediction_B",
        ]
    ].to_numpy(dtype=float)

    if not np.isfinite(responses).all():
        raise ValueError(
            "Responses contain non-finite values."
        )

    return responses, metadata


# ---------------------------------------------------------------------
# JAX likelihood construction
# ---------------------------------------------------------------------

def create_wrapped_likelihood_functions(
    *,
    build_model: Callable,
    input_data: np.ndarray,
    responses: np.ndarray,
    input_precision: float,
    update_type: str,
    max_posterior_precision: float,
):
    """
    Create scalar and pointwise JAX/PyTensor log-likelihood functions.

    All perceptual-model parameters are fixed in this smoke test.
    Only log_response_sd varies.

    The scalar function is used by PyMC/NUTS.

    The pointwise function is retained so the exact same fitting framework
    can later be used for leave-one-out model comparison.
    """

    input_data_jax = jnp.asarray(
        input_data,
        dtype=jnp.float64,
    )

    responses_jax = jnp.asarray(
        responses,
        dtype=jnp.float64,
    )

    n_trials = int(
        len(responses)
    )

    # Trial 0 corresponds to the initial state and is excluded.
    include_mask = jnp.ones(
        n_trials,
        dtype=jnp.float64,
    )

    include_mask = include_mask.at[0].set(
        0.0
    )

    @wrap_jax
    def total_logp(
        value,
        log_response_sd,
    ):
        """
        Scalar log likelihood used for NUTS sampling.
        """

        model = build_model(
            input_precision=input_precision,
            update_type=update_type,
            max_posterior_precision=max_posterior_precision,
        )

        maybe_updated_model = model.input_data(
            input_data=input_data_jax
        )

        if maybe_updated_model is not None:
            model = maybe_updated_model

        response_inputs = jnp.column_stack(
            [
                value[:, 0],
                value[:, 1],
                include_mask,
            ]
        )

        surprise = (
            two_stream_gaussian_prediction_surprise(
                model,
                response_function_inputs=response_inputs,
                response_function_parameters=jnp.atleast_1d(
                    log_response_sd
                ),
            )
        )

        # Drop trial 0 completely rather than retaining a constant
        # zero contribution.
        return -jnp.sum(
            surprise[1:]
        )

    @wrap_jax
    def pointwise_logp(
        log_response_sd,
    ):
        """
        Trial-wise log probabilities for later LOO model comparison.
        """

        model = build_model(
            input_precision=input_precision,
            update_type=update_type,
            max_posterior_precision=max_posterior_precision,
        )

        maybe_updated_model = model.input_data(
            input_data=input_data_jax
        )

        if maybe_updated_model is not None:
            model = maybe_updated_model

        response_inputs = jnp.column_stack(
            [
                responses_jax[:, 0],
                responses_jax[:, 1],
                include_mask,
            ]
        )

        surprise = (
            two_stream_gaussian_prediction_surprise(
                model,
                response_function_inputs=response_inputs,
                response_function_parameters=jnp.atleast_1d(
                    log_response_sd
                ),
            )
        )

        return -surprise[1:]

    return (
        total_logp,
        pointwise_logp,
    )


# ---------------------------------------------------------------------
# Bayesian fit
# ---------------------------------------------------------------------

def run_bayesian_fit(
    *,
    build_model: Callable,
    input_data: np.ndarray,
    responses: np.ndarray,
    input_precision: float,
    update_type: str,
    max_posterior_precision: float,
    prior_median_response_sd: float,
    prior_log_sd: float,
    draws: int,
    tune: int,
    chains: int,
    cores: int,
    target_accept: float,
    random_seed: int,
):
    """
    Estimate response noise with PyMC/NUTS.
    """

    (
        total_logp,
        pointwise_logp,
    ) = create_wrapped_likelihood_functions(
        build_model=build_model,
        input_data=input_data,
        responses=responses,
        input_precision=input_precision,
        update_type=update_type,
        max_posterior_precision=max_posterior_precision,
    )

    with pm.Model() as model:

        # ---------------------------------------------------------
        # Prior on shared response noise
        # ---------------------------------------------------------

        log_response_sd = pm.Normal(
            "log_response_sd",
            mu=np.log(
                prior_median_response_sd
            ),
            sigma=prior_log_sd,
        )

        response_sd = pm.Deterministic(
            "response_sd",
            pm.math.exp(
                log_response_sd
            ),
        )

        # ---------------------------------------------------------
        # Custom HGF response likelihood
        # ---------------------------------------------------------

        pm.CustomDist(
            "response_likelihood",
            log_response_sd,
            logp=total_logp,
            observed=responses,
        )

        # ---------------------------------------------------------
        # Save trial-wise log probabilities.
        #
        # These are not necessary for the current smoke test, but
        # we build the pipeline correctly now so it can later feed
        # directly into ArviZ LOO model comparison.
        # ---------------------------------------------------------

        pm.Deterministic(
            "pointwise_loglikelihood",
            pointwise_logp(
                log_response_sd
            ),
        )

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            backend="jax",
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
        )

    # -------------------------------------------------------------
    # Put pointwise likelihood in the group expected by ArviZ.
    # -------------------------------------------------------------

    idata["log_likelihood"] = (
        idata.posterior[
            "pointwise_loglikelihood"
        ].to_dataset(
            name="log_likelihood"
        )
    )

    return model, idata


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------

def create_diagnostic_summary(
    *,
    idata,
    generating_response_sd: float,
) -> tuple[pd.DataFrame, dict]:
    """
    Create posterior and sampler diagnostic summaries.
    """

    posterior_summary = az.summary(
        idata,
        var_names=[
            "log_response_sd",
            "response_sd",
        ],
        round_to=4,
    )

    n_divergences = int(
        idata.sample_stats[
            "diverging"
        ].sum().values
    )

    response_sd_samples = (
        idata.posterior[
            "response_sd"
        ]
        .values
        .reshape(-1)
    )

    posterior_mean = float(
        np.mean(
            response_sd_samples
        )
    )

    posterior_median = float(
        np.median(
            response_sd_samples
        )
    )

    hdi = az.hdi(
        response_sd_samples,
        hdi_prob=0.94,
    )

    diagnostics = {
        "true_generating_response_sd": float(
            generating_response_sd
        ),
        "posterior_mean_response_sd": (
            posterior_mean
        ),
        "posterior_median_response_sd": (
            posterior_median
        ),
        "response_sd_hdi_94_lower": float(
            hdi[0]
        ),
        "response_sd_hdi_94_upper": float(
            hdi[1]
        ),
        "true_value_inside_94_hdi": bool(
            hdi[0]
            <= generating_response_sd
            <= hdi[1]
        ),
        "n_divergences": n_divergences,
    }

    return (
        posterior_summary,
        diagnostics,
    )


# ---------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------

def save_outputs(
    *,
    output_dir: Path,
    idata,
    posterior_summary: pd.DataFrame,
    diagnostics: dict,
    run_metadata: dict,
) -> None:
    """
    Save Bayesian smoke-test outputs.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    idata_path = (
        output_dir
        / "inference_data.nc"
    )

    summary_path = (
        output_dir
        / "posterior_summary.csv"
    )

    diagnostics_path = (
        output_dir
        / "diagnostics.json"
    )

    metadata_path = (
        output_dir
        / "fit_metadata.json"
    )

    idata.to_netcdf(
        idata_path
    )

    posterior_summary.to_csv(
        summary_path
    )

    with diagnostics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            diagnostics,
            file,
            indent=2,
        )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            run_metadata,
            file,
            indent=2,
        )

    print(
        f"\nSaved inference data:  {idata_path}"
    )

    print(
        f"Saved posterior table: {summary_path}"
    )

    print(
        f"Saved diagnostics:     {diagnostics_path}"
    )

    print(
        f"Saved fit metadata:    {metadata_path}"
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Bayesian response-noise recovery smoke test "
            "using pyHGF + PyMC/NUTS."
        )
    )

    parser.add_argument(
        "--environment-dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--generating-model",
        type=str,
        required=True,
        choices=MODEL_NAMES,
    )

    parser.add_argument(
        "--candidate-model",
        type=str,
        required=True,
        choices=MODEL_NAMES,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
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
        "--prior-median-response-sd",
        type=float,
        default=0.20,
        help=(
            "Median of the LogNormal-equivalent response-SD prior."
        ),
    )

    parser.add_argument(
        "--prior-log-sd",
        type=float,
        default=1.0,
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
        default=2,
        help=(
            "Two chains are sufficient for this smoke test. "
            "Formal fits will use four."
        ),
    )

    parser.add_argument(
        "--cores",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--target-accept",
        type=float,
        default=0.90,
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=2026,
    )

    return parser


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    parser = build_arg_parser()
    args = parser.parse_args()

    environment_dir = Path(
        args.environment_dir
    )

    generating_model = (
        args.generating_model
    )

    candidate_model = (
        args.candidate_model
    )

    if args.output_dir is None:

        output_dir = (
            environment_dir
            / "model_recovery"
            / "bayesian_smoke_test"
            / f"generating_{generating_model}"
            / f"candidate_{candidate_model}"
        )

    else:

        output_dir = Path(
            args.output_dir
        )

    print(
        "\n========================================"
    )

    print(
        "Bayesian response-noise smoke test"
    )

    print(
        "========================================"
    )

    print(
        f"Environment:      {environment_dir}"
    )

    print(
        f"Generating model: {generating_model}"
    )

    print(
        f"Candidate model:  {candidate_model}"
    )

    # -------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------

    input_data = load_model_input(
        environment_dir
    )

    responses, response_metadata = (
        load_synthetic_responses(
            environment_dir=environment_dir,
            generating_model=generating_model,
        )
    )

    if len(input_data) != len(responses):
        raise ValueError(
            f"Input has {len(input_data)} trials "
            f"but responses have {len(responses)}."
        )

    generating_sd_A = float(
        response_metadata[
            "response_sd_A"
        ]
    )

    generating_sd_B = float(
        response_metadata[
            "response_sd_B"
        ]
    )

    if not np.isclose(
        generating_sd_A,
        generating_sd_B,
    ):
        raise ValueError(
            "This smoke test assumes the same response SD "
            "for both streams."
        )

    true_response_sd = (
        generating_sd_A
    )

    print(
        f"Generating response SD: "
        f"{true_response_sd}"
    )

    print(
        f"Trials total: "
        f"{len(responses)}"
    )

    print(
        f"Trials in likelihood: "
        f"{len(responses) - 1}"
    )

    # -------------------------------------------------------------
    # Candidate builder
    # -------------------------------------------------------------

    build_model = MODEL_BUILDERS[
        candidate_model
    ]

    # -------------------------------------------------------------
    # Bayesian inference
    # -------------------------------------------------------------

    _, idata = run_bayesian_fit(
        build_model=build_model,
        input_data=input_data,
        responses=responses,
        input_precision=args.input_precision,
        update_type=args.update_type,
        max_posterior_precision=(
            args.max_posterior_precision
        ),
        prior_median_response_sd=(
            args.prior_median_response_sd
        ),
        prior_log_sd=(
            args.prior_log_sd
        ),
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        cores=args.cores,
        target_accept=args.target_accept,
        random_seed=args.random_seed,
    )

    # -------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------

    (
        posterior_summary,
        diagnostics,
    ) = create_diagnostic_summary(
        idata=idata,
        generating_response_sd=(
            true_response_sd
        ),
    )

    print(
        "\nPosterior summary"
    )

    print(
        posterior_summary.to_string()
    )

    print(
        "\nRecovery diagnostics"
    )

    for key, value in diagnostics.items():
        print(
            f"    {key}: {value}"
        )

    # -------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------

    run_metadata = {
        "environment_dir": str(
            environment_dir
        ),
        "generating_model": (
            generating_model
        ),
        "candidate_model": (
            candidate_model
        ),
        "free_parameters": [
            "log_response_sd",
        ],
        "fixed_perceptual_parameters": True,
        "true_generating_response_sd": float(
            true_response_sd
        ),
        "prior": {
            "log_response_sd": {
                "distribution": "Normal",
                "mu": float(
                    np.log(
                        args.prior_median_response_sd
                    )
                ),
                "sigma": float(
                    args.prior_log_sd
                ),
            }
        },
        "sampling": {
            "backend": "jax",
            "sampler": "NUTS",
            "draws": int(
                args.draws
            ),
            "tune": int(
                args.tune
            ),
            "chains": int(
                args.chains
            ),
            "target_accept": float(
                args.target_accept
            ),
            "random_seed": int(
                args.random_seed
            ),
        },
        "first_trial_excluded": True,
    }

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------

    save_outputs(
        output_dir=output_dir,
        idata=idata,
        posterior_summary=posterior_summary,
        diagnostics=diagnostics,
        run_metadata=run_metadata,
    )

    print(
        "\nFinished Bayesian response-noise smoke test."
    )


if __name__ == "__main__":
    main()