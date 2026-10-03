"""
Bayesian recovery smoke test for the volatility-state tonic volatility.

Free parameters
---------------
1. volatility_tonic_volatility
2. shared response noise SD

Fixed perceptual parameters
---------------------------
value_tonic_volatility = -7.824

All other local-only model parameters remain at their current project values.

Current generating values
-------------------------
volatility_tonic_volatility = -6.0
response_sd = 0.10

Purpose
-------
This test asks whether the tonic volatility of the volatility state can be
identified from synthetic continuous predictions when all other perceptual
parameters are fixed.

This follows the same Bayesian pyHGF -> JAX -> PyTensor -> PyMC/NUTS pipeline
used in the previous successful value-tonic-volatility recovery test.
"""

from __future__ import annotations

from pathlib import Path
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

jax.config.update(
    "jax_enable_x64",
    True,
)


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.models.local_only import (  # noqa: E402
    build_model as build_local_only_model,
)

from src.response_models.continuous_prediction import (  # noqa: E402
    two_stream_gaussian_prediction_surprise,
)


# ---------------------------------------------------------------------
# Known generating values
# ---------------------------------------------------------------------

FIXED_VALUE_TONIC_VOLATILITY = -7.824

TRUE_VOLATILITY_TONIC_VOLATILITY = -6.0


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_model_input(
    environment_dir: Path,
) -> np.ndarray:
    """
    Load the two-stream observations supplied to the HGF.
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

    input_data = np.load(
        input_path
    )

    if input_data.ndim != 2:
        raise ValueError(
            f"Expected a 2D input matrix, got {input_data.shape}."
        )

    if input_data.shape[1] != 2:
        raise ValueError(
            f"Expected two streams, got {input_data.shape}."
        )

    if not np.isfinite(input_data).all():
        raise ValueError(
            "Input data contains non-finite values."
        )

    return input_data.astype(float)


def load_synthetic_responses(
    *,
    environment_dir: Path,
    generating_model: str,
) -> tuple[np.ndarray, dict]:
    """
    Load synthetic continuous predictions and response metadata.
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
        metadata = json.load(
            file
        )

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
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(responses).all():
        raise ValueError(
            "Responses contain non-finite values."
        )

    return responses, metadata


# ---------------------------------------------------------------------
# JAX likelihood
# ---------------------------------------------------------------------

def create_likelihood_functions(
    *,
    input_data: np.ndarray,
    responses: np.ndarray,
    input_precision: float,
    update_type: str,
    max_posterior_precision: float,
):
    """
    Create scalar and pointwise likelihood functions.

    Variable parameters
    -------------------
    volatility_tonic_volatility
    log_response_sd

    Fixed parameter
    ---------------
    value_tonic_volatility = -7.824
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

    include_mask = jnp.ones(
        n_trials,
        dtype=jnp.float64,
    )

    # Exclude the initialization trial.
    include_mask = (
        include_mask
        .at[0]
        .set(0.0)
    )

    @wrap_jax
    def total_logp(
        value,
        log_response_sd,
        volatility_tonic_volatility,
    ):
        """
        Scalar log likelihood for NUTS.
        """

        log_response_sd = (
            jnp.asarray(
                log_response_sd
            ).squeeze()
        )

        volatility_tonic_volatility = (
            jnp.asarray(
                volatility_tonic_volatility
            ).squeeze()
        )

        model = build_local_only_model(
            input_precision=input_precision,

            # Explicitly held fixed.
            value_tonic_volatility=(
                FIXED_VALUE_TONIC_VOLATILITY
            ),

            # Free perceptual parameter.
            volatility_tonic_volatility=(
                volatility_tonic_volatility
            ),

            update_type=update_type,

            max_posterior_precision=(
                max_posterior_precision
            ),
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
                response_function_inputs=(
                    response_inputs
                ),
                response_function_parameters=(
                    jnp.atleast_1d(
                        log_response_sd
                    )
                ),
            )
        )

        return -jnp.sum(
            surprise[1:]
        )

    @wrap_jax
    def pointwise_logp(
        log_response_sd,
        volatility_tonic_volatility,
    ):
        """
        Trial-wise log probabilities.

        These are retained for the later LOO model-comparison pipeline.
        """

        log_response_sd = (
            jnp.asarray(
                log_response_sd
            ).squeeze()
        )

        volatility_tonic_volatility = (
            jnp.asarray(
                volatility_tonic_volatility
            ).squeeze()
        )

        model = build_local_only_model(
            input_precision=input_precision,

            value_tonic_volatility=(
                FIXED_VALUE_TONIC_VOLATILITY
            ),

            volatility_tonic_volatility=(
                volatility_tonic_volatility
            ),

            update_type=update_type,

            max_posterior_precision=(
                max_posterior_precision
            ),
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
                response_function_inputs=(
                    response_inputs
                ),
                response_function_parameters=(
                    jnp.atleast_1d(
                        log_response_sd
                    )
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

def run_fit(
    *,
    input_data: np.ndarray,
    responses: np.ndarray,
    input_precision: float,
    update_type: str,
    max_posterior_precision: float,
    draws: int,
    tune: int,
    chains: int,
    cores: int,
    target_accept: float,
    random_seed: int,
):
    """
    Estimate volatility tonic volatility and response SD jointly.
    """

    (
        total_logp,
        pointwise_logp,
    ) = create_likelihood_functions(
        input_data=input_data,
        responses=responses,
        input_precision=input_precision,
        update_type=update_type,
        max_posterior_precision=(
            max_posterior_precision
        ),
    )

    with pm.Model() as model:

        # ---------------------------------------------------------
        # Response parameter
        # ---------------------------------------------------------

        log_response_sd = pm.Normal(
            "log_response_sd",
            mu=np.log(
                0.20
            ),
            sigma=1.0,
        )

        response_sd = pm.Deterministic(
            "response_sd",
            pm.math.exp(
                log_response_sd
            ),
        )

        # ---------------------------------------------------------
        # Volatility-state tonic volatility
        #
        # Same broad prior form as in the preceding recovery test.
        # ---------------------------------------------------------

        base_volatility_prior = pm.Normal.dist(
            mu=-3.0,
            sigma=5.0,
        )

        volatility_tonic_volatility = pm.Censored(
            "volatility_tonic_volatility",
            base_volatility_prior,
            lower=-12.0,
            upper=2.0,
            initval=-5.0,
        )

        # ---------------------------------------------------------
        # Response likelihood
        # ---------------------------------------------------------

        pm.CustomDist(
            "response_likelihood",
            log_response_sd,
            volatility_tonic_volatility,
            logp=total_logp,
            observed=responses,
        )

        # ---------------------------------------------------------
        # Trial-wise likelihood for future LOO
        # ---------------------------------------------------------

        pm.Deterministic(
            "pointwise_loglikelihood",
            pointwise_logp(
                log_response_sd,
                volatility_tonic_volatility,
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

    # Store trial-wise log probabilities in ArviZ's expected group.
    idata["log_likelihood"] = (
        idata.posterior[
            "pointwise_loglikelihood"
        ]
        .to_dataset(
            name="log_likelihood"
        )
    )

    return model, idata


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------

def summarize_recovery(
    *,
    idata,
    true_response_sd: float,
    true_volatility_tonic_volatility: float,
) -> tuple[pd.DataFrame, dict]:
    """
    Summarize posterior recovery and sampler diagnostics.
    """

    posterior_summary = az.summary(
        idata,
        var_names=[
            "volatility_tonic_volatility",
            "log_response_sd",
            "response_sd",
        ],
        round_to=4,
    )

    n_divergences = int(
        idata.sample_stats[
            "diverging"
        ]
        .sum()
        .values
    )

    # -------------------------------------------------------------
    # Response SD posterior
    # -------------------------------------------------------------

    response_sd_samples = (
        idata.posterior[
            "response_sd"
        ]
        .values
        .reshape(-1)
    )

    response_sd_hdi = az.hdi(
        response_sd_samples,
        hdi_prob=0.94,
    )

    # -------------------------------------------------------------
    # Volatility tonic posterior
    # -------------------------------------------------------------

    volatility_samples = (
        idata.posterior[
            "volatility_tonic_volatility"
        ]
        .values
        .reshape(-1)
    )

    volatility_hdi = az.hdi(
        volatility_samples,
        hdi_prob=0.94,
    )

    diagnostics = {

        "true_response_sd": float(
            true_response_sd
        ),

        "posterior_mean_response_sd": float(
            np.mean(
                response_sd_samples
            )
        ),

        "response_sd_hdi_94_lower": float(
            response_sd_hdi[0]
        ),

        "response_sd_hdi_94_upper": float(
            response_sd_hdi[1]
        ),

        "true_response_sd_inside_94_hdi": bool(
            response_sd_hdi[0]
            <= true_response_sd
            <= response_sd_hdi[1]
        ),

        "fixed_value_tonic_volatility": float(
            FIXED_VALUE_TONIC_VOLATILITY
        ),

        "true_volatility_tonic_volatility": float(
            true_volatility_tonic_volatility
        ),

        "posterior_mean_volatility_tonic_volatility": float(
            np.mean(
                volatility_samples
            )
        ),

        "volatility_tonic_volatility_hdi_94_lower": float(
            volatility_hdi[0]
        ),

        "volatility_tonic_volatility_hdi_94_upper": float(
            volatility_hdi[1]
        ),

        "true_volatility_tonic_volatility_inside_94_hdi": bool(
            volatility_hdi[0]
            <= true_volatility_tonic_volatility
            <= volatility_hdi[1]
        ),

        "n_divergences": int(
            n_divergences
        ),
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
    metadata: dict,
) -> None:
    """
    Save recovery outputs.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    idata_path = (
        output_dir
        / "inference_data.nc"
    )

    posterior_summary_path = (
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
        posterior_summary_path
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
            metadata,
            file,
            indent=2,
        )

    print(
        f"\nSaved inference data: "
        f"{idata_path}"
    )

    print(
        f"Saved posterior summary: "
        f"{posterior_summary_path}"
    )

    print(
        f"Saved diagnostics: "
        f"{diagnostics_path}"
    )

    print(
        f"Saved metadata: "
        f"{metadata_path}"
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """
    Command-line interface.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Recover the local-only volatility-state tonic volatility "
            "jointly with response noise."
        )
    )

    parser.add_argument(
        "--environment-dir",
        type=str,
        required=True,
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
        "--random-seed",
        type=int,
        default=2026,
    )

    return parser


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    """
    Run volatility-tonic-volatility recovery smoke test.
    """

    args = (
        build_arg_parser()
        .parse_args()
    )

    environment_dir = Path(
        args.environment_dir
    )

    if args.output_dir is None:

        output_dir = (
            environment_dir
            / "model_recovery"
            / "bayesian_parameter_smoke_test"
            / "volatility_tonic_volatility"
        )

    else:

        output_dir = Path(
            args.output_dir
        )

    print(
        "\n========================================"
    )

    print(
        "Bayesian volatility-tonic recovery"
    )

    print(
        "========================================"
    )

    print(
        f"Environment: "
        f"{environment_dir}"
    )

    print(
        "Generating model: local_only"
    )

    print(
        "Candidate model: local_only"
    )

    print(
        f"Fixed value tonic volatility: "
        f"{FIXED_VALUE_TONIC_VOLATILITY}"
    )

    print(
        f"True volatility tonic volatility: "
        f"{TRUE_VOLATILITY_TONIC_VOLATILITY}"
    )

    # -------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------

    input_data = load_model_input(
        environment_dir
    )

    (
        responses,
        response_metadata,
    ) = load_synthetic_responses(
        environment_dir=environment_dir,
        generating_model="local_only",
    )

    if len(input_data) != len(responses):
        raise ValueError(
            "Number of model-input trials and "
            "response trials does not match."
        )

    true_response_sd_A = float(
        response_metadata[
            "response_sd_A"
        ]
    )

    true_response_sd_B = float(
        response_metadata[
            "response_sd_B"
        ]
    )

    if not np.isclose(
        true_response_sd_A,
        true_response_sd_B,
    ):
        raise ValueError(
            "This recovery test assumes a shared "
            "response-noise parameter."
        )

    true_response_sd = (
        true_response_sd_A
    )

    print(
        f"True response SD: "
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
    # Bayesian fit
    # -------------------------------------------------------------

    _, idata = run_fit(
        input_data=input_data,
        responses=responses,
        input_precision=(
            args.input_precision
        ),
        update_type=(
            args.update_type
        ),
        max_posterior_precision=(
            args.max_posterior_precision
        ),
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        cores=args.cores,
        target_accept=(
            args.target_accept
        ),
        random_seed=(
            args.random_seed
        ),
    )

    # -------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------

    (
        posterior_summary,
        diagnostics,
    ) = summarize_recovery(
        idata=idata,
        true_response_sd=(
            true_response_sd
        ),
        true_volatility_tonic_volatility=(
            TRUE_VOLATILITY_TONIC_VOLATILITY
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

    metadata = {

        "environment_dir": str(
            environment_dir
        ),

        "generating_model": (
            "local_only"
        ),

        "candidate_model": (
            "local_only"
        ),

        "free_parameters": [
            "volatility_tonic_volatility",
            "log_response_sd",
        ],

        "fixed_parameters": {
            "value_tonic_volatility": float(
                FIXED_VALUE_TONIC_VOLATILITY
            ),
        },

        "true_volatility_tonic_volatility": float(
            TRUE_VOLATILITY_TONIC_VOLATILITY
        ),

        "true_response_sd": float(
            true_response_sd
        ),

        "volatility_tonic_volatility_prior": {
            "base_distribution": (
                "Normal"
            ),
            "mu": -3.0,
            "sigma": 5.0,
            "lower": -12.0,
            "upper": 2.0,
            "initval": -5.0,
        },

        "response_sd_prior": {
            "parameterization": (
                "log_response_sd"
            ),
            "distribution": (
                "Normal"
            ),
            "mu": float(
                np.log(
                    0.20
                )
            ),
            "sigma": 1.0,
        },

        "sampling": {
            "backend": (
                "jax"
            ),
            "sampler": (
                "NUTS"
            ),
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
        posterior_summary=(
            posterior_summary
        ),
        diagnostics=(
            diagnostics
        ),
        metadata=(
            metadata
        ),
    )

    print(
        "\nFinished volatility-tonic-volatility "
        "recovery smoke test."
    )


if __name__ == "__main__":
    main()