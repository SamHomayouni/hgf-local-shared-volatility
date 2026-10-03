"""
Bayesian recovery smoke test for the Hybrid/shared-volatility HGF.

Free parameters
---------------
1. shared_value_coupling
2. shared response noise SD

Fixed perceptual parameters
---------------------------
value_tonic_volatility        = -7.824
local_vol_tonic_volatility    = -6.0
shared_value_tonic_volatility = -4.0
local_volatility_coupling     = 0.7
local_vol_autoconnection      = 0.7
value_autoconnection          = 0.95

Known generating values
-----------------------
shared_value_coupling = 0.75
response_sd           = 0.10

Architecture
------------
value A <- local volatility A <- shared value
value B <- local volatility B <- shared value

The shared parent is value-coupled to both local-volatility nodes.

Implementation
--------------
The topology is constructed with static shared_value_coupling=0.75.

The sampled JAX coupling is then inserted into the already-created
value-coupling edges using pyhgf.utils.set_coupling().
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

from pyhgf.utils import set_coupling


# =====================================================================
# JAX
# =====================================================================

jax.config.update(
    "jax_enable_x64",
    True,
)


# =====================================================================
# Project imports
# =====================================================================

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir() and (p / "config").is_dir())

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.models.shared_volatility import (
    build_model as build_shared_volatility_model,
)

from src.response_models.continuous_prediction import (
    two_stream_gaussian_prediction_surprise,
)


# =====================================================================
# Hybrid node IDs
# =====================================================================

VALUE_A_NODE = 2
VALUE_B_NODE = 3

LOCAL_VOL_A_NODE = 4
LOCAL_VOL_B_NODE = 5

SHARED_VALUE_NODE = 6


# =====================================================================
# Generating / fixed parameters
# =====================================================================

TRUE_SHARED_VALUE_COUPLING = 0.75

FIXED_VALUE_TONIC_VOLATILITY = -7.824
FIXED_VALUE_AUTOCONNECTION = 0.95

FIXED_LOCAL_VOL_TONIC_VOLATILITY = -6.0
FIXED_LOCAL_VOLATILITY_COUPLING = 0.7
FIXED_LOCAL_VOL_AUTOCONNECTION = 0.7

FIXED_SHARED_VALUE_TONIC_VOLATILITY = -4.0


# =====================================================================
# Data loading
# =====================================================================

def load_model_input(
    environment_dir: Path,
) -> np.ndarray:

    input_path = (
        environment_dir
        / "model_runs"
        / "model_input_data.npy"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Could not find model input: {input_path}"
        )

    input_data = np.load(
        input_path
    )

    if input_data.ndim != 2:
        raise ValueError(
            f"Expected 2D input, got {input_data.shape}."
        )

    if input_data.shape[1] != 2:
        raise ValueError(
            f"Expected two streams, got {input_data.shape}."
        )

    if not np.isfinite(
        input_data
    ).all():
        raise ValueError(
            "Input data contains non-finite values."
        )

    return input_data.astype(
        float
    )


def load_synthetic_responses(
    *,
    environment_dir: Path,
) -> tuple[np.ndarray, dict]:

    response_dir = (
        environment_dir
        / "synthetic_responses"
        / "shared_volatility"
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

    missing = (
        required_columns
        - set(response_df.columns)
    )

    if missing:
        raise ValueError(
            "Response file is missing columns: "
            f"{sorted(missing)}"
        )

    generating_model = metadata.get(
        "generating_perceptual_model"
    )

    if generating_model != "shared_volatility":
        raise ValueError(
            "This test requires Hybrid-generated responses, "
            f"but metadata reports {generating_model!r}."
        )

    responses = response_df[
        [
            "participant_prediction_A",
            "participant_prediction_B",
        ]
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        responses
    ).all():
        raise ValueError(
            "Responses contain non-finite values."
        )

    return (
        responses,
        metadata,
    )


# =====================================================================
# Dynamic shared value coupling
# =====================================================================

def set_hybrid_shared_value_coupling(
    model,
    shared_value_coupling,
):
    """
    Set the sampled value coupling on both Hybrid shared edges:

        shared value (6) -> local volatility A (4)
        shared value (6) -> local volatility B (5)

    pyHGF's set_coupling() updates the coupling representation on
    both sides of each value-coupling edge.
    """

    shared_value_coupling = (
        jnp.asarray(
            shared_value_coupling
        )
        .squeeze()
    )

    set_coupling(
        attributes=model.attributes,
        edges=model.edges,
        parent_idx=SHARED_VALUE_NODE,
        child_idx=LOCAL_VOL_A_NODE,
        coupling=shared_value_coupling,
    )

    set_coupling(
        attributes=model.attributes,
        edges=model.edges,
        parent_idx=SHARED_VALUE_NODE,
        child_idx=LOCAL_VOL_B_NODE,
        coupling=shared_value_coupling,
    )

    return model


# =====================================================================
# Dynamic Hybrid model
# =====================================================================

def build_dynamic_hybrid_model(
    *,
    shared_value_coupling,
    input_precision: float,
    update_type: str,
    max_posterior_precision: float,
):
    """
    Construct the Hybrid model while keeping graph topology static.
    """

    shared_value_coupling = (
        jnp.asarray(
            shared_value_coupling
        )
        .squeeze()
    )

    # -----------------------------------------------------------------
    # Static topology.
    #
    # Do not pass the sampled JAX tracer into network construction.
    # -----------------------------------------------------------------

    model = build_shared_volatility_model(
        input_precision=(
            input_precision
        ),

        value_tonic_volatility=(
            FIXED_VALUE_TONIC_VOLATILITY
        ),

        value_autoconnection_strength=(
            FIXED_VALUE_AUTOCONNECTION
        ),

        local_vol_tonic_volatility=(
            FIXED_LOCAL_VOL_TONIC_VOLATILITY
        ),

        local_volatility_coupling=(
            FIXED_LOCAL_VOLATILITY_COUPLING
        ),

        local_vol_autoconnection_strength=(
            FIXED_LOCAL_VOL_AUTOCONNECTION
        ),

        shared_value_tonic_volatility=(
            FIXED_SHARED_VALUE_TONIC_VOLATILITY
        ),

        # Static edge-construction value.
        shared_value_coupling=(
            TRUE_SHARED_VALUE_COUPLING
        ),

        update_type=(
            update_type
        ),

        max_posterior_precision=(
            max_posterior_precision
        ),
    )

    # -----------------------------------------------------------------
    # Insert sampled numerical coupling after topology construction.
    # -----------------------------------------------------------------

    model = set_hybrid_shared_value_coupling(
        model,
        shared_value_coupling,
    )

    return model


# =====================================================================
# Likelihood functions
# =====================================================================

def create_likelihood_functions(
    *,
    input_data: np.ndarray,
    responses: np.ndarray,
    input_precision: float,
    update_type: str,
    max_posterior_precision: float,
):

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

    # Exclude initial-prior trial.
    include_mask = (
        include_mask
        .at[0]
        .set(0.0)
    )

    # =================================================================
    # Total likelihood
    # =================================================================

    @wrap_jax
    def total_logp(
        value,
        log_shared_value_coupling,
        log_response_sd,
    ):

        log_shared_value_coupling = (
            jnp.asarray(
                log_shared_value_coupling
            )
            .squeeze()
        )

        log_response_sd = (
            jnp.asarray(
                log_response_sd
            )
            .squeeze()
        )

        shared_value_coupling = jnp.exp(
            log_shared_value_coupling
        )

        model = build_dynamic_hybrid_model(
            shared_value_coupling=(
                shared_value_coupling
            ),

            input_precision=(
                input_precision
            ),

            update_type=(
                update_type
            ),

            max_posterior_precision=(
                max_posterior_precision
            ),
        )

        maybe_updated_model = (
            model.input_data(
                input_data=(
                    input_data_jax
                )
            )
        )

        if maybe_updated_model is not None:
            model = maybe_updated_model

        response_inputs = (
            jnp.column_stack(
                [
                    value[:, 0],
                    value[:, 1],
                    include_mask,
                ]
            )
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

    # =================================================================
    # Pointwise likelihood
    # =================================================================

    @wrap_jax
    def pointwise_logp(
        log_shared_value_coupling,
        log_response_sd,
    ):

        log_shared_value_coupling = (
            jnp.asarray(
                log_shared_value_coupling
            )
            .squeeze()
        )

        log_response_sd = (
            jnp.asarray(
                log_response_sd
            )
            .squeeze()
        )

        shared_value_coupling = jnp.exp(
            log_shared_value_coupling
        )

        model = build_dynamic_hybrid_model(
            shared_value_coupling=(
                shared_value_coupling
            ),

            input_precision=(
                input_precision
            ),

            update_type=(
                update_type
            ),

            max_posterior_precision=(
                max_posterior_precision
            ),
        )

        maybe_updated_model = (
            model.input_data(
                input_data=(
                    input_data_jax
                )
            )
        )

        if maybe_updated_model is not None:
            model = maybe_updated_model

        response_inputs = (
            jnp.column_stack(
                [
                    responses_jax[:, 0],
                    responses_jax[:, 1],
                    include_mask,
                ]
            )
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


# =====================================================================
# Bayesian fitting
# =====================================================================

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

        # -------------------------------------------------------------
        # Shared value coupling
        #
        # Positive first-pass scientific parameterization:
        #
        # log(alpha) ~ Normal(0, 1)
        # alpha      = exp(log(alpha))
        #
        # The prior median is 1, not the true value 0.75.
        # -------------------------------------------------------------

        log_shared_value_coupling = pm.Normal(
            "log_shared_value_coupling",
            mu=0.0,
            sigma=1.0,
        )

        shared_value_coupling = pm.Deterministic(
            "shared_value_coupling",

            pm.math.exp(
                log_shared_value_coupling
            ),
        )

        # -------------------------------------------------------------
        # Response noise
        # -------------------------------------------------------------

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

        # -------------------------------------------------------------
        # Response likelihood
        # -------------------------------------------------------------

        pm.CustomDist(
            "response_likelihood",

            log_shared_value_coupling,
            log_response_sd,

            logp=total_logp,

            observed=responses,
        )

        # -------------------------------------------------------------
        # Pointwise likelihood
        # -------------------------------------------------------------

        pm.Deterministic(
            "pointwise_loglikelihood",

            pointwise_logp(
                log_shared_value_coupling,
                log_response_sd,
            ),
        )

        idata = pm.sample(
            draws=draws,
            tune=tune,

            chains=chains,
            cores=cores,

            backend="jax",

            target_accept=(
                target_accept
            ),

            random_seed=(
                random_seed
            ),

            return_inferencedata=True,
        )

    idata[
        "log_likelihood"
    ] = (
        idata.posterior[
            "pointwise_loglikelihood"
        ]
        .to_dataset(
            name="log_likelihood"
        )
    )

    return (
        model,
        idata,
    )


# =====================================================================
# Posterior helpers
# =====================================================================

def flatten_samples(
    idata,
    variable_name: str,
) -> np.ndarray:

    return (
        idata.posterior[
            variable_name
        ]
        .values
        .reshape(-1)
    )


# =====================================================================
# Diagnostics
# =====================================================================

def summarize_recovery(
    *,
    idata,
    true_response_sd: float,
) -> tuple[
    pd.DataFrame,
    dict,
]:

    posterior_summary = az.summary(
        idata,

        var_names=[
            "log_shared_value_coupling",
            "shared_value_coupling",
            "log_response_sd",
            "response_sd",
        ],

        round_to=4,
    )

    coupling_samples = flatten_samples(
        idata,
        "shared_value_coupling",
    )

    response_sd_samples = flatten_samples(
        idata,
        "response_sd",
    )

    coupling_hdi = az.hdi(
        coupling_samples,
        hdi_prob=0.94,
    )

    response_sd_hdi = az.hdi(
        response_sd_samples,
        hdi_prob=0.94,
    )

    coupling_response_correlation = float(
        np.corrcoef(
            coupling_samples,
            response_sd_samples,
        )[0, 1]
    )

    n_divergences = int(
        idata.sample_stats[
            "diverging"
        ]
        .sum()
        .values
    )

    diagnostics = {

        # =============================================================
        # Fixed perceptual parameters
        # =============================================================

        "fixed_value_tonic_volatility": float(
            FIXED_VALUE_TONIC_VOLATILITY
        ),

        "fixed_local_vol_tonic_volatility": float(
            FIXED_LOCAL_VOL_TONIC_VOLATILITY
        ),

        "fixed_local_volatility_coupling": float(
            FIXED_LOCAL_VOLATILITY_COUPLING
        ),

        "fixed_local_vol_autoconnection": float(
            FIXED_LOCAL_VOL_AUTOCONNECTION
        ),

        "fixed_shared_value_tonic_volatility": float(
            FIXED_SHARED_VALUE_TONIC_VOLATILITY
        ),

        # =============================================================
        # Shared value coupling
        # =============================================================

        "true_shared_value_coupling": float(
            TRUE_SHARED_VALUE_COUPLING
        ),

        "posterior_mean_shared_value_coupling": float(
            np.mean(
                coupling_samples
            )
        ),

        "posterior_median_shared_value_coupling": float(
            np.median(
                coupling_samples
            )
        ),

        "shared_value_coupling_hdi_94_lower": float(
            coupling_hdi[0]
        ),

        "shared_value_coupling_hdi_94_upper": float(
            coupling_hdi[1]
        ),

        "true_shared_value_coupling_inside_94_hdi": bool(
            coupling_hdi[0]
            <= TRUE_SHARED_VALUE_COUPLING
            <= coupling_hdi[1]
        ),

        # =============================================================
        # Response SD
        # =============================================================

        "true_response_sd": float(
            true_response_sd
        ),

        "posterior_mean_response_sd": float(
            np.mean(
                response_sd_samples
            )
        ),

        "posterior_median_response_sd": float(
            np.median(
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

        # =============================================================
        # Dependence
        # =============================================================

        "posterior_correlation_shared_value_coupling_vs_response_sd": (
            coupling_response_correlation
        ),

        # =============================================================
        # Sampling
        # =============================================================

        "n_divergences": int(
            n_divergences
        ),

        "maximum_r_hat": float(
            posterior_summary[
                "r_hat"
            ].max()
        ),

        "minimum_ess_bulk": float(
            posterior_summary[
                "ess_bulk"
            ].min()
        ),

        "minimum_ess_tail": float(
            posterior_summary[
                "ess_tail"
            ].min()
        ),
    }

    return (
        posterior_summary,
        diagnostics,
    )


# =====================================================================
# Saving
# =====================================================================

def save_outputs(
    *,
    output_dir: Path,
    idata,
    posterior_summary: pd.DataFrame,
    diagnostics: dict,
    metadata: dict,
) -> None:

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
            metadata,
            file,
            indent=2,
        )

    print(
        f"\nSaved inference data: {idata_path}"
    )

    print(
        f"Saved posterior summary: {summary_path}"
    )

    print(
        f"Saved diagnostics: {diagnostics_path}"
    )

    print(
        f"Saved metadata: {metadata_path}"
    )


# =====================================================================
# CLI
# =====================================================================

def build_arg_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Recover Hybrid shared value coupling "
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
        "--random-seed",
        type=int,
        default=2026,
    )

    return parser


# =====================================================================
# Main
# =====================================================================

def main() -> None:

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
            / "hybrid_shared_value_coupling"
        )

    else:

        output_dir = Path(
            args.output_dir
        )

    print(
        "\n========================================"
    )

    print(
        "Hybrid shared-value coupling recovery"
    )

    print(
        "========================================"
    )

    print(
        f"Environment: {environment_dir}"
    )

    print(
        "Generating model: shared_volatility"
    )

    print(
        "Candidate model: shared_volatility"
    )

    print(
        "\nTrue shared value coupling: "
        f"{TRUE_SHARED_VALUE_COUPLING}"
    )

    print(
        "\nFixed perceptual parameters:"
    )

    print(
        f"    value tonic volatility: "
        f"{FIXED_VALUE_TONIC_VOLATILITY}"
    )

    print(
        f"    local volatility tonic: "
        f"{FIXED_LOCAL_VOL_TONIC_VOLATILITY}"
    )

    print(
        f"    local volatility coupling: "
        f"{FIXED_LOCAL_VOLATILITY_COUPLING}"
    )

    print(
        f"    local vol autoconnection: "
        f"{FIXED_LOCAL_VOL_AUTOCONNECTION}"
    )

    print(
        f"    shared value tonic volatility: "
        f"{FIXED_SHARED_VALUE_TONIC_VOLATILITY}"
    )

    # -----------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------

    input_data = load_model_input(
        environment_dir
    )

    (
        responses,
        response_metadata,
    ) = load_synthetic_responses(
        environment_dir=(
            environment_dir
        ),
    )

    if len(input_data) != len(responses):
        raise ValueError(
            "Input and response trial counts do not match."
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
            "This test assumes shared response noise."
        )

    true_response_sd = (
        true_response_sd_A
    )

    print(
        f"\nTrue response SD: "
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

    print(
        f"Chains: {args.chains}"
    )

    print(
        f"Draws per chain: {args.draws}"
    )

    print(
        f"Tuning steps per chain: {args.tune}"
    )

    # -----------------------------------------------------------------
    # Fit
    # -----------------------------------------------------------------

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

        draws=(
            args.draws
        ),

        tune=(
            args.tune
        ),

        chains=(
            args.chains
        ),

        cores=(
            args.cores
        ),

        target_accept=(
            args.target_accept
        ),

        random_seed=(
            args.random_seed
        ),
    )

    # -----------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------

    (
        posterior_summary,
        diagnostics,
    ) = summarize_recovery(
        idata=idata,
        true_response_sd=(
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

    for key, value in (
        diagnostics.items()
    ):
        print(
            f"    {key}: {value}"
        )

    # -----------------------------------------------------------------
    # Metadata
    # -----------------------------------------------------------------

    metadata = {

        "environment_dir": str(
            environment_dir
        ),

        "generating_model": (
            "shared_volatility"
        ),

        "candidate_model": (
            "shared_volatility"
        ),

        "free_parameters": [
            "log_shared_value_coupling",
            "log_response_sd",
        ],

        "derived_parameters": [
            "shared_value_coupling",
            "response_sd",
        ],

        "true_parameters": {

            "shared_value_coupling": float(
                TRUE_SHARED_VALUE_COUPLING
            ),

            "response_sd": float(
                true_response_sd
            ),
        },

        "fixed_parameters": {

            "value_tonic_volatility": float(
                FIXED_VALUE_TONIC_VOLATILITY
            ),

            "value_autoconnection_strength": float(
                FIXED_VALUE_AUTOCONNECTION
            ),

            "local_vol_tonic_volatility": float(
                FIXED_LOCAL_VOL_TONIC_VOLATILITY
            ),

            "local_volatility_coupling": float(
                FIXED_LOCAL_VOLATILITY_COUPLING
            ),

            "local_vol_autoconnection_strength": float(
                FIXED_LOCAL_VOL_AUTOCONNECTION
            ),

            "shared_value_tonic_volatility": float(
                FIXED_SHARED_VALUE_TONIC_VOLATILITY
            ),
        },

        "priors": {

            "log_shared_value_coupling": {
                "distribution": "Normal",
                "mu": 0.0,
                "sigma": 1.0,
                "transformation": (
                    "shared_value_coupling = "
                    "exp(log_shared_value_coupling)"
                ),
            },

            "log_response_sd": {
                "distribution": "Normal",
                "mu": float(
                    np.log(0.20)
                ),
                "sigma": 1.0,
                "transformation": (
                    "response_sd = exp(log_response_sd)"
                ),
            },
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

            "cores": int(
                args.cores
            ),

            "target_accept": float(
                args.target_accept
            ),

            "random_seed": int(
                args.random_seed
            ),
        },

        "first_trial_excluded": True,

        "implementation_note": (
            "Hybrid topology is built using static "
            "shared_value_coupling=0.75. The sampled JAX "
            "value coupling is then inserted into both "
            "shared-parent -> local-volatility edges using "
            "pyhgf.utils.set_coupling before filtering."
        ),
    }

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------

    save_outputs(
        output_dir=(
            output_dir
        ),

        idata=(
            idata
        ),

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
        "\nFinished Hybrid shared-value coupling recovery."
    )


if __name__ == "__main__":
    main()