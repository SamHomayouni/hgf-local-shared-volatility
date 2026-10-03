"""
Bayesian model recovery for the three candidate HGF architectures.

For one synthetic response dataset:

    generating model
        -> local_only
        -> global_only
        -> shared_volatility

fit all three candidate models:

    local_only
    global_only
    shared_volatility

and compare them using PSIS-LOO.

Final first-pass candidate parameterizations
--------------------------------------------

LOCAL
    free:
        value_tonic_volatility
        volatility_coupling
        response_sd

    fixed:
        volatility_tonic_volatility = -6.0

GLOBAL
    free:
        value_tonic_volatility
        shared_volatility_coupling
        response_sd

    fixed:
        shared_volatility_tonic_volatility = -6.0

HYBRID
    free:
        value_tonic_volatility
        shared_value_coupling
        response_sd

    fixed:
        local_vol_tonic_volatility = -6.0
        local_volatility_coupling = 0.7
        local_vol_autoconnection_strength = 0.7
        shared_value_tonic_volatility = -4.0
        value_autoconnection_strength = 0.95

The first trial is excluded.

Pointwise likelihood
--------------------
One pointwise likelihood value corresponds to one trial and contains the
joint likelihood contribution from response A and response B.

Outputs
-------
<environment-dir>/model_recovery/bayesian_model_recovery/<generating-model>/

    local_only/
        inference_data.nc
        posterior_summary.csv
        diagnostics.json

    global_only/
        ...

    shared_volatility/
        ...

    loo_summary.csv
    model_comparison.csv
    recovery_result.json
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.models.local_only import (
    build_model as build_local_only_model,
)

from src.models.global_only import (
    build_model as build_global_only_model,
)

from src.models.shared_volatility import (
    build_model as build_shared_volatility_model,
)

from src.response_models.continuous_prediction import (
    two_stream_gaussian_prediction_surprise,
)


# =====================================================================
# Model names
# =====================================================================

MODEL_NAMES = (
    "local_only",
    "global_only",
    "shared_volatility",
)


# =====================================================================
# Node IDs
# =====================================================================

VALUE_A_NODE = 2
VALUE_B_NODE = 3

LOCAL_VOL_A_NODE = 4
LOCAL_VOL_B_NODE = 5

GLOBAL_SHARED_VOL_NODE = 4

HYBRID_SHARED_VALUE_NODE = 6


# =====================================================================
# Fixed perceptual parameters
# =====================================================================

LOCAL_VOL_TONIC = -6.0

GLOBAL_SHARED_VOL_TONIC = -6.0

HYBRID_LOCAL_VOL_TONIC = -6.0
HYBRID_LOCAL_VOL_COUPLING = 0.7
HYBRID_LOCAL_VOL_AUTOCONNECTION = 0.7

HYBRID_SHARED_VALUE_TONIC = -4.0

HYBRID_VALUE_AUTOCONNECTION = 0.95


# =====================================================================
# Data loading
# =====================================================================

def load_model_input(
    environment_dir: Path,
) -> np.ndarray:
    """
    Load observations supplied to every candidate model.
    """

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
            "Input contains non-finite values."
        )

    return input_data.astype(
        float
    )


def load_synthetic_responses(
    *,
    environment_dir: Path,
    generating_model: str,
) -> tuple[np.ndarray, dict]:
    """
    Load one synthetic participant dataset.
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
            f"Could not find response metadata: {metadata_path}"
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

    actual_generator = metadata.get(
        "generating_perceptual_model"
    )

    if actual_generator != generating_model:
        raise ValueError(
            f"Requested generating model {generating_model!r}, "
            f"but metadata reports {actual_generator!r}."
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
# Volatility-coupling manipulation
# =====================================================================

def set_volatility_edge_coupling(
    model,
    *,
    parent_idx: int,
    child_idx: int,
    coupling,
):
    """
    Replace the numerical coupling of an existing volatility edge.

    Both parent-side and child-side representations are updated.
    """

    coupling = (
        jnp.asarray(
            coupling
        )
        .squeeze()
    )

    volatility_parents = (
        model.edges[
            child_idx
        ].volatility_parents
    )

    if volatility_parents is None:
        raise ValueError(
            f"Node {child_idx} has no volatility parents."
        )

    parent_position = (
        volatility_parents.index(
            parent_idx
        )
    )

    volatility_children = (
        model.edges[
            parent_idx
        ].volatility_children
    )

    if volatility_children is None:
        raise ValueError(
            f"Node {parent_idx} has no volatility children."
        )

    child_position = (
        volatility_children.index(
            child_idx
        )
    )

    child_couplings = (
        model.attributes[
            child_idx
        ][
            "volatility_coupling_parents"
        ]
    )

    model.attributes[
        child_idx
    ][
        "volatility_coupling_parents"
    ] = (
        child_couplings
        .at[parent_position]
        .set(coupling)
    )

    parent_couplings = (
        model.attributes[
            parent_idx
        ][
            "volatility_coupling_children"
        ]
    )

    model.attributes[
        parent_idx
    ][
        "volatility_coupling_children"
    ] = (
        parent_couplings
        .at[child_position]
        .set(coupling)
    )

    return model


def set_local_coupling(
    model,
    coupling,
):
    """
    Local model:

        volatility A -> value A
        volatility B -> value B
    """

    model = set_volatility_edge_coupling(
        model,
        parent_idx=LOCAL_VOL_A_NODE,
        child_idx=VALUE_A_NODE,
        coupling=coupling,
    )

    model = set_volatility_edge_coupling(
        model,
        parent_idx=LOCAL_VOL_B_NODE,
        child_idx=VALUE_B_NODE,
        coupling=coupling,
    )

    return model


def set_global_coupling(
    model,
    coupling,
):
    """
    Global model:

        shared volatility -> value A
        shared volatility -> value B
    """

    model = set_volatility_edge_coupling(
        model,
        parent_idx=GLOBAL_SHARED_VOL_NODE,
        child_idx=VALUE_A_NODE,
        coupling=coupling,
    )

    model = set_volatility_edge_coupling(
        model,
        parent_idx=GLOBAL_SHARED_VOL_NODE,
        child_idx=VALUE_B_NODE,
        coupling=coupling,
    )

    return model


def set_hybrid_shared_value_coupling(
    model,
    coupling,
):
    """
    Hybrid model:

        shared value -> local volatility A
        shared value -> local volatility B
    """

    coupling = (
        jnp.asarray(
            coupling
        )
        .squeeze()
    )

    set_coupling(
        attributes=model.attributes,
        edges=model.edges,
        parent_idx=HYBRID_SHARED_VALUE_NODE,
        child_idx=LOCAL_VOL_A_NODE,
        coupling=coupling,
    )

    set_coupling(
        attributes=model.attributes,
        edges=model.edges,
        parent_idx=HYBRID_SHARED_VALUE_NODE,
        child_idx=LOCAL_VOL_B_NODE,
        coupling=coupling,
    )

    return model


# =====================================================================
# Candidate construction
# =====================================================================

def build_candidate_model(
    *,
    candidate_name: str,
    value_tonic_volatility,
    architecture_coupling,
    input_precision: float,
    update_type: str,
    max_posterior_precision: float,
):
    """
    Construct one candidate using its final smoke-test parameterization.
    """

    value_tonic_volatility = (
        jnp.asarray(
            value_tonic_volatility
        )
        .squeeze()
    )

    architecture_coupling = (
        jnp.asarray(
            architecture_coupling
        )
        .squeeze()
    )

    # -----------------------------------------------------------------
    # Local-only
    # -----------------------------------------------------------------

    if candidate_name == "local_only":

        model = build_local_only_model(
            input_precision=(
                input_precision
            ),

            value_tonic_volatility=(
                value_tonic_volatility
            ),

            volatility_tonic_volatility=(
                LOCAL_VOL_TONIC
            ),

            # Static graph construction.
            volatility_coupling=1.0,

            update_type=(
                update_type
            ),

            max_posterior_precision=(
                max_posterior_precision
            ),
        )

        return set_local_coupling(
            model,
            architecture_coupling,
        )

    # -----------------------------------------------------------------
    # Global-only
    # -----------------------------------------------------------------

    if candidate_name == "global_only":

        model = build_global_only_model(
            input_precision=(
                input_precision
            ),

            value_tonic_volatility=(
                value_tonic_volatility
            ),

            shared_volatility_tonic_volatility=(
                GLOBAL_SHARED_VOL_TONIC
            ),

            # Static graph construction.
            shared_volatility_coupling=1.0,

            update_type=(
                update_type
            ),

            max_posterior_precision=(
                max_posterior_precision
            ),
        )

        return set_global_coupling(
            model,
            architecture_coupling,
        )

    # -----------------------------------------------------------------
    # Hybrid
    # -----------------------------------------------------------------

    if candidate_name == "shared_volatility":

        model = build_shared_volatility_model(
            input_precision=(
                input_precision
            ),

            value_tonic_volatility=(
                value_tonic_volatility
            ),

            value_autoconnection_strength=(
                HYBRID_VALUE_AUTOCONNECTION
            ),

            local_vol_tonic_volatility=(
                HYBRID_LOCAL_VOL_TONIC
            ),

            local_volatility_coupling=(
                HYBRID_LOCAL_VOL_COUPLING
            ),

            local_vol_autoconnection_strength=(
                HYBRID_LOCAL_VOL_AUTOCONNECTION
            ),

            shared_value_tonic_volatility=(
                HYBRID_SHARED_VALUE_TONIC
            ),

            # Static graph construction.
            shared_value_coupling=0.75,

            update_type=(
                update_type
            ),

            max_posterior_precision=(
                max_posterior_precision
            ),
        )

        return set_hybrid_shared_value_coupling(
            model,
            architecture_coupling,
        )

    raise ValueError(
        f"Unknown candidate: {candidate_name}"
    )


# =====================================================================
# Candidate-specific parameter names
# =====================================================================

def coupling_parameter_names(
    candidate_name: str,
) -> tuple[str, str]:
    """
    Return:
        log-scale parameter name
        natural-scale parameter name
    """

    if candidate_name == "local_only":
        return (
            "log_volatility_coupling",
            "volatility_coupling",
        )

    if candidate_name == "global_only":
        return (
            "log_shared_volatility_coupling",
            "shared_volatility_coupling",
        )

    if candidate_name == "shared_volatility":
        return (
            "log_shared_value_coupling",
            "shared_value_coupling",
        )

    raise ValueError(
        f"Unknown candidate: {candidate_name}"
    )


# =====================================================================
# Likelihood
# =====================================================================

def create_likelihood_functions(
    *,
    candidate_name: str,
    input_data: np.ndarray,
    responses: np.ndarray,
    input_precision: float,
    update_type: str,
    max_posterior_precision: float,
):
    """
    Create scalar and pointwise likelihoods for one candidate.
    """

    input_data_jax = jnp.asarray(
        input_data,
        dtype=jnp.float64,
    )

    responses_jax = jnp.asarray(
        responses,
        dtype=jnp.float64,
    )

    n_trials = len(
        responses
    )

    include_mask = jnp.ones(
        n_trials,
        dtype=jnp.float64,
    )

    include_mask = (
        include_mask
        .at[0]
        .set(0.0)
    )

    # -----------------------------------------------------------------
    # Total likelihood
    # -----------------------------------------------------------------

    @wrap_jax
    def total_logp(
        value,
        value_tonic_volatility,
        log_architecture_coupling,
        log_response_sd,
    ):

        value_tonic_volatility = (
            jnp.asarray(
                value_tonic_volatility
            )
            .squeeze()
        )

        log_architecture_coupling = (
            jnp.asarray(
                log_architecture_coupling
            )
            .squeeze()
        )

        log_response_sd = (
            jnp.asarray(
                log_response_sd
            )
            .squeeze()
        )

        architecture_coupling = jnp.exp(
            log_architecture_coupling
        )

        model = build_candidate_model(
            candidate_name=(
                candidate_name
            ),

            value_tonic_volatility=(
                value_tonic_volatility
            ),

            architecture_coupling=(
                architecture_coupling
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

    # -----------------------------------------------------------------
    # Pointwise likelihood
    # -----------------------------------------------------------------

    @wrap_jax
    def pointwise_logp(
        value_tonic_volatility,
        log_architecture_coupling,
        log_response_sd,
    ):

        value_tonic_volatility = (
            jnp.asarray(
                value_tonic_volatility
            )
            .squeeze()
        )

        log_architecture_coupling = (
            jnp.asarray(
                log_architecture_coupling
            )
            .squeeze()
        )

        log_response_sd = (
            jnp.asarray(
                log_response_sd
            )
            .squeeze()
        )

        architecture_coupling = jnp.exp(
            log_architecture_coupling
        )

        model = build_candidate_model(
            candidate_name=(
                candidate_name
            ),

            value_tonic_volatility=(
                value_tonic_volatility
            ),

            architecture_coupling=(
                architecture_coupling
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

        # One likelihood contribution per trial.
        # A and B are combined within each trial.
        return -surprise[1:]

    return (
        total_logp,
        pointwise_logp,
    )


# =====================================================================
# Fit one candidate
# =====================================================================

def fit_candidate(
    *,
    candidate_name: str,
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
    Fit one candidate model.
    """

    (
        log_coupling_name,
        coupling_name,
    ) = coupling_parameter_names(
        candidate_name
    )

    (
        total_logp,
        pointwise_logp,
    ) = create_likelihood_functions(
        candidate_name=(
            candidate_name
        ),

        input_data=(
            input_data
        ),

        responses=(
            responses
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

    with pm.Model() as pymc_model:

        # -------------------------------------------------------------
        # Value tonic volatility
        # -------------------------------------------------------------

        value_base = pm.Normal.dist(
            mu=-3.0,
            sigma=5.0,
        )

        value_tonic_volatility = pm.Censored(
            "value_tonic_volatility",

            value_base,

            lower=-12.0,
            upper=2.0,

            initval=-7.0,
        )

        # -------------------------------------------------------------
        # Architecture-specific coupling
        # -------------------------------------------------------------

        log_architecture_coupling = pm.Normal(
            log_coupling_name,
            mu=0.0,
            sigma=1.0,
        )

        pm.Deterministic(
            coupling_name,

            pm.math.exp(
                log_architecture_coupling
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

        pm.Deterministic(
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

            value_tonic_volatility,
            log_architecture_coupling,
            log_response_sd,

            logp=(
                total_logp
            ),

            observed=(
                responses
            ),
        )

        # -------------------------------------------------------------
        # Pointwise likelihood
        # -------------------------------------------------------------

        pm.Deterministic(
            "pointwise_loglikelihood",

            pointwise_logp(
                value_tonic_volatility,
                log_architecture_coupling,
                log_response_sd,
            ),
        )

        # -------------------------------------------------------------
        # Sampling
        # -------------------------------------------------------------

        idata = pm.sample(
            draws=(
                draws
            ),

            tune=(
                tune
            ),

            chains=(
                chains
            ),

            cores=(
                cores
            ),

            backend="jax",

            target_accept=(
                target_accept
            ),

            random_seed=(
                random_seed
            ),

            return_inferencedata=True,
        )

    # -----------------------------------------------------------------
    # Put manually evaluated trial-wise likelihood in ArviZ's expected
    # log_likelihood group.
    #
    # The variable is called "response".
    # -----------------------------------------------------------------

    idata[
        "log_likelihood"
    ] = (
        idata.posterior[
            "pointwise_loglikelihood"
        ]
        .to_dataset(
            name="response"
        )
    )

    return (
        pymc_model,
        idata,
    )


# =====================================================================
# Diagnostics
# =====================================================================

def candidate_summary(
    *,
    candidate_name: str,
    idata,
) -> tuple[
    pd.DataFrame,
    dict,
]:
    """
    Standard posterior and sampler diagnostics.
    """

    (
        log_coupling_name,
        coupling_name,
    ) = coupling_parameter_names(
        candidate_name
    )

    summary = az.summary(
        idata,

        var_names=[
            "value_tonic_volatility",
            log_coupling_name,
            coupling_name,
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

    diagnostics = {

        "candidate_model": (
            candidate_name
        ),

        "coupling_parameter": (
            coupling_name
        ),

        "n_divergences": (
            n_divergences
        ),

        "maximum_r_hat": float(
            summary[
                "r_hat"
            ].max()
        ),

        "minimum_ess_bulk": float(
            summary[
                "ess_bulk"
            ].min()
        ),

        "minimum_ess_tail": float(
            summary[
                "ess_tail"
            ].min()
        ),
    }

    return (
        summary,
        diagnostics,
    )


# =====================================================================
# Save candidate
# =====================================================================

def save_candidate_outputs(
    *,
    candidate_dir: Path,
    candidate_name: str,
    idata,
    summary: pd.DataFrame,
    diagnostics: dict,
    generating_model: str,
    sampling_metadata: dict,
) -> None:

    candidate_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    idata.to_netcdf(
        candidate_dir
        / "inference_data.nc"
    )

    summary.to_csv(
        candidate_dir
        / "posterior_summary.csv"
    )

    with (
        candidate_dir
        / "diagnostics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            diagnostics,
            file,
            indent=2,
        )

    metadata = {

        "generating_model": (
            generating_model
        ),

        "candidate_model": (
            candidate_name
        ),

        "free_parameterization": {

            "value_tonic_volatility": True,

            "architecture_coupling": (
                coupling_parameter_names(
                    candidate_name
                )[1]
            ),

            "response_sd": True,
        },

        "sampling": (
            sampling_metadata
        ),

        "first_trial_excluded": True,

        "pointwise_loo_unit": (
            "one trial, with stream A and B "
            "response likelihoods combined"
        ),
    }

    with (
        candidate_dir
        / "fit_metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )


# =====================================================================
# PSIS-LOO
# =====================================================================

def calculate_loo_results(
    fitted_models: dict,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict,
]:
    """
    Compute model-specific PSIS-LOO diagnostics and model comparison.
    """

    loo_objects = {}
    loo_rows = []

    for (
        model_name,
        idata,
    ) in fitted_models.items():

        loo = az.loo(
            idata,

            pointwise=True,

            var_name="response",
        )

        loo_objects[
            model_name
        ] = loo

        pareto_k = np.asarray(
            loo.pareto_k
        )

        loo_rows.append(
            {
                "model": model_name,

                "elpd_loo": float(
                    loo.elpd_loo
                ),

                "se": float(
                    loo.se
                ),

                "p_loo": float(
                    loo.p_loo
                ),

                "warning": bool(
                    loo.warning
                ),

                "max_pareto_k": float(
                    np.max(
                        pareto_k
                    )
                ),

                "n_pareto_k_gt_0_7": int(
                    np.sum(
                        pareto_k > 0.7
                    )
                ),

                "n_pareto_k_gt_1_0": int(
                    np.sum(
                        pareto_k > 1.0
                    )
                ),
            }
        )

    loo_summary = (
        pd.DataFrame(
            loo_rows
        )
        .sort_values(
            "elpd_loo",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    # -----------------------------------------------------------------
    # ArviZ model comparison.
    #
    # Higher ELPD is better because scale="log".
    # -----------------------------------------------------------------

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
        .rename(
            columns={
                "index": "model",
            }
        )
    )

    best_model = str(
        comparison.iloc[0][
            "model"
        ]
    )

    result = {

        "best_model_by_psis_loo": (
            best_model
        ),

        "comparison_method": (
            "PSIS-LOO"
        ),

        "weight_method": (
            "stacking"
        ),

        "scale": (
            "log"
        ),
    }

    return (
        loo_summary,
        comparison,
        result,
    )


# =====================================================================
# CLI
# =====================================================================

def build_arg_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Fit Local, Global, and Hybrid candidates "
            "to one synthetic response dataset and "
            "compare them using PSIS-LOO."
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

    generating_model = (
        args.generating_model
    )

    if args.output_dir is None:

        output_dir = (
            environment_dir
            / "model_recovery"
            / "bayesian_model_recovery"
            / generating_model
        )

    else:

        output_dir = Path(
            args.output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
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

        generating_model=(
            generating_model
        ),
    )

    if len(
        input_data
    ) != len(
        responses
    ):
        raise ValueError(
            "Input and response trial counts do not match."
        )

    # -----------------------------------------------------------------
    # Header
    # -----------------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "Bayesian model recovery"
    )

    print(
        "========================================"
    )

    print(
        f"Environment: {environment_dir}"
    )

    print(
        f"Generating model: {generating_model}"
    )

    print(
        f"Trials total: {len(responses)}"
    )

    print(
        f"Trials in likelihood: {len(responses) - 1}"
    )

    print(
        "\nCandidate models:"
    )

    for name in MODEL_NAMES:
        print(
            f"    {name}"
        )

    sampling_metadata = {

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

        "cores": int(
            args.cores
        ),

        "target_accept": float(
            args.target_accept
        ),

        "base_random_seed": int(
            args.random_seed
        ),
    }

    # -----------------------------------------------------------------
    # Fit every candidate to exactly the same responses.
    # -----------------------------------------------------------------

    fitted_models = {}

    for (
        candidate_index,
        candidate_name,
    ) in enumerate(
        MODEL_NAMES
    ):

        print(
            "\n========================================"
        )

        print(
            f"Fitting candidate: {candidate_name}"
        )

        print(
            "========================================"
        )

        candidate_seed = (
            args.random_seed
            + candidate_index
        )

        (
            _,
            idata,
        ) = fit_candidate(
            candidate_name=(
                candidate_name
            ),

            input_data=(
                input_data
            ),

            responses=(
                responses
            ),

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
                candidate_seed
            ),
        )

        fitted_models[
            candidate_name
        ] = idata

        (
            summary,
            diagnostics,
        ) = candidate_summary(
            candidate_name=(
                candidate_name
            ),

            idata=(
                idata
            ),
        )

        print(
            "\nPosterior summary"
        )

        print(
            summary.to_string()
        )

        print(
            "\nSampler diagnostics"
        )

        for (
            key,
            value,
        ) in diagnostics.items():

            print(
                f"    {key}: {value}"
            )

        candidate_dir = (
            output_dir
            / candidate_name
        )

        save_candidate_outputs(
            candidate_dir=(
                candidate_dir
            ),

            candidate_name=(
                candidate_name
            ),

            idata=(
                idata
            ),

            summary=(
                summary
            ),

            diagnostics=(
                diagnostics
            ),

            generating_model=(
                generating_model
            ),

            sampling_metadata=(
                sampling_metadata
            ),
        )

    # -----------------------------------------------------------------
    # PSIS-LOO comparison
    # -----------------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "PSIS-LOO model comparison"
    )

    print(
        "========================================"
    )

    (
        loo_summary,
        comparison,
        recovery_result,
    ) = calculate_loo_results(
        fitted_models
    )

    print(
        "\nLOO diagnostics"
    )

    print(
        loo_summary.to_string(
            index=False
        )
    )

    print(
        "\nModel comparison"
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    # -----------------------------------------------------------------
    # Recovery verdict
    # -----------------------------------------------------------------

    best_model = (
        recovery_result[
            "best_model_by_psis_loo"
        ]
    )

    recovered_correctly = bool(
        best_model
        == generating_model
    )

    recovery_result.update(
        {
            "environment": (
                environment_dir.name
            ),

            "generating_model": (
                generating_model
            ),

            "recovered_correctly": (
                recovered_correctly
            ),

            "n_trials_total": int(
                len(
                    responses
                )
            ),

            "n_trials_in_likelihood": int(
                len(
                    responses
                )
                - 1
            ),

            "response_metadata": (
                response_metadata
            ),
        }
    )

    # -----------------------------------------------------------------
    # Save comparison outputs
    # -----------------------------------------------------------------

    loo_summary.to_csv(
        output_dir
        / "loo_summary.csv",

        index=False,
    )

    comparison.to_csv(
        output_dir
        / "model_comparison.csv",

        index=False,
    )

    with (
        output_dir
        / "recovery_result.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            recovery_result,
            file,
            indent=2,
        )

    # -----------------------------------------------------------------
    # Final terminal result
    # -----------------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "Recovery result"
    )

    print(
        "========================================"
    )

    print(
        f"Generating model: {generating_model}"
    )

    print(
        f"Best model by PSIS-LOO: {best_model}"
    )

    print(
        f"Correctly recovered: {recovered_correctly}"
    )

    print(
        f"\nOutputs saved to: {output_dir}"
    )


if __name__ == "__main__":
    main()