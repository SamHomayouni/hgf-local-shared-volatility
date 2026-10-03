"""
Continuous Gaussian prediction response model.

This module implements the first response model for the two-stream
global-vs-local volatility project.

Conceptual model
----------------
The environment generates observations:

    u_A(t), u_B(t)

The perceptual model receives those observations and forms predictions:

    mu_hat_A(t), mu_hat_B(t)

The response model generates or evaluates continuous participant predictions:

    y_A(t) ~ Normal(mu_hat_A(t), sigma_resp_A^2)
    y_B(t) ~ Normal(mu_hat_B(t), sigma_resp_B^2)

For the first version, sigma_resp_A and sigma_resp_B can be set equal.

Timing
------
The response should be compared to the model's pre-outcome prediction for
the same trial. In pyhgf trajectory exports this is expected to correspond
to:

    x_2_expected_mean  for stream A value prediction
    x_3_expected_mean  for stream B value prediction

where:
    node 2 = value_A
    node 3 = value_B

The module also includes a pyhgf-compatible response function:

    two_stream_gaussian_prediction_surprise

This function returns trial-wise surprise values, i.e. negative log-likelihoods,
for use with pyhgf's surprise machinery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


try:
    import jax.numpy as jnp
except ImportError:  # pragma: no cover
    jnp = None


PredictionColumn = Literal["expected_mean", "mean"]


@dataclass(frozen=True)
class ContinuousPredictionResponseParameters:
    """
    Parameters of the continuous Gaussian response model.

    Parameters
    ----------
    response_sd_A:
        Standard deviation of response noise for stream A.

    response_sd_B:
        Standard deviation of response noise for stream B.
    """

    response_sd_A: float
    response_sd_B: float

    @classmethod
    def shared_noise(
        cls,
        response_sd: float,
    ) -> "ContinuousPredictionResponseParameters":
        """
        Create response parameters with the same noise for A and B.
        """
        return cls(response_sd_A=response_sd, response_sd_B=response_sd)


def validate_positive_sd(value: float, name: str) -> None:
    """
    Validate that a standard deviation is positive.
    """
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")

    if value <= 0.0:
        raise ValueError(f"{name} must be positive.")


def expected_mean_column(node_id: int, prediction_column: PredictionColumn) -> str:
    """
    Construct the expected pyhgf trajectory column name.

    Examples
    --------
    node_id=2, prediction_column="expected_mean"
        -> "x_2_expected_mean"

    node_id=2, prediction_column="mean"
        -> "x_2_mean"
    """
    return f"x_{node_id}_{prediction_column}"


def extract_model_predictions_from_trajectories(
    trajectories: pd.DataFrame,
    *,
    value_node_A: int = 2,
    value_node_B: int = 3,
    prediction_column: PredictionColumn = "expected_mean",
) -> pd.DataFrame:
    """
    Extract model predictions for A and B from pyhgf trajectory output.

    Parameters
    ----------
    trajectories:
        DataFrame created from model.to_pandas(), saved as node_trajectories.csv.

    value_node_A:
        Node ID of value_A. Default is 2.

    value_node_B:
        Node ID of value_B. Default is 3.

    prediction_column:
        Which pyhgf trajectory quantity should be used as the response-model
        prediction.

        Default is "expected_mean", because the participant's response should be
        compared to the model's pre-outcome prediction.

        Use "mean" only for debugging or if expected_mean is unavailable and you
        explicitly want posterior values.

    Returns
    -------
    pandas.DataFrame
        Columns:
            model_prediction_A
            model_prediction_B
    """

    column_A = expected_mean_column(value_node_A, prediction_column)
    column_B = expected_mean_column(value_node_B, prediction_column)

    required_columns = {column_A, column_B}
    missing_columns = required_columns - set(trajectories.columns)

    if missing_columns:
        available_columns = "\n".join(trajectories.columns)
        raise ValueError(
            "Could not extract model predictions. "
            f"Missing columns: {sorted(missing_columns)}\n\n"
            f"Available columns:\n{available_columns}"
        )

    predictions = pd.DataFrame(
        {
            "model_prediction_A": trajectories[column_A].to_numpy(dtype=float),
            "model_prediction_B": trajectories[column_B].to_numpy(dtype=float),
        }
    )

    if predictions.isna().any().any():
        raise ValueError("Extracted model predictions contain NaN values.")

    return predictions


def gaussian_trialwise_surprise_numpy(
    *,
    responses: np.ndarray,
    predictions: np.ndarray,
    response_sd_A: float,
    response_sd_B: float,
    include_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute trial-wise negative log-likelihood for two-stream responses.

    Parameters
    ----------
    responses:
        Array of shape (n_trials, 2):
            responses[:, 0] = participant prediction for stream A
            responses[:, 1] = participant prediction for stream B

    predictions:
        Array of shape (n_trials, 2):
            predictions[:, 0] = model prediction for stream A
            predictions[:, 1] = model prediction for stream B

    response_sd_A:
        Response noise standard deviation for stream A.

    response_sd_B:
        Response noise standard deviation for stream B.

    include_mask:
        Optional boolean array of shape (n_trials,). Trials with False are
        assigned zero surprise. This is useful for excluding trial 0.

    Returns
    -------
    numpy.ndarray
        Trial-wise surprise values, one per trial.
    """

    validate_positive_sd(response_sd_A, "response_sd_A")
    validate_positive_sd(response_sd_B, "response_sd_B")

    responses = np.asarray(responses, dtype=float)
    predictions = np.asarray(predictions, dtype=float)

    if responses.ndim != 2 or responses.shape[1] != 2:
        raise ValueError(
            f"responses must have shape (n_trials, 2), got {responses.shape}."
        )

    if predictions.ndim != 2 or predictions.shape[1] != 2:
        raise ValueError(
            f"predictions must have shape (n_trials, 2), got {predictions.shape}."
        )

    if len(responses) != len(predictions):
        raise ValueError(
            f"responses has {len(responses)} rows, but predictions has "
            f"{len(predictions)} rows."
        )

    if include_mask is None:
        include_mask = np.ones(len(responses), dtype=bool)
    else:
        include_mask = np.asarray(include_mask, dtype=bool)

        if include_mask.shape != (len(responses),):
            raise ValueError(
                "include_mask must have shape (n_trials,), got "
                f"{include_mask.shape}."
            )

    response_A = responses[:, 0]
    response_B = responses[:, 1]

    prediction_A = predictions[:, 0]
    prediction_B = predictions[:, 1]

    valid = (
        include_mask
        & np.isfinite(response_A)
        & np.isfinite(response_B)
        & np.isfinite(prediction_A)
        & np.isfinite(prediction_B)
    )

    surprise_A = (
        0.5 * np.log(2.0 * np.pi * response_sd_A**2)
        + ((response_A - prediction_A) ** 2) / (2.0 * response_sd_A**2)
    )

    surprise_B = (
        0.5 * np.log(2.0 * np.pi * response_sd_B**2)
        + ((response_B - prediction_B) ** 2) / (2.0 * response_sd_B**2)
    )

    surprise = surprise_A + surprise_B
    surprise = np.where(valid, surprise, 0.0)

    return surprise


def gaussian_negative_log_likelihood_numpy(
    *,
    responses: np.ndarray,
    predictions: np.ndarray,
    response_sd_A: float,
    response_sd_B: float,
    include_mask: np.ndarray | None = None,
) -> float:
    """
    Compute total negative log-likelihood for two-stream responses.
    """
    surprise = gaussian_trialwise_surprise_numpy(
        responses=responses,
        predictions=predictions,
        response_sd_A=response_sd_A,
        response_sd_B=response_sd_B,
        include_mask=include_mask,
    )

    return float(np.sum(surprise))


def simulate_continuous_prediction_responses(
    *,
    predictions: pd.DataFrame,
    response_parameters: ContinuousPredictionResponseParameters,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Simulate continuous participant predictions from model predictions.

    Parameters
    ----------
    predictions:
        DataFrame with columns:
            model_prediction_A
            model_prediction_B

    response_parameters:
        Response noise parameters.

    rng:
        NumPy random generator.

    Returns
    -------
    pandas.DataFrame
        Columns:
            participant_prediction_A
            participant_prediction_B
            response_noise_A
            response_noise_B
    """

    required_columns = {
        "model_prediction_A",
        "model_prediction_B",
    }

    missing_columns = required_columns - set(predictions.columns)

    if missing_columns:
        raise ValueError(
            f"predictions is missing required columns: {sorted(missing_columns)}"
        )

    validate_positive_sd(
        response_parameters.response_sd_A,
        "response_sd_A",
    )
    validate_positive_sd(
        response_parameters.response_sd_B,
        "response_sd_B",
    )

    n_trials = len(predictions)

    response_noise_A = rng.normal(
        loc=0.0,
        scale=response_parameters.response_sd_A,
        size=n_trials,
    )

    response_noise_B = rng.normal(
        loc=0.0,
        scale=response_parameters.response_sd_B,
        size=n_trials,
    )

    participant_prediction_A = (
        predictions["model_prediction_A"].to_numpy(dtype=float)
        + response_noise_A
    )

    participant_prediction_B = (
        predictions["model_prediction_B"].to_numpy(dtype=float)
        + response_noise_B
    )

    return pd.DataFrame(
        {
            "participant_prediction_A": participant_prediction_A,
            "participant_prediction_B": participant_prediction_B,
            "response_noise_A": response_noise_A,
            "response_noise_B": response_noise_B,
        }
    )


def add_behavioral_learning_rates(
    response_df: pd.DataFrame,
    *,
    denominator_epsilon: float = 1e-8,
) -> pd.DataFrame:
    """
    Add Nassar-style behavioural learning-rate columns.

    For stream s:

        prediction_error_s(t) = observation_s(t) - participant_prediction_s(t)

        update_s(t) = participant_prediction_s(t + 1)
                      - participant_prediction_s(t)

        behavioural_learning_rate_s(t) = update_s(t) / prediction_error_s(t)

    The final trial has no next response, so its update and behavioural learning
    rate are NaN.
    """

    output = response_df.copy()

    for stream in ["A", "B"]:
        observation_col = f"observation_{stream}"
        prediction_col = f"participant_prediction_{stream}"

        if observation_col not in output.columns:
            raise ValueError(f"Missing required column: {observation_col}")

        if prediction_col not in output.columns:
            raise ValueError(f"Missing required column: {prediction_col}")

        prediction_error_col = f"prediction_error_{stream}"
        update_col = f"response_update_{stream}"
        learning_rate_col = f"behavioral_learning_rate_{stream}"

        prediction_error = (
            output[observation_col].to_numpy(dtype=float)
            - output[prediction_col].to_numpy(dtype=float)
        )

        next_prediction = output[prediction_col].shift(-1).to_numpy(dtype=float)
        current_prediction = output[prediction_col].to_numpy(dtype=float)

        update = next_prediction - current_prediction

        with np.errstate(divide="ignore", invalid="ignore"):
            learning_rate = update / prediction_error

        learning_rate[np.abs(prediction_error) < denominator_epsilon] = np.nan

        output[prediction_error_col] = prediction_error
        output[update_col] = update
        output[learning_rate_col] = learning_rate

    return output


def make_response_function_inputs(
    response_df: pd.DataFrame,
    *,
    include_column: str = "include_in_response_likelihood",
) -> np.ndarray:
    """
    Create a response_function_inputs matrix for pyhgf.

    Output columns:
        0. participant_prediction_A
        1. participant_prediction_B
        2. include_in_response_likelihood

    The third column is a 0/1 mask. Trials with 0 are ignored by the custom
    response function.
    """

    required_columns = {
        "participant_prediction_A",
        "participant_prediction_B",
        include_column,
    }

    missing_columns = required_columns - set(response_df.columns)

    if missing_columns:
        raise ValueError(
            f"response_df is missing required columns: {sorted(missing_columns)}"
        )

    return response_df[
        [
            "participant_prediction_A",
            "participant_prediction_B",
            include_column,
        ]
    ].to_numpy(dtype=float)


def two_stream_gaussian_prediction_surprise(
    hgf,
    response_function_inputs=None,
    response_function_parameters=None,
):
    """
    pyhgf-compatible response function for two-stream continuous predictions.

    Parameters
    ----------
    hgf:
        A pyhgf model after receiving input data.

    response_function_inputs:
        Array-like with shape (n_trials, 2) or (n_trials, 3).

        Column 0:
            participant prediction for stream A.

        Column 1:
            participant prediction for stream B.

        Optional column 2:
            include mask. 1 means include trial in likelihood; 0 means ignore.
            This is useful for excluding the first trial.

    response_function_parameters:
        Array-like response parameters.

        If length 1:
            [log_sigma_resp]
            The same response noise is used for both streams.

        If length 2:
            [log_sigma_resp_A, log_sigma_resp_B]
            Stream-specific response noise is used.

    Returns
    -------
    jax.numpy.ndarray
        Trial-wise surprise values. Sum them to obtain total response surprise.

    Notes
    -----
    The response model uses:
        hgf.node_trajectories[2]["expected_mean"]
        hgf.node_trajectories[3]["expected_mean"]

    These correspond to value_A and value_B pre-outcome predictions in the
    current project node mapping.
    """

    if jnp is None:
        raise ImportError(
            "jax is required for pyhgf-compatible response functions."
        )

    if response_function_inputs is None:
        return jnp.inf

    responses = jnp.asarray(response_function_inputs)

    if responses.ndim != 2:
        raise ValueError(
            "response_function_inputs must be a 2D array with columns "
            "[prediction_A, prediction_B] or "
            "[prediction_A, prediction_B, include_mask]."
        )

    if responses.shape[1] < 2:
        raise ValueError(
            "response_function_inputs must have at least two columns."
        )

    y_A = responses[:, 0]
    y_B = responses[:, 1]

    if responses.shape[1] >= 3:
        include_mask = responses[:, 2] > 0.0
    else:
        include_mask = jnp.ones_like(y_A, dtype=bool)

    if response_function_parameters is None:
        parameters = jnp.asarray([0.0])
    else:
        parameters = jnp.atleast_1d(jnp.asarray(response_function_parameters))

    log_sigma_A = parameters[0]

    if parameters.shape[0] >= 2:
        log_sigma_B = parameters[1]
    else:
        log_sigma_B = parameters[0]

    sigma_A = jnp.exp(log_sigma_A)
    sigma_B = jnp.exp(log_sigma_B)

    mu_A = hgf.node_trajectories[2]["expected_mean"]
    mu_B = hgf.node_trajectories[3]["expected_mean"]

    valid = (
        include_mask
        & jnp.isfinite(y_A)
        & jnp.isfinite(y_B)
        & jnp.isfinite(mu_A)
        & jnp.isfinite(mu_B)
    )

    surprise_A = (
        0.5 * jnp.log(2.0 * jnp.pi * sigma_A**2)
        + ((y_A - mu_A) ** 2) / (2.0 * sigma_A**2)
    )

    surprise_B = (
        0.5 * jnp.log(2.0 * jnp.pi * sigma_B**2)
        + ((y_B - mu_B) ** 2) / (2.0 * sigma_B**2)
    )

    surprise = surprise_A + surprise_B
    surprise = jnp.where(valid, surprise, 0.0)

    model_has_nan = jnp.any(jnp.isnan(mu_A)) | jnp.any(jnp.isnan(mu_B))

    return jnp.where(
        model_has_nan,
        jnp.full_like(surprise, jnp.inf),
        surprise,
    )