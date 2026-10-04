"""Tests for shared model-input preparation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pipeline.model_runner import prepare_model_input


def make_two_stream_environment() -> pd.DataFrame:
    """Create a tiny valid environment in deliberately shuffled row order."""

    return pd.DataFrame(
        [
            {
                "trial_index": 1,
                "trial": 2,
                "stream": "B",
                "block": "S1",
                "regime": "stable",
                "latent_mean": 1.0,
                "observation": 1.2,
                "changepoint": False,
            },
            {
                "trial_index": 0,
                "trial": 1,
                "stream": "A",
                "block": "S1",
                "regime": "stable",
                "latent_mean": 0.0,
                "observation": 0.1,
                "changepoint": False,
            },
            {
                "trial_index": 1,
                "trial": 2,
                "stream": "A",
                "block": "S1",
                "regime": "stable",
                "latent_mean": 0.0,
                "observation": 0.2,
                "changepoint": False,
            },
            {
                "trial_index": 0,
                "trial": 1,
                "stream": "B",
                "block": "S1",
                "regime": "stable",
                "latent_mean": 1.0,
                "observation": 1.1,
                "changepoint": False,
            },
        ]
    )


def test_prepare_model_input_preserves_trial_and_stream_order() -> None:
    environment = make_two_stream_environment()

    prepared = prepare_model_input(environment)

    expected = np.array(
        [
            [0.1, 1.1],
            [0.2, 1.2],
        ]
    )

    np.testing.assert_allclose(
        prepared.input_data,
        expected,
    )

    assert prepared.input_data.shape == (2, 2)
    assert prepared.stream_names == ("A", "B")

    assert prepared.labels["trial_index"].tolist() == [0, 1]

    np.testing.assert_allclose(
        prepared.labels["observation_A"],
        [0.1, 0.2],
    )

    np.testing.assert_allclose(
        prepared.labels["observation_B"],
        [1.1, 1.2],
    )


def test_prepare_model_input_rejects_nan_observation() -> None:
    environment = make_two_stream_environment()

    environment.loc[
        (
            (environment["trial_index"] == 0)
            & (environment["stream"] == "A")
        ),
        "observation",
    ] = np.nan

    with pytest.raises(
        ValueError,
        match="input_data contains NaN values",
    ):
        prepare_model_input(environment)


def test_prepare_model_input_rejects_missing_stream() -> None:
    environment = make_two_stream_environment()

    environment = environment[
        environment["stream"] == "A"
    ].copy()

    with pytest.raises(
        ValueError,
        match="Expected streams",
    ):
        prepare_model_input(environment)