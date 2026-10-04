"""Tests for HGF learning-rate and cross-transfer calculations."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.analysis.learning_rates import (
    calculate_effects_for_model,
    calculate_stream_block_effects,
    calculate_trialwise_learning_rates,
    infer_environment_type,
)


MODEL_NAME = "test_model"


def make_environment_labels(
    environment_type: str,
) -> pd.DataFrame:
    """Create one representative row for each experimental block."""

    if environment_type == "local_A":
        volatile_regimes = ("volatile", "stable")
    elif environment_type == "local_B":
        volatile_regimes = ("stable", "volatile")
    elif environment_type == "global":
        volatile_regimes = ("volatile", "volatile")
    else:
        raise ValueError(environment_type)

    rows = []

    for trial_index, block in enumerate(
        ["S1", "V1", "S2", "V2", "S3"]
    ):
        if block.startswith("S"):
            regime_a = "stable"
            regime_b = "stable"
        else:
            regime_a, regime_b = volatile_regimes

        rows.append(
            {
                "trial_index": trial_index,
                "trial": trial_index + 1,
                "block": block,
                "regime_A": regime_a,
                "regime_B": regime_b,
            }
        )

    return pd.DataFrame(rows)


def make_block_summary() -> pd.DataFrame:
    """Create block means with analytically known effects.

    Stream A:
        S1 = 1
        V1 = 5
        S2 = 3
        V2 = 9
        S3 = 5

        D_A_1 = 5 - mean(1, 3) = 3
        D_A_2 = 9 - mean(3, 5) = 5
        D_A   = mean(3, 5) = 4

    Stream B:
        S1 = 2
        V1 = 4
        S2 = 2
        V2 = 6
        S3 = 2

        D_B_1 = 4 - mean(2, 2) = 2
        D_B_2 = 6 - mean(2, 2) = 4
        D_B   = mean(2, 4) = 3
    """

    values = {
        "Stream A": {
            "S1": 1.0,
            "V1": 5.0,
            "S2": 3.0,
            "V2": 9.0,
            "S3": 5.0,
        },
        "Stream B": {
            "S1": 2.0,
            "V1": 4.0,
            "S2": 2.0,
            "V2": 6.0,
            "S3": 2.0,
        },
    }

    rows = []

    for stream, block_values in values.items():
        for block, mean_value in block_values.items():
            rows.append(
                {
                    "model": MODEL_NAME,
                    "stream": stream,
                    "block": block,
                    "mean_learning_rate": mean_value,
                }
            )

    return pd.DataFrame(rows)


def test_trialwise_learning_rate_is_precision_ratio() -> None:
    labels = pd.DataFrame(
        {
            "trial_index": [0, 1],
            "trial": [1, 2],
            "block": ["S1", "S1"],
            "regime_A": ["stable", "stable"],
            "regime_B": ["stable", "stable"],
        }
    )

    trajectories = pd.DataFrame(
        {
            "x_0_expected_precision": [10.0, 20.0],
            "x_1_expected_precision": [8.0, 12.0],
            "x_2_expected_precision": [5.0, 4.0],
            "x_3_expected_precision": [2.0, 3.0],
        }
    )

    result = calculate_trialwise_learning_rates(
        model_name=MODEL_NAME,
        labels=labels,
        trajectories=trajectories,
    )

    np.testing.assert_allclose(
        result["stream_A_learning_rate"],
        [2.0, 5.0],
    )

    np.testing.assert_allclose(
        result["stream_B_learning_rate"],
        [4.0, 4.0],
    )


def test_block_effect_calculation() -> None:
    block_summary = make_block_summary()

    a_effects = calculate_stream_block_effects(
        block_summary=block_summary,
        model_name=MODEL_NAME,
        stream="Stream A",
    )

    b_effects = calculate_stream_block_effects(
        block_summary=block_summary,
        model_name=MODEL_NAME,
        stream="Stream B",
    )

    assert a_effects["effect_V1"] == pytest.approx(3.0)
    assert a_effects["effect_V2"] == pytest.approx(5.0)
    assert a_effects["effect_average"] == pytest.approx(4.0)

    assert b_effects["effect_V1"] == pytest.approx(2.0)
    assert b_effects["effect_V2"] == pytest.approx(4.0)
    assert b_effects["effect_average"] == pytest.approx(3.0)

@pytest.mark.parametrize(
    ("environment_type", "expected"),
    [
        ("local_A", "local_A"),
        ("local_B", "local_B"),
        ("global", "global"),
    ],
)
def test_environment_type_inference(
    environment_type: str,
    expected: str,
) -> None:
    labels = make_environment_labels(environment_type)

    assert infer_environment_type(labels) == expected

def test_local_A_cti_is_B_effect_divided_by_A_effect() -> None:
    result = calculate_effects_for_model(
        block_summary=make_block_summary(),
        model_name=MODEL_NAME,
        environment_type="local_A",
    )

    assert result["direct_stream"] == "A"
    assert result["cross_stream"] == "B"

    assert result[
        "direct_volatility_effect_average"
    ] == pytest.approx(4.0)

    assert result[
        "cross_stream_change_average"
    ] == pytest.approx(3.0)

    assert result[
        "cross_transfer_index"
    ] == pytest.approx(3.0 / 4.0)

    assert result[
        "cross_transfer_direction"
    ] == "A_to_B"

    assert result[
        "cross_transfer_index_interpretable"
    ] is True


def test_local_B_cti_is_A_effect_divided_by_B_effect() -> None:
    result = calculate_effects_for_model(
        block_summary=make_block_summary(),
        model_name=MODEL_NAME,
        environment_type="local_B",
    )

    assert result["direct_stream"] == "B"
    assert result["cross_stream"] == "A"

    assert result[
        "direct_volatility_effect_average"
    ] == pytest.approx(3.0)

    assert result[
        "cross_stream_change_average"
    ] == pytest.approx(4.0)

    assert result[
        "cross_transfer_index"
    ] == pytest.approx(4.0 / 3.0)

    assert result[
        "cross_transfer_direction"
    ] == "B_to_A"

    assert result[
        "cross_transfer_index_interpretable"
    ] is True


def test_global_environment_has_no_cti() -> None:
    result = calculate_effects_for_model(
        block_summary=make_block_summary(),
        model_name=MODEL_NAME,
        environment_type="global",
    )

    assert result["direct_stream"] == "A_and_B"
    assert result["cross_stream"] == "not_applicable"

    assert math.isnan(
        result["cross_transfer_index"]
    )

    assert (
        result["cross_transfer_index_interpretable"]
        is False
    )

    assert result[
        "cross_transfer_direction"
    ] == "not_applicable"


def test_cti_is_not_interpretable_when_direct_effect_is_nonpositive() -> None:
    block_summary = make_block_summary()

    mask = (
        (block_summary["stream"] == "Stream A")
        & (block_summary["block"].isin(["V1", "V2"]))
    )

    block_summary.loc[
        mask,
        "mean_learning_rate",
    ] = 0.0

    result = calculate_effects_for_model(
        block_summary=block_summary,
        model_name=MODEL_NAME,
        environment_type="local_A",
    )

    assert (
        result["direct_volatility_effect_average"]
        < 0.0
    )

    assert math.isnan(
        result["cross_transfer_index"]
    )

    assert (
        result["cross_transfer_index_interpretable"]
        is False
    )

