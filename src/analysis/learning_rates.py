"""Learning-rate calculations for the two-stream pyHGF analyses.

This module contains the scientific calculations used to derive:

- trial-wise precision-weight learning rates
- block-average learning rates
- volatility effects
- cross-stream transfer indices

It deliberately contains no plotting or command-line code.
"""

from __future__ import annotations

from pathlib import Path
import math
import json

import numpy as np
import pandas as pd


MODEL_NAMES = [
    "local_only",
    "global_only",
    "shared_volatility",
]

BLOCK_ORDER = [
    "S1",
    "V1",
    "S2",
    "V2",
    "S3",
]

VOLATILE_BLOCKS = [
    "V1",
    "V2",
]

DEFAULT_MIN_TRIAL_INDEX = 1

def load_labels(model_runs_dir: Path) -> pd.DataFrame:
    """
    Load trial-wise labels.

    Expected file
    -------------
    <model_runs_dir>/model_labels.csv
    """

    labels_path = model_runs_dir / "model_labels.csv"

    if not labels_path.exists():
        raise FileNotFoundError(
            f"Could not find model labels file: {labels_path}"
        )

    labels = pd.read_csv(labels_path)

    required_columns = {
        "trial_index",
        "trial",
        "block",
        "regime_A",
        "regime_B",
    }

    missing_columns = required_columns - set(labels.columns)

    if missing_columns:
        raise ValueError(
            "model_labels.csv is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    return labels

def load_model_trajectories(
    *,
    model_name: str,
    model_runs_dir: Path,
) -> pd.DataFrame:
    """
    Load node_trajectories.csv for one inference model.
    """

    trajectories_path = (
        model_runs_dir
        / model_name
        / "node_trajectories.csv"
    )

    if not trajectories_path.exists():
        raise FileNotFoundError(
            f"Could not find node trajectories for {model_name}: "
            f"{trajectories_path}"
        )

    trajectories = pd.read_csv(trajectories_path)

    required_columns = {
        "x_0_expected_precision",
        "x_1_expected_precision",
        "x_2_expected_precision",
        "x_3_expected_precision",
        "x_2_precision",
        "x_3_precision"}

    missing_columns = required_columns - set(trajectories.columns)

    if missing_columns:
        available_columns = "\n".join(
            str(column)
            for column in trajectories.columns
        )

        raise ValueError(
            f"{model_name} trajectories are missing required columns: "
            f"{sorted(missing_columns)}\n\n"
            f"Available columns:\n{available_columns}"
        )

    return trajectories


def get_block_regimes(labels: pd.DataFrame) -> pd.DataFrame:
    """
    Return one A/B regime pair for each block.

    Also verifies that each stream has only one regime within each block.
    """

    rows = []

    for block_name in BLOCK_ORDER:
        block_df = labels[labels["block"] == block_name]

        if block_df.empty:
            raise ValueError(
                f"Could not find expected block {block_name!r} "
                "in model_labels.csv."
            )

        regimes_a = block_df["regime_A"].dropna().unique()
        regimes_b = block_df["regime_B"].dropna().unique()

        if len(regimes_a) != 1:
            raise ValueError(
                f"Block {block_name} contains multiple Stream A regimes: "
                f"{regimes_a}"
            )

        if len(regimes_b) != 1:
            raise ValueError(
                f"Block {block_name} contains multiple Stream B regimes: "
                f"{regimes_b}"
            )

        rows.append(
            {
                "block": block_name,
                "regime_A": str(regimes_a[0]),
                "regime_B": str(regimes_b[0]),
            }
        )

    return pd.DataFrame(rows)


def infer_environment_type(labels: pd.DataFrame) -> str:
    """
    Infer the current environment from A/B regime labels.

    Returns
    -------
    str
        One of:
            "local_A"
            "local_B"
            "global"
    """

    # Nominal regime labels alone cannot distinguish Hybrid from Global.
    block_regimes = get_block_regimes(labels)

    stable_blocks = block_regimes[
        block_regimes["block"].isin(["S1", "S2", "S3"])
    ]

    volatile_blocks = block_regimes[
        block_regimes["block"].isin(VOLATILE_BLOCKS)
    ]

    # Stable blocks should be objectively stable for both streams
    # in all three environments currently being studied.
    stable_blocks_valid = (
        (stable_blocks["regime_A"] == "stable").all()
        and
        (stable_blocks["regime_B"] == "stable").all()
    )

    if not stable_blocks_valid:
        raise ValueError(
            "Expected S1, S2, and S3 to be stable for both streams. "
            f"Found:\n{stable_blocks.to_string(index=False)}"
        )

    a_volatile = (
        volatile_blocks["regime_A"] == "volatile"
    ).all()

    b_volatile = (
        volatile_blocks["regime_B"] == "volatile"
    ).all()

    a_stable = (
        volatile_blocks["regime_A"] == "stable"
    ).all()

    b_stable = (
        volatile_blocks["regime_B"] == "stable"
    ).all()

    if a_volatile and b_stable:
        return "local_A"

    if a_stable and b_volatile:
        return "local_B"

    if a_volatile and b_volatile:
        return "global"

    raise ValueError(
        "Could not classify the environment as local_A, local_B, "
        "or global.\n\n"
        f"Observed block regimes:\n"
        f"{block_regimes.to_string(index=False)}"
    )

def resolve_environment_type(
    *,
    labels: pd.DataFrame,
    model_runs_dir: Path,
    requested: str = "auto",
) -> str:
    """Identify the generative environment using provenance rather than block labels.

    Hybrid and Global have the same nominal A/B regime patterns.
    """
    supported = {"local_A", "local_B", "global", "hybrid"}
    if requested != "auto" and requested not in supported:
        raise ValueError(f"Unknown requested environment type: {requested}")

    environment_dir = Path(model_runs_dir).parent
    metadata_path = environment_dir / "environment_metadata.json"
    metadata_type = None
    if metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        metadata_type = metadata.get("environment_type")
        if metadata_type is not None and metadata_type not in supported:
            raise ValueError(
                f"Unrecognized environment_type in {metadata_path}: "
                f"{metadata_type!r}"
            )

    if (
        metadata_type is not None
        and requested != "auto"
        and requested != metadata_type
    ):
        raise ValueError(
            f"Requested {requested!r} conflicts with environment metadata "
            f"({metadata_type!r})."
        )

    if requested != "auto":
        environment_type = requested
    elif metadata_type is not None:
        environment_type = metadata_type
    elif environment_dir.name in supported:
        environment_type = environment_dir.name
    else:
        environment_type = infer_environment_type(labels)

    nominal_label_type = infer_environment_type(labels)
    expected_label_type = (
        "global" if environment_type == "hybrid" else environment_type
    )
    if nominal_label_type != expected_label_type:
        raise ValueError(
            f"Environment {environment_type!r} conflicts with the A/B "
            f"regime pattern {nominal_label_type!r} in model_labels.csv."
        )

    return environment_type


def environment_display_name(environment_type: str) -> str:
    """
    Human-readable environment name.
    """

    mapping = {
        "local_A": "Local-A environment",
        "local_B": "Local-B environment",
        "global": "Global environment",
        "hybrid": "Hybrid environment",
    }

    if environment_type not in mapping:
        raise ValueError(
            f"Unknown environment type: {environment_type}"
        )

    return mapping[environment_type]


def calculate_trialwise_learning_rates(
    *,
    model_name: str,
    labels: pd.DataFrame,
    trajectories: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate trial-wise precision-weight learning rates.

    Stream A:
        input_A expected precision
        /
        value_A posterior precision

    Stream B:
        input_B expected precision
        /
        value_B posterior precision
    """

    if len(labels) != len(trajectories):
        raise ValueError(
            f"Length mismatch for {model_name}: "
            f"labels has {len(labels)} rows, "
            f"but trajectories has {len(trajectories)} rows."
        )

    output = labels.copy()
    output.insert(0, "model", model_name)

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        stream_a_learning_rate = (
            trajectories[
                "x_0_expected_precision"
            ].to_numpy(dtype=float)
            /
            trajectories[
                "x_2_precision"
            ].to_numpy(dtype=float)
        )

        stream_b_learning_rate = (
            trajectories[
                "x_1_expected_precision"
            ].to_numpy(dtype=float)
            /
            trajectories[
                "x_3_precision"
            ].to_numpy(dtype=float)
        )

    output["stream_A_learning_rate"] = (
        stream_a_learning_rate
    )

    output["stream_B_learning_rate"] = (
        stream_b_learning_rate
    )

    output["input_A_expected_precision"] = (
        trajectories[
            "x_0_expected_precision"
        ].to_numpy(dtype=float)
    )

    output["input_B_expected_precision"] = (
        trajectories[
            "x_1_expected_precision"
        ].to_numpy(dtype=float)
    )

    output["value_A_expected_precision"] = (
        trajectories[
            "x_2_expected_precision"
        ].to_numpy(dtype=float)
    )

    output["value_B_expected_precision"] = (
        trajectories[
            "x_3_expected_precision"
        ].to_numpy(dtype=float)
    )

    # Infinite values should not silently enter later summaries.
    output = output.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return output


def calculate_block_summary(
    trialwise_learning_rates: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate mean learning rate per model, stream, and block.
    """

    long_df = trialwise_learning_rates.melt(
        id_vars=[
            "model",
            "trial_index",
            "trial",
            "block",
            "regime_A",
            "regime_B",
        ],
        value_vars=[
            "stream_A_learning_rate",
            "stream_B_learning_rate",
        ],
        var_name="stream",
        value_name="learning_rate",
    )

    long_df["stream"] = long_df["stream"].map(
        {
            "stream_A_learning_rate": "Stream A",
            "stream_B_learning_rate": "Stream B",
        }
    )

    summary = (
        long_df
        .groupby(
            [
                "model",
                "stream",
                "block",
            ],
            as_index=False,
            observed=True,
        )
        .agg(
            mean_learning_rate=(
                "learning_rate",
                "mean",
            ),
            sd_learning_rate=(
                "learning_rate",
                "std",
            ),
            n_trials=(
                "learning_rate",
                "count",
            ),
        )
    )

    summary["sem_learning_rate"] = (
        summary["sd_learning_rate"]
        /
        np.sqrt(summary["n_trials"])
    )

    summary["block"] = pd.Categorical(
        summary["block"],
        categories=BLOCK_ORDER,
        ordered=True,
    )

    summary = (
        summary
        .sort_values(
            [
                "model",
                "stream",
                "block",
            ]
        )
        .reset_index(drop=True)
    )

    return summary


def get_block_mean(
    *,
    block_summary: pd.DataFrame,
    model_name: str,
    stream: str,
    block: str,
) -> float:
    """
    Return one block-average learning rate.
    """

    row = block_summary[
        (block_summary["model"] == model_name)
        &
        (block_summary["stream"] == stream)
        &
        (block_summary["block"] == block)
    ]

    if row.empty:
        raise ValueError(
            f"Could not find block mean for "
            f"model={model_name}, "
            f"stream={stream}, "
            f"block={block}."
        )

    return float(
        row["mean_learning_rate"].iloc[0]
    )


def calculate_stream_block_effects(
    *,
    block_summary: pd.DataFrame,
    model_name: str,
    stream: str,
) -> dict[str, float]:
    """
    Calculate V1 and V2 learning-rate effects for one stream.

    Effect V1:
        V1 - mean(S1, S2)

    Effect V2:
        V2 - mean(S2, S3)

    Average:
        mean(effect V1, effect V2)
    """

    s1 = get_block_mean(
        block_summary=block_summary,
        model_name=model_name,
        stream=stream,
        block="S1",
    )

    v1 = get_block_mean(
        block_summary=block_summary,
        model_name=model_name,
        stream=stream,
        block="V1",
    )

    s2 = get_block_mean(
        block_summary=block_summary,
        model_name=model_name,
        stream=stream,
        block="S2",
    )

    v2 = get_block_mean(
        block_summary=block_summary,
        model_name=model_name,
        stream=stream,
        block="V2",
    )

    s3 = get_block_mean(
        block_summary=block_summary,
        model_name=model_name,
        stream=stream,
        block="S3",
    )

    effect_v1 = (
        v1
        -
        ((s1 + s2) / 2.0)
    )

    effect_v2 = (
        v2
        -
        ((s2 + s3) / 2.0)
    )

    effect_average = (
        effect_v1 + effect_v2
    ) / 2.0

    return {
        "effect_V1": effect_v1,
        "effect_V2": effect_v2,
        "effect_average": effect_average,
    }


def calculate_effects_for_model(
    *,
    block_summary: pd.DataFrame,
    model_name: str,
    environment_type: str,
) -> dict[str, float | str | bool]:
    """
    Calculate A/B volatility-block effects and environment-specific CTI.

    local_A:
        direct effect = A
        cross-stream change = B
        CTI = B / A

    local_B:
        direct effect = B
        cross-stream change = A
        CTI = A / B

    global:
        Both streams are directly manipulated; no CTI is defined.

    hybrid:
        Both streams receive the shared drive, while their process variances
        change stochastically. Block effects are descriptive nominal contrasts;
        no CTI is defined.
    """

    a_effects = calculate_stream_block_effects(
        block_summary=block_summary,
        model_name=model_name,
        stream="Stream A",
    )

    b_effects = calculate_stream_block_effects(
        block_summary=block_summary,
        model_name=model_name,
        stream="Stream B",
    )

    a_average = a_effects["effect_average"]
    b_average = b_effects["effect_average"]

    direct_stream: str
    cross_stream: str
    direct_effect: float
    cross_stream_change: float
    cti_direction: str

    if environment_type == "local_A":
        direct_stream = "A"
        cross_stream = "B"

        direct_effect = a_average
        cross_stream_change = b_average

        cti_direction = "A_to_B"

        cti_is_interpretable = (
            direct_effect > 0.0
        )

        if cti_is_interpretable:
            cross_transfer_index = (
                cross_stream_change
                /
                direct_effect
            )
        else:
            cross_transfer_index = math.nan

        interpretation_note = (
            "Stream A is objectively volatile. "
            "Stream B effect is interpreted as cross-stream change."
        )

    elif environment_type == "local_B":
        direct_stream = "B"
        cross_stream = "A"

        direct_effect = b_average
        cross_stream_change = a_average

        cti_direction = "B_to_A"

        cti_is_interpretable = (
            direct_effect > 0.0
        )

        if cti_is_interpretable:
            cross_transfer_index = (
                cross_stream_change
                /
                direct_effect
            )
        else:
            cross_transfer_index = math.nan

        interpretation_note = (
            "Stream B is objectively volatile. "
            "Stream A effect is interpreted as cross-stream change."
        )

    elif environment_type in {"global", "hybrid"}:
        direct_stream = "A_and_B"
        cross_stream = "not_applicable"

        direct_effect = math.nan
        cross_stream_change = math.nan

        cti_direction = "not_applicable"
        cross_transfer_index = math.nan
        cti_is_interpretable = False

        if environment_type == "hybrid":
            interpretation_note = (
                "Hybrid: V/S block labels are nominal. Both streams receive "
                "the shared drive, while their process variances fluctuate. "
                "V-minus-adjacent-S learning-rate differences are descriptive "
                "block contrasts, not isolated volatility effects; "
                "cross-transfer index is not defined."
            )
        else:
            interpretation_note = (
                "Both streams are objectively volatile. "
                "A and B effects are both direct volatility effects; "
                "cross-transfer index is not applicable."
            )

    else:
        raise ValueError(
            f"Unknown environment type: {environment_type}"
        )

    return {
        "environment_type": environment_type,
        "model": model_name,

        "stream_A_effect_V1": (
            a_effects["effect_V1"]
        ),
        "stream_A_effect_V2": (
            a_effects["effect_V2"]
        ),
        "stream_A_effect_average": (
            a_effects["effect_average"]
        ),

        "stream_B_effect_V1": (
            b_effects["effect_V1"]
        ),
        "stream_B_effect_V2": (
            b_effects["effect_V2"]
        ),
        "stream_B_effect_average": (
            b_effects["effect_average"]
        ),

        "direct_stream": direct_stream,
        "cross_stream": cross_stream,

        "direct_volatility_effect_average": (
            direct_effect
        ),

        "cross_stream_change_average": (
            cross_stream_change
        ),

        "cross_transfer_direction": (
            cti_direction
        ),

        "cross_transfer_index": (
            cross_transfer_index
        ),

        "cross_transfer_index_interpretable": (
            cti_is_interpretable
        ),

        "interpretation_note": (
            interpretation_note
        ),
    }


def calculate_effect_summary(
    *,
    block_summary: pd.DataFrame,
    model_names: list[str],
    environment_type: str,
) -> pd.DataFrame:
    """
    Calculate volatility-effect summary for all models.
    """

    rows = []

    for model_name in model_names:
        rows.append(
            calculate_effects_for_model(
                block_summary=block_summary,
                model_name=model_name,
                environment_type=environment_type,
            )
        )

    return pd.DataFrame(rows)
