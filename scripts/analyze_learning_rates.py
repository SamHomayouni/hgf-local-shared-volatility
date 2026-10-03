"""
Analyze trial-wise HGF learning rates across local-A, local-B, and
global-like volatility environments.

Learning-rate definition
------------------------
For every inference model:

    Stream A:
        learning_rate_A(k)
        = x_0_expected_precision(k) / x_2_expected_precision(k)

    Stream B:
        learning_rate_B(k)
        = x_1_expected_precision(k) / x_3_expected_precision(k)

where:

    node 0 = input_A
    node 1 = input_B
    node 2 = value_A
    node 3 = value_B

These node IDs are shared by all three current inference models:

    local_only
    global_only
    shared_volatility

Environment types
-----------------
The script automatically infers the environment type from model_labels.csv.

local_A:
    V1 and V2:
        A volatile
        B stable

local_B:
    V1 and V2:
        A stable
        B volatile

global:
    V1 and V2:
        A volatile
        B volatile

Block-wise volatility effects
-----------------------------
For Stream A:

    D_A_1 =
        mean(A in V1)
        - [mean(A in S1) + mean(A in S2)] / 2

    D_A_2 =
        mean(A in V2)
        - [mean(A in S2) + mean(A in S3)] / 2

    D_A = (D_A_1 + D_A_2) / 2

For Stream B:

    D_B_1 =
        mean(B in V1)
        - [mean(B in S1) + mean(B in S2)] / 2

    D_B_2 =
        mean(B in V2)
        - [mean(B in S2) + mean(B in S3)] / 2

    D_B = (D_B_1 + D_B_2) / 2

Interpretation
--------------
local_A:
    D_A = direct volatility effect
    D_B = A -> B cross-stream change

    CTI_A_to_B = D_B / D_A

local_B:
    D_B = direct volatility effect
    D_A = B -> A cross-stream change

    CTI_B_to_A = D_A / D_B

global:
    D_A = A's volatility effect
    D_B = B's volatility effect

    CTI is not defined because both streams are objectively volatile.

Inputs
------
<model_runs_dir>/model_labels.csv

<model_runs_dir>/local_only/node_trajectories.csv
<model_runs_dir>/global_only/node_trajectories.csv
<model_runs_dir>/shared_volatility/node_trajectories.csv

Outputs
-------
<output_dir>/trialwise_learning_rates_raw_including_initial_trial.csv
<output_dir>/trialwise_learning_rates.csv
<output_dir>/block_learning_rate_summary.csv
<output_dir>/volatility_effect_summary.csv

Figures
-------
<figure_dir>/<model_name>_learning_rates.png
<figure_dir>/<model_name>_block_average_learning_rates.png
<figure_dir>/volatility_effect_summary.png
"""

from __future__ import annotations

from pathlib import Path
import argparse
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_MODEL_RUNS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "hgf_walk_outputs"
    / "local_A"
    / "model_runs"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "hgf_walk_outputs"
    / "local_A"
    / "learning_rates"
)

DEFAULT_FIGURE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "hgf_walk_outputs"
    / "local_A"
    / "figures"
    / "learning_rates"
)


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


# ---------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------

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
    }

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


# ---------------------------------------------------------------------
# Environment classification
# ---------------------------------------------------------------------

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


def environment_display_name(environment_type: str) -> str:
    """
    Human-readable environment name.
    """

    mapping = {
        "local_A": "Local-A environment",
        "local_B": "Local-B environment",
        "global": "Global environment",
    }

    if environment_type not in mapping:
        raise ValueError(
            f"Unknown environment type: {environment_type}"
        )

    return mapping[environment_type]


# ---------------------------------------------------------------------
# Learning-rate calculation
# ---------------------------------------------------------------------

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
        value_A expected precision

    Stream B:
        input_B expected precision
        /
        value_B expected precision
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
                "x_2_expected_precision"
            ].to_numpy(dtype=float)
        )

        stream_b_learning_rate = (
            trajectories[
                "x_1_expected_precision"
            ].to_numpy(dtype=float)
            /
            trajectories[
                "x_3_expected_precision"
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


# ---------------------------------------------------------------------
# Block summaries
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Volatility effects and cross-stream transfer
# ---------------------------------------------------------------------

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
        A and B are both directly volatile.
        No CTI is defined.
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

    elif environment_type == "global":
        direct_stream = "A_and_B"
        cross_stream = "not_applicable"

        direct_effect = math.nan
        cross_stream_change = math.nan

        cti_direction = "not_applicable"
        cross_transfer_index = math.nan
        cti_is_interpretable = False

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


# ---------------------------------------------------------------------
# Volatility-block plotting helpers
# ---------------------------------------------------------------------

def get_block_volatility_label(
    labels: pd.DataFrame,
    block: str,
) -> str | None:
    """
    Return a human-readable volatility label for one block.

    Possible returns:
        None
        "A volatile"
        "B volatile"
        "A + B volatile"
    """

    block_rows = labels[
        labels["block"] == block
    ]

    if block_rows.empty:
        return None

    a_volatile = (
        block_rows["regime_A"]
        == "volatile"
    ).any()

    b_volatile = (
        block_rows["regime_B"]
        == "volatile"
    ).any()

    if a_volatile and b_volatile:
        return "A + B volatile"

    if a_volatile:
        return "A volatile"

    if b_volatile:
        return "B volatile"

    return None


def add_volatile_block_shading(
    ax: plt.Axes,
    labels: pd.DataFrame,
) -> None:
    """
    Shade all objectively volatile blocks.

    Labels automatically reflect whether:
        A is volatile,
        B is volatile,
        or both are volatile.
    """

    used_labels: set[str] = set()

    for block in BLOCK_ORDER:
        volatility_label = get_block_volatility_label(
            labels,
            block,
        )

        if volatility_label is None:
            continue

        block_rows = labels[
            labels["block"] == block
        ]

        start_trial = int(
            block_rows["trial_index"].min()
        )

        stop_trial = (
            int(block_rows["trial_index"].max())
            + 1
        )

        legend_label = (
            volatility_label
            if volatility_label not in used_labels
            else None
        )

        ax.axvspan(
            start_trial,
            stop_trial,
            alpha=0.12,
            label=legend_label,
        )

        used_labels.add(volatility_label)


def add_block_boundaries(
    ax: plt.Axes,
    labels: pd.DataFrame,
) -> None:
    """
    Add vertical block-boundary lines.
    """

    block_start_indices = (
        labels
        .groupby(
            "block",
            observed=True,
        )["trial_index"]
        .min()
        .reindex(BLOCK_ORDER)
        .dropna()
        .to_numpy()
    )

    for start_index in block_start_indices:
        ax.axvline(
            x=start_index,
            linewidth=0.8,
            linestyle="--",
            alpha=0.4,
        )


# ---------------------------------------------------------------------
# Plot text
# ---------------------------------------------------------------------

def add_effect_text_box(
    *,
    ax: plt.Axes,
    effect_row: pd.Series,
    environment_type: str,
) -> None:
    """
    Add environment-appropriate summary information to a plot.
    """

    a_effect = float(
        effect_row[
            "stream_A_effect_average"
        ]
    )

    b_effect = float(
        effect_row[
            "stream_B_effect_average"
        ]
    )

    if environment_type == "local_A":
        cti = effect_row[
            "cross_transfer_index"
        ]

        if pd.isna(cti):
            cti_text = "not interpretable"
        else:
            cti_text = f"{cti:.3f}"

        text = (
            "Volatility summary\n"
            f"A direct effect: {a_effect:.4f}\n"
            f"B cross-stream change: {b_effect:.4f}\n"
            f"CTI A -> B: {cti_text}"
        )

    elif environment_type == "local_B":
        cti = effect_row[
            "cross_transfer_index"
        ]

        if pd.isna(cti):
            cti_text = "not interpretable"
        else:
            cti_text = f"{cti:.3f}"

        text = (
            "Volatility summary\n"
            f"B direct effect: {b_effect:.4f}\n"
            f"A cross-stream change: {a_effect:.4f}\n"
            f"CTI B -> A: {cti_text}"
        )

    elif environment_type == "global":
        text = (
            "Volatility summary\n"
            f"A volatility effect: {a_effect:.4f}\n"
            f"B volatility effect: {b_effect:.4f}\n"
            "CTI: not applicable"
        )

    else:
        raise ValueError(
            f"Unknown environment type: "
            f"{environment_type}"
        )

    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": "black",
            "alpha": 0.85,
        },
    )


# ---------------------------------------------------------------------
# Trial-wise learning-rate plot
# ---------------------------------------------------------------------

def plot_learning_rates_for_model(
    *,
    model_name: str,
    trialwise: pd.DataFrame,
    effect_summary: pd.DataFrame,
    environment_type: str,
    figure_dir: Path,
) -> None:
    """
    Plot trial-wise learning rates for one model.
    """

    model_df = trialwise[
        trialwise["model"] == model_name
    ].copy()

    if model_df.empty:
        raise ValueError(
            f"No trial-wise learning rates "
            f"found for {model_name}."
        )

    effect_row = effect_summary[
        effect_summary["model"] == model_name
    ].iloc[0]

    fig, ax = plt.subplots(
        figsize=(14, 6)
    )

    ax.plot(
        model_df["trial_index"],
        model_df[
            "stream_A_learning_rate"
        ],
        linewidth=1.5,
        label="Stream A learning rate",
    )

    ax.plot(
        model_df["trial_index"],
        model_df[
            "stream_B_learning_rate"
        ],
        linewidth=1.5,
        label="Stream B learning rate",
    )

    add_volatile_block_shading(
        ax,
        model_df,
    )

    add_block_boundaries(
        ax,
        model_df,
    )

    add_effect_text_box(
        ax=ax,
        effect_row=effect_row,
        environment_type=environment_type,
    )

    environment_name = (
        environment_display_name(
            environment_type
        )
    )

    ax.set_title(
        f"{model_name}: trial-wise learning rates\n"
        f"{environment_name}",
        fontsize=15,
        fontweight="bold",
    )

    ax.set_xlabel("Trial")

    ax.set_ylabel(
        "Learning rate / precision coefficient"
    )

    ax.legend(
        loc="upper right"
    )

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        figure_dir
        / f"{model_name}_learning_rates.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved figure: {output_path}"
    )


# ---------------------------------------------------------------------
# Block-average plot
# ---------------------------------------------------------------------

def plot_block_means_for_model(
    *,
    model_name: str,
    block_summary: pd.DataFrame,
    labels: pd.DataFrame,
    environment_type: str,
    figure_dir: Path,
) -> None:
    """
    Plot block-wise average learning rates for one model.
    """

    model_summary = block_summary[
        block_summary["model"] == model_name
    ].copy()

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    for stream_name in [
        "Stream A",
        "Stream B",
    ]:
        stream_summary = model_summary[
            model_summary["stream"]
            == stream_name
        ].copy()

        stream_summary["block"] = (
            pd.Categorical(
                stream_summary["block"],
                categories=BLOCK_ORDER,
                ordered=True,
            )
        )

        stream_summary = (
            stream_summary
            .sort_values("block")
        )

        ax.plot(
            stream_summary[
                "block"
            ].astype(str),
            stream_summary[
                "mean_learning_rate"
            ],
            marker="o",
            linewidth=2,
            label=(
                f"{stream_name} "
                "average learning rate"
            ),
        )

    used_labels: set[str] = set()

    for block_index, block in enumerate(
        BLOCK_ORDER
    ):
        volatility_label = (
            get_block_volatility_label(
                labels,
                block,
            )
        )

        if volatility_label is None:
            continue

        legend_label = (
            volatility_label
            if volatility_label not in used_labels
            else None
        )

        ax.axvspan(
            block_index - 0.45,
            block_index + 0.45,
            alpha=0.12,
            label=legend_label,
        )

        used_labels.add(
            volatility_label
        )

    environment_name = (
        environment_display_name(
            environment_type
        )
    )

    ax.set_title(
        f"{model_name}: average learning rate by block\n"
        f"{environment_name}",
        fontsize=15,
        fontweight="bold",
    )

    ax.set_xlabel("Block")

    ax.set_ylabel(
        "Average learning rate / precision coefficient"
    )

    ax.legend(
        loc="best"
    )

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        figure_dir
        / (
            f"{model_name}_"
            "block_average_learning_rates.png"
        )
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved figure: {output_path}"
    )


# ---------------------------------------------------------------------
# Across-model volatility-effect summary plot
# ---------------------------------------------------------------------

def plot_effect_summary(
    *,
    effect_summary: pd.DataFrame,
    environment_type: str,
    figure_dir: Path,
) -> None:
    """
    Plot average A and B volatility-block effects across models.

    For local environments, the directly manipulated stream and
    cross-stream change are identified in the title and annotations.

    For the global environment, both effects are direct volatility effects.
    """

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    x_positions = np.arange(
        len(effect_summary)
    )

    width = 0.35

    a_values = effect_summary[
        "stream_A_effect_average"
    ].to_numpy(dtype=float)

    b_values = effect_summary[
        "stream_B_effect_average"
    ].to_numpy(dtype=float)

    ax.bar(
        x_positions - width / 2,
        a_values,
        width=width,
        label="Stream A effect",
    )

    ax.bar(
        x_positions + width / 2,
        b_values,
        width=width,
        label="Stream B effect",
    )

    ax.set_xticks(
        x_positions
    )

    ax.set_xticklabels(
        effect_summary["model"],
        rotation=20,
        ha="right",
    )

    environment_name = (
        environment_display_name(
            environment_type
        )
    )

    if environment_type == "local_A":
        title_detail = (
            "A direct volatility effect; "
            "B cross-stream change"
        )

    elif environment_type == "local_B":
        title_detail = (
            "B direct volatility effect; "
            "A cross-stream change"
        )

    else:
        title_detail = (
            "Both streams objectively volatile"
        )

    ax.set_title(
        "Learning-rate volatility effects\n"
        f"{environment_name}: {title_detail}",
        fontsize=15,
        fontweight="bold",
    )

    ax.set_ylabel(
        "Average learning-rate change"
    )

    ax.legend(
        loc="best"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    all_values = np.concatenate(
        [
            a_values,
            b_values,
        ]
    )

    finite_values = all_values[
        np.isfinite(all_values)
    ]

    if len(finite_values) > 0:
        max_abs_value = max(
            float(
                np.max(
                    np.abs(
                        finite_values
                    )
                )
            ),
            1e-6,
        )
    else:
        max_abs_value = 1.0

    annotation_offset = (
        0.04 * max_abs_value
    )

    for row_index, row in (
        effect_summary
        .reset_index(drop=True)
        .iterrows()
    ):
        a_value = float(
            row[
                "stream_A_effect_average"
            ]
        )

        b_value = float(
            row[
                "stream_B_effect_average"
            ]
        )

        annotation_y = (
            max(a_value, b_value)
            + annotation_offset
        )

        if environment_type in {
            "local_A",
            "local_B",
        }:
            cti = row[
                "cross_transfer_index"
            ]

            if pd.isna(cti):
                annotation = "CTI: NA"
            else:
                direction = (
                    row[
                        "cross_transfer_direction"
                    ]
                    .replace("_", " -> ")
                )

                annotation = (
                    f"CTI {direction}: "
                    f"{cti:.2f}"
                )

        else:
            annotation = "CTI: N/A"

        ax.text(
            x_positions[row_index],
            annotation_y,
            annotation,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        figure_dir
        / "volatility_effect_summary.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved figure: {output_path}"
    )


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build command-line interface.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Calculate trial-wise learning rates, "
            "block volatility effects, and "
            "environment-appropriate cross-stream "
            "transfer from pyhgf trajectories."
        )
    )

    parser.add_argument(
        "--model-runs-dir",
        type=str,
        default=str(
            DEFAULT_MODEL_RUNS_DIR
        ),
        help=(
            "Directory containing model run outputs."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(
            DEFAULT_OUTPUT_DIR
        ),
        help=(
            "Directory for learning-rate CSV outputs."
        ),
    )

    parser.add_argument(
        "--figure-dir",
        type=str,
        default=str(
            DEFAULT_FIGURE_DIR
        ),
        help=(
            "Directory for learning-rate figures."
        ),
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=MODEL_NAMES,
        choices=MODEL_NAMES,
        help=(
            "Which inference models to analyze."
        ),
    )

    parser.add_argument(
        "--min-trial-index",
        type=int,
        default=DEFAULT_MIN_TRIAL_INDEX,
        help=(
            "Minimum trial_index to include in "
            "learning-rate summaries and plots. "
            "Default is 1, which excludes the "
            "initialization trial."
        ),
    )

    return parser


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    """
    Run learning-rate analysis.
    """

    parser = build_arg_parser()
    args = parser.parse_args()

    model_runs_dir = Path(
        args.model_runs_dir
    )

    output_dir = Path(
        args.output_dir
    )

    figure_dir = Path(
        args.figure_dir
    )

    model_names = list(
        args.models
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Load labels and determine which environment is being analyzed.
    # -------------------------------------------------------------

    labels = load_labels(
        model_runs_dir
    )

    environment_type = (
        infer_environment_type(
            labels
        )
    )

    environment_name = (
        environment_display_name(
            environment_type
        )
    )

    print(
        "\nDetected environment:"
    )

    print(
        f"    {environment_type} "
        f"({environment_name})"
    )

    print(
        "\nBlock regimes:"
    )

    print(
        get_block_regimes(
            labels
        ).to_string(
            index=False
        )
    )

    # -------------------------------------------------------------
    # Calculate trial-wise learning rates.
    # -------------------------------------------------------------

    trialwise_tables = []

    for model_name in model_names:
        print(
            f"\nAnalyzing model: "
            f"{model_name}"
        )

        trajectories = (
            load_model_trajectories(
                model_name=model_name,
                model_runs_dir=model_runs_dir,
            )
        )

        model_trialwise = (
            calculate_trialwise_learning_rates(
                model_name=model_name,
                labels=labels,
                trajectories=trajectories,
            )
        )

        trialwise_tables.append(
            model_trialwise
        )

    trialwise_learning_rates_raw = (
        pd.concat(
            trialwise_tables,
            axis=0,
            ignore_index=True,
        )
    )

    # -------------------------------------------------------------
    # Remove initialization trial from analysis by default.
    # -------------------------------------------------------------

    trialwise_learning_rates = (
        trialwise_learning_rates_raw[
            trialwise_learning_rates_raw[
                "trial_index"
            ]
            >= args.min_trial_index
        ]
        .copy()
    )

    if trialwise_learning_rates.empty:
        raise ValueError(
            "No trials remain after applying "
            "--min-trial-index. "
            f"Current value: "
            f"{args.min_trial_index}"
        )

    # -------------------------------------------------------------
    # Block summaries and effects.
    # -------------------------------------------------------------

    block_summary = (
        calculate_block_summary(
            trialwise_learning_rates
        )
    )

    effect_summary = (
        calculate_effect_summary(
            block_summary=block_summary,
            model_names=model_names,
            environment_type=environment_type,
        )
    )

    # -------------------------------------------------------------
    # Save CSV outputs.
    # -------------------------------------------------------------

    raw_trialwise_path = (
        output_dir
        / (
            "trialwise_learning_rates_"
            "raw_including_initial_trial.csv"
        )
    )

    trialwise_path = (
        output_dir
        / "trialwise_learning_rates.csv"
    )

    block_summary_path = (
        output_dir
        / "block_learning_rate_summary.csv"
    )

    effect_summary_path = (
        output_dir
        / "volatility_effect_summary.csv"
    )

    trialwise_learning_rates_raw.to_csv(
        raw_trialwise_path,
        index=False,
    )

    trialwise_learning_rates.to_csv(
        trialwise_path,
        index=False,
    )

    block_summary.to_csv(
        block_summary_path,
        index=False,
    )

    effect_summary.to_csv(
        effect_summary_path,
        index=False,
    )

    print(
        "\nSaved raw trial-wise learning rates:"
    )
    print(
        f"    {raw_trialwise_path}"
    )

    print(
        "\nSaved analyzed trial-wise learning rates:"
    )
    print(
        f"    {trialwise_path}"
    )

    print(
        "\nSaved block summary:"
    )
    print(
        f"    {block_summary_path}"
    )

    print(
        "\nSaved volatility-effect summary:"
    )
    print(
        f"    {effect_summary_path}"
    )

    # -------------------------------------------------------------
    # Print main summary directly in terminal.
    # -------------------------------------------------------------

    summary_columns = [
        "model",
        "stream_A_effect_average",
        "stream_B_effect_average",
        "cross_transfer_direction",
        "cross_transfer_index",
        "cross_transfer_index_interpretable",
    ]

    print(
        "\nMain volatility-effect summary:"
    )

    print(
        effect_summary[
            summary_columns
        ].to_string(
            index=False
        )
    )

    # -------------------------------------------------------------
    # Figures.
    # -------------------------------------------------------------

    for model_name in model_names:

        plot_learning_rates_for_model(
            model_name=model_name,
            trialwise=trialwise_learning_rates,
            effect_summary=effect_summary,
            environment_type=environment_type,
            figure_dir=figure_dir,
        )

        plot_block_means_for_model(
            model_name=model_name,
            block_summary=block_summary,
            labels=labels,
            environment_type=environment_type,
            figure_dir=figure_dir,
        )

    plot_effect_summary(
        effect_summary=effect_summary,
        environment_type=environment_type,
        figure_dir=figure_dir,
    )

    print(
        "\nFinished learning-rate analysis."
    )


if __name__ == "__main__":
    main()