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

# ---------------------------------------------------------------------
# Project paths and project imports
# ---------------------------------------------------------------------

PROJECT_ROOT = next(
    p
    for p in Path(__file__).resolve().parents
    if (p / "src").is_dir()
    and (p / "config").is_dir()
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.utils.paths import (  # noqa: E402
    DEFAULT_FIGURES_DIR,
    DEFAULT_LEARNING_RATES_DIR,
    DEFAULT_MODEL_RUNS_DIR,
)

from src.analysis.learning_rates import (  # noqa: E402
    BLOCK_ORDER,
    DEFAULT_MIN_TRIAL_INDEX,
    MODEL_NAMES,
    calculate_block_summary,
    calculate_effect_summary,
    calculate_trialwise_learning_rates,
    environment_display_name,
    get_block_regimes,
    infer_environment_type,
    load_labels,
    load_model_trajectories,
)


# Keep the historical CLI names used by this script.
DEFAULT_OUTPUT_DIR = DEFAULT_LEARNING_RATES_DIR

DEFAULT_FIGURE_DIR = (
    DEFAULT_FIGURES_DIR / "learning_rates"
)



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