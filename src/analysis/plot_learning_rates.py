"""Plotting functions for two-stream HGF learning-rate analyses.

This module contains visualization only. Scientific learning-rate
calculations and volatility-effect calculations live in
src.analysis.learning_rates.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.learning_rates import (
    BLOCK_ORDER,
    environment_display_name,
)


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