"""
Analyze trial-wise HGF learning rates across Local-A, Global, and
Hybrid environments (also supports Local-B).

Learning-rate definition
------------------------
For every inference model:

    Stream A:
        learning_rate_A(k)
        = x_0_expected_precision(k) / x_2_precision(k)

    Stream B:
        learning_rate_B(k)
        = x_1_expected_precision(k) / x_3_precision(k)

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
The script uses environment metadata or the environment directory name to
identify Hybrid, because Hybrid and Global share the same nominal block labels.
The --environment-type option can override the automatic detection when needed.

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

hybrid:
    V/S labels are NOMINAL, not fixed volatility regimes. Actual process
    variances fluctuate. Both streams receive the shared drive.

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

hybrid:
    D_A and D_B are descriptive nominal V-versus-neighboring-S contrasts.
    CTI is not defined because the shared drive affects both streams.

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
import sys

import pandas as pd


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
    DEFAULT_MIN_TRIAL_INDEX,
    MODEL_NAMES,
    calculate_block_summary,
    calculate_effect_summary,
    calculate_trialwise_learning_rates,
    environment_display_name,
    get_block_regimes,
    resolve_environment_type,
    load_labels,
    load_model_trajectories,
)

from src.analysis.plot_learning_rates import (  # noqa: E402
    plot_block_means_for_model,
    plot_effect_summary,
    plot_learning_rates_for_model,
)


# Keep the historical CLI names used by this script.
DEFAULT_OUTPUT_DIR = DEFAULT_LEARNING_RATES_DIR

DEFAULT_FIGURE_DIR = (
    DEFAULT_FIGURES_DIR / "learning_rates"
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
        "--environment-type",
        choices=["auto", "local_A", "local_B", "global", "hybrid"],
        default="auto",
        help=(
            "Environment identity. 'auto' uses environment_metadata.json "
            "when available, then the environment folder, then regime labels."
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

    environment_type = resolve_environment_type(
        labels=labels,
        model_runs_dir=model_runs_dir,
        requested=args.environment_type,
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

    if environment_type == "hybrid":
        print(
            "\nHybrid note: S/V labels are nominal, not fixed "
            "volatility regimes. Block contrasts are descriptive. "
            "CTI is undefined because both streams receive the shared drive."
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
