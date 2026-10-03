"""
Plot behavioural learning rates from hybrid synthetic responses.

These are NOT HGF latent learning rates. They are Nassar-style behavioural
learning rates computed from the synthetic participant predictions:

    behavioural learning rate = response update / prediction error

For stream s:

    alpha_s(t) = [y_s(t+1) - y_s(t)] / [u_s(t) - y_s(t)]

The script reads:

    synthetic_continuous_predictions.csv

and plots:

    1. raw trial-wise behavioural learning rates
    2. rolling-median smoothed behavioural learning rates
    3. block-wise median behavioural learning rates

Run from project root:

    python scripts/plot_hybrid_behavioral_learning_rates.py

To plot only a trial range:

    python scripts/plot_hybrid_behavioral_learning_rates.py \
        --start-trial-index 150 \
        --end-trial-index 450
"""

from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir() and (p / "config").is_dir())

DEFAULT_RESPONSE_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "hgf_walk_outputs"
    / "synthetic_responses"
    / "shared_volatility_sigma010"
    / "synthetic_continuous_predictions.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "hgf_walk_outputs"
    / "figures"
    / "synthetic_responses"
    / "behavioral_learning_rates"
)


def load_response_data(response_file: str | Path) -> pd.DataFrame:
    response_file = Path(response_file)

    if not response_file.exists():
        raise FileNotFoundError(f"Could not find response file: {response_file}")

    data = pd.read_csv(response_file)

    required_columns = {
        "trial_index",
        "behavioral_learning_rate_A",
        "behavioral_learning_rate_B",
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            "The response file is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    return data


def filter_trials(
    data: pd.DataFrame,
    *,
    start_trial_index: int | None,
    end_trial_index: int | None,
) -> pd.DataFrame:
    filtered = data.copy()

    if start_trial_index is not None:
        filtered = filtered[filtered["trial_index"] >= start_trial_index]

    if end_trial_index is not None:
        filtered = filtered[filtered["trial_index"] <= end_trial_index]

    if filtered.empty:
        raise ValueError(
            "No trials remain after filtering. "
            "Check --start-trial-index and --end-trial-index."
        )

    return filtered.reset_index(drop=True)


def get_a_volatile_spans(data: pd.DataFrame) -> list[tuple[float, float]]:
    """
    Find contiguous A-volatile regions for background shading.
    """

    if "regime_A" in data.columns:
        volatile_mask = data["regime_A"].astype(str).str.lower().eq("volatile")
    elif "block" in data.columns:
        volatile_mask = data["block"].astype(str).str.upper().str.startswith("V")
    else:
        return []

    if not volatile_mask.any():
        return []

    trial_index = data["trial_index"].to_numpy()
    mask = volatile_mask.to_numpy()

    spans: list[tuple[float, float]] = []
    in_span = False
    span_start = None

    for i, is_volatile in enumerate(mask):
        if is_volatile and not in_span:
            in_span = True
            span_start = trial_index[i]

        if in_span and not is_volatile:
            span_end = trial_index[i - 1]
            spans.append((float(span_start), float(span_end)))
            in_span = False
            span_start = None

    if in_span:
        spans.append((float(span_start), float(trial_index[-1])))

    return spans


def shade_volatile_blocks(ax, spans: list[tuple[float, float]]) -> None:
    first = True

    for start, end in spans:
        ax.axvspan(
            start,
            end,
            alpha=0.12,
            label="A volatile block" if first else None,
        )
        first = False


def robust_ylim(
    values: pd.Series,
    *,
    lower_quantile: float,
    upper_quantile: float,
) -> tuple[float, float]:
    """
    Return percentile-based y-limits so extreme ratio artifacts do not dominate.
    """

    clean = values.replace([np.inf, -np.inf], np.nan).dropna()

    if clean.empty:
        return (-1.0, 1.0)

    low = clean.quantile(lower_quantile)
    high = clean.quantile(upper_quantile)

    if np.isclose(low, high):
        padding = max(abs(low) * 0.1, 0.1)
        return (low - padding, high + padding)

    padding = 0.10 * (high - low)

    return (low - padding, high + padding)


def plot_raw_trialwise_learning_rates(
    *,
    data: pd.DataFrame,
    output_dir: Path,
    file_prefix: str,
    lower_quantile: float,
    upper_quantile: float,
    dpi: int,
    show: bool,
) -> Path:
    """
    Plot raw behavioural learning rates with robust y-axis limits.
    """

    x = data["trial_index"]
    lr_A = data["behavioral_learning_rate_A"].replace([np.inf, -np.inf], np.nan)
    lr_B = data["behavioral_learning_rate_B"].replace([np.inf, -np.inf], np.nan)

    all_values = pd.concat([lr_A, lr_B], ignore_index=True)
    y_min, y_max = robust_ylim(
        all_values,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
    )

    volatile_spans = get_a_volatile_spans(data)

    fig, ax = plt.subplots(figsize=(16, 6))

    shade_volatile_blocks(ax, volatile_spans)

    ax.axhline(0.0, linewidth=1.0, alpha=0.8)

    ax.plot(
        x,
        lr_A,
        linewidth=0.8,
        alpha=0.75,
        label="Stream A behavioural learning rate",
    )

    ax.plot(
        x,
        lr_B,
        linewidth=0.8,
        alpha=0.75,
        label="Stream B behavioural learning rate",
    )

    ax.set_ylim(y_min, y_max)

    ax.set_title(
        "Hybrid synthetic responses: trial-wise behavioural learning rates",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xlabel("Trial")
    ax.set_ylabel("Behavioural learning rate")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{file_prefix}_raw_trialwise.png"

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)

    if show:
        plt.show()

    return output_path


def plot_rolling_learning_rates(
    *,
    data: pd.DataFrame,
    output_dir: Path,
    file_prefix: str,
    window: int,
    lower_quantile: float,
    upper_quantile: float,
    dpi: int,
    show: bool,
) -> Path:
    """
    Plot rolling-median behavioural learning rates.
    """

    x = data["trial_index"]

    lr_A = (
        data["behavioral_learning_rate_A"]
        .replace([np.inf, -np.inf], np.nan)
        .rolling(window=window, center=True, min_periods=max(3, window // 4))
        .median()
    )

    lr_B = (
        data["behavioral_learning_rate_B"]
        .replace([np.inf, -np.inf], np.nan)
        .rolling(window=window, center=True, min_periods=max(3, window // 4))
        .median()
    )

    all_values = pd.concat([lr_A, lr_B], ignore_index=True)
    y_min, y_max = robust_ylim(
        all_values,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
    )

    volatile_spans = get_a_volatile_spans(data)

    fig, ax = plt.subplots(figsize=(16, 6))

    shade_volatile_blocks(ax, volatile_spans)

    ax.axhline(0.0, linewidth=1.0, alpha=0.8)

    ax.plot(
        x,
        lr_A,
        linewidth=2.0,
        label=f"Stream A rolling median, window={window}",
    )

    ax.plot(
        x,
        lr_B,
        linewidth=2.0,
        label=f"Stream B rolling median, window={window}",
    )

    ax.set_ylim(y_min, y_max)

    ax.set_title(
        "Hybrid synthetic responses: smoothed behavioural learning rates",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xlabel("Trial")
    ax.set_ylabel("Behavioural learning rate")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{file_prefix}_rolling_median_window_{window}.png"

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)

    if show:
        plt.show()

    return output_path


def plot_block_median_learning_rates(
    *,
    data: pd.DataFrame,
    output_dir: Path,
    file_prefix: str,
    dpi: int,
    show: bool,
) -> Path | None:
    """
    Plot block-wise median behavioural learning rates.

    This is often much easier to interpret than the raw trial-wise ratio.
    """

    if "block" not in data.columns:
        print("Skipping block-wise plot because there is no 'block' column.")
        return None

    block_order = ["S1", "V1", "S2", "V2", "S3"]

    summary = (
        data.groupby("block", dropna=False)
        .agg(
            stream_A_median=(
                "behavioral_learning_rate_A",
                lambda x: x.replace([np.inf, -np.inf], np.nan).median(),
            ),
            stream_B_median=(
                "behavioral_learning_rate_B",
                lambda x: x.replace([np.inf, -np.inf], np.nan).median(),
            ),
            stream_A_valid_n=(
                "behavioral_learning_rate_A",
                lambda x: x.replace([np.inf, -np.inf], np.nan).dropna().shape[0],
            ),
            stream_B_valid_n=(
                "behavioral_learning_rate_B",
                lambda x: x.replace([np.inf, -np.inf], np.nan).dropna().shape[0],
            ),
        )
        .reset_index()
    )

    present_order = [block for block in block_order if block in set(summary["block"])]
    remaining_blocks = [block for block in summary["block"] if block not in present_order]
    ordered_blocks = present_order + remaining_blocks

    summary["block"] = pd.Categorical(
        summary["block"],
        categories=ordered_blocks,
        ordered=True,
    )
    summary = summary.sort_values("block")

    x_positions = np.arange(len(summary))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        x_positions - width / 2,
        summary["stream_A_median"],
        width=width,
        label="Stream A median behavioural learning rate",
    )

    ax.bar(
        x_positions + width / 2,
        summary["stream_B_median"],
        width=width,
        label="Stream B median behavioural learning rate",
    )

    ax.axhline(0.0, linewidth=1.0, alpha=0.8)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(summary["block"].astype(str).tolist())

    ax.set_title(
        "Hybrid synthetic responses: block-wise median behavioural learning rates",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xlabel("Block")
    ax.set_ylabel("Median behavioural learning rate")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{file_prefix}_block_medians.png"

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)

    summary_path = output_dir / f"{file_prefix}_block_medians.csv"
    summary.to_csv(summary_path, index=False)

    if show:
        plt.show()

    return output_path


def print_diagnostics(data: pd.DataFrame) -> None:
    """
    Print simple diagnostics about NaNs and extreme values.
    """

    for stream in ["A", "B"]:
        column = f"behavioral_learning_rate_{stream}"
        values = data[column].replace([np.inf, -np.inf], np.nan)

        print(f"\nStream {stream} behavioural learning rate diagnostics:")
        print(f"  valid trials: {values.dropna().shape[0]}")
        print(f"  NaN/inf trials: {values.isna().sum()}")
        print(f"  median: {values.median():.6f}")
        print(f"  mean: {values.mean():.6f}")
        print(f"  1st percentile: {values.quantile(0.01):.6f}")
        print(f"  99th percentile: {values.quantile(0.99):.6f}")
        print(f"  minimum: {values.min():.6f}")
        print(f"  maximum: {values.max():.6f}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot behavioural learning rates from hybrid synthetic responses."
    )

    parser.add_argument(
        "--response-file",
        type=str,
        default=str(DEFAULT_RESPONSE_FILE),
        help="Path to synthetic_continuous_predictions.csv.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where figures should be saved.",
    )

    parser.add_argument(
        "--start-trial-index",
        type=int,
        default=None,
        help="Optional first trial_index to plot.",
    )

    parser.add_argument(
        "--end-trial-index",
        type=int,
        default=None,
        help="Optional last trial_index to plot.",
    )

    parser.add_argument(
        "--rolling-window",
        type=int,
        default=25,
        help="Rolling-median window size for smoothed plot.",
    )

    parser.add_argument(
        "--lower-quantile",
        type=float,
        default=0.01,
        help="Lower quantile for robust y-axis limits.",
    )

    parser.add_argument(
        "--upper-quantile",
        type=float,
        default=0.99,
        help="Upper quantile for robust y-axis limits.",
    )

    parser.add_argument(
        "--file-prefix",
        type=str,
        default="hybrid_behavioral_learning_rates",
        help="Prefix for output figure filenames.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure resolution.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively after saving.",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    data = load_response_data(args.response_file)

    data = filter_trials(
        data,
        start_trial_index=args.start_trial_index,
        end_trial_index=args.end_trial_index,
    )

    output_dir = Path(args.output_dir)

    print_diagnostics(data)

    saved_paths: list[Path] = []

    saved_paths.append(
        plot_raw_trialwise_learning_rates(
            data=data,
            output_dir=output_dir,
            file_prefix=args.file_prefix,
            lower_quantile=args.lower_quantile,
            upper_quantile=args.upper_quantile,
            dpi=args.dpi,
            show=args.show,
        )
    )

    saved_paths.append(
        plot_rolling_learning_rates(
            data=data,
            output_dir=output_dir,
            file_prefix=args.file_prefix,
            window=args.rolling_window,
            lower_quantile=args.lower_quantile,
            upper_quantile=args.upper_quantile,
            dpi=args.dpi,
            show=args.show,
        )
    )

    block_path = plot_block_median_learning_rates(
        data=data,
        output_dir=output_dir,
        file_prefix=args.file_prefix,
        dpi=args.dpi,
        show=args.show,
    )

    if block_path is not None:
        saved_paths.append(block_path)

    print("\nSaved figures:")
    for path in saved_paths:
        print(f"  {path}")

    print("\nPlotted trial range:")
    print(f"  first trial_index: {data['trial_index'].min()}")
    print(f"  last trial_index:  {data['trial_index'].max()}")
    print(f"  number of trials:  {len(data)}")


if __name__ == "__main__":
    main()