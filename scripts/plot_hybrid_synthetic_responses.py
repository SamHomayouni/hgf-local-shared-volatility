"""
Plot hybrid synthetic continuous prediction responses.

This script visualizes the output of:

    scripts/simulate_continuous_predictions.py

for the shared_volatility / hybrid model.

It plots, for each stream:

    observation
    model prediction
    simulated participant prediction

The main purpose is to check whether the response model behaves as expected:
participant predictions should be noisy around the model predictions, and model
predictions should be smoother than raw observations.

Run from project root
---------------------

python scripts/plot_hybrid_synthetic_responses.py

Or with explicit paths:

python scripts/plot_hybrid_synthetic_responses.py \
    --response-file outputs/hgf_walk_outputs/synthetic_responses/shared_volatility_sigma010/synthetic_continuous_predictions.csv \
    --output-dir outputs/hgf_walk_outputs/figures/synthetic_responses

To plot only a section:

python scripts/plot_hybrid_synthetic_responses.py \
    --start-trial-index 150 \
    --end-trial-index 450
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
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
)


def load_response_data(response_file: str | Path) -> pd.DataFrame:
    """
    Load synthetic response data and check required columns.
    """

    response_file = Path(response_file)

    if not response_file.exists():
        raise FileNotFoundError(f"Could not find response file: {response_file}")

    data = pd.read_csv(response_file)

    required_columns = {
        "trial_index",
        "observation_A",
        "observation_B",
        "model_prediction_A",
        "model_prediction_B",
        "participant_prediction_A",
        "participant_prediction_B",
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
    """
    Optionally restrict the plot to a trial-index range.
    """

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

    Preferred source:
        regime_A == "volatile"

    Fallback:
        block labels that start with "V"
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

        if in_span and (not is_volatile):
            span_end = trial_index[i - 1]
            spans.append((float(span_start), float(span_end)))
            in_span = False
            span_start = None

    if in_span:
        spans.append((float(span_start), float(trial_index[-1])))

    return spans


def shade_volatile_blocks(ax, spans: list[tuple[float, float]]) -> None:
    """
    Add background shading for A-volatile blocks.
    """

    first = True

    for start, end in spans:
        label = "A volatile block" if first else None
        ax.axvspan(
            start,
            end,
            alpha=0.12,
            label=label,
        )
        first = False


def plot_stream(
    *,
    data: pd.DataFrame,
    stream: str,
    output_dir: str | Path,
    file_prefix: str,
    dpi: int,
    show: bool,
) -> Path:
    """
    Plot observations, model predictions, and participant predictions for one stream.
    """

    stream = stream.upper()

    if stream not in {"A", "B"}:
        raise ValueError("stream must be 'A' or 'B'.")

    observation_col = f"observation_{stream}"
    model_prediction_col = f"model_prediction_{stream}"
    participant_prediction_col = f"participant_prediction_{stream}"

    x = data["trial_index"]

    volatile_spans = get_a_volatile_spans(data)

    fig, ax = plt.subplots(figsize=(16, 6))

    shade_volatile_blocks(ax, volatile_spans)

    ax.plot(
        x,
        data[observation_col],
        linewidth=0.8,
        alpha=0.45,
        label=f"Stream {stream} observation",
    )

    ax.plot(
        x,
        data[model_prediction_col],
        linewidth=2.0,
        label=f"Stream {stream} model prediction",
    )

    ax.plot(
        x,
        data[participant_prediction_col],
        linewidth=1.0,
        alpha=0.75,
        label=f"Stream {stream} simulated participant prediction",
    )

    ax.set_title(
        f"Hybrid synthetic continuous predictions: stream {stream}",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xlabel("Trial")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{file_prefix}_stream_{stream}.png"

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)

    if show:
        plt.show()

    return output_path


def plot_prediction_errors(
    *,
    data: pd.DataFrame,
    output_dir: str | Path,
    file_prefix: str,
    dpi: int,
    show: bool,
) -> Path:
    """
    Optional diagnostic plot: response noise around model prediction.

    This plots:

        participant_prediction - model_prediction

    for streams A and B.
    """

    x = data["trial_index"]
    volatile_spans = get_a_volatile_spans(data)

    residual_A = data["participant_prediction_A"] - data["model_prediction_A"]
    residual_B = data["participant_prediction_B"] - data["model_prediction_B"]

    fig, ax = plt.subplots(figsize=(16, 5))

    shade_volatile_blocks(ax, volatile_spans)

    ax.axhline(0.0, linewidth=1.0, alpha=0.8)

    ax.plot(
        x,
        residual_A,
        linewidth=0.9,
        alpha=0.75,
        label="Stream A response noise",
    )

    ax.plot(
        x,
        residual_B,
        linewidth=0.9,
        alpha=0.75,
        label="Stream B response noise",
    )

    ax.set_title(
        "Hybrid synthetic response noise around model predictions",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xlabel("Trial")
    ax.set_ylabel("Participant prediction - model prediction")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{file_prefix}_response_noise.png"

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)

    if show:
        plt.show()

    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build command-line interface.
    """

    parser = argparse.ArgumentParser(
        description="Plot hybrid synthetic continuous prediction responses."
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
        "--file-prefix",
        type=str,
        default="hybrid_synthetic_predictions",
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

    parser.add_argument(
        "--skip-response-noise-plot",
        action="store_true",
        help="Skip the participant-minus-model diagnostic plot.",
    )

    return parser


def main() -> None:
    """
    Load synthetic responses and create plots.
    """

    parser = build_arg_parser()
    args = parser.parse_args()

    response_data = load_response_data(args.response_file)

    response_data = filter_trials(
        response_data,
        start_trial_index=args.start_trial_index,
        end_trial_index=args.end_trial_index,
    )

    output_dir = Path(args.output_dir)

    saved_paths = []

    for stream in ["A", "B"]:
        saved_path = plot_stream(
            data=response_data,
            stream=stream,
            output_dir=output_dir,
            file_prefix=args.file_prefix,
            dpi=args.dpi,
            show=args.show,
        )
        saved_paths.append(saved_path)

    if not args.skip_response_noise_plot:
        saved_path = plot_prediction_errors(
            data=response_data,
            output_dir=output_dir,
            file_prefix=args.file_prefix,
            dpi=args.dpi,
            show=args.show,
        )
        saved_paths.append(saved_path)

    print("Saved figures:")
    for path in saved_paths:
        print(f"  {path}")

    print("\nPlotted trial range:")
    print(f"  first trial_index: {response_data['trial_index'].min()}")
    print(f"  last trial_index:  {response_data['trial_index'].max()}")
    print(f"  number of trials:  {len(response_data)}")


if __name__ == "__main__":
    main()