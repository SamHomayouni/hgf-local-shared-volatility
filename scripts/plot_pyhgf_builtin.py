"""
Plot pyhgf built-in belief/value and volatility trajectories.

This script uses pyhgf's own plotting functions:

    pyhgf.plots.matplotlib.plot_nodes
    pyhgf.plots.matplotlib.plot_trajectories

Purpose
-------
Generate first-pass diagnostic plots for the three pyhgf models:

    local_only
    global_only
    shared_volatility

The script reruns each model on the saved model input matrix because pyhgf's
built-in plotting functions need a live Network object, not only the saved
CSV trajectories.

Expected input
--------------
By default, this script reads:

    outputs/model_runs/model_input_data.npy

This file should have been created by:

    scripts/run_pyhgf_models.py

Expected output
---------------
By default, this script writes figures to:

    outputs/figures/pyhgf_builtin/

For each model, it saves:
    - all_nodes.png
    - value_nodes.png
    - volatility_nodes.png
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------
# Allows running from project root with:
#
#   python scripts/plot_pyhgf_builtin.py
#
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from pyhgf.plots.matplotlib import plot_nodes, plot_trajectories  # noqa: E402

from src.models.local_only import (  # noqa: E402
    MODEL_NAME as LOCAL_ONLY_NAME,
    build_model as build_local_only_model,
    get_node_ids as get_local_only_node_ids,
)

from src.models.global_only import (  # noqa: E402
    MODEL_NAME as GLOBAL_ONLY_NAME,
    build_model as build_global_only_model,
    get_node_ids as get_global_only_node_ids,
)

from src.models.shared_volatility import (  # noqa: E402
    MODEL_NAME as SHARED_VOLATILITY_NAME,
    build_model as build_shared_volatility_model,
    get_node_ids as get_shared_volatility_node_ids,
)


@dataclass(frozen=True)
class PlotModelSpec:
    """
    Information needed to build and plot one model.
    """

    name: str
    build_model: Callable
    get_node_ids: Callable[[], dict[str, int]]
    value_node_names: tuple[str, ...]
    volatility_node_names: tuple[str, ...]


MODEL_REGISTRY: dict[str, PlotModelSpec] = {
    LOCAL_ONLY_NAME: PlotModelSpec(
        name=LOCAL_ONLY_NAME,
        build_model=build_local_only_model,
        get_node_ids=get_local_only_node_ids,
        value_node_names=("value_A", "value_B"),
        volatility_node_names=("volatility_A", "volatility_B"),
    ),
    GLOBAL_ONLY_NAME: PlotModelSpec(
        name=GLOBAL_ONLY_NAME,
        build_model=build_global_only_model,
        get_node_ids=get_global_only_node_ids,
        value_node_names=("value_A", "value_B"),
        volatility_node_names=("shared_volatility",),
    ),
    SHARED_VOLATILITY_NAME: PlotModelSpec(
        name=SHARED_VOLATILITY_NAME,
        build_model=build_shared_volatility_model,
        get_node_ids=get_shared_volatility_node_ids,
        value_node_names=("value_A", "value_B"),
        volatility_node_names=("local_vol_A", "local_vol_B", "shared_volatility"),
    ),
}


def load_input_data(input_path: str | Path) -> np.ndarray:
    """
    Load the model input matrix.

    Expected shape:
        (n_trials, 2)

    Column convention:
        input_data[:, 0] = stream A observations
        input_data[:, 1] = stream B observations
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Could not find input matrix: {input_path}\n"
            "Run scripts/run_pyhgf_models.py first, or pass --input-data."
        )

    input_data = np.load(input_path)

    if input_data.ndim != 2:
        raise ValueError(f"Expected 2D input_data, got shape {input_data.shape}.")

    if input_data.shape[1] != 2:
        raise ValueError(
            f"Expected input_data with 2 columns, got shape {input_data.shape}."
        )

    if np.isnan(input_data).any():
        raise ValueError("input_data contains NaN values.")

    return input_data


def build_and_run_model(
    *,
    model_spec: PlotModelSpec,
    input_data: np.ndarray,
    model_kwargs: dict,
):
    """
    Build one pyhgf model and feed it the observations.
    """
    model = model_spec.build_model(**model_kwargs)

    maybe_updated_model = model.input_data(input_data=input_data)

    # pyhgf examples usually return an updated model, but this guard makes
    # the code safe if the method mutates in place.
    if maybe_updated_model is not None:
        model = maybe_updated_model

    return model


def get_node_indices(
    *,
    node_ids: dict[str, int],
    node_names: tuple[str, ...],
) -> list[int]:
    """
    Convert node names to pyhgf integer node IDs.
    """
    missing_names = [name for name in node_names if name not in node_ids]

    if missing_names:
        raise KeyError(
            f"Missing node names in node_ids: {missing_names}. "
            f"Available names: {sorted(node_ids)}"
        )

    return [node_ids[name] for name in node_names]


def save_current_figure(output_path: Path) -> None:
    """
    Save the current matplotlib figure and close it.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_all_nodes(
    *,
    model,
    output_path: Path,
    show_posterior: bool,
    show_surprise: bool,
    show_total_surprise: bool,
) -> None:
    """
    Save pyhgf's all-node trajectory plot.
    """
    plot_trajectories(
        network=model,
        ci=True,
        show_surprise=show_surprise,
        show_posterior=show_posterior,
        show_total_surprise=show_total_surprise,
        figsize=(18, 10),
    )

    save_current_figure(output_path)


def plot_selected_nodes(
    *,
    model,
    node_idxs: list[int],
    output_path: Path,
    title: str,
    show_posterior: bool,
    show_surprise: bool,
) -> None:
    """
    Save pyhgf's selected-node trajectory plot.

    This is used separately for:
        - value nodes
        - volatility-related nodes
    """
    plot_nodes(
        network=model,
        node_idxs=node_idxs,
        ci=True,
        show_surprise=show_surprise,
        show_posterior=show_posterior,
        figsize=(14, 4 * len(node_idxs)),
    )

    fig = plt.gcf()
    fig.suptitle(title, y=1.02)

    save_current_figure(output_path)


def run_plots_for_one_model(
    *,
    model_spec: PlotModelSpec,
    input_data: np.ndarray,
    output_dir: Path,
    model_kwargs: dict,
    show_posterior: bool,
    show_surprise: bool,
    show_total_surprise: bool,
) -> None:
    """
    Build, run, and plot one model.
    """
    print(f"\nPlotting model: {model_spec.name}")

    model = build_and_run_model(
        model_spec=model_spec,
        input_data=input_data,
        model_kwargs=model_kwargs,
    )

    node_ids = model_spec.get_node_ids()

    value_node_idxs = get_node_indices(
        node_ids=node_ids,
        node_names=model_spec.value_node_names,
    )

    volatility_node_idxs = get_node_indices(
        node_ids=node_ids,
        node_names=model_spec.volatility_node_names,
    )

    model_figure_dir = output_dir / model_spec.name
    model_figure_dir.mkdir(parents=True, exist_ok=True)

    all_nodes_path = model_figure_dir / "all_nodes.png"
    value_nodes_path = model_figure_dir / "value_nodes.png"
    volatility_nodes_path = model_figure_dir / "volatility_nodes.png"

    # 1. Full pyhgf diagnostic plot.
    try:
        plot_all_nodes(
            model=model,
            output_path=all_nodes_path,
            show_posterior=show_posterior,
            show_surprise=show_surprise,
            show_total_surprise=show_total_surprise,
        )
        print(f"Saved: {all_nodes_path}")
    except Exception as error:
        print(
            f"Could not create all-node plot for {model_spec.name}: {error}"
        )

    # 2. Value/state node plot: value_A and value_B.
    plot_selected_nodes(
        model=model,
        node_idxs=value_node_idxs,
        output_path=value_nodes_path,
        title=f"{model_spec.name}: value nodes",
        show_posterior=show_posterior,
        show_surprise=show_surprise,
    )
    print(f"Saved: {value_nodes_path}")

    # 3. Volatility-related node plot.
    plot_selected_nodes(
        model=model,
        node_idxs=volatility_node_idxs,
        output_path=volatility_nodes_path,
        title=f"{model_spec.name}: volatility-related nodes",
        show_posterior=show_posterior,
        show_surprise=show_surprise,
    )
    print(f"Saved: {volatility_nodes_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build command-line interface.
    """
    parser = argparse.ArgumentParser(
        description="Create pyhgf built-in trajectory plots for fitted models."
    )

    parser.add_argument(
        "--input-data",
        type=str,
        default=str(
            PROJECT_ROOT
            / "outputs"
            / "hgf_walk_outputs"
            / "model_runs"
            / "model_input_data.npy"
        ),
        help="Path to model_input_data.npy.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(
            PROJECT_ROOT
            / "outputs"
            / "hgf_walk_outputs"
            / "figures"
            / "pyhgf_builtin"
        ),
        help="Directory where pyhgf built-in plots should be saved.",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            LOCAL_ONLY_NAME,
            GLOBAL_ONLY_NAME,
            SHARED_VOLATILITY_NAME,
        ],
        choices=list(MODEL_REGISTRY.keys()),
        help="Which models to plot.",
    )

    parser.add_argument(
        "--input-precision",
        type=float,
        default=100.0,
        help=(
            "Precision of input nodes. If observation noise sd is 0.10, "
            "precision should be 1 / 0.10**2 = 100."
        ),
    )

    parser.add_argument(
        "--update-type",
        type=str,
        default="eHGF",
        choices=["eHGF", "standard", "unbounded"],
        help="pyhgf update type.",
    )

    parser.add_argument(
        "--max-posterior-precision",
        type=float,
        default=1e10,
        help="Numerical cap for posterior precision.",
    )

    parser.add_argument(
        "--hide-posterior",
        action="store_true",
        help="If set, do not overlay posterior trajectories.",
    )

    parser.add_argument(
        "--hide-surprise",
        action="store_true",
        help="If set, do not show surprise shading.",
    )

    parser.add_argument(
        "--show-total-surprise",
        action="store_true",
        help="If set, show total surprise on all-node plots.",
    )

    return parser


def main() -> None:
    """
    Run built-in pyhgf plotting for selected models.
    """
    parser = build_arg_parser()
    args = parser.parse_args()

    input_data = load_input_data(args.input_data)
    output_dir = Path(args.output_dir)

    print(f"Loaded input data: {args.input_data}")
    print(f"input_data shape: {input_data.shape}")

    model_kwargs = {
        "input_precision": args.input_precision,
        "update_type": args.update_type,
        "max_posterior_precision": args.max_posterior_precision,
    }

    show_posterior = not args.hide_posterior
    show_surprise = not args.hide_surprise

    for model_name in args.models:
        model_spec = MODEL_REGISTRY[model_name]

        run_plots_for_one_model(
            model_spec=model_spec,
            input_data=input_data,
            output_dir=output_dir,
            model_kwargs=model_kwargs,
            show_posterior=show_posterior,
            show_surprise=show_surprise,
            show_total_surprise=args.show_total_surprise,
        )

    print("\nFinished pyhgf built-in plotting.")


if __name__ == "__main__":
    main()