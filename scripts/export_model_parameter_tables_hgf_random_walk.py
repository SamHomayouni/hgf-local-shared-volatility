"""
Export parameter tables for the pyhgf model constructors used with the
HGF-style random-walk environment.

This script reads the default arguments from the three model files:

    src/models/local_only.py
    src/models/global_only.py
    src/models/shared_volatility.py

and creates parameter tables for:

    local_only
    global_only
    shared_volatility

Outputs are saved to:

    outputs/hgf_walk_outputs/tables/model_parameters/

Each model gets:
    - parameters.csv
    - parameters.md
    - parameters.png

Run from project root:

    python scripts/export_model_parameter_tables_hgf_random_walk.py
"""

from __future__ import annotations

from pathlib import Path
import inspect
import sys

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir() and (p / "config").is_dir())

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.models.local_only import (  # noqa: E402
    MODEL_NAME as LOCAL_ONLY_NAME,
    build_local_only_model,
)

from src.models.global_only import (  # noqa: E402
    MODEL_NAME as GLOBAL_ONLY_NAME,
    build_global_only_model,
)

from src.models.shared_volatility import (  # noqa: E402
    MODEL_NAME as SHARED_VOLATILITY_NAME,
    build_shared_vol_model,
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "hgf_walk_outputs"
    / "tables"
    / "model_parameters"
)


PARAMETER_DESCRIPTIONS = {
    # Shared parameters
    "input_precision": (
        "Observation precision of continuous input nodes. For the current "
        "HGF-random-walk environment, observation_noise_sd = 0.10, so "
        "input_precision = 1 / 0.10**2 = 100."
    ),
    "value_mean": "Initial mean of value/state nodes.",
    "value_precision": "Initial precision of value/state nodes.",
    "value_tonic_volatility": (
        "Baseline log-volatility of value/state nodes. For the current "
        "environment, q_low = 0.0004, so log(q_low) is approximately -7.824."
    ),
    "value_autoconnection_strength": (
        "Autoconnection strength lambda for value/state nodes. In the current "
        "environment, lambda_x = 0.95 for both streams."
    ),
    "value_coupling": "Value coupling from value nodes to input nodes.",
    "update_type": "pyhgf volatility update type.",
    "max_posterior_precision": "Numerical upper bound on posterior precision.",

    # Local-only parameters
    "volatility_mean": "Initial mean of local volatility parents.",
    "volatility_precision": "Initial precision of local volatility parents.",
    "volatility_tonic_volatility": (
        "Baseline log-volatility of local volatility parents. This controls "
        "how much the volatility-parent state itself can change over time; it "
        "is not the same as q_low of the value process."
    ),
    "volatility_coupling": (
        "Volatility coupling from local volatility parents to value nodes."
    ),

    # Global-only parameters
    "shared_volatility_mean": "Initial mean of the shared volatility parent.",
    "shared_volatility_precision": (
        "Initial precision of the shared volatility parent."
    ),
    "shared_volatility_tonic_volatility": (
        "Baseline log-volatility of the shared volatility parent. This controls "
        "how much the shared volatility state itself can change over time; it "
        "is not the same as q_low of the value process."
    ),
    "shared_volatility_coupling": (
        "Volatility coupling from shared volatility parent to value nodes."
    ),

    # Shared-volatility / hybrid parameters
    "local_vol_mean": "Initial mean of local volatility parents.",
    "local_vol_precision": "Initial precision of local volatility parents.",
    "local_vol_tonic_volatility": (
        "Baseline log-volatility of local volatility parents in the hybrid "
        "model. This controls volatility-node dynamics, not the value-level "
        "baseline process variance."
    ),
    "shared_value_mean": "Initial mean of the shared higher-level value parent.",
    "shared_value_precision": (
        "Initial precision of the shared higher-level value parent."
    ),
    "shared_value_tonic_volatility": (
        "Baseline log-volatility of the shared higher-level value parent."
    ),
    "local_volatility_coupling": (
        "Volatility coupling from local volatility parents to value nodes."
    ),
    "shared_value_coupling": (
        "Value coupling from shared parent to local volatility nodes."
    ),
    "local_vol_autoconnection_strength": (
        "Autoconnection strength lambda for local volatility nodes in the "
        "hybrid model."
    ),

    # Generic fallback names
    "autoconnection_strength": "Autoconnection strength lambda.",
}


MODEL_BUILDERS = {
    LOCAL_ONLY_NAME: build_local_only_model,
    GLOBAL_ONLY_NAME: build_global_only_model,
    SHARED_VOLATILITY_NAME: build_shared_vol_model,
}


def format_value(value: object) -> str:
    """
    Format parameter values for readable tables.
    """
    if isinstance(value, float):
        if value == 0:
            return "0.0"

        if abs(value) >= 1e5 or abs(value) < 1e-4:
            return f"{value:.1e}"

        return f"{value:g}"

    return str(value)


def extract_default_parameters(builder_function) -> pd.DataFrame:
    """
    Extract default keyword-only parameters from a model constructor.
    """
    signature = inspect.signature(builder_function)

    rows = []

    for name, parameter in signature.parameters.items():
        if parameter.default is inspect.Parameter.empty:
            continue

        value = parameter.default
        description = PARAMETER_DESCRIPTIONS.get(
            name,
            "No description added yet.",
        )

        rows.append(
            {
                "Parameter": name,
                "Value": format_value(value),
                "Description": description,
            }
        )

    return pd.DataFrame(rows)


def save_markdown_table(dataframe: pd.DataFrame, output_path: Path) -> None:
    """
    Save table as markdown.
    """
    markdown = dataframe.to_markdown(index=False)
    output_path.write_text(markdown, encoding="utf-8")


def save_png_table(
    dataframe: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    """
    Save table as a PNG image using matplotlib.
    """
    n_rows = len(dataframe)

    figure_width = 14
    figure_height = max(3, 0.45 * n_rows + 1.4)

    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    ax.axis("off")

    ax.set_title(
        title,
        fontsize=16,
        fontweight="bold",
        pad=18,
    )

    table = ax.table(
        cellText=dataframe.values,
        colLabels=dataframe.columns,
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.25, 0.15, 0.60],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    for column_idx in range(len(dataframe.columns)):
        header_cell = table[(0, column_idx)]
        header_cell.set_text_props(weight="bold")
        header_cell.set_facecolor("#f2f2f2")

    for row_idx in range(1, n_rows + 1):
        table[(row_idx, 1)].set_text_props(ha="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def export_parameter_table(
    *,
    model_name: str,
    builder_function,
    output_dir: Path,
) -> None:
    """
    Export CSV, markdown, and PNG tables for one model.
    """
    model_output_dir = output_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    dataframe = extract_default_parameters(builder_function)

    csv_path = model_output_dir / "parameters.csv"
    markdown_path = model_output_dir / "parameters.md"
    png_path = model_output_dir / "parameters.png"

    dataframe.to_csv(csv_path, index=False)
    save_markdown_table(dataframe, markdown_path)
    save_png_table(
        dataframe=dataframe,
        output_path=png_path,
        title=f"{model_name}: default model parameters",
    )

    print(f"\nSaved parameter table for {model_name}")
    print(f"CSV:      {csv_path}")
    print(f"Markdown: {markdown_path}")
    print(f"PNG:      {png_path}")


def main() -> None:
    """
    Export parameter tables for all model constructors.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for model_name, builder_function in MODEL_BUILDERS.items():
        export_parameter_table(
            model_name=model_name,
            builder_function=builder_function,
            output_dir=OUTPUT_DIR,
        )

    print("\nFinished exporting HGF-random-walk model parameter tables.")


if __name__ == "__main__":
    main()