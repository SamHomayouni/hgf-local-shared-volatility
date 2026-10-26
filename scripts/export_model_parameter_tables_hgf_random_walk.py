"""Export presentable tables of current fixed inference-model constructor parameters.

Run from the repository root:
    python scripts/export_model_parameter_tables_hgf_random_walk.py

Reads Python function defaults without importing or running pyHGF. This reports
current BUILDER DEFAULTS (not parameters estimated by fitting).
Outputs detail tables and a side-by-side model comparison as CSV, PNG, Markdown.
"""
from __future__ import annotations

import argparse
import ast
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.table import Table
import pandas as pd

PROJECT_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "src").is_dir() and (p / "config").is_dir()),
    None,
)
if PROJECT_ROOT is None:
    raise RuntimeError("Place this file in your repository's scripts/ directory.")

MODEL_SOURCES = {
    "local_only": ("local_only.py", "build_local_only_model"),
    "global_only": ("global_only.py", "build_global_only_model"),
    "shared_volatility": ("shared_volatility.py", "build_shared_vol_model"),
}
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "hgf_walk_outputs" / "tables" / "model_parameters"
DESCRIPTIONS = {
    "input_precision": "Precision of observed inputs, e.g. 100 for observation SD 0.10.",
    "value_mean": "Initial mean of each inferred value state.",
    "value_precision": "Initial precision of each inferred value state.",
    "value_tonic_volatility": "Tonic log process variance of inferred value states; depends on volatility-parent beliefs.",
    "value_autoconnection_strength": "Value-state autoconnection; 1 means no mean reversion.",
    "value_coupling": "Coupling from the value state to its corresponding input node.",
    "volatility_mean": "Initial mean of each local volatility parent (Local-only).",
    "volatility_precision": "Initial precision of each local volatility parent (Local-only).",
    "volatility_tonic_volatility": "Tonic log innovation variance of local volatility states (Local-only).",
    "volatility_coupling": "Volatility coupling from a local parent to its value state.",
    "shared_volatility_mean": "Initial mean of shared volatility parent (Global-only).",
    "shared_volatility_precision": "Initial precision of shared volatility parent (Global-only).",
    "shared_volatility_tonic_volatility": "Tonic log innovation variance of shared volatility parent (Global-only).",
    "shared_volatility_coupling": "Shared volatility parent coupling to both value states (Global-only).",
    "local_vol_mean": "Initial mean of each local volatility node (Hybrid).",
    "local_vol_precision": "Initial precision of each local volatility node (Hybrid).",
    "local_vol_tonic_volatility": "Tonic log innovation variance of local volatility nodes (Hybrid).",
    "local_volatility_coupling": "Local-volatility-to-value volatility coupling (Hybrid).",
    "local_vol_autoconnection_strength": "Autoconnection of Hybrid local volatility nodes; values below 1 give mean reversion.",
    "shared_value_mean": "Initial mean of higher shared value parent (Hybrid).",
    "shared_value_precision": "Initial precision of higher shared value parent (Hybrid).",
    "shared_value_tonic_volatility": "Tonic log innovation variance of higher shared value parent (Hybrid).",
    "shared_value_coupling": "Shared value parent mean coupling to both local volatility nodes (Hybrid).",
    "update_type": "pyHGF update approximation (e.g. eHGF).",
    "max_posterior_precision": "Upper numerical cap on posterior precision.",
}


def display(v: object) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if v == 0:
            return "0"
        if v.is_integer() and abs(v) < 1e8:
            return str(int(v))
        return f"{v:.10g}"
    return str(v)


def source_defaults(path: Path, function_name: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing model module: {path}")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    f = next((node for node in tree.body
              if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name), None)
    if f is None:
        raise ValueError(f"Function {function_name} not found in {path}")
    parameters: dict[str, object] = {}
    for arg, default in zip(f.args.kwonlyargs, f.args.kw_defaults):
        if default is None:
            continue
        try:
            parameters[arg.arg] = ast.literal_eval(default)
        except (ValueError, TypeError, SyntaxError) as exc:
            raise ValueError(f"Parameter {arg.arg} in {path} has a computed default. "
                             "Cannot safely extract its runtime value by static parsing.") from exc
    if not parameters:
        raise ValueError(f"No literal keyword defaults found in {path}")
    return parameters


def save_markdown(df: pd.DataFrame, path: Path) -> None:
    def quote(v: object) -> str:
        return str(v).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(quote(c) for c in df.columns) + " |",
             "|" + "|".join("---" for _ in df.columns) + "|"]
    for row in df.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(quote(v) for v in row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_image(df: pd.DataFrame, path: Path, title: str, widths: list[float], note: str = "") -> None:
    chars = [max(12, int(128 * w)) for w in widths]
    values = [list(map(str, df.columns))] + [list(map(str, row)) for row in df.itertuples(index=False, name=None)]
    wrapped, heights = [], []
    for i, row in enumerate(values):
        cells = ["\n".join(textwrap.wrap(v, width=chars[j], break_long_words=False,
                                         break_on_hyphens=False) or [""]) for j, v in enumerate(row)]
        wrapped.append(cells)
        heights.append((0.42 if i == 0 else 0.31) + 0.22 * (max(c.count("\n") + 1 for c in cells) - 1))
    fig_height = max(2.2, sum(heights) + 1.15)
    fig, ax = plt.subplots(figsize=(15.5, fig_height))
    ax.axis("off")
    table = Table(ax, bbox=[0, 0, 1, 1])
    for i, row in enumerate(wrapped):
        for j, value in enumerate(row):
            face = "#23354b" if i == 0 else ("#f1f4f7" if i % 2 == 0 else "white")
            cell = table.add_cell(i, j, widths[j], heights[i] / sum(heights),
                                  text=value, facecolor=face, loc="left", edgecolor="#dee4e9")
            cell.PAD = 0.025
            cell.set_linewidth(0.45)
            cell.get_text().set_color("white" if i == 0 else "#223042")
            cell.get_text().set_fontsize(9 if i == 0 else 8.5)
            cell.get_text().set_fontweight("bold" if i == 0 else "normal")
    ax.add_table(table)
    fig.suptitle(title, fontsize=17, fontweight="bold", y=0.997, color="#23354b")
    if note:
        fig.text(0.02, 0.008, note, ha="left", fontsize=8, color="#515a67")
    fig.subplots_adjust(left=0.02, right=0.98, top=1 - 0.74 / fig_height,
                        bottom=(0.38 if note else 0.08) / fig_height)
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def save_table(df: pd.DataFrame, directory: Path, stem: str, title: str,
               widths: list[float], note: str = "") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    df.to_csv(directory / f"{stem}.csv", index=False)
    save_markdown(df, directory / f"{stem}.md")
    save_image(df, directory / f"{stem}.png", title, widths, note)
    print(f"  {directory / (stem + '.png')}")


def check_run_metadata(models: dict[str, dict], runs_root: Path) -> None:
    """Warn if runner overrides disagree with current source defaults."""
    if not runs_root.exists():
        return
    discrepancies = []
    for environment in ("local_A", "local_B", "global", "hybrid"):
        for model_name, defaults in models.items():
            metadata = runs_root / environment / "model_runs" / model_name / "run_metadata.json"
            if not metadata.is_file():
                continue
            run = json.loads(metadata.read_text(encoding="utf-8"))
            for key, value in run.get("model_kwargs", {}).items():
                if key in defaults and defaults[key] != value:
                    discrepancies.append(f"{environment}/{model_name}: {key} source={defaults[key]} run={value}")
    if discrepancies:
        print("\nWARNING: saved run metadata overrides do not match current builder defaults:")
        for diff in discrepancies:
            print("  " + diff)
        print("The exported tables report builder defaults, not these differing run overrides.")
    else:
        print("No conflicting builder defaults were found in available run_metadata.json overrides.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "src" / "models")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runs-root", type=Path, default=PROJECT_ROOT / "outputs" / "hgf_walk_outputs")
    args = parser.parse_args()
    models: dict[str, dict[str, object]] = {}
    parameter_order: list[str] = []
    for model_name, (filename, function) in MODEL_SOURCES.items():
        path = args.model_dir / filename
        params = source_defaults(path, function)
        models[model_name] = params
        for key in params:
            if key not in parameter_order:
                parameter_order.append(key)
        df = pd.DataFrame([
            {"Parameter": name, "Value": display(value),
             "Explanation": DESCRIPTIONS.get(name, "Current constructor default.")}
            for name, value in params.items()
        ])
        print(f"Exporting {model_name} from {path}")
        save_table(df, args.output_dir / model_name, "parameters", f"{model_name}: inference parameters",
                   [0.29, 0.14, 0.57],
                   note=f"Source: src/models/{filename} defaults; NOT Bayesian-fitted parameters.")
    comparison = pd.DataFrame([
        {"Parameter": p, **{model: display(models[model][p]) if p in models[model] else "not applicable"
                            for model in MODEL_SOURCES}}
        for p in parameter_order
    ])
    save_table(comparison, args.output_dir, "model_comparison",
               "Comparison of three fixed inference models", [0.28, 0.24, 0.24, 0.24],
               note="Current constructor defaults; different models have structurally different parameters.")
    check_run_metadata(models, args.runs_root)
    print("Model parameter tables finished. No models or saved trajectories were modified.")


if __name__ == "__main__":
    main()
