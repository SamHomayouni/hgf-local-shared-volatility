"""Export presentation-ready parameter tables for all four current environments.

Run from the repository root:
    python scripts/export_environment_parameter_tables_hgf_random_walk.py

Reads current YAML configurations; does not generate environments or run models.
Creates an overview and detailed CSV, PNG, and Markdown tables.
"""
from __future__ import annotations

import argparse
import math
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.table import Table
import pandas as pd
import yaml

PROJECT_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "src").is_dir() and (p / "config").is_dir()),
    None,
)
if PROJECT_ROOT is None:
    raise RuntimeError("Place this file in your repository's scripts/ directory.")

CONFIG_NAMES = {
    "local_A": "hgf_random_walk_local_A.yaml",
    "local_B": "hgf_random_walk_local_B.yaml",
    "global": "hgf_random_walk_global.yaml",
    "hybrid": "hgf_hybrid_weber.yaml",
}
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "hgf_walk_outputs" / "tables" / "environment_parameters"

DESCRIPTIONS = {
    "simulation.name": "Environment simulation identifier.",
    "simulation.random_seed": "Random seed for generating this environment.",
    "simulation.n_trials": "Number of observation trials per stream.",
    "latent_dynamics.lambda_x": "Latent-value autoconnection (1 = random walk).",
    "latent_dynamics.rho_x": "Constant latent-value drift.",
    "latent_dynamics.q_low": "Fixed stable-block process variance; Local/Global designs only.",
    "latent_dynamics.q_high": "Fixed volatile-block process variance; Local/Global designs only.",
    "latent_dynamics.observation_noise_sd": "Standard deviation of environmental observation noise.",
    "latent_dynamics.value_tonic_volatility": "Tonic log-process-variance term for the Hybrid value states.",
    "latent_dynamics.volatility_coupling": "Local-volatility-to-value process-variance coupling (Hybrid).",
    "local_volatility_dynamics.lambda_v": "Autoconnection of the generative local-volatility states.",
    "local_volatility_dynamics.rho_v": "Constant drift of generative local volatility.",
    "local_volatility_dynamics.tonic_volatility": "Log innovation variance of generative local-volatility states.",
    "local_volatility_dynamics.shared_value_coupling": "Shared-state-to-local-volatility mean coupling (PROCESS).",
    "shared_signal.baseline_mean": "Mean shared-state signal outside prescribed pulses.",
    "shared_signal.noise_sd": "Per-trial standard deviation of shared-signal jitter.",
    "observation.log_variance": "Natural log of environmental observation variance; SD = exp(log_variance/2).",
    "stream_a.initial_mean": "Pre-first-trial latent value in stream A.",
    "stream_b.initial_mean": "Pre-first-trial latent value in stream B.",
    "stream_a.initial_volatility": "Pre-first-trial generative local volatility in stream A (Hybrid).",
    "stream_b.initial_volatility": "Pre-first-trial generative local volatility in stream B (Hybrid).",
    "derived.observation_noise_sd": "Environmental observation standard deviation, regardless of YAML parameterization.",
    "derived.observation_precision": "Inverse observation variance; matching input precision when correctly specified.",
    "derived.sqrt_q_low": "Stable-block process standard deviation (Local/Global).",
    "derived.sqrt_q_high": "Volatile-block process standard deviation (Local/Global).",
    "derived.q_high_to_q_low_ratio": "Ratio of volatile to stable process variances (Local/Global).",
    "derived.local_vol_innovation_variance": "Generative local-volatility innovation variance (Hybrid).",
    "derived.initial_q_a_reference": "Hybrid A reference process variance BEFORE first random state update.",
    "derived.initial_q_b_reference": "Hybrid B reference process variance BEFORE first random state update.",
}


def display(value: object) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if value == 0:
            return "0"
        if value.is_integer() and abs(value) < 1e8:
            return str(int(value))
        return f"{value:.10g}"
    return str(value)


def save_markdown(df: pd.DataFrame, path: Path) -> None:
    def quote(v: object) -> str:
        return str(v).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(quote(c) for c in df.columns) + " |",
             "|" + "|".join("---" for _ in df.columns) + "|"]
    for row in df.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(quote(v) for v in row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_image(df: pd.DataFrame, path: Path, title: str, widths: list[float], note: str = "") -> None:
    # Use adaptive line wrapping and row heights to avoid truncating longer descriptions.
    chars = [max(12, int(128 * w)) for w in widths]
    values = [list(map(str, df.columns))] + [list(map(str, row)) for row in df.itertuples(index=False, name=None)]
    wrapped = []
    heights = []
    for i, row in enumerate(values):
        cells = ["\n".join(textwrap.wrap(v, width=chars[j], break_long_words=False,
                                         break_on_hyphens=False) or [""]) for j, v in enumerate(row)]
        wrapped.append(cells)
        lines = max(x.count("\n") + 1 for x in cells)
        heights.append((0.42 if i == 0 else 0.31) + 0.22 * (lines - 1))
    fig_height = max(2.2, sum(heights) + 1.15)
    fig, ax = plt.subplots(figsize=(15.5, fig_height))
    ax.axis("off")
    table = Table(ax, bbox=[0, 0, 1, 1])
    total_height = sum(heights)
    for i, row in enumerate(wrapped):
        for j, val in enumerate(row):
            face = "#23354b" if i == 0 else ("#f1f4f7" if i % 2 == 0 else "#ffffff")
            cell = table.add_cell(i, j, width=widths[j], height=heights[i] / total_height,
                                  text=val, facecolor=face, loc="left", edgecolor="#dee4e9")
            cell.PAD = 0.025
            cell.set_linewidth(0.45)
            cell.get_text().set_color("white" if i == 0 else "#223042")
            cell.get_text().set_fontweight("bold" if i == 0 else "normal")
            cell.get_text().set_fontsize(9 if i == 0 else 8.5)
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


def get_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing configuration: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("environment"), dict):
        raise ValueError(f"Invalid environment YAML: {path}")
    return config


def flatten_params(name: str, config: dict) -> pd.DataFrame:
    env = config["environment"]
    sections = [
        ("Simulation", "simulation", config.get("simulation", {})),
        ("Value dynamics", "latent_dynamics", env.get("latent_dynamics", {})),
        ("Local volatility dynamics", "local_volatility_dynamics", env.get("local_volatility_dynamics", {})),
        ("Shared signal", "shared_signal", env.get("shared_signal", {})),
        ("Observation", "observation", env.get("observation", {})),
        ("Stream A initial state", "stream_a", env.get("stream_a", {})),
        ("Stream B initial state", "stream_b", env.get("stream_b", {})),
    ]
    rows = []
    for section, prefix, params in sections:
        for key, value in params.items():
            if key == "pulses":  # exported as a separate table
                continue
            full_key = f"{prefix}.{key}"
            if isinstance(value, (dict, list)):
                raise ValueError(f"Unexpected structured config parameter: {name}/{full_key}")
            rows.append({"Section": section, "Parameter": full_key, "Value": display(value),
                         "Explanation": DESCRIPTIONS.get(full_key, "Configuration parameter.")})
    return pd.DataFrame(rows)


def derived_parameters(name: str, config: dict) -> pd.DataFrame:
    dynamics = config["environment"].get("latent_dynamics", {})
    env = config["environment"]
    derived: dict[str, float] = {}
    if "q_low" in dynamics and "q_high" in dynamics:
        ql, qh = float(dynamics["q_low"]), float(dynamics["q_high"])
        if ql <= 0 or qh <= 0:
            raise ValueError(f"Nonpositive process variance in {name}")
        derived.update(sqrt_q_low=math.sqrt(ql), sqrt_q_high=math.sqrt(qh),
                       q_high_to_q_low_ratio=qh / ql)
    if "observation_noise_sd" in dynamics:
        sigma = float(dynamics["observation_noise_sd"])
    elif "log_variance" in env.get("observation", {}):
        sigma = math.exp(float(env["observation"]["log_variance"]) / 2)
    else:
        raise ValueError(f"Cannot find observation noise specification for {name}")
    if sigma <= 0:
        raise ValueError(f"Observation SD must be positive for {name}")
    derived["observation_noise_sd"] = sigma
    derived["observation_precision"] = 1.0 / (sigma * sigma)
    if "local_volatility_dynamics" in env:
        omega_v = float(env["local_volatility_dynamics"]["tonic_volatility"])
        omega_x = float(dynamics["value_tonic_volatility"])
        kappa = float(dynamics["volatility_coupling"])
        derived["local_vol_innovation_variance"] = math.exp(omega_v)
        for key, output in (("stream_a", "initial_q_a_reference"),
                            ("stream_b", "initial_q_b_reference")):
            v0 = float(env[key]["initial_volatility"])
            derived[output] = math.exp(omega_x + kappa * v0)
    return pd.DataFrame([
        {"Parameter": key, "Value": display(val),
         "Explanation": DESCRIPTIONS.get("derived." + key, "Derived quantity.")}
        for key, val in derived.items()
    ])


def blocks_table(config: dict) -> pd.DataFrame:
    rows = []
    for b in config.get("blocks", []):
        s, e = int(b["start_index"]), int(b["stop_index"])
        rows.append({"Block": b["name"], "First trial": s + 1, "Last trial": e,
                     "N trials": e - s, "Stream A": b["stream_a_regime"],
                     "Stream B": b["stream_b_regime"]})
    return pd.DataFrame(rows)


def pulse_table(config: dict) -> pd.DataFrame:
    pulses = config["environment"].get("shared_signal", {}).get("pulses", [])
    return pd.DataFrame([
        {"Pulse": p["name"], "First trial": int(p["start_index"]) + 1,
         "Last trial": int(p["stop_index"]),
         "N trials": int(p["stop_index"]) - int(p["start_index"]),
         "Shared mean": display(float(p["mean"]))}
        for p in pulses
    ])


def overview_row(config: dict, name: str) -> dict[str, str]:
    env = config["environment"]
    lat = env.get("latent_dynamics", {})
    v = env.get("local_volatility_dynamics", {})
    signal = env.get("shared_signal", {})
    ddf = derived_parameters(name, config)
    d = dict(zip(ddf["Parameter"], ddf["Value"]))
    return {
        "Environment": name,
        "Trials": display(config["simulation"].get("n_trials", "")),
        "Seed": display(config["simulation"].get("random_seed", "")),
        "Obs SD": d["observation_noise_sd"],
        "Obs precision": d["observation_precision"],
        "Value lambda": display(lat.get("lambda_x", "")),
        "q low": display(lat.get("q_low", "not applicable")),
        "q high": display(lat.get("q_high", "not applicable")),
        "Value omega": display(lat.get("value_tonic_volatility", "not applicable")),
        "Local vol omega": display(v.get("tonic_volatility", "not applicable")),
        "Local vol kappa": display(lat.get("volatility_coupling", "not applicable")),
        "Shared coupling": display(v.get("shared_value_coupling", "not applicable")),
        "Shared pulse count": display(len(signal.get("pulses", []))) if signal else "not applicable",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=PROJECT_ROOT / "config")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--environments", nargs="+", choices=list(CONFIG_NAMES), default=list(CONFIG_NAMES))
    args = parser.parse_args()
    overview = []
    for name in args.environments:
        path = args.config_dir / CONFIG_NAMES[name]
        config = get_config(path)
        target = args.output_dir / name
        print(f"Exporting {name} from {path}")
        save_table(flatten_params(name, config), target, "parameters", f"{name}: environment parameters",
                   [0.17, 0.32, 0.13, 0.38], note=f"Source: config/{path.name} (current YAML)")
        save_table(derived_parameters(name, config), target, "derived_parameters",
                   f"{name}: derived parameters", [0.34, 0.16, 0.50],
                   note="Computed from current YAML; Hybrid initial q is pre-first-update reference.")
        blocks = blocks_table(config)
        if not blocks.empty:
            save_table(blocks, target, "block_schedule", f"{name}: block schedule",
                       [0.15, 0.16, 0.16, 0.15, 0.19, 0.19])
        pulses = pulse_table(config)
        if not pulses.empty:
            save_table(pulses, target, "shared_pulses", f"{name}: shared pulses",
                       [0.24, 0.19, 0.19, 0.17, 0.21],
                       note="Trial positions are 1-based here; original YAML indices are 0-based.")
        overview.append(overview_row(config, name))
    compare = pd.DataFrame(overview)
    save_table(compare, args.output_dir, "environment_overview",
               "Environment comparison: four generative conditions",
               [0.18] + [0.82 / (len(compare.columns) - 1)] * (len(compare.columns) - 1),
               note="not applicable = distinct generative structure; Hybrid has dynamic q, not fixed q_low/q_high.")
    print("Environment tables finished. No simulation files were changed.")


if __name__ == "__main__":
    main()
