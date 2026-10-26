"""Existing plotting command, extended for saved Local, Global and Hybrid data.

Run from the repository root, for example:
    python scripts/plot_environment_hgf_random_walk.py --environment outputs/hgf_walk_outputs/hybrid/environment_long.csv

The reusable implementation lives in src/analysis/plot_environments.py.
This command never regenerates or changes environment data or model parameters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = next(
    (p for p in Path(__file__).resolve().parents
     if (p / "src").is_dir() and (p / "config").is_dir()), None)
if PROJECT_ROOT is None:
    raise RuntimeError("Place this script under your repository's scripts/ directory.")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")  # File output only; no GUI is needed on a server.

from src.utils.paths import DEFAULT_ENVIRONMENT_PATH, DEFAULT_FIGURES_DIR
# Re-export old helpers so existing Python imports remain usable.
from src.analysis.plot_environments import (
    add_volatile_block_shading, get_volatile_blocks, load_environment,
    plot_both_latent_means, plot_environment, plot_latent_innovations,
    plot_observation_noise, plot_process_quantity, plot_stream, save_figure,
)

DEFAULT_FIGURE_DIR = Path(DEFAULT_FIGURES_DIR) / "environment"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot saved two-stream environment diagnostics, including available volatility states.")
    parser.add_argument("--environment", default=str(DEFAULT_ENVIRONMENT_PATH),
                        help="Path to environment_long.csv (not the wide file).")
    parser.add_argument("--output-dir", default=None,
                        help="Figure directory. Default: <input CSV folder>/figures/environment/.")
    parser.add_argument("--volatility-only", action="store_true",
                        help="Only write the new volatility diagnostics, leaving the seven old plots alone.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    environment_path = Path(args.environment).expanduser().resolve()
    if args.output_dir is not None:
        figure_dir = Path(args.output_dir).expanduser().resolve()
    elif environment_path == Path(DEFAULT_ENVIRONMENT_PATH).resolve():
        figure_dir = DEFAULT_FIGURE_DIR
    else:
        figure_dir = environment_path.parent / "figures" / "environment"
    try:
        df = load_environment(environment_path)
        report = plot_environment(environment_df=df, figure_dir=figure_dir,
                                  volatility_only=args.volatility_only)
        report["environment_csv"] = str(environment_path)
        report["environment_csv_sha256"] = hashlib.sha256(environment_path.read_bytes()).hexdigest()
        report["note"] = (
            "Only listed figures were written in this invocation. Existing PNGs of the same "
            "names were replaced; unrelated files were not deleted. Input CSV was not modified.")
        figure_dir.mkdir(parents=True, exist_ok=True)
        manifest = figure_dir / "environment_plot_manifest.json"
        manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Plotting failed: {exc}\n")
    print(f"\nWrote {len(report['figures_written_this_run'])} figures to {figure_dir}")
    print(f"Plot manifest: {manifest}")


if __name__ == "__main__":
    main()
