"""Canonical project paths.

This module centralizes filesystem locations used by the current
Gaussian-random-walk modelling pipeline.

Generated outputs are organized by environment condition:

    outputs/hgf_walk_outputs/local_A/
    outputs/hgf_walk_outputs/local_B/
    outputs/hgf_walk_outputs/global/
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "src").is_dir() and (parent / "config").is_dir()
)

CONFIG_DIR = PROJECT_ROOT / "config"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
HGF_WALK_OUTPUTS_DIR = OUTPUTS_DIR / "hgf_walk_outputs"

ENVIRONMENT_NAMES = (
    "local_A",
    "local_B",
    "global",
)

DEFAULT_ENVIRONMENT_NAME = "local_A"


def get_environment_output_dir(
    environment_name: str = DEFAULT_ENVIRONMENT_NAME,
) -> Path:
    """Return the output directory for one environment condition."""

    if environment_name not in ENVIRONMENT_NAMES:
        raise ValueError(
            f"Unknown environment {environment_name!r}. "
            f"Expected one of {ENVIRONMENT_NAMES}."
        )

    return HGF_WALK_OUTPUTS_DIR / environment_name


DEFAULT_ENVIRONMENT_DIR = get_environment_output_dir()

DEFAULT_ENVIRONMENT_PATH = (
    DEFAULT_ENVIRONMENT_DIR / "environment_long.csv"
)

DEFAULT_MODEL_RUNS_DIR = (
    DEFAULT_ENVIRONMENT_DIR / "model_runs"
)

DEFAULT_LEARNING_RATES_DIR = (
    DEFAULT_ENVIRONMENT_DIR / "learning_rates"
)

DEFAULT_SYNTHETIC_RESPONSES_DIR = (
    DEFAULT_ENVIRONMENT_DIR / "synthetic_responses"
)

DEFAULT_FIGURES_DIR = (
    DEFAULT_ENVIRONMENT_DIR / "figures"
)