"""Plot saved Local, Global and Hybrid environment data, not HGF beliefs.

The old Local/Global generator stores prescribed process variances, not hidden
volatility nodes. Its extra volatility plot is therefore log(process_variance),
explicitly labelled as a schedule. Only saved local_volatility/shared_state
columns are plotted as states. No HGF parameters are assumed or reconstructed.

All functions are read-only with respect to input data. No generator, fitting
code, clipping, normalization, smoothing or seed selection is invoked here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

STREAMS = ("A", "B")
REQUIRED_COLUMNS = {
    "trial_index", "trial", "stream", "block", "regime", "latent_mean",
    "observation", "changepoint", "process_variance", "process_sd",
    "latent_innovation", "observation_noise", "is_a_volatile_block",
}
NUMERIC_COLUMNS = (
    "latent_mean", "observation", "process_variance", "process_sd",
    "latent_innovation", "observation_noise", "local_volatility",
    "local_volatility_innovation", "deterministic_local_volatility",
    "log_process_variance", "deterministic_log_process_variance",
    "shared_state", "shared_mean", "shared_noise", "shared_volatility_drift",
)
SHARED_COLUMNS = (
    "shared_state", "shared_mean", "shared_noise", "shared_volatility_drift",
    "pulse_name", "is_transition", "regime_is_nominal", "environment_type",
)
PREFIX = "hgf_environment"


def _booleans(series: pd.Series, name: str) -> pd.Series:
    """Parse CSV boolean flags without treating the text 'False' as true."""
    values = series.astype(str).str.strip().str.lower().map({
        "true": True, "false": False, "1": True, "0": False,
        "1.0": True, "0.0": False,
    })
    if values.isna().any():
        raise ValueError(f"{name} must contain only true/false or 0/1, without missing values.")
    return values.astype(bool)


def validate_environment(environment_df: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted, validated copy; never repair or mutate saved trajectories."""
    missing = REQUIRED_COLUMNS - set(environment_df.columns)
    if missing:
        raise ValueError(f"Environment file is missing required columns: {sorted(missing)}")
    df = environment_df.copy(deep=True)
    if df.empty:
        raise ValueError("The environment table is empty.")
    if set(df["stream"].dropna().unique()) != set(STREAMS) or df["stream"].isna().any():
        raise ValueError("The long table must contain exactly streams A and B.")
    for column in ("trial_index", "trial"):
        values = pd.to_numeric(df[column], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values != np.floor(values)).any():
            raise ValueError(f"{column} must contain finite integer indices.")
        if (values < (0 if column == "trial_index" else 1)).any():
            raise ValueError(f"Invalid negative/zero values in {column}.")
        df[column] = values.astype(np.int64)
    if df.duplicated(["trial_index", "stream"]).any():
        raise ValueError("Duplicate (trial_index, stream) rows; refusing to average them.")
    for column in ("block", "regime"):
        if df[column].isna().any():
            raise ValueError(f"Missing values in {column}.")
    if not set(df["regime"]).issubset({"stable", "volatile"}):
        raise ValueError("regime must be 'stable' or 'volatile'.")
    for column in NUMERIC_COLUMNS:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="raise")
            if not np.isfinite(df[column].to_numpy(dtype=float)).all():
                raise ValueError(f"{column} contains missing or non-finite values.")
    if (df["process_variance"] <= 0).any():
        raise ValueError("process_variance must be strictly positive; values are not clipped.")
    if not np.allclose(df["process_sd"], np.sqrt(df["process_variance"]),
                       rtol=1e-7, atol=1e-12):
        raise ValueError("process_sd is inconsistent with sqrt(process_variance).")
    for column in ("changepoint", "is_a_volatile_block", "is_transition", "regime_is_nominal"):
        if column in df:
            df[column] = _booleans(df[column], column)
    df = df.sort_values(["trial_index", "stream"]).reset_index(drop=True)
    a = df.loc[df["stream"] == "A"].reset_index(drop=True)
    b = df.loc[df["stream"] == "B"].reset_index(drop=True)
    if not np.array_equal(a["trial_index"], b["trial_index"]):
        raise ValueError("A and B must have the same trial indices; a stream/trial is missing.")
    for column in ("trial", "block", "is_a_volatile_block") + SHARED_COLUMNS:
        if column in df:
            if df[column].isna().any() or not np.array_equal(a[column], b[column]):
                raise ValueError(f"Shared column {column!r} disagrees between A and B rows.")
    if "deterministic_local_volatility" in df and "local_volatility" not in df:
        raise ValueError("A deterministic reference was saved without local_volatility.")
    if "shared_mean" in df and "shared_state" not in df:
        raise ValueError("shared_mean was saved without shared_state.")
    if "log_process_variance" in df and not np.allclose(
        df["log_process_variance"], np.log(df["process_variance"]), rtol=1e-8, atol=1e-10
    ):
        raise ValueError("log_process_variance is inconsistent with the natural log of Q.")
    return df


def load_environment(environment_path: str | Path) -> pd.DataFrame:
    """Load either generator's existing environment_long.csv, not a wide CSV."""
    path = Path(environment_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Could not find the generated environment CSV: {path}")
    return validate_environment(pd.read_csv(path))


def _stream(df: pd.DataFrame, stream: str) -> pd.DataFrame:
    return df.loc[df["stream"] == stream].sort_values("trial_index")


def has_nominal_regimes(df: pd.DataFrame) -> bool:
    if "regime_is_nominal" in df:
        return bool(_booleans(df["regime_is_nominal"], "regime_is_nominal").any())
    # Hybrid truth columns identify the current generator even in an older export.
    return {"shared_state", "local_volatility"}.issubset(df.columns)


def equal_process_schedules(df: pd.DataFrame) -> bool:
    """Equality of saved Q values, not a test for an inferred global architecture."""
    return np.array_equal(_stream(df, "A")["process_variance"].to_numpy(),
                          _stream(df, "B")["process_variance"].to_numpy())


def environment_title(df: pd.DataFrame) -> str:
    if {"shared_state", "local_volatility"}.issubset(df.columns):
        return "Hybrid environment"
    if "local_volatility" in df or "shared_state" in df:
        return "Environment with saved volatility states"
    a = _stream(df, "A")["regime"].eq("volatile").to_numpy()
    b = _stream(df, "B")["regime"].eq("volatile").to_numpy()
    if a.any() and not b.any():
        return "Local-A environment"
    if b.any() and not a.any():
        return "Local-B environment"
    if a.any() and np.array_equal(a, b) and equal_process_schedules(df):
        return "Global environment"
    return "Two-stream environment"


def get_volatile_blocks(environment_df: pd.DataFrame) -> list[tuple[int, int, str, str]]:
    """Contiguous labelled intervals; in Hybrid these are NOMINAL, not measured."""
    df = environment_df
    a, b = _stream(df, "A"), _stream(df, "B")
    times = a["trial_index"].to_numpy()
    nominal = has_nominal_regimes(df)
    ranges: list[tuple[int, int, str, str]] = []
    previous: tuple[str, bool, bool] | None = None
    start = last = 0

    def append_interval(key: tuple[str, bool, bool] | None, left: int, right: int) -> None:
        if key is None or not (key[1] or key[2]):
            return
        who = "A + B" if key[1] and key[2] else ("A" if key[1] else "B")
        label = f"Nominal volatile block ({who})" if nominal else f"{who} volatile"
        ranges.append((left, right, key[0], label))

    for t, block, ra, rb in zip(times, a["block"], a["regime"], b["regime"]):
        key = (str(block), ra == "volatile", rb == "volatile")
        if previous != key or (previous is not None and t != last + 1):
            append_interval(previous, start, last + 1)
            start = int(t)
        previous, last = key, int(t)
    append_interval(previous, start, last + 1)
    return ranges


def get_transition_ranges(df: pd.DataFrame) -> list[tuple[int, int]]:
    """Read pulse intervals from saved flags; never hard-code the four boundaries."""
    if "is_transition" not in df:
        return []
    a = _stream(df, "A")
    times = a.loc[_booleans(a["is_transition"], "is_transition"), "trial_index"].to_numpy()
    if len(times) == 0:
        return []
    groups = np.split(times, np.flatnonzero(np.diff(times) != 1) + 1)
    return [(int(g[0]), int(g[-1]) + 1) for g in groups]


def add_volatile_block_shading(axis: Axes, environment_df: pd.DataFrame) -> None:
    """Show planned blocks and, when saved, hatched shared-pulse transitions."""
    used: set[str] = set()
    for start, stop, name, label in get_volatile_blocks(environment_df):
        axis.axvspan(start, stop, alpha=0.12, linewidth=0, zorder=0,
                     label=label if label not in used else None)
        used.add(label)
        axis.text((start + stop) / 2, 1.015, name,
                  transform=axis.get_xaxis_transform(), ha="center", va="bottom",
                  fontsize=9, clip_on=False)
    for i, (start, stop) in enumerate(get_transition_ranges(environment_df)):
        axis.axvspan(start, stop, alpha=0.22, hatch="////", fill=False,
                     linewidth=0.6, zorder=0.5,
                     label="Shared-pulse transition" if i == 0 else None)


def _finish(axis: Axes, df: pd.DataFrame, detail: str, ylabel: str) -> None:
    add_volatile_block_shading(axis, df)
    axis.set_title(f"{environment_title(df)} - {detail}", pad=28, fontsize=13)
    axis.set_xlabel("Trial index")
    axis.set_ylabel(ylabel)
    axis.legend(loc="upper left", bbox_to_anchor=(0, -0.18), ncol=2,
                fontsize=9, frameon=False)


def save_figure(fig: Figure, output_path: Path) -> Path:
    """Only generated figures are overwritten; no input data is modified."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.tight_layout(pad=1.2)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


def _two_stream_plot(*, environment_df: pd.DataFrame, column: str, detail: str,
                     ylabel: str, line_label: str, filename: str,
                     figure_dir: Path, linewidth: float = 1.8) -> Path:
    fig, axis = plt.subplots(figsize=(12, 4.8))
    for stream in STREAMS:
        table = _stream(environment_df, stream)
        axis.plot(table["trial_index"], table[column], linewidth=linewidth,
                  label=f"{line_label} {stream}")
    _finish(axis, environment_df, detail, ylabel)
    return save_figure(fig, Path(figure_dir) / filename)


def plot_stream(*, environment_df: pd.DataFrame, stream: str, figure_dir: Path) -> Path:
    if stream not in STREAMS:
        raise ValueError("stream must be A or B.")
    table = _stream(environment_df, stream)
    fig, axis = plt.subplots(figsize=(12, 4.8))
    axis.plot(table["trial_index"], table["observation"], linewidth=0.7,
              label=f"Observation {stream}")
    axis.plot(table["trial_index"], table["latent_mean"], linewidth=2,
              label=f"Latent value {stream}")
    _finish(axis, environment_df, f"stream {stream}", "Value")
    return save_figure(fig, Path(figure_dir) / f"hgf_random_walk_stream_{stream}.png")


def plot_both_latent_means(*, environment_df: pd.DataFrame, figure_dir: Path) -> Path:
    return _two_stream_plot(
        environment_df=environment_df, column="latent_mean", detail="latent values",
        ylabel="Latent value (mean of observations)", line_label="Latent value",
        filename="hgf_random_walk_latent_means_both_streams.png", figure_dir=figure_dir)


def plot_process_quantity(*, environment_df: pd.DataFrame, value_column: str,
                          ylabel: str, title: str, output_filename: str,
                          figure_dir: Path) -> Path:
    # Keep the old public function signature for callers importing this helper.
    detail = {"process_variance": "process variance", "process_sd": "process standard deviation"}.get(
        value_column, title)
    label = {"process_variance": "Process variance", "process_sd": "Process SD"}.get(
        value_column, value_column.replace("_", " "))
    return _two_stream_plot(environment_df=environment_df, column=value_column,
                            detail=detail, ylabel=ylabel, line_label=label,
                            filename=output_filename, figure_dir=figure_dir)


def plot_latent_innovations(*, environment_df: pd.DataFrame, figure_dir: Path) -> Path:
    return _two_stream_plot(
        environment_df=environment_df, column="latent_innovation", detail="latent value innovations",
        ylabel="Value innovation", line_label="Value innovation", linewidth=0.8,
        filename="hgf_random_walk_latent_innovations.png", figure_dir=figure_dir)


def plot_observation_noise(*, environment_df: pd.DataFrame, figure_dir: Path) -> Path:
    return _two_stream_plot(
        environment_df=environment_df, column="observation_noise", detail="observation noise",
        ylabel="Observation minus latent value", line_label="Observation noise", linewidth=0.8,
        filename="hgf_random_walk_observation_noise.png", figure_dir=figure_dir)


def plot_log_process_variance(*, environment_df: pd.DataFrame, figure_dir: Path) -> Path:
    """Plot ln(Q), NOT an invented HGF volatility state (ln(Q)-omega)/kappa."""
    df = environment_df
    fig, axis = plt.subplots(figsize=(12, 4.8))
    no_states = not ({"local_volatility", "shared_state"} & set(df.columns))
    common = no_states and equal_process_schedules(df)
    for stream in (("A",) if common else STREAMS):
        table = _stream(df, stream)
        axis.plot(table["trial_index"], np.log(table["process_variance"]), linewidth=1.8,
                  label="Common log process variance (A = B)" if common
                  else f"Log process variance {stream}")
    detail = "log process-volatility schedule" if no_states else "log process variance"
    _finish(axis, df, detail, "Natural log of process variance, ln(Q)")
    return save_figure(fig, Path(figure_dir) / f"{PREFIX}_log_process_variance.png")


def plot_shared_state(*, environment_df: pd.DataFrame, figure_dir: Path) -> Path:
    """A shared signal is plotted ONCE, after checking A/B copies agree."""
    table = _stream(environment_df, "A")
    fig, axis = plt.subplots(figsize=(12, 4.8))
    axis.plot(table["trial_index"], table["shared_state"], linewidth=0.85,
              label="Realized shared state")
    if "shared_mean" in table:
        axis.plot(table["trial_index"], table["shared_mean"], linestyle="--", linewidth=1.6,
                  label="Prescribed shared-state mean")
    detail = "shared drift state x_c" if "local_volatility" in table else "saved shared state"
    _finish(axis, environment_df, detail, "Shared state")
    return save_figure(fig, Path(figure_dir) / f"{PREFIX}_shared_state.png")


def plot_local_volatility(*, environment_df: pd.DataFrame, figure_dir: Path) -> Path:
    return _two_stream_plot(
        environment_df=environment_df, column="local_volatility", detail="local volatility states",
        ylabel="Local volatility state, v", line_label="Local volatility state",
        filename=f"{PREFIX}_local_volatility.png", figure_dir=figure_dir)


def plot_local_volatility_reference(*, environment_df: pd.DataFrame, figure_dir: Path) -> Path:
    fig, axis = plt.subplots(figsize=(12, 5.1))
    for stream in STREAMS:
        table = _stream(environment_df, stream)
        axis.plot(table["trial_index"], table["local_volatility"], linewidth=1.8,
                  label=f"Actual local volatility {stream}")
    for stream in STREAMS:
        table = _stream(environment_df, stream)
        axis.plot(table["trial_index"], table["deterministic_local_volatility"],
                  linestyle="--", linewidth=1.5,
                  label=f"{stream}: prescribed pulses only, no random terms")
    _finish(axis, environment_df, "local volatility and no-noise reference", "Local volatility state, v")
    return save_figure(fig, Path(figure_dir) / f"{PREFIX}_local_volatility_reference.png")


def random_volatility_component(environment_df: pd.DataFrame) -> pd.DataFrame:
    """Actual minus saved reference includes shared jitter AND local randomness."""
    columns = ("local_volatility", "deterministic_local_volatility")
    if not set(columns).issubset(environment_df.columns):
        raise ValueError("Random component requires both actual local volatility and its saved reference.")
    result = environment_df[["trial_index", "stream"]].copy()
    result["random_component"] = (environment_df[columns[0]].to_numpy()
                                  - environment_df[columns[1]].to_numpy())
    return result


def plot_volatility_random_component(*, environment_df: pd.DataFrame, figure_dir: Path) -> Path:
    component = random_volatility_component(environment_df)
    fig, axis = plt.subplots(figsize=(12, 4.8))
    for stream in STREAMS:
        table = _stream(component, stream)
        axis.plot(table["trial_index"], table["random_component"], linewidth=1.7,
                  label=f"Accumulated random contribution {stream}")
    axis.axhline(0, linestyle=":", linewidth=0.8)
    _finish(axis, environment_df, "accumulated random contribution to local volatility",
            "Actual local volatility minus no-noise reference")
    return save_figure(fig, Path(figure_dir) / f"{PREFIX}_volatility_random_component.png")


def plot_local_volatility_innovations(*, environment_df: pd.DataFrame, figure_dir: Path) -> Path:
    return _two_stream_plot(
        environment_df=environment_df, column="local_volatility_innovation",
        detail="local volatility innovations (not value innovations)",
        ylabel="Local-volatility innovation", line_label="Local-volatility innovation",
        linewidth=0.8, filename=f"{PREFIX}_local_volatility_innovations.png", figure_dir=figure_dir)


def plot_environment(*, environment_df: pd.DataFrame, figure_dir: str | Path,
                     volatility_only: bool = False) -> dict[str, Any]:
    """Keep the seven familiar plots and add only scientifically supported diagnostics."""
    df = validate_environment(environment_df)
    out = Path(figure_dir)
    paths: list[Path] = []
    kwargs = {"environment_df": df, "figure_dir": out}
    if not volatility_only:
        paths.extend(plot_stream(stream=s, **kwargs) for s in STREAMS)
        paths.append(plot_both_latent_means(**kwargs))
        paths.append(plot_process_quantity(
            value_column="process_variance", ylabel="Process variance Q", title="Process variance",
            output_filename="hgf_random_walk_process_variance.png", **kwargs))
        paths.append(plot_process_quantity(
            value_column="process_sd", ylabel="Process SD sqrt(Q)", title="Process SD",
            output_filename="hgf_random_walk_process_sd.png", **kwargs))
        paths.append(plot_latent_innovations(**kwargs))
        paths.append(plot_observation_noise(**kwargs))
    paths.append(plot_log_process_variance(**kwargs))
    if "shared_state" in df:
        paths.append(plot_shared_state(**kwargs))
    if "local_volatility" in df:
        paths.append(plot_local_volatility(**kwargs))
        if "deterministic_local_volatility" in df:
            paths.append(plot_local_volatility_reference(**kwargs))
            paths.append(plot_volatility_random_component(**kwargs))
    if "local_volatility_innovation" in df:
        paths.append(plot_local_volatility_innovations(**kwargs))
    has_states = bool({"local_volatility", "shared_state"} & set(df.columns))
    note = (
        "State plots use saved environmental states, not HGF posterior beliefs. "
        "The no-noise reference is a diagnostic, not a fitted or required trajectory. "
        "Actual minus reference includes shared-signal jitter and local randomness."
        if has_states else
        "No local/shared hidden volatility state was saved by this generator. "
        "The volatility plot shows ln(Q) from the saved process-variance schedule. "
        "Equal A/B schedules are drawn once; this does not reconstruct a hidden parent."
    )
    print(f"\n{note}")
    if has_nominal_regimes(df):
        print("Block shading is NOMINAL. Hatched intervals mark saved shared-pulse transitions.")
    return {
        "environment_description": environment_title(df),
        "n_trials": int(df["trial_index"].nunique()),
        "volatility_only": bool(volatility_only),
        "has_saved_local_volatility": "local_volatility" in df,
        "has_saved_shared_state": "shared_state" in df,
        "nominal_regimes": has_nominal_regimes(df),
        "equal_process_schedules": equal_process_schedules(df),
        "new_log_variance_plot": "natural log of saved Q, not an inferred hidden state",
        "interpretation": note,
        "figures_written_this_run": [p.name for p in paths],
    }
