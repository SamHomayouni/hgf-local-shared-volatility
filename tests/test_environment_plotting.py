"""Read-only plotting and representation tests; no pyHGF fitting required."""
from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.analysis import plot_environments as plotting


@pytest.fixture
def local_data() -> pd.DataFrame:
    n = 12
    records = []
    for t in range(n):
        block = "S1" if t < 4 else ("V1" if t < 8 else "S2")
        for s in ("A", "B"):
            volatile = s == "A" and 4 <= t < 8
            q = 1.0 if volatile else 0.25
            records.append({
                "trial_index": t, "trial": t + 1, "stream": s, "block": block,
                "regime": "volatile" if volatile else "stable", "changepoint": False,
                "is_a_volatile_block": 4 <= t < 8, "process_variance": q,
                "process_sd": np.sqrt(q), "latent_mean": float(t),
                "observation": float(t) + 0.1, "latent_innovation": 1.0,
                "observation_noise": 0.1,
            })
    return pd.DataFrame(records)


@pytest.fixture
def global_data(local_data: pd.DataFrame) -> pd.DataFrame:
    df = local_data.copy()
    for t in df["trial_index"].unique():
        a = df[(df["trial_index"] == t) & (df["stream"] == "A")].iloc[0]
        mask = (df["trial_index"] == t) & (df["stream"] == "B")
        df.loc[mask, ["regime", "process_variance", "process_sd"]] = [
            a["regime"], a["process_variance"], a["process_sd"]]
    return df


@pytest.fixture
def hybrid_data(global_data: pd.DataFrame) -> pd.DataFrame:
    df = global_data.copy()
    n = df["trial_index"].nunique()
    m = np.zeros(n)
    m[4:6], m[8:10] = 0.1, -0.1
    jitter = 0.002 * np.sin(np.arange(n))
    c = m + jitter
    for s, initial, sign in (("A", 6.0, 1.0), ("B", 4.0, -1.0)):
        mask = df["stream"] == s
        innovation = sign * 0.03 * np.cos(np.arange(n))
        reference = initial + 2 * np.cumsum(m)
        v = initial + np.cumsum(2 * c + innovation)
        df.loc[mask, "shared_mean"] = m
        df.loc[mask, "shared_noise"] = jitter
        df.loc[mask, "shared_state"] = c
        df.loc[mask, "shared_volatility_drift"] = 2 * c
        df.loc[mask, "local_volatility_innovation"] = innovation
        df.loc[mask, "local_volatility"] = v
        df.loc[mask, "deterministic_local_volatility"] = reference
        df.loc[mask, "log_process_variance"] = -2 + 0.5 * v
        df.loc[mask, "deterministic_log_process_variance"] = -2 + 0.5 * reference
        df.loc[mask, "process_variance"] = np.exp(-2 + 0.5 * v)
        df.loc[mask, "process_sd"] = np.exp((-2 + 0.5 * v) / 2)
    df["is_transition"] = df["trial_index"].isin([4, 5, 8, 9])
    df["regime_is_nominal"] = True
    df["environment_type"] = "hybrid"
    return df


@pytest.fixture
def capture(monkeypatch: pytest.MonkeyPatch):
    figures = {}
    def save(fig, output_path):
        figures[Path(output_path).name] = fig
        plt.close(fig)
        return Path(output_path)
    monkeypatch.setattr(plotting, "save_figure", save)
    return figures


def test_sort_and_read_only(local_data):
    shuffled = local_data.sample(frac=1, random_state=5)
    before = shuffled.copy(deep=True)
    validated = plotting.validate_environment(shuffled)
    pd.testing.assert_frame_equal(shuffled, before)
    pd.testing.assert_frame_equal(validated, local_data)


@pytest.mark.parametrize("field,bad", [
    ("process_variance", 0.0), ("process_variance", -1.0),
    ("latent_mean", np.nan), ("observation_noise", np.inf),
    ("trial_index", 1.5), ("trial_index", -1), ("process_sd", 999.0),
])
def test_reject_bad_numbers(local_data, field, bad):
    df = local_data.copy()
    if field == "trial_index":
        df[field] = df[field].astype(float)
    df.loc[0, field] = bad
    with pytest.raises(ValueError):
        plotting.validate_environment(df)


def test_reject_duplicates(local_data):
    with pytest.raises(ValueError, match="Duplicate"):
        plotting.validate_environment(pd.concat([local_data, local_data.iloc[[0]]]))


def test_reject_missing_stream_trial(local_data):
    with pytest.raises(ValueError, match="same trial indices"):
        plotting.validate_environment(local_data.iloc[1:])


def test_reject_unknown_stream(local_data):
    local_data.loc[0, "stream"] = "C"
    with pytest.raises(ValueError, match="exactly streams"):
        plotting.validate_environment(local_data)


def test_reject_conflicting_shared_signal(hybrid_data):
    hybrid_data.loc[1, "shared_state"] += 0.01
    with pytest.raises(ValueError, match="shared_state.*disagrees"):
        plotting.validate_environment(hybrid_data)


def test_reject_inconsistent_log_q(hybrid_data):
    hybrid_data.loc[0, "log_process_variance"] += 1
    with pytest.raises(ValueError, match="log_process_variance is inconsistent"):
        plotting.validate_environment(hybrid_data)


def test_reference_requires_actual_state(hybrid_data):
    with pytest.raises(ValueError, match="reference.*without local_volatility"):
        plotting.validate_environment(hybrid_data.drop(columns="local_volatility"))


def test_boolean_strings(local_data):
    local_data["regime_is_nominal"] = "False"
    local_data["is_transition"] = "False"
    df = plotting.validate_environment(local_data)
    assert plotting.has_nominal_regimes(df) is False
    assert plotting.get_transition_ranges(df) == []


def test_shared_boolean_mismatch(hybrid_data):
    hybrid_data.loc[1, "is_transition"] = True
    with pytest.raises(ValueError, match="is_transition.*disagrees"):
        plotting.validate_environment(hybrid_data)


def test_nominal_labels_and_transitions(hybrid_data):
    df = plotting.validate_environment(hybrid_data)
    assert plotting.get_volatile_blocks(df) == [(4, 8, "V1", "Nominal volatile block (A + B)")]
    assert plotting.get_transition_ranges(df) == [(4, 6), (8, 10)]


def test_local_block_labels(local_data):
    assert plotting.get_volatile_blocks(local_data) == [(4, 8, "V1", "A volatile")]


def test_log_plot_local_does_not_invent_state(local_data, capture, tmp_path):
    report = plotting.plot_environment(environment_df=local_data, figure_dir=tmp_path)
    assert len(report["figures_written_this_run"]) == 8
    assert not report["has_saved_local_volatility"]
    assert not report["has_saved_shared_state"]
    ax = capture["hgf_environment_log_process_variance.png"].axes[0]
    assert len(ax.lines) == 2
    for s, line in zip(("A", "B"), ax.lines):
        np.testing.assert_array_equal(line.get_ydata(), np.log(local_data.loc[local_data.stream == s, "process_variance"]))
    assert "local_volatility" not in local_data
    assert not any("local_volatility" in name for name in capture)


def test_global_log_schedule_is_single_curve(global_data, capture, tmp_path):
    report = plotting.plot_environment(environment_df=global_data, figure_dir=tmp_path)
    assert len(report["figures_written_this_run"]) == 8
    assert report["environment_description"] == "Global environment"
    ax = capture["hgf_environment_log_process_variance.png"].axes[0]
    assert len(ax.lines) == 1
    assert "A = B" in ax.lines[0].get_label()


def test_unequal_schedules_never_collapsed(global_data, capture, tmp_path):
    global_data.loc[1, "process_variance"] = 0.5
    global_data.loc[1, "process_sd"] = np.sqrt(0.5)
    plotting.plot_environment(environment_df=global_data, figure_dir=tmp_path, volatility_only=True)
    assert len(capture["hgf_environment_log_process_variance.png"].axes[0].lines) == 2


def test_hybrid_uses_actual_saved_states(hybrid_data, capture, tmp_path):
    before = hybrid_data.copy(deep=True)
    report = plotting.plot_environment(environment_df=hybrid_data, figure_dir=tmp_path)
    assert len(report["figures_written_this_run"]) == 13
    ax = capture["hgf_environment_local_volatility.png"].axes[0]
    for s, line in zip(("A", "B"), ax.lines):
        np.testing.assert_array_equal(line.get_ydata(), hybrid_data.loc[hybrid_data.stream == s, "local_volatility"])
    shared = capture["hgf_environment_shared_state.png"].axes[0]
    assert len(shared.lines) == 2
    np.testing.assert_array_equal(shared.lines[0].get_ydata(), hybrid_data.loc[hybrid_data.stream == "A", "shared_state"])
    pd.testing.assert_frame_equal(hybrid_data, before)


def test_residual_is_saved_actual_minus_reference(hybrid_data):
    component = plotting.random_volatility_component(hybrid_data)
    np.testing.assert_array_equal(component["random_component"],
                                  hybrid_data.local_volatility - hybrid_data.deterministic_local_volatility)


def test_actual_states_without_references_still_plot(hybrid_data, capture, tmp_path):
    report = plotting.plot_environment(
        environment_df=hybrid_data.drop(columns="deterministic_local_volatility"),
        figure_dir=tmp_path, volatility_only=True)
    assert len(report["figures_written_this_run"]) == 4
    assert not any("reference" in name or "random_component" in name for name in capture)


@pytest.mark.parametrize("kind", ["local", "global", "hybrid"])
def test_cli_real_pngs_and_unchanged_csv(kind, request, tmp_path):
    df = request.getfixturevalue(f"{kind}_data")
    path = tmp_path / "environment_long.csv"
    df.to_csv(path, index=False)
    before = path.read_bytes()
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts" / "plot_environment_hgf_random_walk.py"),
        "--environment", str(path), "--volatility-only",
    ], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert path.read_bytes() == before
    out = tmp_path / "figures" / "environment"
    manifest = json.loads((out / "environment_plot_manifest.json").read_text())
    assert len(manifest["figures_written_this_run"]) == (6 if kind == "hybrid" else 1)
    for name in manifest["figures_written_this_run"]:
        assert (out / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert not list(tmp_path.glob("*metadata*"))
