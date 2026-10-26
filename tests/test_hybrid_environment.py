"""Tests for the Hybrid environment and the supplied model-input runner.

No pyHGF dependency or inference-model fit is needed for these tests.
"""
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.environments import two_stream_hgf_hybrid as h
from src.pipeline.model_runner import load_environment_long, prepare_model_input
from src.utils.paths import ENVIRONMENT_NAMES, get_environment_output_dir


@pytest.fixture(scope="module")
def config():
    return h.parse_environment_config(ROOT / "config" / "hgf_hybrid_weber.yaml")


@pytest.fixture(scope="module")
def df(config):
    return h.simulate_two_stream_environment(config)


def stream_table(df, stream):
    return df.loc[df.stream == stream].reset_index(drop=True)


def test_weber_process_values_and_eq65(config):
    d = config.dynamics
    assert (d.lambda_x, d.lambda_v, d.rho_x, d.rho_v) == (1, 1, 0, 0)
    assert (d.value_tonic_volatility, d.local_volatility_tonic_volatility) == (-2, -3)
    assert (d.shared_value_coupling, d.volatility_coupling) == (2, .5)
    assert (config.stream_a.initial_volatility, config.stream_b.initial_volatility) == (6, 4)
    assert d.observation_log_variance == 1
    assert np.exp(d.observation_log_variance / 2) == pytest.approx(1.6487212707)


def test_twenty_trial_pulses_and_zero_baseline(df):
    a = stream_table(df, "A")
    expected = np.zeros(1000)
    for start, sign in ((200, 1), (400, -1), (600, 1), (800, -1)):
        expected[start:start + 20] = sign * .1
    np.testing.assert_array_equal(a.shared_mean, expected)
    assert a.is_transition.sum() == 80
    assert a.shared_mean.sum() == pytest.approx(0, abs=1e-14)
    assert not np.allclose(a.shared_state, a.shared_mean)
    assert a.shared_noise.std() == pytest.approx(.005, rel=.15)
    assert abs(a.shared_noise.sum()) > 1e-8  # Realized noise is NOT recentered.
    np.testing.assert_allclose(a.shared_state - a.shared_mean, a.shared_noise, atol=1e-16)


def test_initial_update_order_and_recurrences(config, df):
    for stream, initial in (("A", config.stream_a), ("B", config.stream_b)):
        a = stream_table(df, stream)
        previous_v = np.r_[initial.initial_volatility, a.local_volatility.to_numpy()[:-1]]
        np.testing.assert_allclose(a.local_volatility,
            previous_v + 2 * a.shared_state + a.local_volatility_innovation)
        previous_x = np.r_[initial.initial_mean, a.latent_mean.to_numpy()[:-1]]
        np.testing.assert_allclose(a.latent_mean, previous_x + a.latent_innovation)
        np.testing.assert_allclose(a.log_process_variance, -2 + .5 * a.local_volatility)
        np.testing.assert_allclose(a.process_variance, np.exp(a.log_process_variance))
        np.testing.assert_allclose(a.process_sd ** 2, a.process_variance)
        np.testing.assert_allclose(a.observation, a.latent_mean + a.observation_noise)
        assert a.observation_noise.std() == pytest.approx(np.exp(.5), rel=.15)
        assert a.local_volatility_innovation.std() == pytest.approx(np.exp(-1.5), rel=.15)


def test_shared_drive_but_independent_local_innovations(config, df):
    a, b = stream_table(df, "A"), stream_table(df, "B")
    np.testing.assert_array_equal(a.shared_state, b.shared_state)
    for column in ("local_volatility_innovation", "observation_noise"):
        assert not np.array_equal(a[column], b[column])
        assert abs(np.corrcoef(a[column], b[column])[0, 1]) < .15
    previous_difference = np.r_[2, (a.local_volatility - b.local_volatility).to_numpy()[:-1]]
    np.testing.assert_allclose(a.local_volatility - b.local_volatility,
        previous_difference + a.local_volatility_innovation - b.local_volatility_innovation,
        atol=1e-13)
    standardized_a = a.latent_innovation / a.process_sd
    standardized_b = b.latent_innovation / b.process_sd
    assert abs(np.corrcoef(standardized_a, standardized_b)[0, 1]) < .15


def test_deterministic_reference_and_balanced_pulses(config, df):
    for stream, initial in (("A", 6), ("B", 4)):
        a = stream_table(df, stream)
        np.testing.assert_allclose(a.deterministic_local_volatility,
            initial + 2 * np.cumsum(a.shared_mean), atol=1e-12)
        assert a.deterministic_local_volatility.iloc[199] == pytest.approx(initial)
        assert a.deterministic_local_volatility.iloc[200] == pytest.approx(initial + .2)
        assert a.deterministic_local_volatility.iloc[219] == pytest.approx(initial + 4)
        assert a.deterministic_local_volatility.iloc[419] == pytest.approx(initial)
        assert a.deterministic_local_volatility.iloc[619] == pytest.approx(initial + 4)
        assert a.deterministic_local_volatility.iloc[819] == pytest.approx(initial)
    pulses = h.summarize_pulses(df, config)
    np.testing.assert_allclose(pulses.deterministic_delta_local_volatility,
                              [4, 4, -4, -4, 4, 4, -4, -4])
    for _, row in pulses.iterrows():
        a = stream_table(df, row.stream)
        actual_change = (a.local_volatility.iloc[row.stop_index - 1]
                         - a.local_volatility.iloc[row.start_index - 1])
        assert actual_change == pytest.approx(row.realized_delta_local_volatility)


def test_reproducibility_and_observation_noise_separation(config, df):
    pd.testing.assert_frame_equal(df, h.simulate_two_stream_environment(config))
    different = h.simulate_two_stream_environment(replace(config, random_seed=124))
    assert not np.array_equal(df.observation, different.observation)
    noisier = h.simulate_two_stream_environment(replace(config,
        dynamics=replace(config.dynamics, observation_log_variance=3)))
    for column in ("shared_state", "local_volatility", "latent_mean", "process_variance"):
        np.testing.assert_array_equal(df[column], noisier[column])
    np.testing.assert_allclose(noisier.observation_noise, np.e * df.observation_noise)


def test_stream_b_is_not_changed_by_stream_a_initial_value(config, df):
    changed = h.simulate_two_stream_environment(replace(config,
        stream_a=replace(config.stream_a, initial_mean=17)))
    pd.testing.assert_frame_equal(stream_table(df, "B"), stream_table(changed, "B"))
    np.testing.assert_allclose(stream_table(changed, "A").latent_mean,
                               stream_table(df, "A").latent_mean + 17)


def test_saved_output_roundtrip_and_real_model_input_loader(config, df, tmp_path):
    h.save_environment(environment_df=df, config=config, output_dir=tmp_path)
    loaded = load_environment_long(tmp_path / "environment_long.csv")
    prepared = prepare_model_input(loaded)
    assert prepared.input_data.shape == (1000, 2)
    assert prepared.stream_names == ("A", "B")
    for column, stream in enumerate(("A", "B")):
        np.testing.assert_allclose(prepared.input_data[:, column], stream_table(df, stream).observation)
    assert "local_volatility" not in prepared.labels.columns
    assert "shared_state" not in prepared.labels.columns
    wide = pd.read_csv(tmp_path / "environment_wide.csv")
    assert len(wide) == 1000
    assert {"local_volatility_A", "local_volatility_B", "shared_state", "is_transition"} <= set(wide)
    summary = pd.read_csv(tmp_path / "environment_summary.csv")
    assert len(summary) == 10
    assert summary.n_transition_trials.sum() == 160  # 80 per stream.
    assert summary.regime_is_nominal.all()
    assert not loaded.changepoint.any()
    meta = json.loads((tmp_path / "environment_metadata.json").read_text())
    assert meta["derived_parameters"]["observation_noise_variance"] == pytest.approx(np.e)
    assert meta["derived_parameters"]["corresponding_input_precision"] == pytest.approx(np.exp(-1))
    restored_config = h.parse_environment_config(tmp_path / "environment_config_resolved.yaml")
    assert restored_config == config
    pd.testing.assert_frame_equal(df, h.simulate_two_stream_environment(restored_config))
    with pytest.raises(FileExistsError):
        h.save_environment(environment_df=df, config=config, output_dir=tmp_path)


def test_new_name_and_existing_directories_are_protected(config, df):
    assert ENVIRONMENT_NAMES == ("local_A", "local_B", "global", "hybrid")
    assert get_environment_output_dir("hybrid").name == "hybrid"
    assert get_environment_output_dir().name == "local_A"
    for old in ("local_A", "local_B", "global"):
        with pytest.raises(ValueError, match="Do not save Hybrid"):
            h.save_environment(environment_df=df, config=config,
                               output_dir=get_environment_output_dir(old))


@pytest.mark.parametrize("bad_parameter", [
    {"lambda_v": .7}, {"lambda_x": .95}, {"rho_v": .01},
    {"volatility_coupling": -.5}, {"observation_log_variance": float("nan")},
    {"observation_log_variance": 1500},
])
def test_invalid_dynamics_rejected(config, bad_parameter):
    with pytest.raises(ValueError):
        h.validate_environment_config(replace(config,
            dynamics=replace(config.dynamics, **bad_parameter)))


def test_invalid_pulses_and_indices_rejected(config):
    with pytest.raises(ValueError):
        h.validate_environment_config(replace(config, random_seed=-1))
    bad = replace(config.shared_signal.pulses[1], start_index=210)
    with pytest.raises(ValueError):
        h.validate_environment_config(replace(config, shared_signal=replace(config.shared_signal,
            pulses=(config.shared_signal.pulses[0], bad))))
    with pytest.raises(ValueError):
        h.validate_environment_config(replace(config,
            shared_signal=replace(config.shared_signal, noise_sd=-.1)))


def test_yaml_fractional_index_is_not_silently_truncated(tmp_path):
    raw = yaml.safe_load((ROOT / "config/hgf_hybrid_weber.yaml").read_text())
    raw["blocks"][0]["start_index"] = .5
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        h.parse_environment_config(path)


def test_numerical_failures_are_not_clipped_or_retried(config, tmp_path):
    extreme = replace(config, dynamics=replace(config.dynamics, value_tonic_volatility=1500))
    with pytest.raises(FloatingPointError, match="No clipping or retry"):
        h.simulate_two_stream_environment(extreme)
    n_finite, n_fail = h.check_environment_seeds(extreme, range(2), tmp_path / "failed_seeds")
    assert (n_finite, n_fail) == (0, 2)
    report = json.loads((tmp_path / "failed_seeds/summary.json").read_text())
    assert report["seeds"] == [0, 1]
    failed = pd.read_csv(tmp_path / "failed_seeds/seed_diagnostics.csv")
    assert (failed.status == "numerical_failure").all()


def test_multiseed_summary_keeps_every_seed(config, tmp_path):
    n_finite, n_fail = h.check_environment_seeds(config, range(3), tmp_path / "checks")
    assert (n_finite, n_fail) == (3, 0)
    summary = pd.read_csv(tmp_path / "checks/seed_diagnostics.csv")
    assert list(summary.random_seed) == [0, 0, 1, 1, 2, 2]
    assert len(pd.read_csv(tmp_path / "checks/block_summaries.csv")) == 30
