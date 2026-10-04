"""Tests for the two-stream Gaussian random-walk environments."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from src.environments.two_stream_hgf_random_walk import (
    parse_environment_config,
    run_from_config,
    validate_environment_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_DIR = PROJECT_ROOT / "config"

CONFIG_PATHS = {
    "local_A": (
        CONFIG_DIR
        / "hgf_random_walk_local_A.yaml"
    ),
    "local_B": (
        CONFIG_DIR
        / "hgf_random_walk_local_B.yaml"
    ),
    "global": (
        CONFIG_DIR
        / "hgf_random_walk_global.yaml"
    ),
}


BLOCK_ORDER = [
    "S1",
    "V1",
    "S2",
    "V2",
    "S3",
]


BLOCK_BOUNDS = {
    "S1": (0, 200),
    "V1": (200, 400),
    "S2": (400, 600),
    "V2": (600, 800),
    "S3": (800, 1000),
}


EXPECTED_VOLATILE_REGIMES = {
    "local_A": {
        "A": "volatile",
        "B": "stable",
    },
    "local_B": {
        "A": "stable",
        "B": "volatile",
    },
    "global": {
        "A": "volatile",
        "B": "volatile",
    },
}


@pytest.mark.parametrize(
    "environment_name",
    CONFIG_PATHS,
)
def test_config_core_parameters(
    environment_name: str,
) -> None:
    """All current environments share the same basic dynamics."""

    config = parse_environment_config(
        CONFIG_PATHS[environment_name]
    )

    assert config.random_seed == 123
    assert config.n_trials == 1000

    assert config.latent_dynamics.lambda_x == pytest.approx(
        1.0
    )

    assert config.latent_dynamics.rho_x == pytest.approx(
        0.0
    )

    assert config.latent_dynamics.q_low == pytest.approx(
        0.0004
    )

    assert config.latent_dynamics.q_high == pytest.approx(
        0.01
    )

    assert (
        config.latent_dynamics.observation_noise_sd
        == pytest.approx(0.10)
    )

    assert config.stream_a.initial_mean == pytest.approx(
        0.0
    )

    assert config.stream_b.initial_mean == pytest.approx(
        0.0
    )


@pytest.mark.parametrize(
    "environment_name",
    CONFIG_PATHS,
)
def test_config_block_structure(
    environment_name: str,
) -> None:
    """Protect the five-block 1000-trial experimental design."""

    config = parse_environment_config(
        CONFIG_PATHS[environment_name]
    )

    assert [
        block.name
        for block in config.blocks
    ] == BLOCK_ORDER

    for block in config.blocks:
        expected_start, expected_stop = (
            BLOCK_BOUNDS[block.name]
        )

        assert block.start_index == expected_start
        assert block.stop_index == expected_stop

        assert (
            block.stop_index - block.start_index
        ) == 200


@pytest.mark.parametrize(
    "environment_name",
    CONFIG_PATHS,
)
def test_generated_environment_shape_and_trial_order(
    environment_name: str,
) -> None:
    """Each of 1000 trials must contain exactly A and B."""

    environment = run_from_config(
        config_path=CONFIG_PATHS[environment_name]
    )

    assert len(environment) == 2000

    assert set(environment["stream"]) == {
        "A",
        "B",
    }

    for stream in ("A", "B"):
        stream_df = environment[
            environment["stream"] == stream
        ].reset_index(drop=True)

        assert len(stream_df) == 1000

        np.testing.assert_array_equal(
            stream_df["trial_index"].to_numpy(),
            np.arange(1000),
        )

        np.testing.assert_array_equal(
            stream_df["trial"].to_numpy(),
            np.arange(1, 1001),
        )

        assert (
            stream_df["block"]
            .drop_duplicates()
            .tolist()
            == BLOCK_ORDER
        )


@pytest.mark.parametrize(
    "environment_name",
    CONFIG_PATHS,
)
def test_generated_regime_schedule(
    environment_name: str,
) -> None:
    """Check which stream is objectively volatile in V1/V2."""

    environment = run_from_config(
        config_path=CONFIG_PATHS[environment_name]
    )

    volatile_expectation = (
        EXPECTED_VOLATILE_REGIMES[
            environment_name
        ]
    )

    for block in ("S1", "S2", "S3"):
        for stream in ("A", "B"):
            rows = environment[
                (environment["block"] == block)
                & (environment["stream"] == stream)
            ]

            assert len(rows) == 200

            assert set(rows["regime"]) == {
                "stable"
            }

    for block in ("V1", "V2"):
        for stream in ("A", "B"):
            rows = environment[
                (environment["block"] == block)
                & (environment["stream"] == stream)
            ]

            assert len(rows) == 200

            assert set(rows["regime"]) == {
                volatile_expectation[stream]
            }


@pytest.mark.parametrize(
    "environment_name",
    CONFIG_PATHS,
)
def test_process_variance_matches_regime(
    environment_name: str,
) -> None:
    """Q must be q_low in stable rows and q_high in volatile rows."""

    config = parse_environment_config(
        CONFIG_PATHS[environment_name]
    )

    environment = run_from_config(
        config_path=CONFIG_PATHS[environment_name]
    )

    expected_variance = (
        environment["regime"]
        .map(
            {
                "stable": (
                    config.latent_dynamics.q_low
                ),
                "volatile": (
                    config.latent_dynamics.q_high
                ),
            }
        )
        .to_numpy(dtype=float)
    )

    np.testing.assert_allclose(
        environment["process_variance"],
        expected_variance,
    )

    np.testing.assert_allclose(
        environment["process_sd"],
        np.sqrt(expected_variance),
    )


@pytest.mark.parametrize(
    "environment_name",
    CONFIG_PATHS,
)
def test_random_walk_environment_has_no_changepoints(
    environment_name: str,
) -> None:
    """The Gaussian random-walk environments must not contain changepoints."""

    environment = run_from_config(
        config_path=CONFIG_PATHS[environment_name]
    )

    assert not environment["changepoint"].any()


@pytest.mark.parametrize(
    "environment_name",
    CONFIG_PATHS,
)
def test_fixed_seed_is_exactly_reproducible(
    environment_name: str,
) -> None:
    """The same configuration must reproduce the same environment."""

    first = run_from_config(
        config_path=CONFIG_PATHS[environment_name]
    )

    second = run_from_config(
        config_path=CONFIG_PATHS[environment_name]
    )

    pd.testing.assert_frame_equal(
        first,
        second,
        check_exact=True,
    )


def test_global_environment_shares_variance_schedule_not_innovations(
) -> None:
    """Global volatility must not make A and B use the same random shocks."""

    environment = run_from_config(
        config_path=CONFIG_PATHS["global"]
    )

    innovation_table = environment.pivot(
        index="trial_index",
        columns="stream",
        values="latent_innovation",
    ).sort_index()

    variance_table = environment.pivot(
        index="trial_index",
        columns="stream",
        values="process_variance",
    ).sort_index()

    # Same volatility schedule.
    np.testing.assert_allclose(
        variance_table["A"],
        variance_table["B"],
    )

    # Different innovation sequences.
    assert not np.array_equal(
        innovation_table["A"].to_numpy(),
        innovation_table["B"].to_numpy(),
    )


def test_validation_rejects_inference_style_lambda() -> None:
    """The generative environment must remain a pure random walk."""

    config = parse_environment_config(
        CONFIG_PATHS["local_A"]
    )

    altered_dynamics = replace(
        config.latent_dynamics,
        lambda_x=0.95,
    )

    altered_config = replace(
        config,
        latent_dynamics=altered_dynamics,
    )

    with pytest.raises(
        ValueError,
        match="lambda_x must be 1.0",
    ):
        validate_environment_config(
            altered_config
        )


def test_saved_environment_metadata_matches_configuration(
    tmp_path: Path,
) -> None:
    """Saved data and metadata must describe the same environment."""

    config_path = CONFIG_PATHS["local_A"]

    config = parse_environment_config(
        config_path
    )

    output_dir = tmp_path / "environment"

    generated = run_from_config(
        config_path=config_path,
        output_dir=output_dir,
    )

    expected_files = {
        "environment_long.csv",
        "environment_wide.csv",
        "environment_summary.csv",
        "environment_metadata.json",
    }

    assert {
        path.name
        for path in output_dir.iterdir()
        if path.is_file()
    } == expected_files

    saved_long = pd.read_csv(
        output_dir / "environment_long.csv"
    )

    saved_wide = pd.read_csv(
        output_dir / "environment_wide.csv"
    )

    summary = pd.read_csv(
        output_dir / "environment_summary.csv"
    )

    with (
        output_dir
        / "environment_metadata.json"
    ).open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    assert len(saved_long) == len(generated)
    assert len(saved_long) == 2000
    assert len(saved_wide) == 1000

    assert metadata["name"] == config.name
    assert (
        metadata["random_seed"]
        == config.random_seed
    )
    assert metadata["n_trials"] == 1000

    dynamics_metadata = metadata[
        "latent_dynamics"
    ]

    assert dynamics_metadata[
        "lambda_x"
    ] == pytest.approx(1.0)

    assert dynamics_metadata[
        "rho_x"
    ] == pytest.approx(0.0)

    assert dynamics_metadata[
        "q_low"
    ] == pytest.approx(0.0004)

    assert dynamics_metadata[
        "q_high"
    ] == pytest.approx(0.01)

    assert dynamics_metadata[
        "observation_noise_sd"
    ] == pytest.approx(0.10)

    assert metadata[
        "columns_long"
    ] == list(generated.columns)

    assert len(summary) == 10

    assert (
        summary["n_trials"] == 200
    ).all()

    assert (
        summary["n_changepoints"] == 0
    ).all()

    expected_summary_variance = (
        summary["regime"]
        .map(
            {
                "stable": 0.0004,
                "volatile": 0.01,
            }
        )
        .to_numpy(dtype=float)
    )

    np.testing.assert_allclose(
        summary["mean_process_variance"],
        expected_summary_variance,
    )