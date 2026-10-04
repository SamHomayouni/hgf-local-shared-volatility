"""Run the shared-volatility / hybrid pyHGF model."""

from __future__ import annotations

from pathlib import Path
import argparse
import sys


PROJECT_ROOT = next(
    p
    for p in Path(__file__).resolve().parents
    if (p / "src").is_dir()
    and (p / "config").is_dir()
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.models.shared_volatility import (  # noqa: E402
    MODEL_NAME,
    build_model,
    get_node_ids,
)

from src.pipeline.model_runner import (  # noqa: E402
    add_common_runner_arguments,
    execute_model_run,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run shared-volatility / hybrid pyHGF model."
        )
    )

    add_common_runner_arguments(parser)

    parser.add_argument(
        "--shared-value-coupling",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--shared-value-mean",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--shared-value-precision",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--shared-value-tonic-volatility",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--local-vol-mean",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--local-vol-precision",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--local-vol-tonic-volatility",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--local-volatility-coupling",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--local-vol-autoconnection-strength",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--value-tonic-volatility",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--value-autoconnection-strength",
        type=float,
        default=None,
    )

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    model_kwargs = {
        "input_precision": args.input_precision,
        "update_type": args.update_type,
        "max_posterior_precision": (
            args.max_posterior_precision
        ),
    }

    optional_model_args = {
        "shared_value_coupling": (
            args.shared_value_coupling
        ),
        "shared_value_mean": args.shared_value_mean,
        "shared_value_precision": (
            args.shared_value_precision
        ),
        "shared_value_tonic_volatility": (
            args.shared_value_tonic_volatility
        ),
        "local_vol_mean": args.local_vol_mean,
        "local_vol_precision": (
            args.local_vol_precision
        ),
        "local_vol_tonic_volatility": (
            args.local_vol_tonic_volatility
        ),
        "local_volatility_coupling": (
            args.local_volatility_coupling
        ),
        "local_vol_autoconnection_strength": (
            args.local_vol_autoconnection_strength
        ),
        "value_tonic_volatility": (
            args.value_tonic_volatility
        ),
        "value_autoconnection_strength": (
            args.value_autoconnection_strength
        ),
    }

    for parameter_name, parameter_value in (
        optional_model_args.items()
    ):
        if parameter_value is not None:
            model_kwargs[parameter_name] = (
                parameter_value
            )

    execute_model_run(
        environment_path=args.environment,
        output_dir=args.output_dir,
        model_name=MODEL_NAME,
        build_model=build_model,
        get_node_ids=get_node_ids,
        model_kwargs=model_kwargs,
        print_model_kwargs=True,
    )

    print(
        "\nFinished shared-volatility model run."
    )


if __name__ == "__main__":
    main()