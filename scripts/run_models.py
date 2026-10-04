"""Run one or more current pyHGF models on a generated environment.

Examples
--------
Run all three models on the default local_A environment:

    python scripts/run_models.py

Run all models on the global environment:

    python scripts/run_models.py --environment-name global

Run only the local-only and global-only models:

    python scripts/run_models.py \
        --models local_only global_only

Run models on a custom environment file:

    python scripts/run_models.py \
        --environment /path/to/environment_long.csv \
        --output-dir /path/to/model_runs

For model-specific parameter overrides, use the dedicated runner scripts.
"""

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


from src.models.global_only import (  # noqa: E402
    MODEL_NAME as GLOBAL_MODEL_NAME,
    build_model as build_global_model,
    get_node_ids as get_global_node_ids,
)

from src.models.local_only import (  # noqa: E402
    MODEL_NAME as LOCAL_MODEL_NAME,
    build_model as build_local_model,
    get_node_ids as get_local_node_ids,
)

from src.models.shared_volatility import (  # noqa: E402
    MODEL_NAME as SHARED_MODEL_NAME,
    build_model as build_shared_model,
    get_node_ids as get_shared_node_ids,
)

from src.pipeline.model_runner import (  # noqa: E402
    execute_model_run,
)

from src.utils.paths import (  # noqa: E402
    DEFAULT_ENVIRONMENT_NAME,
    ENVIRONMENT_NAMES,
    get_environment_output_dir,
)


MODEL_SPECS = {
    "local_only": {
        "model_name": LOCAL_MODEL_NAME,
        "build_model": build_local_model,
        "get_node_ids": get_local_node_ids,
    },
    "global_only": {
        "model_name": GLOBAL_MODEL_NAME,
        "build_model": build_global_model,
        "get_node_ids": get_global_node_ids,
    },
    "shared_volatility": {
        "model_name": SHARED_MODEL_NAME,
        "build_model": build_shared_model,
        "get_node_ids": get_shared_node_ids,
    },
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one or more current pyHGF models."
        )
    )

    parser.add_argument(
        "--environment-name",
        type=str,
        choices=ENVIRONMENT_NAMES,
        default=DEFAULT_ENVIRONMENT_NAME,
        help=(
            "Named generated environment to use when "
            "--environment is not supplied."
        ),
    )

    parser.add_argument(
        "--environment",
        type=str,
        default=None,
        help=(
            "Optional explicit path to environment_long.csv. "
            "Overrides --environment-name."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Optional model output directory. By default, "
            "uses the model_runs directory belonging to the "
            "selected environment."
        ),
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_SPECS),
        default=list(MODEL_SPECS),
        help=(
            "Models to run. Defaults to all three."
        ),
    )

    parser.add_argument(
        "--input-precision",
        type=float,
        default=100.0,
    )

    parser.add_argument(
        "--update-type",
        type=str,
        default="eHGF",
        choices=[
            "eHGF",
            "standard",
            "unbounded",
        ],
    )

    parser.add_argument(
        "--max-posterior-precision",
        type=float,
        default=1e10,
    )

    return parser


def resolve_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    """Resolve environment input and model output paths."""

    environment_dir = get_environment_output_dir(
        args.environment_name
    )

    if args.environment is None:
        environment_path = (
            environment_dir / "environment_long.csv"
        )
    else:
        environment_path = Path(args.environment)

    if args.output_dir is None:
        output_dir = environment_dir / "model_runs"
    else:
        output_dir = Path(args.output_dir)

    return environment_path, output_dir


def main() -> None:
    args = build_arg_parser().parse_args()

    environment_path, output_dir = resolve_paths(args)

    common_model_kwargs = {
        "input_precision": args.input_precision,
        "update_type": args.update_type,
        "max_posterior_precision": (
            args.max_posterior_precision
        ),
    }

    print(f"Environment: {environment_path}")
    print(f"Output dir:  {output_dir}")
    print(
        "Models:      "
        + ", ".join(args.models)
    )

    for model_key in args.models:
        specification = MODEL_SPECS[model_key]

        execute_model_run(
            environment_path=environment_path,
            output_dir=output_dir,
            model_name=specification["model_name"],
            build_model=specification["build_model"],
            get_node_ids=specification["get_node_ids"],
            model_kwargs=common_model_kwargs.copy(),
            print_model_kwargs=(
                model_key == "shared_volatility"
            ),
        )

    print("\nFinished requested model runs.")


if __name__ == "__main__":
    main()