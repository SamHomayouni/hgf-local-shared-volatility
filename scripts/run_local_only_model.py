"""Run the local-only pyHGF model."""

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


from src.models.local_only import (  # noqa: E402
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
        description="Run local-only pyHGF model."
    )

    return add_common_runner_arguments(parser)


def main() -> None:
    args = build_arg_parser().parse_args()

    model_kwargs = {
        "input_precision": args.input_precision,
        "update_type": args.update_type,
        "max_posterior_precision": (
            args.max_posterior_precision
        ),
    }

    execute_model_run(
        environment_path=args.environment,
        output_dir=args.output_dir,
        model_name=MODEL_NAME,
        build_model=build_model,
        get_node_ids=get_node_ids,
        model_kwargs=model_kwargs,
    )

    print("\nFinished local-only model run.")


if __name__ == "__main__":
    main()