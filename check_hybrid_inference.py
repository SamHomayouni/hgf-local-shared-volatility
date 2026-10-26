"""Diagnose the Hybrid-input failure using your installed pyHGF and model builders.

This is a diagnostic, not a replacement model runner. It does not regenerate
observations, edit model defaults, write trajectories, or overwrite figures.
All tests use exactly the same input matrix and a fresh Network.

Save anywhere inside the repository, for example:
    experiments/diagnostics/check_hybrid_inference.py

Run from the repository root:
    python experiments/diagnostics/check_hybrid_inference.py \
        --environment outputs/hgf_walk_outputs/hybrid/environment_long.csv \
        --report outputs/hgf_walk_outputs/hybrid/diagnostics/inference_scale_check.json

A successful state check means finite means and finite, positive precisions on
all trials. It does not establish accuracy, parameter recovery, or valid CTI.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
from importlib.metadata import PackageNotFoundError, version
import inspect
import json
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np
import pandas as pd


MODEL_NAMES = ('local_only', 'global_only', 'shared_volatility')
TESTS = (
    ('unchanged', {}),
    ('only_value_tonic_volatility_minus2', {'value_tonic_volatility': -2.0}),
    ('only_value_autoconnection_1', {'value_autoconnection_strength': 1.0}),
)
CORE_FIELDS = ('expected_mean', 'expected_precision', 'mean', 'precision')


def find_root() -> Path:
    """Find the repository without relying on a particular script subdirectory."""
    starts = (Path.cwd().resolve(), Path(__file__).resolve().parent)
    for start in starts:
        for candidate in (start, *start.parents):
            if (candidate / 'src' / 'models').is_dir() and (candidate / 'config').is_dir():
                return candidate
    raise FileNotFoundError('Run this diagnostic inside the repository containing src/models and config.')


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(x) for x in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    try:
        array = np.asarray(value)
        if array.ndim == 0:
            item = array.item()
            return jsonable(item) if not isinstance(item, type(value)) else str(item)
        return jsonable(array.tolist())
    except (TypeError, ValueError):
        return repr(value)


def load_observations(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    needed = {'trial_index', 'stream', 'observation'}
    if not needed.issubset(df.columns):
        raise ValueError(f'Missing environment columns: {sorted(needed - set(df.columns))}')
    if df.duplicated(['trial_index', 'stream']).any():
        raise ValueError('Duplicate trial/stream rows in environment.')
    wide = df.pivot(index='trial_index', columns='stream', values='observation').sort_index()
    if set(wide.columns) != {'A', 'B'}:
        raise ValueError('Exactly streams A and B are required.')
    if not np.array_equal(wide.index.to_numpy(), np.arange(len(wide))):
        raise ValueError('Trial indices must be consecutive integers starting at zero.')
    values = wide[['A', 'B']].to_numpy(dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Input matrix is empty or contains NaN/Inf/missing stream observations.')
    return values


def inspect_states(model: Any, node_ids: dict[str, int], n_trials: int) -> dict[str, Any]:
    failures = []
    for node_id in sorted(set(node_ids.values())):
        node = model.node_trajectories[node_id]
        for field in CORE_FIELDS:
            if field not in node:
                raise KeyError(f'Node {node_id} has no {field!r} trajectory.')
            values = np.asarray(node[field], dtype=float).reshape(-1)
            if len(values) != n_trials:
                raise ValueError(f'Node {node_id}/{field}: expected {n_trials} values, got {len(values)}.')
            nonfinite = ~np.isfinite(values)
            bad = nonfinite.copy()
            if 'precision' in field:
                bad |= values <= 0
            indices = np.flatnonzero(bad)
            if len(indices):
                t = int(indices[0])
                failures.append({
                    'column': f'x_{node_id}_{field}',
                    'first_trial_index': t, 'first_observation_number': t + 1,
                    'reason': 'nonfinite' if nonfinite[t] else 'nonpositive precision',
                    'n_invalid': int(bad.sum()),
                })
    failures.sort(key=lambda r: (r['first_trial_index'], r['column']))
    result: dict[str, Any] = {'states_valid': not failures, 'failures': failures}
    if not failures:
        # For these builders, the direct continuous value update has unit value
        # coupling and weight input precision / VALUE POSTERIOR precision.
        result['direct_value_update_weights'] = {}
        for stream in ('A', 'B'):
            input_node = node_ids[f'input_{stream}']
            value_node = node_ids[f'value_{stream}']
            numerator = np.asarray(model.node_trajectories[input_node]['expected_precision'], float)
            denominator = np.asarray(model.node_trajectories[value_node]['precision'], float)
            weights = numerator / denominator
            result['direct_value_update_weights'][stream] = {
                'definition': f'x_{input_node}_expected_precision / x_{value_node}_precision',
                'min': float(weights.min()), 'median': float(np.median(weights)),
                'max': float(weights.max()),
            }
    return result


def capture_installed_sources() -> dict[str, str]:
    """Capture exact installed implementations, not potentially newer web docs."""
    result = {}
    modules = (
        'pyhgf.updates.posterior.continuous.posterior_update_precision_continuous_node',
        'pyhgf.updates.posterior.continuous.continuous_node_posterior_update_ehgf',
        'pyhgf.updates.prediction.continuous',
    )
    for name in modules:
        try:
            result[name] = inspect.getsource(importlib.import_module(name))
        except (ImportError, OSError, TypeError) as exc:
            result[name] = f'Source unavailable: {type(exc).__name__}: {exc}'
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--environment', type=Path, default=Path('outputs/hgf_walk_outputs/hybrid/environment_long.csv'))
    parser.add_argument('--report', type=Path, help='Optional JSON report; refuses to overwrite an existing file.')
    args = parser.parse_args()
    root = find_root()
    sys.path.insert(0, str(root))
    path = args.environment if args.environment.is_absolute() else root / args.environment
    report_path = None if args.report is None else (args.report if args.report.is_absolute() else root / args.report)
    if report_path is not None and report_path.exists():
        raise FileExistsError(f'Report already exists: {report_path}. Rename it or choose a new report filename.')
    u = load_observations(path)
    versions = {'python': platform.python_version()}
    for package in ('pyhgf', 'jax', 'jaxlib', 'numpy', 'pandas'):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = 'not installed'
    print('Versions:', versions)
    print('Input:', path)
    print('Input shape:', u.shape, '(A first, B second)')
    if versions['pyhgf'] != '0.2.12':
        print('WARNING: This is not pyhgf 0.2.12; report results as version-specific.')
    report = {
        'versions': versions, 'environment': str(path),
        'environment_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'input_shape': list(u.shape), 'records': [],
        'scope': 'Same input for all tests; original model builders; no fitting; no saved runs overwritten.',
    }
    for model_name in MODEL_NAMES:
        module = importlib.import_module(f'src.models.{model_name}')
        model_source = Path(inspect.getfile(module))
        for test_name, overrides in TESTS:
            kwargs = {'input_precision': 100.0, 'update_type': 'eHGF', 'max_posterior_precision': 1e10, **overrides}
            record = {'model': model_name, 'test': test_name, 'kwargs': kwargs,
                      'model_source_sha256': hashlib.sha256(model_source.read_bytes()).hexdigest()}
            try:
                model = module.build_model(**kwargs)
                record['initial_attributes'] = jsonable(model.attributes)
                returned = model.input_data(input_data=u.copy())
                if returned is not None:
                    model = returned
                record.update(inspect_states(model, module.get_node_ids(), len(u)))
                if record['states_valid']:
                    message = f'PASS: finite means, positive finite precisions for all {len(u)} observations'
                else:
                    first = record['failures'][0]
                    message = f"FAIL: observation {first['first_observation_number']}, {first['column']} ({first['reason']})"
            except Exception as exc:
                record.update({'states_valid': False, 'error': f'{type(exc).__name__}: {exc}'})
                message = f"ERROR: {record['error']}"
            report['records'].append(record)
            print(f'{model_name:18s} {test_name:38s} {message}', flush=True)
    if report_path is not None:
        report['installed_update_sources'] = capture_installed_sources()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open('x', encoding='utf-8') as handle:
            json.dump(jsonable(report), handle, indent=2, allow_nan=False)
        print('Report:', report_path)
    print('No environment, trajectory, model, or figure files were modified.')
    print('PASS is a numerical state check, not evidence of scientific validity or recoverability.')


if __name__ == '__main__':
    main()
