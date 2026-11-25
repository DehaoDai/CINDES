"""Immutable study identities, atomic outputs, and read-only validation."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

from study_config import HERE, profile_config, task_plan

CRITICAL_SOURCES = (
    'study_config.py', 'common.py', 'prepare_reference.py', 'run_experiment.py',
    'preflight.py', 'protocol_origin.json',
    'experiment/models.py', 'experiment/reference.py', 'experiment/training.py',
    'experiment/sampling.py', 'experiment/sampling_variants.py',
    'experiment/dgp.py', 'experiment/evaluation.py',
    'experiment/reference_score_table.npz',
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def canonical_hash(value):
    data = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    with Path(path).open(encoding='utf-8') as handle:
        return json.load(handle)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
    os.replace(temporary, path)


def atomic_npz(path, **arrays):
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def resolve_root(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (HERE/path).resolve()


@contextmanager
def lock(path):
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Another process holds the experiment lock: {path}') from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def expected_identity(profile):
    return {'profile': profile, 'config': profile_config(profile),
            'tasks': task_plan(profile),
            'source_sha256': {name: sha256(HERE/name) for name in CRITICAL_SOURCES}}


def load_study(root, profile):
    root = resolve_root(root)
    manifest = read_json(root/'manifest.json')
    identity = manifest.get('identity')
    if manifest.get('identity_sha256') != canonical_hash(identity):
        raise ValueError('The saved study identity is invalid')
    if identity != expected_identity(profile):
        raise ValueError('The profile, protocol, or scientific source code changed. Use a new output directory.')
    for name, value in identity['source_sha256'].items():
        if sha256(root/'source_snapshot'/name) != value:
            raise ValueError(f'The saved scientific source snapshot changed: {name}')
    return manifest


def ensure_study(root, profile):
    root = resolve_root(root)
    root.mkdir(parents=True, exist_ok=True)
    with lock(root/'.study.lock'):
        if (root/'manifest.json').exists():
            return load_study(root, profile)
        if any((root/name).exists() for name in ('runs', 'reference')):
            raise ValueError('Unrecorded experiment output exists; use a new output directory.')
        identity = expected_identity(profile)
        recorded = set(CRITICAL_SOURCES)
        for pattern in ('*.py', '*.sh', '*.qsub', 'README.md', 'requirements.txt'):
            recorded.update(p.relative_to(HERE).as_posix() for p in HERE.glob(pattern) if p.is_file())
        snapshot_hashes = {}
        for name in sorted(recorded):
            destination = root/'source_snapshot'/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(HERE/name, destination)
            snapshot_hashes[name] = sha256(destination)
        manifest = {'created_utc': utc_now(), 'identity': identity,
                    'identity_sha256': canonical_hash(identity),
                    'recorded_source_sha256': snapshot_hashes,
                    'reporting_note': 'Report-format and scheduler scripts are recorded separately from the scientific identity.'}
        write_json(root/'manifest.json', manifest)
        write_json(root/'study_config.json', identity['config'])
        write_json(root/'task_plan.json', identity['tasks'])
        return manifest


def safe_child(parent, relative):
    parent = Path(parent).resolve()
    value = Path(relative)
    if value.is_absolute():
        raise ValueError('An artifact path must be relative')
    candidate = (parent/value).resolve()
    if not candidate.is_relative_to(parent):
        raise ValueError('An artifact path leaves its experiment directory')
    return candidate


def validate_reference(root, manifest):
    root = resolve_root(root)
    record = read_json(root/'reference/metadata.json')
    cfg = manifest['identity']['config']
    expected = {'study_identity_sha256': manifest['identity_sha256'],
                'n': cfg['reference_size'], 'd': cfg['d'],
                'seed': cfg['reference_seed'], 'target_path': 'reference/target.npz'}
    if record.get('status') != 'complete' or any(record.get(k) != v for k, v in expected.items()):
        raise ValueError('Reference preparation is incomplete or incompatible')
    if sha256(safe_child(root, record['target_path'])) != record.get('reference_sha256'):
        raise ValueError('The independent target reference sample changed')
    return record


def validate_task_record(task_dir, manifest, task):
    task_dir = Path(task_dir)
    path = task_dir/'result.json'
    if not path.exists():
        return None
    record = read_json(path)
    if record.get('task_identity') != task or record.get('study_identity_sha256') != manifest['identity_sha256']:
        raise ValueError(f'Incompatible saved task: {task_dir}')
    if record.get('status') != 'complete':
        return None
    cfg = manifest['identity']['config']
    methods = record.get('methods', {})
    if set(methods) != {'cindes', 'dsm'}:
        raise ValueError(f'Expected exactly one result for each method: {task_dir}')
    for method, row in methods.items():
        for key, expected in {'method': method, 'n': task['n'], 'seed': task['seed'],
                              'd': cfg['d'], 'n_gen': cfg['n_gen'],
                              'fit_seed': task['fit_seed'], 'task_id': task['task_id'],
                              'profile': manifest['identity']['profile'],
                              'study_identity_sha256': manifest['identity_sha256'],
                              'experiment_config': cfg}.items():
            if row.get(key) != expected:
                raise ValueError(f'Invalid {method} result field {key}: {task_dir}')
        for metric in cfg['metrics']:
            if not isinstance(row.get(metric), (int, float)) or not math.isfinite(row[metric]):
                raise ValueError(f'Nonfinite or missing {method} metric {metric}: {task_dir}')
    hashes = record.get('artifact_sha256')
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError(f'Task artifact hashes are missing: {task_dir}')
    required = {'data.npz', 'data.json', 'task_manifest.json'}
    for method, row in methods.items():
        metric_path = f'methods/{method}/metrics.json'
        required.add(metric_path)
        if read_json(task_dir/metric_path) != row:
            raise ValueError(f'The paired result differs from its saved {method} metrics')
        for field in ('data', 'model', 'selection', 'generated'):
            filename = row[field+'_path']
            required.add(filename)
            if hashes.get(filename) != row[field+'_sha256']:
                raise ValueError(f'The {method} {field} hash disagrees with the task record')
    if not required.issubset(hashes):
        raise ValueError(f'The task omits required scientific artifacts: {task_dir}')
    for name, expected in hashes.items():
        if sha256(safe_child(task_dir, name)) != expected:
            raise ValueError(f'Saved task artifact changed: {task_dir/name}')
    return record
