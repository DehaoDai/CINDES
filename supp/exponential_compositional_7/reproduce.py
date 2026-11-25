#!/usr/bin/env python3
"""Portable driver for the unchanged CINDES exponential experiment entrypoints."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

from study_config import PROFILES, task_plan

HERE = Path(__file__).resolve().parent


def parse_tasks(value, count):
    if value is None:
        return list(range(1, count + 1))
    chosen = set()
    for part in value.split(','):
        part = part.strip()
        if '-' in part:
            endpoints = part.split('-')
            if len(endpoints) != 2:
                raise ValueError('Use task IDs or ranges, for example 1-20 or 2,50.')
            first, last = map(int, endpoints)
            if first > last:
                raise ValueError('A task range must be increasing.')
            chosen.update(range(first, last + 1))
        else:
            chosen.add(int(part))
    if not chosen or min(chosen) < 1 or max(chosen) > count:
        raise ValueError(f'Task IDs must lie between 1 and {count}.')
    return sorted(chosen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=PROFILES, default='smoke')
    parser.add_argument('--root', type=Path,
                        help='Output path; relative paths are relative to this package.')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--tasks', help='Optional task IDs/ranges, such as 1-20 or 2,50.')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    count = len(task_plan(args.profile))
    try:
        selected = parse_tasks(args.tasks, count)
    except ValueError as error:
        parser.error(str(error))
    requested = (args.root or Path('outputs') / f'exponential_{args.profile}_v1').expanduser()
    root = requested.resolve() if requested.is_absolute() else (HERE / requested).resolve()
    common = ['--root', str(root), '--profile', args.profile]
    commands = [[sys.executable, str(HERE / 'prepare_reference.py'), *common]]
    commands.extend([
        sys.executable, str(HERE / 'run_experiment.py'), *common,
        '--task-id', str(task_id), '--device', args.device, '--threads', str(args.threads),
    ] for task_id in selected)
    summary = [sys.executable, str(HERE / 'summarize_results.py'), *common]
    if len(selected) != count:
        summary.append('--status')
    commands.append(summary)
    print(f'Profile: {args.profile}; selected paired tasks: {len(selected)}/{count}', flush=True)
    print(f'Output: {root}', flush=True)
    if args.profile == 'smoke':
        print('Execution check only: smoke results are not scientific results.', flush=True)
    if args.dry_run:
        for i, command in enumerate(commands):
            if len(commands) <= 6 or i < 2 or i >= len(commands) - 2:
                print(shlex.join(command))
            elif i == 2:
                print(f'... {len(commands) - 4} further paired task commands ...')
        return
    environment = os.environ.copy()
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        environment[name] = str(args.threads)
    environment.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    for command in commands:
        subprocess.run(command, cwd=HERE, env=environment, check=True)
    if len(selected) != count:
        print('Selected tasks completed. Run the strict summary after all tasks finish.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from error
