"""Interrupt a small real run, resume it, and verify completed-stage reuse."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from common import load_study, read_json, sha256, validate_task_record, write_json
from study_config import task_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    work = args.work.resolve()
    if work.exists():
        raise RuntimeError('Use a new verification work directory')
    work.mkdir(parents=True)
    log = work/'execution.log'
    common = ['--root', str(work), '--profile', 'smoke']
    run = [sys.executable, str(PACKAGE/'run_experiment.py'), *common,
           '--task-id', '1', '--device', 'cpu', '--threads', '1']
    with log.open('w') as handle:
        subprocess.run([sys.executable, str(PACKAGE/'prepare_reference.py'), *common],
                       stdout=handle, stderr=subprocess.STDOUT, check=True, timeout=60)
        process = subprocess.Popen(run, stdout=handle, stderr=subprocess.STDOUT)
        task_dir = work/'runs/n_48/seed_9000'
        deadline = time.monotonic()+60
        while time.monotonic() < deadline:
            path = task_dir/'progress.json'
            if path.exists() and read_json(path).get('stage') == 'cindes/generation/shard_01':
                process.send_signal(signal.SIGINT)
                break
            if process.poll() is not None:
                raise RuntimeError('The test run ended before the intended interruption')
            time.sleep(.01)
        else:
            process.terminate()
            process.wait(timeout=30)
            raise TimeoutError('The test did not reach the intended interruption point')
        interrupted_code = process.wait(timeout=30)
        if interrupted_code == 0 or (task_dir/'result.json').exists():
            raise AssertionError('The intended unfinished run completed unexpectedly')
        before_paths = list((task_dir/'methods/cindes/cv').rglob('metadata.json'))
        before_paths += list((task_dir/'methods/cindes/cv').rglob('model.pt'))
        before_paths += [task_dir/'methods/cindes/refit/metadata.json',
                         task_dir/'methods/cindes/refit/model.pt',
                         task_dir/'methods/cindes/generation/shard_00.npz',
                         task_dir/'methods/cindes/generation/shard_00.json']
        before = {p.relative_to(task_dir).as_posix(): sha256(p) for p in before_paths}
        subprocess.run(run, stdout=handle, stderr=subprocess.STDOUT, check=True, timeout=60)
    manifest = load_study(work, 'smoke')
    result = validate_task_record(task_dir, manifest, task_plan('smoke')[0])
    assert result is not None
    for name, expected in before.items():
        assert sha256(task_dir/name) == expected, name
    attempts = [read_json(p) for p in (task_dir/'attempts').glob('*.json')]
    assert any(x['status'] == 'failed' for x in attempts)
    assert any(x['status'] == 'complete' for x in attempts)
    before_result = sha256(task_dir/'result.json')
    repeated = subprocess.run([sys.executable, str(PACKAGE/'run_experiment.py'), *common,
        '--task-id', '1', '--device', 'cuda', '--threads', '1'],
        capture_output=True, text=True, check=True, timeout=60)
    assert 'paired_replication_reused' in repeated.stdout
    assert sha256(task_dir/'result.json') == before_result
    record = {'status': 'passed', 'scope': 'Execution check only; profile smoke',
              'interrupted_returncode': interrupted_code,
              'completed_artifacts_reused_unchanged': len(before),
              'interrupted_and_completed_attempts_retained': True,
              'finished_replication_validated': True,
              'complete_reuse_before_cuda_preflight': True,
              'complete_result_unchanged_by_reuse': True,
              'study_identity_sha256': manifest['identity_sha256'],
              'source_sha256': manifest['identity']['source_sha256'],
              'test_source_sha256': sha256(Path(__file__)),
              'actual_cuda_execution_tested': False}
    write_json(PACKAGE/'verification/resume_check.json', record)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
