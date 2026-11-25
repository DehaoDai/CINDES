"""Prepare one independent evaluation sample before the GPU array starts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from common import (atomic_npz, ensure_study, lock, resolve_root, sha256,
                    utc_now, validate_reference, write_json)
from study_config import HERE, PROFILES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--profile', choices=PROFILES, default='n1000')
    args = parser.parse_args()
    root = resolve_root(args.root)
    manifest = ensure_study(root, args.profile)
    cfg = manifest['identity']['config']
    with lock(root/'reference/.preparation.lock'):
        if (root/'reference/metadata.json').exists():
            record = validate_reference(root, manifest)
            print(json.dumps({'status': 'reused', 'reference': record['target_path'], 'n': record['n']}))
            return
        import numpy as np
        sys.path.insert(0, str(HERE/'experiment'))
        from dgp import ExponentialCompositionalDGP
        dgp = ExponentialCompositionalDGP(d=cfg['d'], pot_seed=cfg['dgp']['pot_seed'],
                amplitude=cfg['dgp']['amplitude'], kappa=cfg['dgp']['kappa'])
        started = time.perf_counter()
        values = dgp.sample(cfg['reference_size'], np.random.default_rng(cfg['reference_seed']))
        if values.shape != (cfg['reference_size'], cfg['d']) or not np.isfinite(values).all() or not (values > 0).all():
            raise FloatingPointError('Invalid evaluation reference sample')
        # Recreate an unrecorded partial reference deterministically after an interruption.
        path = root/'reference/target.npz'
        atomic_npz(path, Y=values)
        record = {'status': 'complete', 'created_utc': utc_now(),
                  'study_identity_sha256': manifest['identity_sha256'],
                  'target_path': 'reference/target.npz', 'reference_sha256': sha256(path),
                  'n': len(values), 'd': cfg['d'], 'seed': cfg['reference_seed'],
                  'sampler': dgp.last_sample_stats, 'dgp': dgp.config,
                  'preparation_seconds': time.perf_counter()-started,
                  'usage': 'Evaluation only; never supplied to training or cross-validation'}
        write_json(root/'reference/metadata.json', record)
        validate_reference(root, manifest)
        print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
