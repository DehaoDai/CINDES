"""Check numerical operations and record the environment on the allocated node."""
from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path


def check_environment(device='cuda', threads=1):
    import numpy as np
    import scipy
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent/'experiment'))
    from reference import LogExponentialReference, ReferenceScore
    target = torch.device(device)
    torch.set_num_threads(int(threads))
    if target.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('No usable CUDA allocation. Submit the GPU array with qsub.')
    if target.type == 'cuda':
        target = torch.device('cuda', torch.cuda.current_device() if target.index is None else target.index)
        torch.cuda.synchronize()
    x = torch.tensor([[-3., -.5, .5, 1., -1., 0., .3],
                      [-4., -.2, .2, .8, -2., .1, .7]], device=target)
    matrix = torch.ones((16, 16), device=target)
    matrix = matrix @ matrix
    posterior = LogExponentialReference(device=target)
    exact = posterior.score(x, .1)
    table = ReferenceScore(device=target).score(x, .1)
    gen = torch.Generator(device=target).manual_seed(7219)
    _, particles = posterior.antithetic(x, .1, 8, gen)
    if not all(bool(torch.isfinite(v).all()) for v in (matrix, exact, table, particles)):
        raise FloatingPointError('Device preflight produced nonfinite values')
    error = float((exact-table).abs().max())
    if error > 0.01:
        raise ArithmeticError(f'Reference interpolation check failed: {error}')
    if target.type == 'cuda':
        torch.cuda.synchronize()
    gpu = None
    if target.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        gpu = {'name': props.name, 'total_memory_bytes': props.total_memory,
               'compute_capability': [props.major, props.minor]}
    return {'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__,
            'torch': str(torch.__version__), 'cuda_runtime': torch.version.cuda,
            'device': str(target), 'gpu': gpu, 'host': platform.node(),
            'torch_cpu_threads': torch.get_num_threads(),
            'reference_preflight_max_absolute_difference': error,
            'scheduler': {key: os.environ.get(key) for key in
                          ('JOB_ID', 'SGE_TASK_ID', 'NSLOTS', 'QUEUE', 'CUDA_VISIBLE_DEVICES')},
            'torch_deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cuda_matmul_allow_tf32': bool(torch.backends.cuda.matmul.allow_tf32)}


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--threads', type=int, default=int(os.environ.get('NSLOTS', '1')))
    args = parser.parse_args()
    print(json.dumps(check_environment(args.device, args.threads), indent=2))


if __name__ == '__main__':
    main()
