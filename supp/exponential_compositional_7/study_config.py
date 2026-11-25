"""Fixed scientific settings and array-task mapping; standard library only."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROFILES = ('n1000', 'full', 'smoke')

BASE = {
    'protocol': 'centered_exponential_scc_confirmation_v1',
    'd': 7, 'sample_sizes': [1000], 'seeds': list(range(4000, 4020)),
    'dgp': {'pot_seed': 11, 'amplitude': 1.5, 'kappa': 1.0,
            'feature': 'psi(y)=1-2*exp(-y)', 'base': 'product Exp(1)'},
    'coordinate_transform': 'Z=log(Y), common to both estimators',
    'reference': 'product log-Exp(1), known to both estimators',
    'network': {'hidden_layers': 2, 'width': 64, 'activation': 'Softplus',
                'softplus_beta': 5.0, 'dsm_time_frequencies': 8},
    'lambdas': [0.0, 0.01, 0.1, 0.3, 1.0, 3.0], 'folds': 3,
    'epochs': 2000, 'checks': 100, 'batch_size': 256,
    'normalizer_size': 65536, 'fake_ratio': 2,
    'dsm_validation_draws_per_observation': 8,
    'optimizer': 'Adam', 'learning_rate': 0.001, 'weight_decay': 0.0,
    'selection': 'Separate three-fold lambda and duration selection; CINDES normalized density NLL, DSM denoising loss',
    'scheduler': 'Cosine over full cap; same prefix retained in refit',
    'cindes_output': 'b-softplus(g); no fixed R and no added tail term',
    'dsm_parameterization': 'reference_residual',
    'initialization': 'reference_T', 'M': 400, 'K': 1000,
    'T': 3.0, 'delta': 0.001, 'n_gen': 5000,
    'generation_shards': 10, 'generation_batch': 64,
    'last_step_noise': True, 'terminal_correction': False, 'support_clipping': False,
    'reference_size': 524288, 'reference_seed': 53000001,
    'metrics': ['tv1', 'tv2', 'llgap', 'h_tv', 'h_w1'],
    'histogram_bins': 20,
    'marginal_bins': 'Exp(1) quantiles, including zero and infinity',
    'H_bins': 'standard-normal quantiles, including both infinite tails',
    'llgap': 'signed reference minus generated mean unnormalized target log density',
    'fit_seed_rule': '100000*n + replication_seed',
    'data_seed_rule': 'fit_seed', 'fold_seed_rule': '700000 + fit_seed',
    'training_truth_access': False,
    'method_order': 'CINDES first for even replication seeds; DSM first for odd seeds',
}


def profile_config(name):
    if name not in PROFILES:
        raise ValueError(f'Unknown profile: {name}')
    value = deepcopy(BASE)
    value['profile'] = name
    value['test_only'] = name == 'smoke'
    if name == 'full':
        value['sample_sizes'] = [500, 1000, 1500, 2000]
    elif name == 'smoke':
        value.update(sample_sizes=[48], seeds=[9000], epochs=2, checks=2,
                     lambdas=[0.0, 0.1], batch_size=24, normalizer_size=128,
                     n_gen=32, generation_shards=2, generation_batch=16, K=16,
                     reference_size=2048, reference_seed=53000002)
    return value


def task_plan(profile):
    config = profile_config(profile)
    tasks = []
    for n in config['sample_sizes']:
        for seed in config['seeds']:
            fit_seed = 100000*n + seed
            tasks.append({'task_id': len(tasks)+1, 'n': n, 'seed': seed,
                          'fit_seed': fit_seed, 'data_seed': fit_seed,
                          'fold_seed': 700000+fit_seed})
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=PROFILES, default='n1000')
    parser.add_argument('--task-count', action='store_true')
    parser.add_argument('--tasks', action='store_true')
    args = parser.parse_args()
    if args.task_count:
        print(len(task_plan(args.profile)))
    elif args.tasks:
        print(json.dumps(task_plan(args.profile), indent=2))
    else:
        print(json.dumps(profile_config(args.profile), indent=2))


if __name__ == '__main__':
    main()
