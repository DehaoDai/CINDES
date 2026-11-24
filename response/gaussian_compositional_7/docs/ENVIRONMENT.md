# Software and hardware

## Environment recorded in the reported study

The following versions come from the saved SCC execution records. They describe the environment that produced the supplied results, rather than a promise that the same named environment or binary wheels are available on every machine.

| Component | Recorded value |
| --- | --- |
| Operating environment | Linux on BU SCC |
| Python | 3.13.5 |
| NumPy | 2.5.2 |
| SciPy | 1.18.1 |
| PyTorch | 2.13.0+cu132 |
| CUDA runtime reported by PyTorch | 13.2 |
| Training device | CPU |
| Cross-validation workers | 4; each fit used one numerical thread |
| Generation device | CUDA GPU |
| Generation arithmetic | float32; TF32 disabled; deterministic algorithms enabled |

Across 80 paired replications, 64 used an NVIDIA L40S, 14 used an NVIDIA A40, and 2 used an NVIDIA RTX PRO 6000 Blackwell Server Edition. CINDES and DSM used the same GPU within each replication. The cluster scripts requested at least 16 GB of GPU memory and compute capability at least 8.0.

The SCC activation commands were:

```bash
module load miniconda
module load academic-ml/fall-2026
conda activate fall-2026-pyt
```

The supplied activation script unloads a conflicting `python3` module first. On another system, use a Python environment containing NumPy, SciPy, and a compatible PyTorch build.

## Environment used to check the submission package

The package checks use Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, and PyTorch 2.15.0+cpu under Linux. This environment is used for the reduced execution check and arithmetic checks, not to regenerate the reported full-study values. The numerical table builder uses only the Python standard library.

The exact checks and their outcomes are recorded under `verification/`. A full CUDA rerun was not performed while assembling the journal archive.

## Installing PyTorch

Use the [official installation selector](https://pytorch.org/get-started/locally/) to choose a build compatible with the operating system and GPU driver. Only `torch` is used by this experiment; torchvision and torchaudio are unnecessary.

For a CPU-only reduced check, the following installs a CPU build:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy scipy
```

For a full run, first install a suitable CUDA build and verify `torch.cuda.is_available()` on the allocated GPU machine. Installing the three generic requirements alone does not establish that a GPU is usable.

PyTorch documents that fixed seeds do not guarantee identical results across releases or platforms. The original source, protocol, seeds, and observed software versions are retained so that differences can be traced. The recorded table values can be reproduced exactly by aggregating the supplied per-replication outcomes.
