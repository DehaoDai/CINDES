#!/usr/bin/env python3
"""Run the preserved SCC driver with an explicit allocated CPU thread count.

The SCC helper already reads NSLOTS/OMP_NUM_THREADS. This bridge also enforces
the launcher's requested cap after import, including an optional lower cap.
The model, random seeds, optimization, sampler, and metrics are unchanged.
"""
from __future__ import annotations
import os
from pathlib import Path
import sys


def cpu_thread_budget():
    raw = os.environ.get("CINDES_CPU_THREADS") or os.environ.get("NSLOTS") or "4"
    try:
        count = int(raw)
    except ValueError as exc:
        raise ValueError("CINDES_CPU_THREADS/NSLOTS must be a positive integer.") from exc
    if count < 1:
        raise ValueError("CINDES_CPU_THREADS/NSLOTS must be positive.")
    slots = os.environ.get("NSLOTS")
    if slots and (not slots.isdigit() or int(slots) < 1):
        raise ValueError("NSLOTS must be a positive integer.")
    if slots and count > int(slots):
        raise ValueError("CINDES_CPU_THREADS must not exceed the allocated NSLOTS.")
    return count


def set_thread_environment(count):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(count)


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent / "experiment"))
    if "--dry-run" not in sys.argv and "--help" not in sys.argv and "-h" not in sys.argv:
        count = cpu_thread_budget()
        set_thread_environment(count)
        import cindes_vs_dsm_mrf
        import torch
        torch.set_num_threads(count)
        print(f"Execution thread cap: {torch.get_num_threads()} (after original helper import)", flush=True)
    import cindes_vs_dsm_tree
    cindes_vs_dsm_tree.main()


if __name__ == "__main__":
    main()
