"""Minimal user-facing toolbox run (Part 1 sanity check)."""

from __future__ import annotations

import numpy as np

from l2ws.data import LTISystem
from l2ws.perturbations import PerturbationSpec
from l2ws.runner import ExperimentConfig, run_experiment_on_system


def _random_stable_system(n: int = 4, m_w: int = 2, p_z: int = 2, seed: int = 7) -> LTISystem:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    max_real = float(np.max(np.real(np.linalg.eigvals(A))))
    # Shift left-half plane for reliable solver convergence.
    A = A - (max_real + 1.0) * np.eye(n, dtype=float)
    Bw = rng.standard_normal((n, m_w))
    Cz = rng.standard_normal((p_z, n))
    return LTISystem(A=A, Bw=Bw, Cz=Cz)


if __name__ == "__main__":
    system = _random_stable_system()
    cfg = ExperimentConfig(
        application="l2_gain",
        num_train=5,
        num_test=2,
        perturb=PerturbationSpec(kind="entrywise_uniform", A_scale=0.02, B_scale=0.02, seed=11),
        seed=11,
        epochs=20,
        batch_size=16,
        lr=3e-4,
        algorithms=("IPM", "L2CA"),
    )
    run_experiment_on_system(system=system, system_name="tiny_random_l2_gain", config=cfg)
