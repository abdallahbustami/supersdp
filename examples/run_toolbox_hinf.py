"""Small H-infinity toolbox run (Part 2 sanity check)."""

from __future__ import annotations

import numpy as np

from l2ws.data import LTISystem
from l2ws.perturbations import PerturbationSpec
from l2ws.runner import ExperimentConfig, run_experiment_on_system


def _random_stable_system(n: int = 5, m_w: int = 2, p_z: int = 2, seed: int = 21) -> LTISystem:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    max_real = float(np.max(np.real(np.linalg.eigvals(A))))
    A = A - (max_real + 1.0) * np.eye(n, dtype=float)
    Bw = rng.standard_normal((n, m_w))
    Cz = rng.standard_normal((p_z, n))
    return LTISystem(A=A, Bw=Bw, Cz=Cz)


if __name__ == "__main__":
    system = _random_stable_system()
    cfg = ExperimentConfig(
        application="hinf_norm",
        num_train=15,
        num_test=5,
        perturb=PerturbationSpec(kind="entrywise_uniform", A_scale=0.02, B_scale=0.02, seed=3),
        seed=3,
        epochs=25,
        batch_size=16,
        algorithms=("MOSEK", "IPM", "L2CA"),
        prestabilize="off",
        prestabilize_margin=1e-3,
    )
    run_experiment_on_system(system=system, system_name="tiny_random_hinf", config=cfg)
