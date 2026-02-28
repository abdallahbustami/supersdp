"""Structured perturbations for toolbox instance generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import LTISystem


@dataclass(frozen=True)
class PerturbationSpec:
    kind: str
    A_scale: float = 0.0
    B_scale: float = 0.0
    seed: int = 0
    A_mask: np.ndarray | None = None
    B_mask: np.ndarray | None = None


def _apply_mask(delta: np.ndarray, mask: np.ndarray | None, name: str) -> np.ndarray:
    if mask is None:
        return delta
    mask_arr = np.asarray(mask, dtype=float)
    if mask_arr.shape != delta.shape:
        raise ValueError(f"{name}_mask shape mismatch: expected {delta.shape}, got {mask_arr.shape}.")
    return delta * mask_arr


def perturb_system(system: LTISystem, spec: PerturbationSpec, idx: int) -> LTISystem:
    """Deterministic per-index perturbation using seed+idx."""
    if not isinstance(spec, PerturbationSpec):
        raise TypeError("spec must be a PerturbationSpec instance.")

    kind = str(spec.kind).strip().lower()
    A = np.asarray(system.A, dtype=float)
    Bw = np.asarray(system.Bw, dtype=float)
    Cz = np.asarray(system.Cz, dtype=float)

    if kind == "none":
        return LTISystem(A=A.copy(), Bw=Bw.copy(), Cz=Cz.copy())

    rng = np.random.default_rng(int(spec.seed) + int(idx))
    A_scale = float(spec.A_scale)
    B_scale = float(spec.B_scale)

    if kind == "entrywise_uniform":
        dA = rng.uniform(-A_scale, A_scale, size=A.shape)
        dB = rng.uniform(-B_scale, B_scale, size=Bw.shape)
    elif kind == "entrywise_gaussian":
        dA = rng.normal(loc=0.0, scale=A_scale, size=A.shape)
        dB = rng.normal(loc=0.0, scale=B_scale, size=Bw.shape)
    elif kind == "diagonal":
        dA = np.zeros_like(A)
        dB = np.zeros_like(Bw)
        dA[np.diag_indices(A.shape[0])] = rng.uniform(-A_scale, A_scale, size=A.shape[0])
        n_diag_b = min(Bw.shape[0], Bw.shape[1])
        if n_diag_b > 0:
            dB[np.arange(n_diag_b), np.arange(n_diag_b)] = rng.uniform(-B_scale, B_scale, size=n_diag_b)
    else:
        raise ValueError("spec.kind must be one of: none, entrywise_uniform, entrywise_gaussian, diagonal.")

    dA = _apply_mask(dA, spec.A_mask, "A")
    dB = _apply_mask(dB, spec.B_mask, "B")

    return LTISystem(A=A + dA, Bw=Bw + dB, Cz=Cz.copy())


__all__ = [
    "PerturbationSpec",
    "perturb_system",
]
