"""Data structures and helpers for semidefinite programs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

ArrayLike = np.ndarray


def _to_numpy_symmetric(matrix: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {arr.shape}")
    return 0.5 * (arr + arr.T)


def _stack_matrices(mats: Sequence[ArrayLike]) -> np.ndarray:
    mats = [0.5 * (np.asarray(mat, dtype=float) + np.asarray(mat, dtype=float).T) for mat in mats]
    return np.stack(mats, axis=0)


@dataclass(frozen=True)
class SDPInstance:
    """Standard-form SDP instance used across the codebase.

    The primal problem solved throughout the project is

        minimize    C • X
        subject to  Ai • X = bi,  i = 1, …, m
                    X ⪰ 0,

    whose dual uses the same {Ai} and objective vector b.
    """

    C: ArrayLike
    A: Sequence[ArrayLike]
    b: ArrayLike
    name: str = "unnamed"

    def __post_init__(self) -> None:
        C = _to_numpy_symmetric(self.C, "C")
        A = np.asarray(self.A, dtype=float)
        if A.ndim != 3:
            A = _stack_matrices(self.A)
        if A.shape[1] != A.shape[2]:
            raise ValueError("Constraint matrices Ai must be square and share the same shape.")
        if A.shape[1] != C.shape[0]:
            raise ValueError("C and Ai must have matching dimensions.")
        b = np.asarray(self.b, dtype=float).reshape(-1)
        if b.shape[0] != A.shape[0]:
            raise ValueError("Length of b must equal the number of constraint matrices.")

        object.__setattr__(self, "C", C)
        object.__setattr__(self, "A", 0.5 * (A + A.transpose(0, 2, 1)))
        object.__setattr__(self, "b", b)

    @property
    def dim(self) -> int:
        """Number of rows/columns of the primal matrix variable X."""
        return int(self.C.shape[0])

    @property
    def num_constraints(self) -> int:
        """Number of equality constraints."""
        return int(self.A.shape[0])

    def apply_A(self, X: ArrayLike) -> np.ndarray:
        """Evaluate A(X) = [Ai • X]_i."""
        X = 0.5 * (np.asarray(X, dtype=float) + np.asarray(X, dtype=float).T)
        return np.tensordot(self.A, X, axes=((1, 2), (0, 1)))

    def apply_AT(self, y: ArrayLike) -> np.ndarray:
        """Evaluate A*(y) = sum_i y_i Ai."""
        y = np.asarray(y, dtype=float).reshape(-1)
        if y.shape[0] != self.num_constraints:
            raise ValueError("Dimension mismatch for y in A*(y).")
        return np.tensordot(y, self.A, axes=(0, 0))

    def residuals(self, X: ArrayLike, y: ArrayLike, S: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Return primal (r_p) and dual (r_d) residuals."""
        r_p = self.b - self.apply_A(X)
        r_d = self.C - self.apply_AT(y) - S
        return r_p, r_d

    def trace_inner(self, A: ArrayLike, B: ArrayLike) -> float:
        """Compute the trace inner product A • B."""
        return float(np.tensordot(A, B, axes=((0, 1), (0, 1))))

    def identity(self, scale: float = 1.0) -> np.ndarray:
        """Convenience helper to return scale * I."""
        return scale * np.eye(self.dim)

    def zeros_matrix(self) -> np.ndarray:
        return np.zeros((self.dim, self.dim), dtype=float)

    def zeros_vector(self) -> np.ndarray:
        return np.zeros(self.num_constraints, dtype=float)
