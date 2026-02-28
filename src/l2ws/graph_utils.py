"""Graph construction utilities for LTI-system-conditioned learning."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

try:  # Optional dependency.
    from torch_geometric.data import Data
except Exception:  # pragma: no cover - optional dependency path
    Data = None  # type: ignore[assignment]


def has_torch_geometric() -> bool:
    """Return ``True`` when ``torch_geometric`` is importable."""
    return Data is not None


class SystemGraphBuilder:
    """Convert continuous-time LTI matrices into a PyG graph.

    Nodes represent state variables. Directed edges correspond to nonzero
    off-diagonal couplings in ``A``. Node features are built from local entries
    ``[diag(A), Bw_row, Cz_col]``.
    """

    def __init__(self, edge_threshold: float = 1e-10) -> None:
        self.edge_threshold = float(edge_threshold)

    def build(self, A: np.ndarray, Bw: np.ndarray, Cz: np.ndarray) -> Any:
        """Return a ``torch_geometric.data.Data`` graph.

        Raises:
            ModuleNotFoundError: If ``torch_geometric`` is unavailable.
            ValueError: If matrix dimensions are inconsistent.
        """
        if Data is None:
            raise ModuleNotFoundError(
                "torch_geometric is required to build graph inputs. "
                "Install with: pip install torch-geometric"
            )

        A = np.asarray(A, dtype=float)
        Bw = np.asarray(Bw, dtype=float)
        Cz = np.asarray(Cz, dtype=float)

        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"A must be square, got shape {A.shape}.")
        n = A.shape[0]
        if Bw.ndim != 2 or Bw.shape[0] != n:
            raise ValueError(f"Bw must have shape (n, p) with n={n}, got {Bw.shape}.")
        if Cz.ndim != 2 or Cz.shape[1] != n:
            raise ValueError(f"Cz must have shape (q, n) with n={n}, got {Cz.shape}.")

        rows, cols = np.nonzero(np.abs(A) > self.edge_threshold)
        off_diag_mask = rows != cols
        off_rows = rows[off_diag_mask]
        off_cols = cols[off_diag_mask]

        if off_rows.size > 0:
            edge_index = np.stack([off_rows, off_cols], axis=0)
            edge_attr = A[off_rows, off_cols]
        else:
            edge_index = np.empty((2, 0), dtype=np.int64)
            edge_attr = np.empty((0,), dtype=float)

        self_loops = np.arange(n, dtype=np.int64)
        edge_index = np.concatenate(
            [edge_index, np.stack([self_loops, self_loops], axis=0)],
            axis=1,
        )
        edge_attr = np.concatenate([edge_attr, np.diag(A)], axis=0)

        node_feats = np.concatenate(
            [
                np.diag(A).reshape(-1, 1),
                Bw,
                Cz.T,
            ],
            axis=1,
        )

        return Data(
            x=torch.as_tensor(node_feats, dtype=torch.float32),
            edge_index=torch.as_tensor(edge_index, dtype=torch.long),
            edge_attr=torch.as_tensor(edge_attr, dtype=torch.float32).unsqueeze(-1),
        )
