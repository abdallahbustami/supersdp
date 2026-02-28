"""L2A algorithm entrypoints.

This module exposes the direct objective-approximation model API in one place.
"""

from __future__ import annotations

from .learning import ScalarL2ANet, train_scalar_l2a

__all__ = [
    "ScalarL2ANet",
    "train_scalar_l2a",
]

