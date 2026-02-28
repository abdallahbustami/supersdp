"""Low-level solver API entrypoints.

This module provides a focused import path for the stable low-level solver API.
"""

from __future__ import annotations

from .toolbox import L2CAConfig, ProblemConfig, SolveResult, SolverConfig, SuperSDP, TrainingConfig

L2WSSolver = SuperSDP

__all__ = [
    "L2CAConfig",
    "SuperSDP",
    "L2WSSolver",
    "ProblemConfig",
    "SolveResult",
    "SolverConfig",
    "TrainingConfig",
]
