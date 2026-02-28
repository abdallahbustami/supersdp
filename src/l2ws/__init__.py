"""Learning-accelerated SDP toolkit.

The stable top-level API is intentionally centered on the low-level solver
interface plus a small set of commonly used SDP/system data helpers.

Advanced algorithm internals and experimental utilities remain available from
their submodules (for example ``l2ws.l2ca`` or ``l2ws.learning``) but are not
re-exported here, which keeps ``import l2ws`` lighter and makes the public
surface clearer.
"""

from .data import (
    L2GainInstance,
    LTISystem,
    build_h2_norm_sdp,
    build_hinf_norm_sdp,
    build_l2_gain_sdp,
    build_lqr_feas_sdp,
    build_lyapunov_sdp,
    extract_lqr_gain_from_solution,
    build_system_graph,
    generate_l2_gain_instances,
    generate_sdp_instances,
)
from .graph_utils import SystemGraphBuilder, has_torch_geometric
from .l2ws import L2CAConfig, ProblemConfig, SolveResult, SolverConfig, SuperSDP, TrainingConfig
from .problem import SDPInstance

L2WSSolver = SuperSDP

__all__ = [
    "SDPInstance",
    "L2GainInstance",
    "LTISystem",
    "build_h2_norm_sdp",
    "build_hinf_norm_sdp",
    "build_l2_gain_sdp",
    "build_lqr_feas_sdp",
    "build_lyapunov_sdp",
    "extract_lqr_gain_from_solution",
    "build_system_graph",
    "generate_l2_gain_instances",
    "generate_sdp_instances",
    "SystemGraphBuilder",
    "has_torch_geometric",
    "SuperSDP",
    "L2WSSolver",
    "L2CAConfig",
    "ProblemConfig",
    "SolveResult",
    "SolverConfig",
    "TrainingConfig",
]
