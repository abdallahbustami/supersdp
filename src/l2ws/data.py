"""Dataset utilities for time-varying SDP instances."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Sequence, Tuple

import numpy as np

from .graph_utils import SystemGraphBuilder
from .problem import SDPInstance


def symmetric_basis(n: int) -> List[np.ndarray]:
    """Return a basis for symmetric ``n x n`` matrices."""
    basis = []
    for i in range(n):
        mat = np.zeros((n, n), dtype=float)
        mat[i, i] = 1.0
        basis.append(mat)
    for i in range(n):
        for j in range(i + 1, n):
            mat = np.zeros((n, n), dtype=float)
            mat[i, j] = mat[j, i] = 1.0
            basis.append(mat)
    return basis


def build_l2_gain_sdp(A: np.ndarray, Bw: np.ndarray, Cz: np.ndarray, name: str = "") -> SDPInstance:
    """Construct the standard-form SDP instance for the L2 gain problem."""
    A = np.asarray(A, dtype=float)
    Bw = np.asarray(Bw, dtype=float)
    Cz = np.asarray(Cz, dtype=float)
    n = A.shape[0]
    p = Bw.shape[1]

    def zeros(shape):
        return np.zeros(shape, dtype=float)

    def block_matrix(blocks: Sequence[Sequence[np.ndarray]]) -> np.ndarray:
        return np.block(blocks)

    basis = symmetric_basis(n)
    Ai: List[np.ndarray] = []
    for P in basis:
        top_left = A.T @ P + P @ A
        top_right = P @ Bw
        mid_left = Bw.T @ P
        third_block = -P
        Ai.append(
            block_matrix(
                [
                    [top_left, top_right, zeros((n, n)), zeros((n, 1))],
                    [mid_left, zeros((p, p)), zeros((p, n)), zeros((p, 1))],
                    [zeros((n, n)), zeros((n, p)), third_block, zeros((n, 1))],
                    [zeros((1, n)), zeros((1, p)), zeros((1, n)), zeros((1, 1))],
                ]
            )
        )

    last_mat = block_matrix(
        [
            [zeros((n, n)), zeros((n, p)), zeros((n, n)), zeros((n, 1))],
            [zeros((p, n)), -np.eye(p), zeros((p, n)), zeros((p, 1))],
            [zeros((n, n)), zeros((n, p)), zeros((n, n)), zeros((n, 1))],
            [zeros((1, n)), zeros((1, p)), zeros((1, n)), np.array([[-1.0]])],
        ]
    )
    Ai.append(last_mat)

    C = block_matrix(
        [
            [-Cz.T @ Cz, zeros((n, p)), zeros((n, n)), zeros((n, 1))],
            [zeros((p, n)), zeros((p, p)), zeros((p, n)), zeros((p, 1))],
            [zeros((n, n)), zeros((n, p)), zeros((n, n)), zeros((n, 1))],
            [zeros((1, n)), zeros((1, p)), zeros((1, n)), zeros((1, 1))],
        ]
    )

    b = np.zeros(len(Ai), dtype=float)
    b[-1] = -1.0

    instance_name = name or "l2_gain"
    return SDPInstance(C=C, A=Ai, b=b, name=instance_name)


def build_hinf_norm_sdp(system: "LTISystem", name: str = "") -> SDPInstance:
    """Construct continuous-time H-infinity norm SDP.

    This toolbox uses the same bounded-real SDP encoding as ``build_l2_gain_sdp``.
    The objective/dual interpretation is therefore consistent with L2 gain:
    recover gamma from the dual variable via ``extract_gamma_from_dual``.
    """
    return build_l2_gain_sdp(
        np.asarray(system.A, dtype=float),
        np.asarray(system.Bw, dtype=float),
        np.asarray(system.Cz, dtype=float),
        name=name or "hinf_norm",
    )

def build_lqr_feas_sdp(
    A: np.ndarray,
    Bw: np.ndarray,
    Cz: np.ndarray,
    name: str = "",
    rho: float = 0.1,
    eps: float = 1e-3,
) -> SDPInstance:
    """Construct an SDP feasibility problem for continuous-time LQR design.

    We encode the LMI
        [ S A' + A S + B Z + Z' B' ,  S ,  Z' ]
        [ S                        , -Qinv , 0 ] <= 0,
        [ Z                        ,  0    , -Rinv ]
    with Qinv = I and Rinv = (1/rho) I,
    plus S >= eps I, using one PSD primal variable X in S^{3n+m}.

    The function signature matches generic instance generation; ``Cz`` is unused.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(Bw, dtype=float)
    _ = Cz  # API compatibility with generate_sdp_instances
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square.")
    n = A.shape[0]
    if B.ndim != 2 or B.shape[0] != n:
        raise ValueError("Bw/B must have shape (n, m).")
    m = B.shape[1]
    if rho <= 0.0:
        raise ValueError("rho must be positive.")
    if eps <= 0.0:
        raise ValueError("eps must be positive.")

    qinv = np.eye(n, dtype=float)
    rinv = (1.0 / float(rho)) * np.eye(m, dtype=float)

    # Block layout [blk1, blk2, blk3, blk4] with sizes [n, n, m, n].
    b1 = slice(0, n)
    b2 = slice(n, 2 * n)
    b3 = slice(2 * n, 2 * n + m)
    b4 = slice(2 * n + m, 3 * n + m)
    N = 3 * n + m

    Ai: List[np.ndarray] = []
    b_vals: List[float] = []

    def add_entry_constraint(r: int, c: int, value: float) -> None:
        M = np.zeros((N, N), dtype=float)
        if r == c:
            M[r, c] = 1.0
        else:
            M[r, c] = 0.5
            M[c, r] = 0.5
        Ai.append(M)
        b_vals.append(float(value))

    # (A) X11 = A X12 + X12 A' + B X13' + X13 B' on symmetric part.
    for E in symmetric_basis(n):
        M = np.zeros((N, N), dtype=float)
        F = A.T @ E + E @ A
        G = E @ B
        M[b1, b1] = E
        M[b1, b2] = -0.5 * F
        M[b2, b1] = -0.5 * F.T
        M[b1, b3] = -1.0 * G
        M[b3, b1] = -1.0 * G.T
        Ai.append(M)
        b_vals.append(0.0)

    # (B1) Fix X22 = I.
    for i in range(n):
        for j in range(i, n):
            add_entry_constraint(n + i, n + j, float(qinv[i, j]))

    # (B2) Fix X33 = (1/rho) I.
    off3 = 2 * n
    for i in range(m):
        for j in range(i, m):
            add_entry_constraint(off3 + i, off3 + j, float(rinv[i, j]))

    # (B3) Fix X23 = 0.
    for i in range(n):
        for j in range(m):
            add_entry_constraint(n + i, off3 + j, 0.0)

    # Keep blk4 as an independent PSD slack: force cross blocks with blk4 to zero.
    off4 = 2 * n + m
    for i in range(n):
        for j in range(n):
            add_entry_constraint(i, off4 + j, 0.0)      # X14
            add_entry_constraint(n + i, off4 + j, 0.0)  # X24
    for i in range(m):
        for j in range(n):
            add_entry_constraint(off3 + i, off4 + j, 0.0)  # X34

    # (C) Enforce symmetry of X12.
    for i in range(n):
        for j in range(i + 1, n):
            M = np.zeros((N, N), dtype=float)
            M[b1.start + i, b2.start + j] = 0.5
            M[b2.start + j, b1.start + i] = 0.5
            M[b1.start + j, b2.start + i] = -0.5
            M[b2.start + i, b1.start + j] = -0.5
            Ai.append(M)
            b_vals.append(0.0)

    # (D) U + X12 + eps I = 0 on symmetric entries.
    for i in range(n):
        for j in range(i, n):
            M = np.zeros((N, N), dtype=float)
            # U_{ij}
            if i == j:
                M[off4 + i, off4 + j] += 1.0
            else:
                M[off4 + i, off4 + j] += 0.5
                M[off4 + j, off4 + i] += 0.5
            # X12_{ij}
            M[b1.start + i, b2.start + j] += 0.5
            M[b2.start + j, b1.start + i] += 0.5
            Ai.append(M)
            b_vals.append(-float(eps) if i == j else 0.0)

    C = np.zeros((N, N), dtype=float)
    b = np.asarray(b_vals, dtype=float)
    instance_name = name or "lqr_feas"
    return SDPInstance(C=C, A=Ai, b=b, name=instance_name)


def extract_lqr_gain_from_solution(X: np.ndarray, n: int, m: int) -> np.ndarray:
    """Extract feedback gain K = Z S^{-1} from an LQR-feasibility SDP solution."""
    X = np.asarray(X, dtype=float)
    if X.shape[0] != X.shape[1]:
        raise ValueError("X must be square.")
    if X.shape[0] < (3 * n + m):
        raise ValueError("X is too small for the provided (n, m) block layout.")

    b1 = slice(0, n)
    b2 = slice(n, 2 * n)
    b3 = slice(2 * n, 2 * n + m)

    S = -X[b1, b2]
    S = 0.5 * (S + S.T)
    Z = -X[b1, b3].T

    eigvals = np.linalg.eigvalsh(S)
    min_eig = float(np.min(eigvals))
    if min_eig <= 1e-10:
        S = S + (1e-10 - min_eig + 1e-10) * np.eye(n, dtype=float)

    K = np.linalg.solve(S.T, Z.T).T
    return K


def _add_cross_block_zero_constraints(Ai: List[np.ndarray], b_vals: List[float], n: int) -> None:
    """Enforce block-diagonal structure for a 2n x 2n symmetric matrix variable."""
    dim = 2 * n
    for i in range(n):
        for j in range(n):
            M = np.zeros((dim, dim), dtype=float)
            M[i, n + j] = 0.5
            M[n + j, i] = 0.5
            Ai.append(M)
            b_vals.append(0.0)


def build_h2_norm_sdp(A: np.ndarray, Bw: np.ndarray, Cz: np.ndarray, name: str = "") -> SDPInstance:
    """Construct a standard-form SDP for continuous-time H2 norm minimization.

    Objective: minimize tr(Bw^T P Bw)
    Constraint: A^T P + P A + Cz^T Cz + Q = 0, with P,Q >= 0.
    Variable is block-diagonal X = diag(P, Q) in S^{2n}.
    """
    A = np.asarray(A, dtype=float)
    Bw = np.asarray(Bw, dtype=float)
    Cz = np.asarray(Cz, dtype=float)
    n = A.shape[0]
    if A.shape[1] != n:
        raise ValueError("A must be square.")
    if Bw.shape[0] != n:
        raise ValueError("Bw row dimension must match A.")
    if Cz.shape[1] != n:
        raise ValueError("Cz column dimension must match A.")

    basis = symmetric_basis(n)
    dim = 2 * n
    Ai: List[np.ndarray] = []
    b_vals: List[float] = []
    Czc = Cz.T @ Cz

    for E in basis:
        P_coeff = A @ E + E @ A.T
        Q_coeff = E
        M = np.zeros((dim, dim), dtype=float)
        M[:n, :n] = P_coeff
        M[n:, n:] = Q_coeff
        Ai.append(M)
        b_vals.append(-float(np.trace(E @ Czc)))

    _add_cross_block_zero_constraints(Ai, b_vals, n)

    C = np.zeros((dim, dim), dtype=float)
    C[:n, :n] = Bw @ Bw.T
    b = np.asarray(b_vals, dtype=float)
    instance_name = name or "h2_norm"
    return SDPInstance(C=C, A=Ai, b=b, name=instance_name)


def build_lyapunov_sdp(A: np.ndarray, Bw: np.ndarray, Cz: np.ndarray, name: str = "") -> SDPInstance:
    """Lyapunov certificate SDP with normalization.

    min trace(Cz'Cz P)
    s.t. A'P + PA + Q = 0, trace(P)=1, P,Q >= 0.

    ``Bw`` is accepted for API compatibility with generic instance generation and
    is unused in this formulation.
    """
    A = np.asarray(A, dtype=float)
    Cz = np.asarray(Cz, dtype=float)
    _ = Bw  # API-compatibility parameter; intentionally unused.
    n = A.shape[0]
    if A.shape[1] != n:
        raise ValueError("A must be square.")
    if Cz.shape[1] != n:
        raise ValueError("Cz column dimension must match A.")

    basis = symmetric_basis(n)
    dim = 2 * n
    Ai: List[np.ndarray] = []
    b_vals: List[float] = []

    # Lyapunov equality: A'P + PA + Q = 0
    for E in basis:
        M = np.zeros((dim, dim), dtype=float)
        M[:n, :n] = A @ E + E @ A.T
        M[n:, n:] = E
        Ai.append(M)
        b_vals.append(0.0)

    # Enforce block-diagonal X = diag(P, Q)
    _add_cross_block_zero_constraints(Ai, b_vals, n)

    # Normalization trace(P) = 1
    M_trace = np.zeros((dim, dim), dtype=float)
    M_trace[:n, :n] = np.eye(n, dtype=float)
    Ai.append(M_trace)
    b_vals.append(1.0)

    C = np.zeros((dim, dim), dtype=float)
    C[:n, :n] = Cz.T @ Cz
    b = np.asarray(b_vals, dtype=float)
    instance_name = name or "lyapunov"
    return SDPInstance(C=C, A=Ai, b=b, name=instance_name)


def _build_lyapunov_q_matrix(
    A: np.ndarray,
    n: int,
    q_mode: str = "identity",
    q_scale: float = 1.0,
    custom_diag: np.ndarray | None = None,
) -> np.ndarray:
    """Construct deterministic objective matrix Q for Lyapunov regularization."""
    mode = str(q_mode).strip().lower()
    if mode == "identity":
        # trace(P)=1 makes pure tr(I P) constant; add a tiny deterministic tilt
        # so identity mode still selects a canonical optimizer.
        q_diag = 1.0 + 1e-3 * (np.arange(n, dtype=float) + 1.0) / float(max(n, 1))
    elif mode == "diag_from_a":
        q_diag = 1.0 + np.abs(np.diag(np.asarray(A, dtype=float)))
    elif mode == "custom_diag":
        if custom_diag is None:
            q_diag = np.ones(n, dtype=float)
        else:
            q_diag = np.asarray(custom_diag, dtype=float).reshape(-1)
            if q_diag.shape[0] != n:
                raise ValueError("custom_diag must have length n.")
            q_diag = np.clip(q_diag, 1e-12, None)
    else:
        raise ValueError("q_mode must be one of: identity, diag_from_A, custom_diag.")
    return float(q_scale) * np.diag(q_diag)


def build_lyapunov_regularized_sdp(
    A: np.ndarray,
    Bw: np.ndarray | None = None,
    Cz: np.ndarray | None = None,
    name: str = "",
    Q_mode: str = "identity",
    Q_scale: float = 1.0,
    eps: float = 0.0,
    custom_diag: np.ndarray | None = None,
) -> SDPInstance:
    """Lyapunov certificate SDP with canonical objective and optional strict margin.

    minimize   tr(Q P)
    subject to A'P + P A + Qs = -eps*I, trace(P)=1, P,Qs >= 0.
    """
    A = np.asarray(A, dtype=float)
    _ = Bw  # API compatibility with generic instance generation.
    _ = Cz
    n = A.shape[0]
    if A.shape[1] != n:
        raise ValueError("A must be square.")
    if float(eps) < 0.0:
        raise ValueError("eps must be nonnegative.")

    basis = symmetric_basis(n)
    dim = 2 * n
    Ai: List[np.ndarray] = []
    b_vals: List[float] = []

    # A'P + P A + Qs = -eps I
    for E in basis:
        M = np.zeros((dim, dim), dtype=float)
        M[:n, :n] = A @ E + E @ A.T
        M[n:, n:] = E
        Ai.append(M)
        b_vals.append(-float(eps) * float(np.trace(E)))

    _add_cross_block_zero_constraints(Ai, b_vals, n)

    M_trace = np.zeros((dim, dim), dtype=float)
    M_trace[:n, :n] = np.eye(n, dtype=float)
    Ai.append(M_trace)
    b_vals.append(1.0)

    Q_obj = _build_lyapunov_q_matrix(A, n=n, q_mode=Q_mode, q_scale=Q_scale, custom_diag=custom_diag)
    C = np.zeros((dim, dim), dtype=float)
    C[:n, :n] = Q_obj
    b = np.asarray(b_vals, dtype=float)
    instance_name = name or "lyapunov_reg"
    return SDPInstance(C=C, A=Ai, b=b, name=instance_name)


@dataclass
class LTISystem:
    """Continuous-time LTI system matrices."""

    A: np.ndarray
    Bw: np.ndarray
    Cz: np.ndarray


@dataclass
class L2GainInstance:
    """Holds the data needed for one perturbed SDP instance."""

    index: int
    delta: float
    system: LTISystem
    sdp: SDPInstance
    features: np.ndarray
    graph: Any | None = None
    true_gamma: float | None = None
    baseline_iterations: int | None = None
    warmstart_iterations: int | None = None


def build_system_graph(system: LTISystem):
    """Build a PyG graph from an LTI system.

    This helper is used when warm-start models use a GNN backbone.
    """
    return SystemGraphBuilder().build(system.A, system.Bw, system.Cz)


def generate_l2_gain_instances(
    base_system: LTISystem,
    num_instances: int,
    perturb_range: Tuple[float, float] = (-0.5, 0.5),
    seed: int | None = None,
    feature_mode: str = "A",
    cache_graph: bool = False,
) -> List[L2GainInstance]:
    """Create a list of perturbed SDP instances for the L2 gain study."""
    rng = np.random.default_rng(seed)
    deltas = rng.uniform(perturb_range[0], perturb_range[1], size=num_instances)
    instances: List[L2GainInstance] = []
    Bw = base_system.Bw
    Cz = base_system.Cz

    for idx, delta in enumerate(deltas):
        A_t = base_system.A + delta * np.eye(base_system.A.shape[0])
        system = LTISystem(A=A_t, Bw=Bw, Cz=Cz)
        sdp = build_l2_gain_sdp(A_t, Bw, Cz, name=f"l2_gain_{idx}")

        if feature_mode.upper() == "A":
            features = A_t.reshape(-1).astype(float)
        elif feature_mode.upper() == "ABC":
            features = np.concatenate([A_t.reshape(-1), Bw.reshape(-1), Cz.reshape(-1)]).astype(float)
        else:
            raise ValueError("feature_mode must be 'A' or 'ABC'.")

        graph = None
        if cache_graph:
            try:
                graph = build_system_graph(system)
            except ModuleNotFoundError:
                graph = None

        instances.append(
            L2GainInstance(
                index=idx,
                delta=float(delta),
                system=system,
                sdp=sdp,
                features=features,
                graph=graph,
            )
        )

    return instances


def generate_sdp_instances(
    base_system: LTISystem,
    num: int,
    sdp_builder: Callable[..., SDPInstance],
    perturb_range: Tuple[float, float] = (-0.5, 0.5),
    seed: int | None = None,
    feature_mode: str = "A",
    cache_graph: bool = False,
) -> List[L2GainInstance]:
    """Create perturbed SDP instances for a generic builder.

    The builder is called as ``sdp_builder(A_t, Bw, Cz, name=...)`` when possible,
    with a fallback to ``sdp_builder(A_t, name=...)`` for single-matrix builders.
    """
    rng = np.random.default_rng(seed)
    deltas = rng.uniform(perturb_range[0], perturb_range[1], size=int(num))
    instances: List[L2GainInstance] = []
    Bw = np.asarray(base_system.Bw, dtype=float)
    Cz = np.asarray(base_system.Cz, dtype=float)
    n = base_system.A.shape[0]

    for idx, delta in enumerate(deltas):
        A_t = np.asarray(base_system.A, dtype=float) + float(delta) * np.eye(n, dtype=float)
        system = LTISystem(A=A_t, Bw=Bw, Cz=Cz)

        try:
            sdp = sdp_builder(A_t, Bw, Cz, name=f"sdp_{idx}")
        except TypeError:
            sdp = sdp_builder(A_t, name=f"sdp_{idx}")

        if feature_mode.upper() == "A":
            features = A_t.reshape(-1).astype(float)
        elif feature_mode.upper() == "ABC":
            features = np.concatenate([A_t.reshape(-1), Bw.reshape(-1), Cz.reshape(-1)]).astype(float)
        else:
            raise ValueError("feature_mode must be 'A' or 'ABC'.")

        graph = None
        if cache_graph:
            try:
                graph = build_system_graph(system)
            except ModuleNotFoundError:
                graph = None

        instances.append(
            L2GainInstance(
                index=idx,
                delta=float(delta),
                system=system,
                sdp=sdp,
                features=features,
                graph=graph,
            )
        )

    return instances
