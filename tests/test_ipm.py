import numpy as np

from l2ws.ipm import IPMSettings, IPMState, InfeasibleIPMSolver, dF_value, nu_value
from l2ws.problem import SDPInstance


def identity_sdp(scale: float = 10.0) -> SDPInstance:
    A1 = np.array([[1.0, 0.0], [0.0, 0.0]])
    A2 = np.array([[0.0, 0.0], [0.0, 1.0]])
    A3 = np.array([[0.0, 1.0], [1.0, 0.0]])
    b = np.array([1.0, 1.0, 0.0])
    C = scale * np.eye(2)
    return SDPInstance(C=C, A=[A1, A2, A3], b=b, name=f"identity_{scale}")


def test_ipm_converges_identity_problem():
    inst = identity_sdp(scale=5.0)
    solver = InfeasibleIPMSolver(IPMSettings(max_iters=80, tol_abs=1e-6, tol_rel=1e-6))
    result = solver.solve(inst)
    assert result.converged
    assert result.iterations <= 80


def test_initialize_fallback_from_bad_state():
    inst = identity_sdp(scale=3.0)
    solver = InfeasibleIPMSolver(IPMSettings(max_iters=40))

    # Indefinite and non-finite entries should trigger safety fallback.
    X_bad = np.array([[1.0, 0.0], [0.0, -1.0]])
    S_bad = np.array([[np.nan, 0.0], [0.0, 1.0]])
    y0 = np.zeros(inst.num_constraints)

    result = solver.solve(inst, initial_state=IPMState(X_bad, y0, S_bad))
    assert result.converged


def test_dF_zero_for_commuting_scaled_identity_pair():
    X = 2.0 * np.eye(3)
    S = 3.0 * np.eye(3)
    assert abs(dF_value(X, S)) < 1e-12
    assert abs(nu_value(X, S) - 6.0) < 1e-12
