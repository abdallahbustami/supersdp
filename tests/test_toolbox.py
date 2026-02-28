import numpy as np
import pytest
import torch

from l2ws.ipm import IPMSettings, IPMState, InfeasibleIPMSolver, dF_value, in_neighborhood
from l2ws.models import GainApproxNet, WarmStartNet
from l2ws.problem import SDPInstance
from l2ws.graph_utils import SystemGraphBuilder, has_torch_geometric
from l2ws.toolbox import (
    L2CAConfig,
    SuperSDP,
    ProblemConfig,
    SolverConfig,
    TrainingConfig,
    _warmstart_from_cholesky,
    _warmstart_targets_cholesky,
)


def identity_sdp(scale: float = 10.0) -> SDPInstance:
    A1 = np.array([[1.0, 0.0], [0.0, 0.0]])
    A2 = np.array([[0.0, 0.0], [0.0, 1.0]])
    A3 = np.array([[0.0, 1.0], [1.0, 0.0]])
    b = np.array([1.0, 1.0, 0.0])
    C = scale * np.eye(2)
    return SDPInstance(C=C, A=[A1, A2, A3], b=b, name=f"identity_{scale}")


def identity_solution(scale: float = 10.0):
    X = np.eye(2)
    y = np.zeros(3)
    S = scale * np.eye(2)
    return X, y, S


def _parse_target_vector(target: np.ndarray, n: int):
    n_tril = n * (n + 1) // 2
    L_flat = target[:n_tril]
    nu = target[n_tril]

    L = np.zeros((n, n), dtype=float)
    tri = np.tril_indices(n)
    L[tri] = L_flat
    return L, float(nu)


def l2ca_train_data():
    instances = [identity_sdp(scale=s) for s in (3.0, 4.0, 5.0, 6.0)]
    solutions = [identity_solution(scale=s) for s in (3.0, 4.0, 5.0, 6.0)]
    return instances, solutions


def test_config_validation():
    with pytest.raises(ValueError):
        ProblemConfig(n=0, m=1)
    with pytest.raises(ValueError):
        SolverConfig(mode="bad")
    with pytest.raises(ValueError):
        SolverConfig(warmstart_type="bad")
    with pytest.raises(ValueError):
        TrainingConfig(dropout=1.5)
    with pytest.raises(ValueError):
        TrainingConfig(backbone="bad")
    with pytest.raises(ValueError):
        L2CAConfig(anchor_mode="bad")
    with pytest.raises(ValueError):
        L2CAConfig(tier0_fallback="bad")
    with pytest.raises(ValueError):
        L2CAConfig(feas_loss_mode="bad")
    with pytest.raises(ValueError):
        L2CAConfig(feas_margin=-1.0)


def test_solver_config_accepts_l2ca():
    cfg = SolverConfig(mode="L2CA", warmstart_type="cholesky", include_gain_approx=True)
    assert cfg.mode == "L2CA"
    assert cfg.warmstart_type == "none"
    assert not cfg.include_gain_approx


def test_auto_labeling_with_backend():
    try:
        import cvxpy as cp
    except Exception:
        pytest.skip("cvxpy not available")

    available = {solver.lower() for solver in cp.installed_solvers()}
    if "scs" not in available:
        pytest.skip("SCS not available")

    inst = identity_sdp(scale=2.0)
    solver = SuperSDP(
        problem_config=ProblemConfig(n=2, m=3),
        training_config=TrainingConfig(epochs=1, batch_size=1, hidden_dims=(8,)),
        solver_config=SolverConfig(mode="L2WS", backend="scs", warmstart_type="cholesky"),
    )
    solver.fit([inst], solutions=None)
    assert solver._trained


def test_l2a_returns_no_ipm():
    torch.manual_seed(0)
    inst = identity_sdp(scale=3.0)
    sol = identity_solution(scale=3.0)
    solver = SuperSDP(
        problem_config=ProblemConfig(n=2, m=3),
        training_config=TrainingConfig(epochs=2, batch_size=1, hidden_dims=(8,)),
        solver_config=SolverConfig(mode="L2A"),
    )
    solver.fit([inst], solutions=[sol])
    result = solver.solve(inst)
    assert result.mode_used == "L2A"
    assert result.iterations == 0
    assert result.converged


def test_cholesky_warmstart_valid_neighborhood():
    """Verify that X=LL^T, S=nu*X^{-1} always satisfies d_F=0."""
    n = 5
    rng = np.random.default_rng(42)
    L = np.tril(rng.standard_normal((n, n)))
    np.fill_diagonal(L, np.abs(np.diag(L)) + 0.1)
    nu_scalar = 2.5
    X = L @ L.T
    S = nu_scalar * np.linalg.solve(X, np.eye(n))
    assert dF_value(X, S) < 1e-10
    assert in_neighborhood(X, S, gamma=0.01)


def test_cholesky_warmstart_ipm_convergence():
    """Verify IPM converges from a Cholesky warm-start."""
    inst = identity_sdp(scale=10.0)
    n = inst.dim
    rng = np.random.default_rng(0)
    L = np.tril(rng.standard_normal((n, n)))
    np.fill_diagonal(L, np.abs(np.diag(L)) + 0.5)
    X0 = L @ L.T
    nu0 = 1.0
    S0 = nu0 * np.linalg.solve(X0, np.eye(n))
    y0 = np.zeros(inst.num_constraints)
    state = IPMState(X0, y0, S0)
    solver = InfeasibleIPMSolver(IPMSettings())
    result = solver.solve(inst, initial_state=state)
    assert result.converged


def test_warmstart_targets_cholesky_roundtrip():
    """Verify target extraction produces valid interiorized Cholesky factors."""
    n = 3
    m = 2
    rng = np.random.default_rng(7)
    L_true = np.tril(rng.standard_normal((n, n)))
    np.fill_diagonal(L_true, np.abs(np.diag(L_true)) + 1.0)
    X_true = L_true @ L_true.T
    nu_true = 1.7
    S_true = nu_true * np.linalg.solve(X_true, np.eye(n))
    y_true = rng.standard_normal(m)

    target = _warmstart_targets_cholesky((X_true, y_true, S_true), n=n, m=m)
    L_rec, nu_rec = _parse_target_vector(target, n=n)

    state = _warmstart_from_cholesky(L_rec, nu_rec, y_true)
    interior_shift = 1e-6
    X_int = X_true + interior_shift * np.eye(n)
    S_int = S_true + interior_shift * np.eye(n)
    nu_int = float(np.trace(X_int @ S_int) / n)

    assert np.allclose(state.X, X_int, atol=1e-6)
    assert np.isclose(nu_rec, nu_int, atol=1e-10)
    assert dF_value(state.X, state.S) < 1e-8


def test_gnn_backbone_fallback_to_mlp(monkeypatch):
    """If torch_geometric is not installed, WarmStartNet should fall back to MLP."""
    import l2ws.models as models_mod

    monkeypatch.setattr(models_mod, "has_torch_geometric", lambda: False)
    with pytest.warns(RuntimeWarning):
        model = WarmStartNet(n=2, m=3, input_dim=4, backbone="gnn", node_feat_dim=3)
    assert model.backbone_name == "mlp"


def test_solver_gnn_request_falls_back_without_pyg(monkeypatch):
    """Solver fit/solve should use MLP fallback when GNN is requested without PyG."""
    import l2ws.models as models_mod
    import l2ws.toolbox as toolbox_mod

    monkeypatch.setattr(models_mod, "has_torch_geometric", lambda: False)
    monkeypatch.setattr(toolbox_mod, "has_torch_geometric", lambda: False)

    inst = identity_sdp(scale=5.0)
    sol = identity_solution(scale=5.0)

    solver = SuperSDP(
        problem_config=ProblemConfig(n=2, m=3),
        training_config=TrainingConfig(
            epochs=2,
            batch_size=1,
            hidden_dims=(8,),
            backbone="gnn",
        ),
        solver_config=SolverConfig(mode="L2WS", warmstart_type="cholesky"),
    )
    with pytest.warns(RuntimeWarning):
        solver.fit([inst], solutions=[sol])

    assert isinstance(solver._model, WarmStartNet)
    assert solver._model.backbone_name == "mlp"
    result = solver.solve(inst)
    assert result.converged


def test_solver_l2a_gnn_request_falls_back_without_pyg(monkeypatch):
    import l2ws.models as models_mod
    import l2ws.toolbox as toolbox_mod

    monkeypatch.setattr(models_mod, "has_torch_geometric", lambda: False)
    monkeypatch.setattr(toolbox_mod, "has_torch_geometric", lambda: False)

    inst = identity_sdp(scale=5.0)
    sol = identity_solution(scale=5.0)

    solver = SuperSDP(
        problem_config=ProblemConfig(n=2, m=3),
        training_config=TrainingConfig(
            epochs=2,
            batch_size=1,
            hidden_dims=(8,),
            backbone="gnn",
        ),
        solver_config=SolverConfig(mode="L2A"),
    )
    with pytest.warns(RuntimeWarning):
        solver.fit([inst], solutions=[sol])

    assert isinstance(solver._model, GainApproxNet)
    assert solver._model.backbone_name == "mlp"
    result = solver.solve(inst)
    assert result.mode_used == "L2A"
    assert result.converged


@pytest.mark.skipif(not has_torch_geometric(), reason="torch_geometric not available")
def test_solver_l2a_gnn_fit_and_solve_with_graphs():
    inst = identity_sdp(scale=4.0)
    sol = identity_solution(scale=4.0)
    graph = SystemGraphBuilder().build(
        A=np.array([[-1.0, 0.2], [0.1, -2.0]]),
        Bw=np.array([[1.0], [0.0]]),
        Cz=np.array([[1.0, 2.0]]),
    )

    solver = SuperSDP(
        problem_config=ProblemConfig(n=2, m=3),
        training_config=TrainingConfig(
            epochs=2,
            batch_size=1,
            hidden_dims=(8,),
            backbone="gnn",
        ),
        solver_config=SolverConfig(mode="L2A"),
    )
    solver.fit([inst], solutions=[sol], graphs=[graph])
    assert isinstance(solver._model, GainApproxNet)
    assert solver._model.backbone_name == "gnn"

    result = solver.solve(inst, graph=graph)
    assert result.mode_used == "L2A"
    assert result.converged


def test_l2ws_cholesky_fewer_iterations_than_identity():
    """A good Cholesky warm-start should not be worse than cold-start identity."""
    inst = identity_sdp(scale=10.0)
    solver = InfeasibleIPMSolver(IPMSettings())

    X_star, y_star, S_star = identity_solution(scale=10.0)
    L = np.linalg.cholesky(X_star)
    nu = float(np.trace(X_star @ S_star) / X_star.shape[0])
    warm_state = _warmstart_from_cholesky(L, nu, y_star)

    warm_res = solver.solve(inst, initial_state=warm_state)
    cold_res = solver.solve(inst)

    assert warm_res.converged
    assert cold_res.converged
    assert warm_res.iterations <= cold_res.iterations


def test_lifelong_update_improves_prediction_signal():
    torch.manual_seed(0)
    inst1 = identity_sdp(scale=4.0)
    inst2 = identity_sdp(scale=8.0)
    sol1 = identity_solution(scale=4.0)
    sol2 = identity_solution(scale=8.0)

    solver = SuperSDP(
        problem_config=ProblemConfig(n=2, m=3),
        training_config=TrainingConfig(epochs=10, batch_size=2, hidden_dims=(8,), dropout=0.0),
        solver_config=SolverConfig(mode="L2WS", lifelong=True, lifelong_strategy="retrain"),
    )
    solver.fit([inst1], solutions=[sol1])

    L_before, nu_before = solver._predict_warmstart(inst2)
    X_before = L_before @ L_before.T
    err_before = abs(np.trace(X_before) / 2.0 - 1.0) + abs(nu_before - 8.0)

    solver.update([inst2], [sol2])
    L_after, nu_after = solver._predict_warmstart(inst2)
    X_after = L_after @ L_after.T
    err_after = abs(np.trace(X_after) / 2.0 - 1.0) + abs(nu_after - 8.0)

    assert err_after <= err_before + 1e-4


def test_save_load_roundtrip(tmp_path):
    torch.manual_seed(0)
    inst = identity_sdp(scale=6.0)
    sol = identity_solution(scale=6.0)
    solver = SuperSDP(
        problem_config=ProblemConfig(n=2, m=3),
        training_config=TrainingConfig(epochs=2, batch_size=1, hidden_dims=(8,)),
        solver_config=SolverConfig(mode="L2WS", warmstart_type="cholesky"),
    )
    solver.fit([inst], solutions=[sol])
    L_before, nu_before = solver._predict_warmstart(inst)

    path = tmp_path / "solver.pt"
    solver.save(str(path))
    loaded = SuperSDP.load(str(path))
    L_after, nu_after = loaded._predict_warmstart(inst)

    assert np.allclose(L_before, L_after, atol=1e-4)
    assert np.isclose(nu_before, nu_after, atol=1e-4)


def test_batch_solving():
    torch.manual_seed(0)
    inst = identity_sdp(scale=7.0)
    sol = identity_solution(scale=7.0)
    solver = SuperSDP(
        problem_config=ProblemConfig(n=2, m=3),
        training_config=TrainingConfig(epochs=2, batch_size=1, hidden_dims=(8,)),
        solver_config=SolverConfig(mode="L2A"),
    )
    solver.fit([inst], solutions=[sol])
    results = solver.solve_batch([inst, inst])
    assert len(results) == 2
    assert all(r.mode_used == "L2A" for r in results)


def test_l2ca_fit_trains_dual_model():
    torch.manual_seed(0)
    instances, solutions = l2ca_train_data()
    solver = SuperSDP(
        problem_config=ProblemConfig(n=2, m=3),
        training_config=TrainingConfig(epochs=2, batch_size=2, hidden_dims=(8,)),
        solver_config=SolverConfig(mode="L2CA"),
        l2ca_config=L2CAConfig(),
    )
    solver.fit(instances, solutions=solutions)

    assert solver.dual_model is not None
    assert solver._model is solver.dual_model
    assert solver.warm_model is None
    assert solver.gain_model is None
    assert solver._l2ca_x_train_norm is not None
    assert solver._l2ca_y_train is not None
    assert solver._l2ca_robust_anchor is not None


def test_l2ca_solve_returns_extended_fields():
    torch.manual_seed(0)
    instances, solutions = l2ca_train_data()
    solver = SuperSDP(
        problem_config=ProblemConfig(n=2, m=3),
        training_config=TrainingConfig(epochs=2, batch_size=2, hidden_dims=(8,)),
        solver_config=SolverConfig(mode="L2CA"),
        l2ca_config=L2CAConfig(),
    )
    solver.fit(instances, solutions=solutions)
    result = solver.solve(identity_sdp(scale=6.0))

    assert result.mode_used == "L2CA"
    assert result.y is not None
    assert result.S is not None
    assert result.dual_feasible is not None
    assert result.fast_path_accept is not None
    assert result.repair_ok is not None
    assert result.anchor_info is not None
    assert result.tier_level is not None
    assert result.dual_obj is not None


def test_l2ca_save_load_roundtrip(tmp_path):
    torch.manual_seed(0)
    instances, solutions = l2ca_train_data()
    solver = SuperSDP(
        problem_config=ProblemConfig(n=2, m=3),
        training_config=TrainingConfig(epochs=2, batch_size=2, hidden_dims=(8,)),
        solver_config=SolverConfig(mode="L2CA"),
        l2ca_config=L2CAConfig(),
    )
    solver.fit(instances, solutions=solutions)

    path = tmp_path / "solver_l2ca.pt"
    solver.save(str(path))
    loaded = SuperSDP.load(str(path))
    result = loaded.solve(identity_sdp(scale=6.5))

    assert loaded.dual_model is not None
    assert result.mode_used == "L2CA"
    assert result.dual_obj is not None


def test_l2ca_solve_batch():
    torch.manual_seed(0)
    instances, solutions = l2ca_train_data()
    solver = SuperSDP(
        problem_config=ProblemConfig(n=2, m=3),
        training_config=TrainingConfig(epochs=2, batch_size=2, hidden_dims=(8,)),
        solver_config=SolverConfig(mode="L2CA"),
        l2ca_config=L2CAConfig(),
    )
    solver.fit(instances, solutions=solutions)
    results = solver.solve_batch([identity_sdp(scale=6.0), identity_sdp(scale=7.0)])

    assert len(results) == 2
    assert all(r.mode_used == "L2CA" for r in results)


def test_l2ca_lifelong_update():
    torch.manual_seed(0)
    instances, solutions = l2ca_train_data()
    solver = SuperSDP(
        problem_config=ProblemConfig(n=2, m=3),
        training_config=TrainingConfig(epochs=2, batch_size=2, hidden_dims=(8,)),
        solver_config=SolverConfig(mode="L2CA", lifelong=True, lifelong_strategy="retrain"),
        l2ca_config=L2CAConfig(),
    )
    solver.fit(instances[:2], solutions=solutions[:2])
    solver.update(instances[2:], solutions[2:])
    result = solver.solve(identity_sdp(scale=8.0))

    assert solver._trained
    assert solver.dual_model is not None
    assert result.mode_used == "L2CA"


def test_public_import_smoke():
    import l2ws
    from l2ws import L2CAConfig as PublicL2CAConfig
    from l2ws import SuperSDP as PublicSolver

    assert l2ws is not None
    assert PublicSolver is SuperSDP
    assert PublicL2CAConfig is L2CAConfig


def test_invalid_inputs():
    inst = identity_sdp(scale=2.0)
    sol = identity_solution(scale=2.0)
    solver = SuperSDP(
        problem_config=ProblemConfig(n=2, m=3),
        training_config=TrainingConfig(epochs=1, batch_size=1, hidden_dims=(8,)),
        solver_config=SolverConfig(mode="L2WS"),
    )
    solver.fit([inst], solutions=[sol])

    bad_inst = SDPInstance(C=np.eye(3), A=[np.eye(3)], b=np.array([1.0]))
    with pytest.raises(ValueError):
        solver.solve(bad_inst)

    with pytest.raises(RuntimeError):
        solver.update([inst], [sol])
