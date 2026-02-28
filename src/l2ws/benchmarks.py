"""Benchmark utilities for L2A/L2WS experiments."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager, redirect_stderr
from dataclasses import dataclass
import io
from typing import Any, Sequence

import numpy as np

try:  # Optional at import time; required for model training/inference.
    import torch
except Exception:  # pragma: no cover - dependency gate
    torch = None  # type: ignore[assignment]

from .data import LTISystem, build_system_graph, generate_l2_gain_instances
from .graph_utils import has_torch_geometric
from .models import GainApproxNet
from .training import TrainerConfig, train_gain_model


_CP_MODULE: Any | None = None
_CP_IMPORT_FAILED = False


def _cvxpy_loggers() -> list[logging.Logger]:
    return [logging.getLogger("__cvxpy__"), logging.getLogger("cvxpy")]


def _get_cvxpy() -> Any | None:
    """Lazy-import cvxpy only when a benchmark path needs it."""
    global _CP_MODULE, _CP_IMPORT_FAILED
    if _CP_MODULE is not None:
        return _CP_MODULE
    if _CP_IMPORT_FAILED:
        return None

    loggers = _cvxpy_loggers()
    prev_levels = [lg.level for lg in loggers]
    try:
        # Suppress optional-backend probe chatter (e.g., GLPK via cvxopt) at import time.
        for lg in loggers:
            lg.setLevel(logging.ERROR)
        # CVXPY configures its own stderr logger at import time; silence only this step.
        with redirect_stderr(io.StringIO()):
            import cvxpy as cp_mod

        _CP_MODULE = cp_mod
        return _CP_MODULE
    except Exception:  # pragma: no cover - dependency gate
        _CP_IMPORT_FAILED = True
        return None
    finally:
        for lg, lvl in zip(loggers, prev_levels):
            lg.setLevel(lvl)


class _CvxpyOptionalBackendFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = str(record.getMessage())
        if "Encountered unexpected exception importing solver GLPK" in msg:
            return False
        if "Encountered unexpected exception importing solver GLPK_MI" in msg:
            return False
        return True


@contextmanager
def _suppress_cvxpy_optional_backend_logs():
    filt = _CvxpyOptionalBackendFilter()
    loggers = _cvxpy_loggers()
    for lg in loggers:
        lg.addFilter(filt)
    try:
        yield
    finally:
        for lg in loggers:
            lg.removeFilter(filt)


def _require_cvxpy() -> Any:
    cp_mod = _get_cvxpy()
    if cp_mod is None:
        raise ModuleNotFoundError("cvxpy is required. Install with: pip install cvxpy")
    return cp_mod


def _require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError("torch is required. Install with: pip install torch")


def normalize_solver(name: str) -> str:
    return str(name).upper()


def installed_solvers() -> set[str]:
    cp = _require_cvxpy()
    with _suppress_cvxpy_optional_backend_logs():
        return {normalize_solver(s) for s in cp.installed_solvers()}


def pick_first_available(candidates: Sequence[str], installed: set[str]) -> str | None:
    for cand in candidates:
        if normalize_solver(cand) in installed:
            return normalize_solver(cand)
    return None


def solver_kwargs(solver_name: str, solver_tol: float, scs_max_iters: int) -> dict[str, Any]:
    """Return solver-specific keyword arguments with aligned tolerances."""
    sname = normalize_solver(solver_name)
    tol = float(solver_tol)
    if sname == "SCS":
        return {"eps": tol, "max_iters": int(scs_max_iters)}
    if sname == "MOSEK":
        return {
            "mosek_params": {
                "MSK_DPAR_INTPNT_CO_TOL_PFEAS": tol,
                "MSK_DPAR_INTPNT_CO_TOL_DFEAS": tol,
                "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": tol,
            }
        }
    return {}


def feature_vector(system: LTISystem, mode: str) -> np.ndarray:
    mode_u = mode.upper()
    if mode_u == "A":
        return np.asarray(system.A, dtype=float).reshape(-1)
    if mode_u == "ABC":
        return np.concatenate(
            [
                np.asarray(system.A, dtype=float).reshape(-1),
                np.asarray(system.Bw, dtype=float).reshape(-1),
                np.asarray(system.Cz, dtype=float).reshape(-1),
            ]
        )
    raise ValueError("feature_mode must be 'A' or 'ABC'.")


def build_cvxpy_opt_problem(system: LTISystem, eps: float):
    cp = _require_cvxpy()

    A = np.asarray(system.A, dtype=float)
    Bw = np.asarray(system.Bw, dtype=float)
    Cz = np.asarray(system.Cz, dtype=float)
    n = A.shape[0]
    p = Bw.shape[1]

    P = cp.Variable((n, n), PSD=True)
    gamma_sq = cp.Variable(nonneg=True)

    top_left = A.T @ P + P @ A + Cz.T @ Cz
    top_right = P @ Bw
    bottom_left = Bw.T @ P
    block = cp.bmat([[top_left, top_right], [bottom_left, -gamma_sq * np.eye(p)]])

    constraints = [
        P >> float(eps) * np.eye(n),
        block << -float(eps) * np.eye(n + p),
    ]
    prob = cp.Problem(cp.Minimize(gamma_sq), constraints)
    return prob, P, gamma_sq


def check_primal_feasibility(
    system: LTISystem,
    P: np.ndarray,
    gamma_sq: float,
    eps: float,
    tol: float,
) -> bool:
    A = np.asarray(system.A, dtype=float)
    Bw = np.asarray(system.Bw, dtype=float)
    Cz = np.asarray(system.Cz, dtype=float)
    n = A.shape[0]
    p = Bw.shape[1]

    P = 0.5 * (P + P.T)
    top_left = A.T @ P + P @ A + Cz.T @ Cz
    top_right = P @ Bw
    bottom_left = Bw.T @ P
    block = np.block([[top_left, top_right], [bottom_left, -float(gamma_sq) * np.eye(p)]])
    block = 0.5 * (block + block.T)

    min_eig_P = float(np.min(np.linalg.eigvalsh(P - float(eps) * np.eye(n))))
    max_eig_block = float(np.max(np.linalg.eigvalsh(block + float(eps) * np.eye(n + p))))
    return (min_eig_P >= -float(tol)) and (max_eig_block <= float(tol))


@dataclass
class SolverRow:
    system_name: str
    n_states: int
    method: str
    value_gamma: float
    time_sec: float
    feasible: bool
    status: str


def solve_gamma_with_solver(
    system: LTISystem,
    solver_name: str,
    eps: float = 1e-6,
    feas_tol: float = 1e-4,
    solver_tol: float = 1e-5,
    scs_max_iters: int = 20000,
) -> SolverRow:
    cp = _require_cvxpy()

    prob, P_var, gamma_sq_var = build_cvxpy_opt_problem(system, eps=eps)
    start = time.perf_counter()
    try:
        kwargs = solver_kwargs(solver_name, solver_tol=solver_tol, scs_max_iters=scs_max_iters)
        with _suppress_cvxpy_optional_backend_logs():
            prob.solve(solver=solver_name, verbose=False, **kwargs)
        elapsed = time.perf_counter() - start
    except Exception as exc:  # noqa: BLE001
        return SolverRow(
            system_name="",
            n_states=int(system.A.shape[0]),
            method=normalize_solver(solver_name),
            value_gamma=float("nan"),
            time_sec=time.perf_counter() - start,
            feasible=False,
            status=f"error: {exc}",
        )

    status = str(prob.status)
    opt_statuses = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
    if status not in opt_statuses or P_var.value is None or gamma_sq_var.value is None:
        return SolverRow(
            system_name="",
            n_states=int(system.A.shape[0]),
            method=normalize_solver(solver_name),
            value_gamma=float("nan"),
            time_sec=elapsed,
            feasible=False,
            status=status,
        )

    gamma_sq = max(float(gamma_sq_var.value), 0.0)
    gamma = float(np.sqrt(gamma_sq))
    feasible = check_primal_feasibility(
        system=system,
        P=np.asarray(P_var.value, dtype=float),
        gamma_sq=gamma_sq,
        eps=eps,
        tol=feas_tol,
    )
    return SolverRow(
        system_name="",
        n_states=int(system.A.shape[0]),
        method=normalize_solver(solver_name),
        value_gamma=gamma,
        time_sec=elapsed,
        feasible=feasible,
        status=status,
    )


def check_gamma_feasible_via_sdp(
    system: LTISystem,
    gamma: float,
    solver_name: str,
    eps: float = 1e-6,
    feas_tol: float = 1e-4,
    solver_tol: float = 1e-5,
    scs_max_iters: int = 20000,
) -> tuple[bool, str]:
    cp = _require_cvxpy()

    A = np.asarray(system.A, dtype=float)
    Bw = np.asarray(system.Bw, dtype=float)
    Cz = np.asarray(system.Cz, dtype=float)
    n = A.shape[0]
    p = Bw.shape[1]
    gamma_sq = max(float(gamma) ** 2, 0.0)

    P = cp.Variable((n, n), PSD=True)
    top_left = A.T @ P + P @ A + Cz.T @ Cz
    top_right = P @ Bw
    bottom_left = Bw.T @ P
    block = cp.bmat([[top_left, top_right], [bottom_left, -gamma_sq * np.eye(p)]])
    constraints = [P >> float(eps) * np.eye(n), block << -float(eps) * np.eye(n + p)]
    prob = cp.Problem(cp.Minimize(0), constraints)

    try:
        kwargs = solver_kwargs(solver_name, solver_tol=solver_tol, scs_max_iters=scs_max_iters)
        with _suppress_cvxpy_optional_backend_logs():
            prob.solve(solver=solver_name, verbose=False, **kwargs)
        status = str(prob.status)
        opt_statuses = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
        if status not in opt_statuses or P.value is None:
            return False, status
        feasible = check_primal_feasibility(
            system=system,
            P=np.asarray(P.value, dtype=float),
            gamma_sq=gamma_sq,
            eps=eps,
            tol=feas_tol,
        )
        return feasible, status
    except Exception as exc:  # noqa: BLE001
        return False, f"error: {exc}"


def solve_tridiagonal_mass_spring_chain(n_states: int) -> LTISystem:
    """Build a deterministic canonical mass-spring-damper chain LTI system."""
    if n_states % 2 != 0 or n_states <= 0:
        raise ValueError("n_states must be a positive even integer.")
    masses = n_states // 2

    k = 2.0
    d = 0.6
    K = 2.0 * k * np.eye(masses)
    D = 2.0 * d * np.eye(masses)
    for i in range(masses - 1):
        K[i, i + 1] = -k
        K[i + 1, i] = -k
        D[i, i + 1] = -0.2 * d
        D[i + 1, i] = -0.2 * d

    A = np.block(
        [
            [np.zeros((masses, masses)), np.eye(masses)],
            [-K, -D],
        ]
    )

    Bw = np.zeros((2 * masses, 1), dtype=float)
    Bw[masses:, 0] = 1.0 / np.sqrt(float(masses))

    Cz = np.zeros((1, 2 * masses), dtype=float)
    Cz[0, :masses] = 1.0 / float(masses)
    Cz[0, masses:] = 0.1 / float(masses)
    return LTISystem(A=A, Bw=Bw, Cz=Cz)


def train_and_predict_l2a_gamma(
    system: LTISystem,
    feature_mode: str = "ABC",
    train_instances: int = 80,
    perturb_min: float = -0.05,
    perturb_max: float = 0.02,
    seed: int = 2,
    label_solver: str = "MOSEK",
    backbone: str = "mlp",
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 3e-4,
    device: str = "cpu",
    lmi_eps: float = 1e-6,
    feas_tol: float = 1e-4,
    solver_tol: float = 1e-5,
    scs_max_iters: int = 20000,
    inference_repeats: int = 1,
) -> tuple[float, float, float, str]:
    """Train L2A and predict gamma for the unperturbed base system.

    Returns:
        gamma_pred, train_time_sec, infer_time_sec, used_backbone
    """
    _require_torch()
    assert torch is not None

    effective_backbone = backbone
    if backbone == "gnn" and not has_torch_geometric():
        effective_backbone = "mlp"

    train_start = time.perf_counter()

    dataset = generate_l2_gain_instances(
        system,
        int(train_instances),
        perturb_range=(float(perturb_min), float(perturb_max)),
        seed=int(seed),
        feature_mode=feature_mode,
        cache_graph=(effective_backbone == "gnn"),
    )

    labels: list[float] = []
    feat_rows: list[np.ndarray] = []
    graph_rows = []

    for inst in dataset:
        row = solve_gamma_with_solver(
            system=inst.system,
            solver_name=label_solver,
            eps=lmi_eps,
            feas_tol=feas_tol,
            solver_tol=solver_tol,
            scs_max_iters=scs_max_iters,
        )
        if not np.isfinite(row.value_gamma):
            continue
        labels.append(float(row.value_gamma))
        feat_rows.append(np.asarray(inst.features, dtype=float))
        if effective_backbone == "gnn":
            graph = inst.graph if inst.graph is not None else build_system_graph(inst.system)
            graph_rows.append(graph)

    if len(labels) < 2:
        raise RuntimeError("Not enough valid labels to train L2A.")

    labels_arr = np.asarray(labels, dtype=float)

    if effective_backbone == "gnn":
        train_x_all = torch.cat([g.x for g in graph_rows], dim=0)
        mean_x = train_x_all.mean(dim=0, keepdim=True)
        std_x = train_x_all.std(dim=0, keepdim=True)
        std_x = torch.where(std_x < 1e-8, torch.ones_like(std_x), std_x)
        for g in graph_rows:
            g.x = (g.x - mean_x) / std_x

        base_graph = build_system_graph(system)
        base_graph.x = (base_graph.x - mean_x) / std_x
        node_feat_dim = int(graph_rows[0].x.shape[1])
        model = GainApproxNet(
            input_dim=None,
            hidden_sizes=(128, 64, 32),
            backbone="gnn",
            node_feat_dim=node_feat_dim,
            dropout=0.1,
        ).to(device)
        train_input = graph_rows
        infer_input = base_graph
    else:
        feat_arr = np.stack(feat_rows, axis=0)
        mean = feat_arr.mean(axis=0)
        std = feat_arr.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        feat_scaled = (feat_arr - mean) / std
        base_feat = feature_vector(system, feature_mode)
        base_feat_scaled = (base_feat - mean) / std

        model = GainApproxNet(
            input_dim=int(feat_scaled.shape[1]),
            hidden_sizes=(128, 64, 32),
            backbone="mlp",
            dropout=0.1,
        ).to(device)
        train_input = feat_scaled
        infer_input = base_feat_scaled

    cfg = TrainerConfig(
        epochs=int(epochs),
        batch_size=int(batch_size),
        learning_rate=float(lr),
        weight_decay=1e-5,
        verbose=False,
    )
    train_gain_model(
        model,
        train_input,
        labels_arr,
        cfg,
        device=device,
        use_graph=(effective_backbone == "gnn"),
    )
    train_time = time.perf_counter() - train_start

    model.eval()
    repeats = max(int(inference_repeats), 1)
    infer_start = time.perf_counter()
    pred_tensor = None
    with torch.no_grad():
        for _ in range(repeats):
            if effective_backbone == "gnn":
                pred_tensor = model(infer_input.to(device))
            else:
                x = torch.as_tensor(infer_input, dtype=torch.float32, device=device).unsqueeze(0)
                pred_tensor = model(x)
    infer_total = time.perf_counter() - infer_start
    infer_time = infer_total / repeats

    if pred_tensor is None:
        raise RuntimeError("Inference failed to produce an output tensor.")
    gamma_pred = float(pred_tensor.squeeze().cpu().item())
    if not np.isfinite(gamma_pred):
        raise RuntimeError("Predicted gamma is NaN/Inf.")
    gamma_pred = max(gamma_pred, 0.0)
    return gamma_pred, float(train_time), float(infer_time), effective_backbone
