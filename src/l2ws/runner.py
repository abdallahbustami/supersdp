"""User-facing experiment runner for the L2WS toolbox.

This module intentionally does not depend on ``case_studies.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager, redirect_stderr
import logging
import time
import io
from typing import Any, Callable, Dict, List, Optional, Sequence
import warnings

import numpy as np
import torch

from .applications import ensure_system_io, get_application_spec
from .benchmarks import installed_solvers, solve_gamma_with_solver
from .data import L2GainInstance, LTISystem
from .ipm import IPMResult, IPMSettings, InfeasibleIPMSolver
from .l2a import ScalarL2ANet, train_scalar_l2a
from .l2ca import (
    DualNet,
    detect_tier,
    extract_gamma_from_dual,
    run_l2ca_inference,
    select_robust_anchor,
    train_dual_net,
)
from .l2ws import ProblemConfig, SolverConfig, SuperSDP, TrainingConfig
from .problem import SDPInstance
from .perturbations import PerturbationSpec, perturb_system

_CP_MODULE: Any | None = None
_CP_IMPORT_FAILED = False


def _cvxpy_loggers() -> list[logging.Logger]:
    return [logging.getLogger("__cvxpy__"), logging.getLogger("cvxpy")]


def _get_cvxpy() -> Any | None:
    """Lazy-import cvxpy only if MOSEK/SCS paths are requested."""
    global _CP_MODULE, _CP_IMPORT_FAILED
    if _CP_MODULE is not None:
        return _CP_MODULE
    if _CP_IMPORT_FAILED:
        return None

    loggers = _cvxpy_loggers()
    prev_levels = [lg.level for lg in loggers]
    try:
        # Avoid noisy optional-backend probe warnings during import.
        for lg in loggers:
            lg.setLevel(logging.ERROR)
        # CVXPY attaches a stderr handler on import; silence only this import step.
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


def _require_cvxpy() -> Any:
    cp_mod = _get_cvxpy()
    if cp_mod is None:
        raise RuntimeError("cvxpy is required for MOSEK/SCS comparisons.")
    return cp_mod


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


@dataclass
class ExperimentConfig:
    application: str
    num_train: int = 50
    num_test: int = 20
    perturb: object = None
    seed: int = 0

    epochs: int = 50
    batch_size: int = 64
    hidden_sizes: tuple = (128, 64, 32)
    lr: float = 3e-4

    l2ca_anchor: str = "knn5_best"
    l2ca_feas_margin: float = 1e-8
    l2ca_bisect_iters: int = 20
    l2ca_tier_auto: str = "on"
    l2ca_tier0_fallback: str = "off"
    l2ca_lambda_obj: float = 0.0
    l2ca_lambda_feas: float = 0.0
    l2ca_feas_loss: str = "shift"
    l2ca_feas_margin_train: float = 0.0
    l2ca_label_margin_debug: bool = False
    l2ca_interiorize_labels: bool = True
    l2ca_interior_delta: float = 1e-3
    l2ca_lambda_y: float = 1.0

    cert_regularize: str = "auto"
    cert_eps: float = 1e-6
    cert_Q: str = "identity"
    cert_Q_scale: float = 1.0

    prestabilize: str = "off"
    prestabilize_margin: float = 1e-3

    algorithms: Sequence[str] = ("IPM", "L2WS", "L2A", "L2CA")


@dataclass
class AlgorithmResult:
    feas_rate: float
    mean_rel_err: Optional[float]
    std_rel_err: Optional[float]
    max_rel_err: Optional[float]
    mean_time_ms: float


@dataclass
class ExperimentResult:
    app: str
    system_name: str
    n: int
    sdp_dim: int
    results: Dict[str, AlgorithmResult]


_STABILITY_WARNED: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key in _STABILITY_WARNED:
        return
    _STABILITY_WARNED.add(key)
    warnings.warn(message, RuntimeWarning, stacklevel=3)


def _normalize_algorithms(algs: Sequence[str]) -> List[str]:
    aliases = {
        "mosek": "MOSEK",
        "scs": "SCS",
        "ipm": "IPM",
        "l2ws": "L2WS",
        "l2a": "L2A",
        "l2ca": "L2CA",
    }
    out: List[str] = []
    for a in algs:
        key = str(a).strip().lower()
        if key not in aliases:
            raise ValueError(f"Unknown algorithm '{a}'.")
        canon = aliases[key]
        if canon not in out:
            out.append(canon)
    return out


def _clone_system(system: LTISystem) -> LTISystem:
    A = np.asarray(system.A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("system.A must be square.")
    n = A.shape[0]
    bw_raw = getattr(system, "Bw", None)
    cz_raw = getattr(system, "Cz", None)
    bw = np.asarray(bw_raw, dtype=float) if bw_raw is not None and np.asarray(bw_raw).size > 0 else np.eye(n, dtype=float)
    cz = np.asarray(cz_raw, dtype=float) if cz_raw is not None and np.asarray(cz_raw).size > 0 else np.eye(n, dtype=float)
    return LTISystem(
        A=A.copy(),
        Bw=bw.copy(),
        Cz=cz.copy(),
    )


def _max_real_part(A: np.ndarray) -> float:
    eigs = np.linalg.eigvals(np.asarray(A, dtype=float))
    return float(np.max(np.real(eigs)))


def _apply_stability_policy(
    system: LTISystem,
    app_key: str,
    prestabilize: str,
    margin: float,
    warn_token: str,
) -> tuple[LTISystem | None, bool]:
    """Enforce stability assumptions for gain/norm applications."""
    mode = str(prestabilize).strip().lower()
    if mode not in {"off", "shift"}:
        raise ValueError("config.prestabilize must be one of {'off', 'shift'}.")

    if app_key not in {"l2_gain", "h2_norm", "hinf_norm"}:
        return system, True

    A = np.asarray(system.A, dtype=float)
    max_real = _max_real_part(A)
    if max_real < 0.0:
        return system, True

    if mode == "off":
        _warn_once(
            f"{warn_token}:{app_key}:skip_unstable",
            (
                f"{app_key}: unstable A encountered (max Re(lambda)={max_real:.3e}) "
                "and prestabilize='off'; skipping this instance."
            ),
        )
        return None, False

    shift = max_real + float(margin)
    A_shift = A - shift * np.eye(A.shape[0], dtype=float)
    _warn_once(
        f"{warn_token}:{app_key}:shift_unstable",
        (
            f"{app_key}: unstable A encountered (max Re(lambda)={max_real:.3e}); "
            f"applied prestabilization shift {shift:.3e}."
        ),
    )
    return LTISystem(A=A_shift, Bw=np.asarray(system.Bw, dtype=float), Cz=np.asarray(system.Cz, dtype=float)), True


def _build_instance_sdp(
    builder: Callable[..., SDPInstance],
    system: LTISystem,
    idx: int,
    builder_kwargs: Dict[str, object],
) -> SDPInstance:
    name = f"sdp_{idx}"
    try:
        return builder(system, name=name, **builder_kwargs)
    except TypeError:
        try:
            return builder(system, **builder_kwargs)
        except TypeError:
            return builder(system)


def _build_instances(
    base_system: LTISystem,
    num: int,
    builder: Callable[..., SDPInstance],
    feature_fn: Callable[[LTISystem], np.ndarray],
    perturb: object,
    seed: int,
    builder_kwargs: Dict[str, object],
    app_key: str,
    prestabilize: str,
    prestabilize_margin: float,
    system_name: str,
) -> List[L2GainInstance]:
    instances: List[L2GainInstance] = []

    if perturb is None:
        spec = PerturbationSpec(kind="none", seed=seed)
    elif isinstance(perturb, PerturbationSpec):
        spec = perturb
    else:
        spec = None

    low_high: tuple[float, float] | None = None
    if spec is None:
        if isinstance(perturb, (tuple, list)) and len(perturb) == 2:
            low_high = (float(perturb[0]), float(perturb[1]))
        else:
            raise ValueError("config.perturb must be None, PerturbationSpec, or a (low, high) tuple.")

    target = int(num)
    max_attempts = max(10 * target, target + 10)
    cand_idx = 0

    while len(instances) < target and cand_idx < max_attempts:
        idx = cand_idx
        if spec is not None:
            sys_i = perturb_system(base_system, spec, idx)
        else:
            assert low_high is not None
            rng = np.random.default_rng(int(seed) + int(idx))
            A = np.asarray(base_system.A, dtype=float)
            Bw = np.asarray(base_system.Bw, dtype=float)
            Cz = np.asarray(base_system.Cz, dtype=float)
            dA = rng.uniform(low_high[0], low_high[1], size=A.shape)
            dB = rng.uniform(low_high[0], low_high[1], size=Bw.shape)
            sys_i = LTISystem(A=A + dA, Bw=Bw + dB, Cz=Cz.copy())

        sys_i, keep = _apply_stability_policy(
            sys_i,
            app_key=app_key,
            prestabilize=prestabilize,
            margin=float(prestabilize_margin),
            warn_token=system_name,
        )
        cand_idx += 1
        if not keep or sys_i is None:
            continue

        sdp = _build_instance_sdp(builder, sys_i, idx=idx, builder_kwargs=builder_kwargs)
        feat = np.asarray(feature_fn(sys_i), dtype=float).reshape(-1)

        instances.append(
            L2GainInstance(
                index=idx,
                delta=0.0,
                system=sys_i,
                sdp=sdp,
                features=feat,
            )
        )

    if len(instances) < target:
        _warn_once(
            f"{system_name}:{app_key}:insufficient_instances",
            (
                f"{system_name}: generated {len(instances)} instances out of requested {target}; "
                "consider prestabilize='shift' or reducing perturbation magnitude."
            ),
        )

    if not instances:
        return instances

    d_feat = instances[0].features.shape[0]
    for inst in instances:
        if inst.features.shape[0] != d_feat:
            raise ValueError(
                "Feature dimensionality mismatch across generated instances. "
                f"Expected {d_feat}, got {inst.features.shape[0]} for idx={inst.index}."
            )
    return instances


def _objective_from_ipm_result(result: IPMResult, instance: L2GainInstance, app_key: str) -> float:
    if app_key in {"l2_gain", "hinf_norm"}:
        return extract_gamma_from_dual(result.y, instance.sdp.b)
    return float(np.trace(instance.sdp.C @ result.X))


def _solve_h2_with_solver(system: LTISystem, solver_name: str, eps: float = 1e-6) -> tuple[float, bool, float]:
    cp = _require_cvxpy()
    A = np.asarray(system.A, dtype=float)
    Bw = np.asarray(system.Bw, dtype=float)
    Cz = np.asarray(system.Cz, dtype=float)
    n = A.shape[0]
    P = cp.Variable((n, n), PSD=True)
    obj = cp.trace(Bw.T @ P @ Bw)
    constraints = [A.T @ P + P @ A + Cz.T @ Cz << -float(eps) * np.eye(n)]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    t0 = time.perf_counter()
    with _suppress_cvxpy_optional_backend_logs():
        prob.solve(solver=solver_name, verbose=False)
    t_ms = (time.perf_counter() - t0) * 1e3
    ok = prob.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
    value = float(prob.value) if ok and prob.value is not None else float("nan")
    return value, ok, t_ms


def _solve_lyapunov_reg_with_solver(
    system: LTISystem,
    solver_name: str,
    Q_obj: np.ndarray,
    eps: float,
) -> tuple[float, bool, float]:
    cp = _require_cvxpy()
    A = np.asarray(system.A, dtype=float)
    n = A.shape[0]
    Q_obj = 0.5 * (np.asarray(Q_obj, dtype=float) + np.asarray(Q_obj, dtype=float).T)
    P = cp.Variable((n, n), PSD=True)
    constraints = [A.T @ P + P @ A << -float(eps) * np.eye(n), cp.trace(P) == 1.0]
    prob = cp.Problem(cp.Minimize(cp.trace(Q_obj @ P)), constraints)
    t0 = time.perf_counter()
    with _suppress_cvxpy_optional_backend_logs():
        prob.solve(solver=solver_name, verbose=False)
    t_ms = (time.perf_counter() - t0) * 1e3
    ok = prob.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
    value = float(prob.value) if ok and prob.value is not None else float("nan")
    return value, ok, t_ms


def _summarize_row(
    values: Sequence[float],
    feas: Optional[Sequence[bool]],
    times_ms: Sequence[float],
    ref_values: Optional[np.ndarray],
) -> AlgorithmResult:
    vals = np.asarray(values, dtype=float)
    tms = np.asarray(times_ms, dtype=float)

    if feas is None:
        feas_rate = float("nan")
    else:
        feas_arr = np.asarray(feas, dtype=bool)
        feas_rate = float(np.mean(feas_arr.astype(float))) if feas_arr.size else float("nan")

    finite_times = tms[np.isfinite(tms)]
    mean_time_ms = float(np.mean(finite_times)) if finite_times.size else float("nan")

    if ref_values is None:
        return AlgorithmResult(
            feas_rate=feas_rate,
            mean_rel_err=None,
            std_rel_err=None,
            max_rel_err=None,
            mean_time_ms=mean_time_ms,
        )

    ref = np.asarray(ref_values, dtype=float)
    mask = np.isfinite(vals) & np.isfinite(ref)
    if not np.any(mask):
        return AlgorithmResult(
            feas_rate=feas_rate,
            mean_rel_err=None,
            std_rel_err=None,
            max_rel_err=None,
            mean_time_ms=mean_time_ms,
        )

    rel = np.abs(vals[mask] - ref[mask]) / np.maximum(np.abs(ref[mask]), 1e-12)
    return AlgorithmResult(
        feas_rate=feas_rate,
        mean_rel_err=float(np.mean(rel)),
        std_rel_err=float(np.std(rel)),
        max_rel_err=float(np.max(rel)),
        mean_time_ms=mean_time_ms,
    )


def _print_table(
    app: str,
    system_name: str,
    n_states: int,
    sdp_dim: int,
    ordered_methods: Sequence[str],
    results: Dict[str, AlgorithmResult],
) -> None:
    print("")
    print(f"=== Toolbox — {app} — {system_name} (n={n_states}, SDP dim={sdp_dim}) ===")
    print(
        f"{'Algorithm':<10} | {'Feas %':>6} | {'Mean Err ± Std':>18} | {'Max Err':>8} | {'Mean Time':>9}"
    )
    print("-" * 70)
    for m in ordered_methods:
        if m not in results:
            continue
        row = results[m]
        feas_txt = "--" if not np.isfinite(row.feas_rate) else f"{100.0 * row.feas_rate:6.1f}"
        if row.mean_rel_err is None:
            err_txt = "--"
            max_txt = "--"
        else:
            err_txt = f"{100.0 * row.mean_rel_err:.3f}% ± {100.0 * row.std_rel_err:.3f}%"
            max_txt = f"{100.0 * row.max_rel_err:.3f}%"
        time_txt = "nan" if not np.isfinite(row.mean_time_ms) else f"{row.mean_time_ms:.3f}ms"
        print(f"{m:<10} | {feas_txt:>6} | {err_txt:>18} | {max_txt:>8} | {time_txt:>9}")


def run_experiment_on_system(
    system: LTISystem,
    system_name: str,
    config: ExperimentConfig,
    custom_sdp_builder: Optional[Callable[..., SDPInstance]] = None,
    custom_feature_fn: Optional[Callable[[LTISystem], np.ndarray]] = None,
) -> ExperimentResult:
    """Run one full train/eval experiment on a single system."""
    app_key = str(config.application).strip().lower()
    is_certificate_app = app_key in {"lyapunov_reg"}
    app_spec = get_application_spec(app_key)
    algorithms = _normalize_algorithms(config.algorithms)
    prestabilize_mode = str(config.prestabilize).strip().lower()
    if prestabilize_mode not in {"off", "shift"}:
        raise ValueError("config.prestabilize must be one of {'off', 'shift'}.")
    if float(config.prestabilize_margin) <= 0.0:
        raise ValueError("config.prestabilize_margin must be positive.")

    builder = custom_sdp_builder if custom_sdp_builder is not None else app_spec.builder
    feature_fn = custom_feature_fn if custom_feature_fn is not None else app_spec.feature_fn
    base_system = ensure_system_io(_clone_system(system), app_key, warn_token=system_name)

    builder_kwargs: Dict[str, object] = {}
    if app_key == "lyapunov_reg":
        builder_kwargs = {
            "cert_Q": config.cert_Q,
            "cert_Q_scale": float(config.cert_Q_scale),
            "cert_eps": float(config.cert_eps),
        }
    builder_kwargs["_warn_token"] = str(system_name)

    instances = _build_instances(
        base_system=base_system,
        num=int(config.num_train) + int(config.num_test),
        builder=builder,
        feature_fn=feature_fn,
        perturb=config.perturb,
        seed=int(config.seed),
        builder_kwargs=builder_kwargs,
        app_key=app_key,
        prestabilize=prestabilize_mode,
        prestabilize_margin=float(config.prestabilize_margin),
        system_name=str(system_name),
    )
    train_instances = instances[: int(config.num_train)]
    test_instances = instances[int(config.num_train) :]
    if not train_instances or not test_instances:
        raise ValueError("Empty train/test split. Check num_train and num_test.")

    # --- cold IPM solves used for labels and baseline reference ---
    ipm_solver = InfeasibleIPMSolver(IPMSettings(max_iters=120, tol_abs=1e-6, tol_rel=1e-5, linear_solve="sylvester"))

    need_train_ipm = any(a in algorithms for a in {"L2WS", "L2A", "L2CA"})
    train_ipm_results: List[IPMResult] = []
    if need_train_ipm:
        for inst in train_instances:
            train_ipm_results.append(ipm_solver.solve(inst.sdp))

    need_test_ipm = ("IPM" in algorithms) or ("MOSEK" not in algorithms and "SCS" not in algorithms)
    ipm_values: List[float] = []
    ipm_feas: List[bool] = []
    ipm_times_ms: List[float] = []
    test_ipm_results: List[IPMResult] = []
    if need_test_ipm:
        for inst in test_instances:
            t0 = time.perf_counter()
            res = ipm_solver.solve(inst.sdp)
            ipm_times_ms.append((time.perf_counter() - t0) * 1e3)
            ipm_feas.append(bool(res.converged))
            test_ipm_results.append(res)
            ipm_values.append(_objective_from_ipm_result(res, inst, app_key) if res.converged else float("nan"))

    # --- optional external solvers ---
    ext_rows: Dict[str, Dict[str, List[object]]] = {}
    ext_requested = [a for a in algorithms if a in {"MOSEK", "SCS"}]
    if ext_requested:
        if _get_cvxpy() is None:
            print("[warning] cvxpy is not available; skipping MOSEK/SCS.")
        else:
            installed = installed_solvers()
            for solver_name in ext_requested:
                if solver_name not in installed:
                    print(f"[warning] {solver_name} requested but not installed; skipping.")
                    continue
                vals: List[float] = []
                feas: List[bool] = []
                tms: List[float] = []
                for inst in test_instances:
                    try:
                        if app_key in {"l2_gain", "hinf_norm"}:
                            row = solve_gamma_with_solver(inst.system, solver_name)
                            vals.append(float(row.value_gamma))
                            feas.append(bool(row.feasible))
                            tms.append(1000.0 * float(row.time_sec))
                        elif app_key == "h2_norm":
                            v, ok, t = _solve_h2_with_solver(inst.system, solver_name)
                            vals.append(v)
                            feas.append(ok)
                            tms.append(t)
                        elif app_key == "lyapunov_reg":
                            n = inst.system.A.shape[0]
                            Q_obj = np.asarray(inst.sdp.C[:n, :n], dtype=float)
                            v, ok, t = _solve_lyapunov_reg_with_solver(
                                inst.system,
                                solver_name,
                                Q_obj=Q_obj,
                                eps=float(config.cert_eps),
                            )
                            vals.append(v)
                            feas.append(ok)
                            tms.append(t)
                        else:
                            vals.append(float("nan"))
                            feas.append(False)
                            tms.append(float("nan"))
                    except Exception as exc:  # noqa: BLE001
                        print(f"[warning] {solver_name} failed on {system_name} instance {inst.index}: {exc}")
                        vals.append(float("nan"))
                        feas.append(False)
                        tms.append(float("nan"))
                ext_rows[solver_name] = {"values": vals, "feas": feas, "times": tms}

    # --- pick reference values (for error reporting) ---
    if "MOSEK" in ext_rows:
        ref_values = np.asarray(ext_rows["MOSEK"]["values"], dtype=float)
    elif "SCS" in ext_rows:
        ref_values = np.asarray(ext_rows["SCS"]["values"], dtype=float)
    elif ipm_values:
        ref_values = np.asarray(ipm_values, dtype=float)
    else:
        ref_values = None

    method_order: List[str] = []
    results: Dict[str, AlgorithmResult] = {}

    # --- external rows ---
    for solver_name in ["MOSEK", "SCS"]:
        if solver_name in ext_rows:
            method_order.append(solver_name)
            ref_for_solver = None if solver_name == "MOSEK" else ref_values
            results[solver_name] = _summarize_row(
                values=ext_rows[solver_name]["values"],
                feas=ext_rows[solver_name]["feas"],
                times_ms=ext_rows[solver_name]["times"],
                ref_values=ref_for_solver,
            )

    # --- IPM baseline ---
    if "IPM" in algorithms and need_test_ipm:
        method_order.append("IPM")
        results["IPM"] = _summarize_row(ipm_values, ipm_feas, ipm_times_ms, ref_values)

    # --- L2WS ---
    if "L2WS" in algorithms:
        values: List[float] = []
        feas: List[bool] = []
        tms: List[float] = []
        kept_idx = [i for i, res in enumerate(train_ipm_results) if res.converged]
        if len(kept_idx) < 5:
            values = [float("nan")] * len(test_instances)
            feas = [False] * len(test_instances)
            tms = [float("nan")] * len(test_instances)
            print(f"[warning] {system_name}: too few converged training solves for L2WS ({len(kept_idx)}).")
        else:
            train_kept = [train_instances[i] for i in kept_idx]
            sols_kept = [
                (
                    np.asarray(train_ipm_results[i].X, dtype=float),
                    np.asarray(train_ipm_results[i].y, dtype=float),
                    np.asarray(train_ipm_results[i].S, dtype=float),
                )
                for i in kept_idx
            ]
            n = train_kept[0].sdp.dim
            m = train_kept[0].sdp.num_constraints
            ws_solver = SuperSDP(
                problem_config=ProblemConfig(n=n, m=m),
                training_config=TrainingConfig(
                    epochs=int(config.epochs),
                    batch_size=int(config.batch_size),
                    lr=float(config.lr),
                    hidden_dims=tuple(config.hidden_sizes),
                    dropout=0.0,
                    backbone="mlp",
                ),
                solver_config=SolverConfig(mode="L2WS", warmstart_type="cholesky"),
                device="cpu",
            )
            ws_solver.fit([inst.sdp for inst in train_kept], sols_kept)
            for inst in test_instances:
                t0 = time.perf_counter()
                res = ws_solver.solve(inst.sdp)
                tms.append((time.perf_counter() - t0) * 1e3)
                feas.append(bool(res.converged))
                if res.converged and res.y is not None and res.X is not None:
                    if app_key in {"l2_gain", "hinf_norm"}:
                        values.append(extract_gamma_from_dual(np.asarray(res.y, dtype=float), inst.sdp.b))
                    else:
                        values.append(float(np.trace(inst.sdp.C @ np.asarray(res.X, dtype=float))))
                else:
                    values.append(float("nan"))
        method_order.append("L2WS")
        results["L2WS"] = _summarize_row(values, feas, tms, ref_values)

    # --- L2A ---
    if "L2A" in algorithms:
        values: List[float] = []
        tms: List[float] = []
        train_feat = np.stack([inst.features for inst in train_instances], axis=0)
        train_labels = [
            _objective_from_ipm_result(res, inst, app_key) if res.converged else float("nan")
            for inst, res in zip(train_instances, train_ipm_results)
        ]
        labels_arr = np.asarray(train_labels, dtype=float)
        mask = np.isfinite(labels_arr)
        if np.sum(mask) < 5:
            values = [float("nan")] * len(test_instances)
            tms = [float("nan")] * len(test_instances)
            print(f"[warning] {system_name}: too few converged training solves for L2A ({int(np.sum(mask))}).")
        else:
            x_train = train_feat[mask]
            y_train = labels_arr[mask]
            x_mean = x_train.mean(axis=0)
            x_std = np.where(x_train.std(axis=0) < 1e-8, 1.0, x_train.std(axis=0))
            y_mean = float(np.mean(y_train))
            y_std = float(np.std(y_train))
            if y_std < 1e-8:
                y_std = 1.0

            model = ScalarL2ANet(input_dim=x_train.shape[1], hidden_sizes=(128, 64, 32))
            train_scalar_l2a(
                model,
                (x_train - x_mean) / x_std,
                (y_train - y_mean) / y_std,
                epochs=int(config.epochs),
                lr=float(config.lr),
                batch_size=int(config.batch_size),
                device="cpu",
            )
            dev = torch.device("cpu")
            for inst in test_instances:
                x = (np.asarray(inst.features, dtype=float) - x_mean) / x_std
                xt = torch.as_tensor(x, dtype=torch.float32, device=dev).unsqueeze(0)
                t0 = time.perf_counter()
                with torch.no_grad():
                    y_hat = float(model(xt).cpu().numpy().reshape(-1)[0])
                tms.append((time.perf_counter() - t0) * 1e3)
                values.append(y_hat * y_std + y_mean)

        method_order.append("L2A")
        # L2A feasibility is typically N/A (objective predictor only).
        results["L2A"] = _summarize_row(values, feas=None, times_ms=tms, ref_values=ref_values)

    # --- L2CA ---
    if "L2CA" in algorithms:
        values: List[float] = []
        feas: List[bool] = []
        tms: List[float] = []

        kept_idx = [i for i, res in enumerate(train_ipm_results) if res.converged]
        if len(kept_idx) < 5:
            values = [float("nan")] * len(test_instances)
            feas = [False] * len(test_instances)
            tms = [float("nan")] * len(test_instances)
            print(f"[warning] {system_name}: too few converged training solves for L2CA ({len(kept_idx)}).")
        else:
            train_kept_instances = [train_instances[i] for i in kept_idx]
            x_train = np.stack([inst.features for inst in train_kept_instances], axis=0)
            y_train = np.stack([np.asarray(train_ipm_results[i].y, dtype=float) for i in kept_idx], axis=0)

            x_mean = x_train.mean(axis=0)
            x_std = np.where(x_train.std(axis=0) < 1e-8, 1.0, x_train.std(axis=0))
            y_mean = y_train.mean(axis=0)
            y_std = np.where(y_train.std(axis=0) < 1e-8, 1.0, y_train.std(axis=0))

            model = DualNet(input_dim=x_train.shape[1], output_dim=y_train.shape[1])
            train_dual_net(
                model,
                (x_train - x_mean) / x_std,
                (y_train - y_mean) / y_std,
                epochs=int(config.epochs),
                lr=float(config.lr),
                batch_size=int(config.batch_size),
                device="cpu",
                b_targets=np.stack([np.asarray(inst.sdp.b, dtype=float) for inst in train_kept_instances], axis=0),
                lambda_obj=float(config.l2ca_lambda_obj),
                y_targets_unscaled=y_train,
                y_mean=y_mean,
                y_std=y_std,
                sdp_instances=[inst.sdp for inst in train_kept_instances],
                lambda_feas=float(config.l2ca_lambda_feas),
                feas_margin=float(config.l2ca_feas_margin),
                feas_loss_mode=str(config.l2ca_feas_loss),
                feas_margin_train=float(config.l2ca_feas_margin_train),
                label_margin_debug=bool(config.l2ca_label_margin_debug),
                interiorize_labels=bool(config.l2ca_interiorize_labels and is_certificate_app),
                interior_delta=float(config.l2ca_interior_delta),
                lambda_y=float(config.l2ca_lambda_y),
                obj_debug=False,
            )

            x_train_norm = (x_train - x_mean) / x_std
            train_sdps = [inst.sdp for inst in train_kept_instances]
            robust_anchor = select_robust_anchor(train_sdps, y_train)
            global_mean_anchor = np.mean(y_train, axis=0)
            anchor_mode = str(config.l2ca_anchor).strip().lower()
            if anchor_mode not in {"knn1", "knn5_best", "global_mean"}:
                anchor_mode = "knn5_best"
            force_global_anchor = anchor_mode == "global_mean"
            anchor_k = 1 if anchor_mode == "knn1" else 5
            tier_auto_enabled = str(config.l2ca_tier_auto).strip().lower() == "on"
            tier0_fallback_mode = str(config.l2ca_tier0_fallback).strip().lower()
            tier_level = 0
            tier_j_idx: int | None = None
            if tier_auto_enabled:
                tier_level, tier_j_idx = detect_tier(
                    train_kept_instances[0].sdp,
                    tol=float(config.l2ca_feas_margin),
                )

            dev = torch.device("cpu")
            for inst in test_instances:
                x = (np.asarray(inst.features, dtype=float) - x_mean) / x_std
                xt = torch.as_tensor(x, dtype=torch.float32, device=dev).unsqueeze(0)
                t0 = time.perf_counter()
                with torch.no_grad():
                    y_scaled = model(xt).cpu().numpy().reshape(-1)
                y_pred = y_scaled * y_std + y_mean

                l2ca_result = run_l2ca_inference(
                    sdp=inst.sdp,
                    y_pred=y_pred,
                    x_feat=x,
                    x_train_norm=x_train_norm,
                    y_train=y_train,
                    cached_feasible_anchor=robust_anchor,
                    feas_margin=float(config.l2ca_feas_margin),
                    bisect_iters=int(config.l2ca_bisect_iters),
                    tier_auto=bool(tier_auto_enabled),
                    tier_level=int(tier_level),
                    tier_j_idx=tier_j_idx,
                    tier0_fallback=str(tier0_fallback_mode),
                    force_global_anchor=bool(force_global_anchor),
                    global_mean_anchor=global_mean_anchor,
                    anchor_k=int(anchor_k),
                    skip_lift=app_key == "lqr_feas",
                    refine_solver=None,
                    refine_iters=0,
                    enable_refine=False,
                    debug=False,
                )
                y_out = np.asarray(l2ca_result["y_out"], dtype=float).reshape(-1)
                ok = bool(l2ca_result["final_ok"])

                tms.append((time.perf_counter() - t0) * 1e3)
                feas.append(bool(ok))
                values.append(float(np.dot(inst.sdp.b, y_out)))

        method_order.append("L2CA")
        ref_for_l2ca = None
        if ref_values is not None:
            if app_key in {"l2_gain", "hinf_norm"}:
                ref_for_l2ca = -np.square(ref_values)
            else:
                ref_for_l2ca = ref_values
        results["L2CA"] = _summarize_row(values, feas, tms, ref_for_l2ca)

    # restrict to requested algorithm order
    requested_order = [a for a in algorithms if a in results]
    _print_table(
        app=app_key,
        system_name=system_name,
        n_states=int(system.A.shape[0]),
        sdp_dim=int(test_instances[0].sdp.dim),
        ordered_methods=requested_order,
        results=results,
    )

    return ExperimentResult(
        app=app_key,
        system_name=system_name,
        n=int(system.A.shape[0]),
        sdp_dim=int(test_instances[0].sdp.dim),
        results=results,
    )


def run_experiment(
    systems: Dict[str, LTISystem],
    config: ExperimentConfig,
    **kwargs,
) -> Dict[str, ExperimentResult]:
    """Run one experiment config on a dict of named systems."""
    out: Dict[str, ExperimentResult] = {}
    for name, system in systems.items():
        out[name] = run_experiment_on_system(
            system=system,
            system_name=name,
            config=config,
            custom_sdp_builder=kwargs.get("custom_sdp_builder"),
            custom_feature_fn=kwargs.get("custom_feature_fn"),
        )
    return out


__all__ = [
    "ExperimentConfig",
    "AlgorithmResult",
    "ExperimentResult",
    "run_experiment_on_system",
    "run_experiment",
]
