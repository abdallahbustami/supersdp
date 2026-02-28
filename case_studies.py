"""Unified case-study runner for L2WS/L2A/L2CA experiments."""

from __future__ import annotations

import argparse
import csv
import os
import re
import time
from functools import partial
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy import sparse as sp_sparse

try:
    import cvxpy as cp
except Exception:  # pragma: no cover
    cp = None

try:
    import h5py    # type: ignore
except Exception:  # pragma: no cover
    h5py = None

try:
    import scipy.io
except Exception: # pragma: no cover
    scipy = None  # type: ignore

import torch

from l2ws.benchmarks import installed_solvers, solve_gamma_with_solver
from l2ws.data import (
    L2GainInstance,
    LTISystem,
    build_h2_norm_sdp,
    build_l2_gain_sdp,
    build_lqr_feas_sdp,
    build_lyapunov_sdp,
    build_lyapunov_regularized_sdp,
    generate_sdp_instances,
)
from l2ws.graph_utils import has_torch_geometric
from l2ws.ipm import IPMResult, IPMSettings, InfeasibleIPMSolver
from l2ws.lifecycle import (
    build_warm_state,
    predict_warmstart,
    run_l2ws_lifecycle,
)
from l2ws.l2a import ScalarL2ANet, train_scalar_l2a
from l2ws.l2ca import (
    DualNet,
    _hamiltonian_feasible,
    build_anchor,
    bisection_to_feasible,
    choose_anchor_best_knn,
    detect_tier,
    extract_gamma_from_dual,
    hamiltonian_bisection,
    is_psd_cholesky,
    is_psd_with_margin,
    maximize_dual_along_b,
    run_l2ca_inference,
    slack_from_y,
    select_robust_anchor,
    tier0_repair_cascade,
    train_dual_net,
)
from l2ws.models import WarmStartNet
from l2ws.training import TrainerConfig


CASE_STUDIES: Dict[str, Dict[str, Any]] = {
    "msd_l2gain": {
        "description": "Mass-Spring-Damper — L₂ Gain",
        "sdp_builder": "l2_gain",
        "systems": [("MSD-4", 4), ("MSD-8", 8), ("MSD-16", 16), ("MSD-32", 32)],
        "perturb": (-0.1, 0.1),
    },
    "msd_lyapunov_reg": {
        "description": "Mass-Spring-Damper — Lyapunov Certificate",
        "sdp_builder": "lyapunov",
        "systems": [("MSD-4", 4), ("MSD-8", 8), ("MSD-16", 16), ("MSD-32", 32)],
        "perturb": (-0.1, 0.1),
    },
    "power_lqr": {
        "description": "Power System — LQR (CARE->LMI feasibility)",
        "sdp_builder": "lqr_feas",
        "systems": [
            ("9bus", "data/PowerSystemData_Consolidated.mat", "9"),
            ("14bus", "data/PowerSystemData_Consolidated.mat", "14"),
            ("30bus", "data/PowerSystemData_Consolidated.mat", "30"),
            ("39bus", "data/PowerSystemData_Consolidated.mat", "39"),
            ("57bus", "data/PowerSystemData_Consolidated.mat", "57"),
        ],
        "perturb": (-0.1, 0.1),
    },
    "power_l2gain": {
        "description": "Power System — L2 Gain (pre-stabilized)",
        "sdp_builder": "l2_gain",
        "systems": [
            ("9bus", "data/PowerSystemData_Consolidated.mat", "9"),
            ("14bus", "data/PowerSystemData_Consolidated.mat", "14"),
            ("30bus", "data/PowerSystemData_Consolidated.mat", "30"),
            ("39bus", "data/PowerSystemData_Consolidated.mat", "39"),
            ("57bus", "data/PowerSystemData_Consolidated.mat", "57"),
        ],
        "perturb": (-0.1, 0.1),
    },
    "power_h2": {
        "description": "Power System — H₂ Norm (pre-stabilized)",
        "sdp_builder": "h2_norm",
        "systems": [
            ("9bus", "data/PowerSystemData_Consolidated.mat", "9"),
            ("14bus", "data/PowerSystemData_Consolidated.mat", "14"),
            ("30bus", "data/PowerSystemData_Consolidated.mat", "30"),
            ("39bus", "data/PowerSystemData_Consolidated.mat", "39"),
            ("57bus", "data/PowerSystemData_Consolidated.mat", "57"),
        ],
        "perturb": (-0.1, 0.1),
    },
    "compleib_l2gain": {
        "description": "COMPleib Aircraft — L₂ Gain",
        "sdp_builder": "l2_gain",
        "systems": [
            ("HF2D12", "COMPlib_r1_1/hf2d12.mat"),
            ("HF2D13", "COMPlib_r1_1/hf2d13.mat"),
            ("HF2D_CD4", "COMPlib_r1_1/hf2d_cd4.mat"),
            ("CM1", "COMPlib_r1_1/cm1.mat"),
        ],
        "perturb": (-0.1, 0.1),
    },
    "compleib_lyapunov": {
        "description": "COMPleib Aircraft — Lyapunov Certificate",
        "sdp_builder": "lyapunov",
        "systems": [
            ("HF2D13", "COMPlib_r1_1/hf2d13.mat"),
            ("HF2D_CD5", "COMPlib_r1_1/hf2d_cd5.mat"),
            ("HE6", "COMPlib_r1_1/he6.mat"),
        ],
        "perturb": (-0.05, 0.05),
    },
    "compleib_lyapunov_reg": {
        "description": "COMPleib Aircraft — Lyapunov Certificate",
        "sdp_builder": "lyapunov",
        "systems": [
            ("HF2D13", "COMPlib_r1_1/hf2d13.mat"),
            ("HF2D_CD5", "COMPlib_r1_1/hf2d_cd5.mat"),
            ("HE6", "COMPlib_r1_1/he6.mat"),
        ],
        "perturb": (-0.05, 0.05),
    },
}

ALGORITHM_CANONICAL = ("MOSEK", "SCS", "IPM", "L2WS", "L2A", "L2CA", "L2CA-H")
ALGORITHM_ALIASES = {
    "mosek": "MOSEK",
    "scs": "SCS",
    "ipm": "IPM",
    "ipm(cold)": "IPM",
    "ipm_cold": "IPM",
    "cold": "IPM",
    "l2ws": "L2WS",
    "l2a": "L2A",
    "l2ca": "L2CA",
    "l2ca-h": "L2CA-H",
    "l2cah": "L2CA-H",
}


@dataclass
class MethodSummary:
    method: str
    feas_rate: float
    mean_err: float
    std_err: float
    max_err: float
    mean_time_ms: float
    total_time_ms: float
    speedup: float


def _normalize_algorithm_selection(values: Sequence[str]) -> set[str]:
    raw = [str(v).strip() for v in values if str(v).strip()]
    if not raw or any(v.lower() == "all" for v in raw):
        return set(ALGORITHM_CANONICAL)
    out: set[str] = set()
    for v in raw:
        key = v.lower()
        if key not in ALGORITHM_ALIASES:
            allowed = ", ".join(list(ALGORITHM_CANONICAL) + ["all"])
            raise ValueError(f"Unknown algorithm '{v}'. Allowed values: {allowed}")
        out.add(ALGORITHM_ALIASES[key])
    return out


def build_mass_spring_damper(n_states: int) -> LTISystem:
    """MSD chain (n_states must be even)."""
    if n_states <= 0 or (n_states % 2) != 0:
        raise ValueError("n_states must be a positive even integer.")
    nm = n_states // 2
    k_spring = 2.0
    d_damp = 0.6

    K = 2.0 * k_spring * np.eye(nm, dtype=float)
    D = 2.0 * d_damp * np.eye(nm, dtype=float)
    for i in range(nm - 1):
        K[i, i + 1] = -k_spring
        K[i + 1, i] = -k_spring
        D[i, i + 1] = -0.2 * d_damp
        D[i + 1, i] = -0.2 * d_damp

    A = np.block([[np.zeros((nm, nm), dtype=float), np.eye(nm, dtype=float)], [-K, -D]])
    Bw = np.zeros((n_states, 1), dtype=float)
    Bw[nm:, 0] = 1.0 / np.sqrt(float(nm))
    Cz = np.zeros((1, n_states), dtype=float)
    Cz[0, :nm] = 1.0 / float(nm)
    Cz[0, nm:] = 0.1 / float(nm)
    return LTISystem(A=A, Bw=Bw, Cz=Cz)


def _iter_mat_struct(data: Any, prefix: str = "") -> Iterable[Tuple[str, np.ndarray]]:
    if isinstance(data, np.ndarray) and data.ndim == 2 and np.isrealobj(data):
        yield prefix, np.asarray(data, dtype=float)
        return

    if isinstance(data, dict):
        for key, val in data.items():
            if str(key).startswith("__"):
                continue
            next_prefix = f"{prefix}/{key}" if prefix else str(key)
            yield from _iter_mat_struct(val, next_prefix)
        return

    fields = getattr(data, "_fieldnames", None)
    if fields:
        for field in fields:
            next_prefix = f"{prefix}/{field}" if prefix else str(field)
            yield from _iter_mat_struct(getattr(data, field), next_prefix)


def _select_power_case_name(case_names: Sequence[str], bus: str) -> str:
    """Select exact power-system case by bus number (e.g., 9, 14, 30, 57)."""
    bus_digits = "".join(ch for ch in str(bus) if ch.isdigit())
    if not bus_digits:
        raise ValueError(f"Invalid bus identifier: {bus!r}")

    # Match case<bus><non-digit-or-end>, so bus=30 does not match case3120sp.
    pat = re.compile(rf"^case{re.escape(bus_digits)}(?:[^0-9]|$)")
    matches = [name for name in case_names if pat.search(name.lower())]
    if matches:
        return sorted(matches, key=lambda s: (len(s), s))[0]

    raise ValueError(
        f"No exact power-system case for bus={bus_digits}. "
        f"Available cases: {', '.join(sorted(case_names))}"
    )


def _h5_resolve_ref(file_obj: h5py.File, obj: Any) -> Any:
    """Resolve MATLAB v7.3 object references recursively."""
    if isinstance(obj, h5py.Dataset) and obj.dtype == h5py.ref_dtype:
        refs = np.array(obj)
        if refs.size == 0:
            return obj
        if refs.size == 1:
            return _h5_resolve_ref(file_obj, file_obj[refs.reshape(-1)[0]])
    return obj


def _h5_scalar_int(ds: h5py.Dataset) -> int:
    arr = np.asarray(ds)
    return int(arr.reshape(-1)[0])


def _h5_decode_sparse(group: h5py.Group, shape: tuple[int, int]) -> np.ndarray:
    """Decode MATLAB sparse matrix stored as {data, ir, jc}."""
    data = np.asarray(group["data"]).reshape(-1).astype(float)
    ir = np.asarray(group["ir"]).reshape(-1).astype(int)
    jc = np.asarray(group["jc"]).reshape(-1).astype(int)
    rows, cols = shape
    mat = np.zeros((rows, cols), dtype=float)
    for j in range(cols):
        start = int(jc[j])
        end = int(jc[j + 1])
        if end > start:
            mat[ir[start:end], j] = data[start:end]
    return mat


def load_power_system(mat_path: str, bus: str) -> LTISystem:
    """Load from PowerSystemData_Consolidated.mat and return LTISystem with Cz=I."""
    path = Path(mat_path)
    if not path.exists():
        raise FileNotFoundError(f"Power data file not found: {path}")

    if h5py is not None:
        try:
            with h5py.File(path, "r") as f:
                if "PowerData" in f:
                    power_data = f["PowerData"]
                    case_name = _select_power_case_name(list(power_data.keys()), str(bus))
                    case_group = _h5_resolve_ref(f, power_data[case_name])
                    if not isinstance(case_group, h5py.Group):
                        raise RuntimeError(f"Unexpected case object type for {case_name}.")

                    if "A" not in case_group or "B" not in case_group:
                        raise RuntimeError(f"Case {case_name} missing A/B entries.")
                    if "n" not in case_group:
                        raise RuntimeError(f"Case {case_name} missing n entry.")

                    n = _h5_scalar_int(case_group["n"])
                    m = _h5_scalar_int(case_group["m"]) if "m" in case_group else 1

                    A_obj = _h5_resolve_ref(f, case_group["A"])
                    B_obj = _h5_resolve_ref(f, case_group["B"])
                    if not isinstance(A_obj, h5py.Group) or not isinstance(B_obj, h5py.Group):
                        raise RuntimeError(f"Case {case_name} A/B are not sparse groups.")

                    A = _h5_decode_sparse(A_obj, (n, n))
                    B = _h5_decode_sparse(B_obj, (n, m))
                    Cz = np.eye(n, dtype=float)
                    print(f"  [info] bus {bus}: loaded {case_name} with A{A.shape}, B{B.shape}")
                    return LTISystem(A=A, Bw=B, Cz=Cz)
        except Exception:
            pass

    # Fallback for non-v7.3 MAT files.
    if scipy is None:
        raise RuntimeError("scipy is required to load non-hdf5 .mat files.")
    data = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    mats = list(_iter_mat_struct(data))
    bus_digits = "".join(ch for ch in str(bus) if ch.isdigit())
    pat = re.compile(rf"case{re.escape(bus_digits)}(?:[^0-9]|$)")
    A_candidates = [(name, arr) for name, arr in mats if arr.shape[0] == arr.shape[1] and pat.search(name.lower())]
    if not A_candidates:
        raise RuntimeError(f"No exact dense A matrix found for bus={bus_digits} in {path}")

    A_name, A = sorted(A_candidates, key=lambda x: x[1].shape[0])[-1]
    n = A.shape[0]
    B_candidates = [
        (name, arr)
        for name, arr in mats
        if arr.shape[0] == n and arr.shape[1] > 0 and pat.search(name.lower())
    ]
    if not B_candidates:
        B = np.eye(n, 1, dtype=float)
    else:
        _, B = sorted(B_candidates, key=lambda x: x[1].shape[1])[-1]

    Cz = np.eye(n, dtype=float)
    print(f"  [info] bus {bus}: loaded {A_name} with A{A.shape}, B{B.shape}")
    return LTISystem(A=np.asarray(A, dtype=float), Bw=np.asarray(B, dtype=float), Cz=Cz)


def _select_matrix(data: dict, keys: Sequence[str]) -> np.ndarray | None:
    for key in keys:
        if key in data:
            arr = data[key]
            if sp_sparse.issparse(arr):
                return np.asarray(arr.toarray(), dtype=float)
            return np.asarray(arr, dtype=float)
    return None


def load_compleib_system(mat_path: str) -> LTISystem:
    """Load from COMPleib .mat file. A→A, B/B1→Bw, C/C1→Cz."""
    path = Path(mat_path)
    if not path.exists():
        raise FileNotFoundError(f"COMPleib file not found: {path}")
    if scipy is None:
        raise RuntimeError("scipy is required to load COMPleib files.")

    data = scipy.io.loadmat(path)
    if "A" not in data:
        raise ValueError(f"{path} does not contain an A matrix.")

    A_raw = data["A"]
    if sp_sparse.issparse(A_raw):
        A = np.asarray(A_raw.toarray(), dtype=float)
    else:
        A = np.asarray(A_raw, dtype=float)
    Bw = _select_matrix(data, ["B", "B1"])
    Cz = _select_matrix(data, ["C", "C1"])
    if Bw is None:
        raise ValueError(f"Could not find B/B1 matrix in {path}.")
    if Cz is None:
        raise ValueError(f"Could not find C/C1 matrix in {path}.")
    return LTISystem(A=A, Bw=np.asarray(Bw, dtype=float), Cz=np.asarray(Cz, dtype=float))


def _system_stable(system: LTISystem) -> bool:
    vals = np.linalg.eigvals(system.A)
    return float(np.max(np.real(vals))) < 0.0


def _prestabilize(system: LTISystem, margin: float = 0.1) -> LTISystem:
    """Shift A so all eigenvalues satisfy Re(lambda) <= -margin."""
    eigs = np.linalg.eigvals(system.A)
    max_real = float(np.max(np.real(eigs)))
    if max_real >= -margin:
        shift = max_real + margin
        A_stable = system.A - shift * np.eye(system.A.shape[0], dtype=float)
        print(f"    [info] Pre-stabilized: shifted A by {-shift:.4f} (max Re(λ) was {max_real:.4f})")
        return LTISystem(A=A_stable, Bw=system.Bw, Cz=system.Cz)
    return system


def _objective_from_ipm_result(result: IPMResult, instance: L2GainInstance, builder_kind: str) -> float:
    if builder_kind == "l2_gain":
        return extract_gamma_from_dual(result.y, instance.sdp.b)
    return float(np.trace(instance.sdp.C @ result.X))


def _build_solver_row(values: List[float], feas: List[bool], times_ms: List[float], ref: np.ndarray, ref_total: float, method: str) -> MethodSummary:
    values_arr = np.asarray(values, dtype=float)
    feas_arr = np.asarray(feas, dtype=bool)
    times_arr = np.asarray(times_ms, dtype=float)

    mask = np.isfinite(values_arr) & np.isfinite(ref)
    if np.any(mask):
        rel_err = np.abs(values_arr[mask] - ref[mask]) / np.maximum(np.abs(ref[mask]), 1e-12)
        mean_err = float(np.mean(rel_err))
        std_err = float(np.std(rel_err))
        max_err = float(np.max(rel_err))
    else:
        mean_err = float("nan")
        std_err = float("nan")
        max_err = float("nan")

    finite_times = times_arr[np.isfinite(times_arr)]
    if finite_times.size:
        total_time = float(np.sum(finite_times))
        mean_time = float(np.mean(finite_times))
    else:
        total_time = float("nan")
        mean_time = float("nan")
    if np.isfinite(ref_total) and np.isfinite(total_time) and total_time > 0.0:
        speedup = float(ref_total / total_time)
    else:
        speedup = float("nan")

    return MethodSummary(
        method=method,
        feas_rate=100.0 * float(np.mean(feas_arr.astype(float))) if feas_arr.size else 0.0,
        mean_err=100.0 * mean_err if np.isfinite(mean_err) else float("nan"),
        std_err=100.0 * std_err if np.isfinite(std_err) else float("nan"),
        max_err=100.0 * max_err if np.isfinite(max_err) else float("nan"),
        mean_time_ms=mean_time,
        total_time_ms=total_time,
        speedup=speedup,
    )


def _print_table(title: str, summaries: Sequence[MethodSummary], n_states: int, sdp_dim: int) -> None:
    print("")
    print(f"=== {title} (n={n_states}, SDP dim={sdp_dim}) ===")
    print(
        f"{'Algorithm':<15} | {'Feas %':>6} | {'Mean Err ± Std':>18} | {'Max Err':>8} | "
        f"{'Mean Time':>9} | {'Total Time':>10} | {'Speedup':>8}"
    )
    print("-" * 96)
    for s in summaries:
        err_txt = "--" if not np.isfinite(s.mean_err) else f"{s.mean_err:.3f}% ± {s.std_err:.3f}%"
        max_txt = "--" if not np.isfinite(s.max_err) else f"{s.max_err:.3f}%"
        print(
            f"{s.method:<15} | {s.feas_rate:>6.1f} | {err_txt:>18} | {max_txt:>8} | "
            f"{s.mean_time_ms:>8.3f}ms | {s.total_time_ms:>9.3f}ms | {s.speedup:>7.1f}x"
        )


def _save_csv(path: Path, summaries: Sequence[MethodSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "feas_rate", "mean_err_pct", "std_err_pct", "max_err_pct", "mean_time_ms", "total_time_ms", "speedup"])
        for s in summaries:
            writer.writerow([s.method, s.feas_rate, s.mean_err, s.std_err, s.max_err, s.mean_time_ms, s.total_time_ms, s.speedup])


def _solve_h2_with_solver(system: LTISystem, solver_name: str, eps: float = 1e-6) -> Tuple[float, bool, float]:
    if cp is None:
        raise RuntimeError("cvxpy is required for external solver comparisons.")
    A = system.A
    Bw = system.Bw
    Cz = system.Cz
    n = A.shape[0]
    P = cp.Variable((n, n), PSD=True)
    obj = cp.trace(Bw.T @ P @ Bw)
    constraints = [A.T @ P + P @ A + Cz.T @ Cz << -eps * np.eye(n)]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    t0 = time.perf_counter()
    prob.solve(solver=solver_name, verbose=False)
    t_ms = (time.perf_counter() - t0) * 1e3
    ok = prob.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
    value = float(prob.value) if ok and prob.value is not None else float("nan")
    return value, ok, t_ms


def _solve_lyapunov_with_solver(
    system: LTISystem,
    solver_name: str,
    eps: float = 1e-6,
    Q: np.ndarray | None = None,
) -> Tuple[float, bool, float]:
    """Solve Lyapunov certificate SDP via CVXPY."""
    if cp is None:
        raise RuntimeError("cvxpy is required for external solver comparisons.")
    A = system.A
    n = A.shape[0]
    if Q is None:
        Q_obj = np.asarray(system.Cz, dtype=float).T @ np.asarray(system.Cz, dtype=float)
    else:
        Q_obj = np.asarray(Q, dtype=float)
    Q_obj = 0.5 * (Q_obj + Q_obj.T)
    if Q_obj.shape != (n, n):
        raise ValueError("Q must have shape (n, n).")
    P = cp.Variable((n, n), PSD=True)
    constraints = [A.T @ P + P @ A << -float(eps) * np.eye(n), cp.trace(P) == 1.0]
    prob = cp.Problem(cp.Minimize(cp.trace(Q_obj @ P)), constraints)
    t0 = time.perf_counter()
    prob.solve(solver=solver_name, verbose=False)
    t_ms = (time.perf_counter() - t0) * 1e3
    ok = prob.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
    value = float(prob.value) if ok and prob.value is not None else float("nan")
    return value, ok, t_ms


def _solve_lqr_feas_with_solver(
    system: LTISystem,
    solver_name: str,
    rho: float = 0.1,
    eps: float = 1e-3,
) -> Tuple[float, bool, float]:
    """Solve LQR CARE->LMI feasibility SDP via CVXPY."""
    if cp is None:
        raise RuntimeError("cvxpy is required for external solver comparisons.")
    A = system.A
    B = system.Bw
    n = A.shape[0]
    m = B.shape[1]

    S = cp.Variable((n, n), PSD=True)
    Z = cp.Variable((m, n))
    qinv = np.eye(n)
    rinv = (1.0 / float(rho)) * np.eye(m)
    top_left = S @ A.T + A @ S + B @ Z + Z.T @ B.T
    lmi = cp.bmat(
        [
            [top_left, S, Z.T],
            [S, -qinv, np.zeros((n, m))],
            [Z, np.zeros((m, n)), -rinv],
        ]
    )
    constraints = [lmi << 0, S >> float(eps) * np.eye(n)]
    prob = cp.Problem(cp.Minimize(0), constraints)
    t0 = time.perf_counter()
    prob.solve(solver=solver_name, verbose=False)
    t_ms = (time.perf_counter() - t0) * 1e3
    ok = prob.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
    value = 0.0 if ok else float("nan")
    return value, ok, t_ms


def _builder_from_kind(kind: str) -> Callable[..., Any]:
    if kind == "l2_gain":
        return build_l2_gain_sdp
    if kind == "h2_norm":
        return build_h2_norm_sdp
    if kind == "lyapunov":
        return build_lyapunov_sdp
    if kind == "lqr_feas":
        return build_lqr_feas_sdp
    raise ValueError(f"Unknown builder kind: {kind}")


def _is_certificate_builder(kind: str) -> bool:
    return kind in {"lyapunov"}


def _load_system_for_entry(study_key: str, entry: tuple[Any, ...], args: argparse.Namespace) -> tuple[str, LTISystem] | None:
    name = str(entry[0])
    try:
        if study_key.startswith("msd_"):
            return name, build_mass_spring_damper(int(entry[1]))
        if study_key in {"power_h2", "power_l2gain", "power_lqr"}:
            return name, load_power_system(args.power_mat, str(entry[2]))
        compleib_path = Path(entry[1])
        if not compleib_path.exists():
            compleib_path = Path(args.compleib_dir) / compleib_path.name
        return name, load_compleib_system(str(compleib_path))
    except Exception as exc:
        print(f"[warning] Skipping {name}: {exc}")
        return None


def _evaluate_system(
    study_key: str,
    description: str,
    system_name: str,
    system: LTISystem,
    builder_kind: str,
    perturb: tuple[float, float],
    args: argparse.Namespace,
) -> list[MethodSummary]:
    selected = set(getattr(args, "_algorithm_set", set(ALGORITHM_CANONICAL)))
    use_mosek = "MOSEK" in selected
    use_scs = "SCS" in selected
    use_ipm = "IPM" in selected
    use_l2ws = "L2WS" in selected
    use_l2a = "L2A" in selected
    use_l2ca = "L2CA" in selected
    use_l2cah = "L2CA-H" in selected and builder_kind == "l2_gain"
    if "L2CA-H" in selected and builder_kind != "l2_gain":
        print(f"[info] {system_name}: L2CA-H is only defined for l2_gain; skipping it.")

    if study_key.startswith("power_") and builder_kind in {"h2_norm", "l2_gain"} and not _system_stable(system):
        system = _prestabilize(system, margin=0.1)
    if study_key.startswith("compleib") and builder_kind in {"l2_gain", "lyapunov"}:
        # COMPlib benchmarks are often near-marginal and can stall primal residual
        # progress in this educational IPM. Enforce a stronger decay margin for
        # numerically stable comparisons across all algorithms.
        system = _prestabilize(system, margin=float(args.compleib_decay_margin))

    max_real = float(np.max(np.real(np.linalg.eigvals(system.A))))
    if builder_kind == "l2_gain" and max_real >= 0.0:
        print(f"[warning] Skipping {system_name}: unstable A (required for L₂ gain).")
        return []
    if builder_kind == "h2_norm" and max_real >= 0.0:
        print(f"[warning] Skipping {system_name}: failed to pre-stabilize for H₂.")
        return []

    sdp_builder = _builder_from_kind(builder_kind)
    title_description = description
    cert_regularization_active = False
    cert_policy_reason = "not_certificate"
    cert_eps_active = 0.0
    cert_q_mode_active = str(args.cert_Q).strip().lower()
    cert_q_scale_active = float(args.cert_Q_scale)
    if _is_certificate_builder(builder_kind):
        cert_mode = str(args.cert_regularize).strip().lower()
        deg_mode = str(args.degenerate_mode).strip().lower()
        if study_key.endswith("_reg"):
            cert_mode = "on"
            cert_policy_reason = "study_suffix_reg"
        if cert_mode == "on":
            cert_regularization_active = True
            cert_policy_reason = cert_policy_reason if cert_policy_reason == "study_suffix_reg" else "forced_on"
        elif cert_mode == "off":
            cert_regularization_active = False
            cert_policy_reason = "forced_off"
        else:
            # We regularize certificate SDPs with a small linear objective to select
            # a canonical solution and avoid degeneracy; feasibility-only variants
            # can be recovered by --cert-regularize off.
            cert_regularization_active = deg_mode in {"auto", "on", "force"}
            cert_policy_reason = f"auto_degmode_{deg_mode}"

        if cert_regularization_active:
            cert_eps_active = float(args.cert_eps)
            sdp_builder = partial(
                build_lyapunov_regularized_sdp,
                Q_mode=cert_q_mode_active,
                Q_scale=cert_q_scale_active,
                eps=cert_eps_active,
            )
            title_description = (
                f"{description} (regularized: Q={cert_q_mode_active}, "
                f"scale={cert_q_scale_active:g}, eps={cert_eps_active:g})"
            )

        print(
            f"    [info] certificate_regularization={'on' if cert_regularization_active else 'off'} "
            f"(policy={cert_policy_reason}, Q={cert_q_mode_active}, "
            f"scale={cert_q_scale_active:g}, eps={float(args.cert_eps):g})"
        )

    effective_backbone = args.backbone
    if use_l2ws and effective_backbone == "gnn" and not has_torch_geometric():
        print("[warning] torch_geometric not available; using MLP backbone for L2WS.")
        effective_backbone = "mlp"
    if not use_l2ws:
        effective_backbone = "mlp"

    instances = generate_sdp_instances(
        base_system=system,
        num=args.num_train + args.num_test,
        sdp_builder=sdp_builder,
        perturb_range=perturb,
        seed=args.seed,
        feature_mode="ABC",
        cache_graph=(effective_backbone == "gnn"),
    )
    train_instances = instances[: args.num_train]
    test_instances = instances[args.num_train :]
    if not train_instances or not test_instances:
        print(f"[warning] Skipping {system_name}: empty train/test split.")
        return []

    device = torch.device(args.device)
    ipm_solver = InfeasibleIPMSolver(IPMSettings(max_iters=120, tol_abs=1e-6, tol_rel=1e-5, linear_solve="sylvester"))

    # Cache training solves once; reused by L2WS baseline stats, L2A labels, and L2CA targets.
    need_train_ipm = use_l2ws or use_l2a or use_l2ca
    train_ipm_results: list[IPMResult] = []
    if need_train_ipm:
        for inst in train_instances:
            res = ipm_solver.solve(
                inst.sdp,
                capture_iterates=False,
                max_captured_iters=0,
            )
            train_ipm_results.append(res)
            inst.baseline_iterations = int(res.iterations) if res.converged else 0
            inst.true_gamma = _objective_from_ipm_result(res, inst, builder_kind) if res.converged else float("nan")

    # Cold IPM on test set (used only when explicitly requested or as reference fallback).
    ipm_values: list[float] = []
    ipm_feas: list[bool] = []
    ipm_times_ms: list[float] = []
    need_test_ipm = use_ipm or use_l2ws or not (use_mosek or use_scs)
    if need_test_ipm:
        for inst in test_instances:
            t0 = time.perf_counter()
            res = ipm_solver.solve(inst.sdp)
            t_ms = (time.perf_counter() - t0) * 1e3
            ipm_times_ms.append(t_ms)
            ipm_feas.append(bool(res.converged))
            ipm_values.append(_objective_from_ipm_result(res, inst, builder_kind) if res.converged else float("nan"))

    installed = installed_solvers() if cp is not None else set()
    ext_rows: Dict[str, Dict[str, list[Any]]] = {}
    selected_external: list[str] = []
    if use_mosek:
        selected_external.append("MOSEK")
    if use_scs:
        selected_external.append("SCS")
    for solver_name in selected_external:
        if solver_name not in installed:
            print(f"[warning] {solver_name} requested but not available; skipping.")
            continue
        vals: list[float] = []
        feas: list[bool] = []
        tms: list[float] = []
        for inst in test_instances:
            try:
                if builder_kind == "l2_gain":
                    row = solve_gamma_with_solver(inst.system, solver_name)
                    vals.append(float(row.value_gamma))
                    feas.append(bool(row.feasible))
                    tms.append(1000.0 * float(row.time_sec))
                elif builder_kind == "h2_norm":
                    v, ok, t = _solve_h2_with_solver(inst.system, solver_name)
                    vals.append(v)
                    feas.append(ok)
                    tms.append(t)
                elif builder_kind == "lyapunov":
                    q_override = None
                    eps_override = 1e-6
                    if cert_regularization_active:
                        n_cert = inst.system.A.shape[0]
                        q_override = 0.5 * (
                            np.asarray(inst.sdp.C[:n_cert, :n_cert], dtype=float)
                            + np.asarray(inst.sdp.C[:n_cert, :n_cert], dtype=float).T
                        )
                        eps_override = float(cert_eps_active)
                    v, ok, t = _solve_lyapunov_with_solver(
                        inst.system,
                        solver_name,
                        eps=eps_override,
                        Q=q_override,
                    )
                    vals.append(v)
                    feas.append(ok)
                    tms.append(t)
                elif builder_kind == "lqr_feas":
                    v, ok, t = _solve_lqr_feas_with_solver(inst.system, solver_name)
                    vals.append(v)
                    feas.append(ok)
                    tms.append(t)
                else:
                    raise ValueError(f"Unsupported builder kind for external solver: {builder_kind}")
            except Exception as exc:
                print(f"  [warning] {solver_name} failed on {system_name} instance {inst.index}: {exc}")
                vals.append(float("nan"))
                feas.append(False)
                tms.append(float("nan"))
        ext_rows[solver_name] = {"values": vals, "feas": feas, "times": tms}

    if "MOSEK" in ext_rows:
        ref_values = np.asarray(ext_rows["MOSEK"]["values"], dtype=float)
        ref_total = float(np.nansum(np.asarray(ext_rows["MOSEK"]["times"], dtype=float)))
    elif "SCS" in ext_rows:
        ref_values = np.asarray(ext_rows["SCS"]["values"], dtype=float)
        ref_total = float(np.nansum(np.asarray(ext_rows["SCS"]["times"], dtype=float)))
    elif ipm_values:
        ref_values = np.asarray(ipm_values, dtype=float)
        ref_total = float(np.nansum(np.asarray(ipm_times_ms, dtype=float)))
    else:
        ref_values = np.full(len(test_instances), np.nan, dtype=float)
        ref_total = 1.0

    # L2WS training (episodic) and test evaluation.
    warm_values: list[float] = []
    warm_feas: list[bool] = []
    warm_times_ms: list[float] = []
    if use_l2ws:
        l2ws_kept_idx = [i for i, res in enumerate(train_ipm_results) if res.converged]
        l2ws_train_instances = [train_instances[i] for i in l2ws_kept_idx]

        if len(l2ws_train_instances) < 5:
            print(
                f"  [warning] {system_name}: only {len(l2ws_train_instances)}/{len(train_instances)} "
                "training instances converged under cold IPM; skipping L2WS."
            )
            warm_values = [float("nan")] * len(test_instances)
            warm_feas = [False] * len(test_instances)
            warm_times_ms = [float("nan")] * len(test_instances)
        else:
            sdp_n = l2ws_train_instances[0].sdp.dim
            sdp_m = l2ws_train_instances[0].sdp.num_constraints
            feat_dim = l2ws_train_instances[0].features.shape[0]
            node_feat_dim = None
            if effective_backbone == "gnn":
                first_graph = l2ws_train_instances[0].graph
                if first_graph is None:
                    from l2ws.data import build_system_graph

                    first_graph = build_system_graph(l2ws_train_instances[0].system)
                    l2ws_train_instances[0].graph = first_graph
                node_feat_dim = int(first_graph.x.shape[1])

            warm_model = WarmStartNet(
                n=sdp_n,
                m=sdp_m,
                input_dim=feat_dim,
                hidden_sizes=tuple(args.hidden_dims),
                dropout=float(args.dropout),
                use_batchnorm=False,
                backbone=effective_backbone,
                node_feat_dim=node_feat_dim,
                warmstart_type="cholesky",
            )
            warm_cfg = TrainerConfig(
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                learning_rate=float(args.warm_lr),
                weight_decay=1e-5,
                verbose=False,
            )
            ep_size = max(1, min(int(args.episode_size), len(l2ws_train_instances)))

            try:
                run_l2ws_lifecycle(
                    l2ws_train_instances,
                    ipm_solver,
                    warm_model,
                    warm_cfg,
                    episode_size=ep_size,
                    device=device,
                    backbone=effective_backbone,
                    retrain_strategy="finetune",
                    replay_capacity=max(ep_size * 10, 200),
                    standardize_inputs=True,
                )
                feature_mean = getattr(warm_model, "_feature_mean", None)
                feature_std = getattr(warm_model, "_feature_std", None)

                for inst in test_instances:
                    t0 = time.perf_counter()
                    pred = predict_warmstart(
                        warm_model,
                        inst,
                        device=device,
                        backbone=effective_backbone,
                        feature_mean=feature_mean,
                        feature_std=feature_std,
                    )
                    state = build_warm_state(inst, pred)
                    res = ipm_solver.solve(inst.sdp, initial_state=state)
                    if not res.converged:
                        res = ipm_solver.solve(inst.sdp)
                    t_ms = (time.perf_counter() - t0) * 1e3
                    warm_times_ms.append(t_ms)
                    warm_feas.append(bool(res.converged))
                    warm_values.append(_objective_from_ipm_result(res, inst, builder_kind) if res.converged else float("nan"))
            except RuntimeError as exc:
                print(f"  [warning] {system_name}: L2WS lifecycle failed ({exc}); skipping L2WS.")
                warm_values = [float("nan")] * len(test_instances)
                warm_feas = [False] * len(test_instances)
                warm_times_ms = [float("nan")] * len(test_instances)

    # L2A training/evaluation.
    l2a_values: list[float] = []
    l2a_feas: list[bool] = []
    l2a_times_ms: list[float] = []
    if use_l2a:
        train_feat = np.stack([inst.features for inst in train_instances], axis=0)
        train_labels = [
            _objective_from_ipm_result(res, inst, builder_kind) if res.converged else float("nan")
            for inst, res in zip(train_instances, train_ipm_results)
        ]
        train_labels_arr = np.asarray(train_labels, dtype=float)
        mask = np.isfinite(train_labels_arr)
        if np.sum(mask) < 5:
            l2a_values = [float("nan")] * len(test_instances)
            l2a_feas = [False] * len(test_instances)
            l2a_times_ms = [float("nan")] * len(test_instances)
        else:
            x_train = train_feat[mask]
            y_train = train_labels_arr[mask]
            x_mean = x_train.mean(axis=0)
            x_std = np.where(x_train.std(axis=0) < 1e-8, 1.0, x_train.std(axis=0))
            y_mean = float(np.mean(y_train))
            y_std = float(np.std(y_train))
            if y_std < 1e-8:
                y_std = 1.0

            l2a_model = ScalarL2ANet(input_dim=x_train.shape[1], hidden_sizes=(128, 64, 32))
            train_scalar_l2a(
                l2a_model,
                (x_train - x_mean) / x_std,
                (y_train - y_mean) / y_std,
                epochs=int(args.epochs),
                lr=float(args.warm_lr),
                batch_size=int(args.batch_size),
                device=args.device,
            )

            for inst in test_instances:
                x = (np.asarray(inst.features, dtype=float) - x_mean) / x_std
                xt = torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
                t0 = time.perf_counter()
                with torch.no_grad():
                    yhat = float(l2a_model(xt).cpu().numpy().reshape(-1)[0])
                val = yhat * y_std + y_mean
                if builder_kind == "lqr_feas":
                    val = 0.0
                t_ms = (time.perf_counter() - t0) * 1e3
                l2a_times_ms.append(t_ms)
                l2a_values.append(float(val))
                if builder_kind == "l2_gain":
                    l2a_feas.append(_hamiltonian_feasible(inst.system.A, inst.system.Bw, inst.system.Cz, max(float(val), 1e-8), 1e-7))
                else:
                    l2a_feas.append(np.isfinite(val))
    # L2CA training/evaluation.
    l2ca_values: list[float] = []
    l2ca_feas: list[bool] = []
    l2ca_times_ms: list[float] = []
    l2ca_gamma_inits: list[float] = []
    if use_l2ca:
        kept_idx = [i for i, res in enumerate(train_ipm_results) if res.converged]
        if len(kept_idx) < 5:
            l2ca_values = [float("nan")] * len(test_instances)
            l2ca_feas = [False] * len(test_instances)
            l2ca_times_ms = [float("nan")] * len(test_instances)
            l2ca_gamma_inits = [float("nan")] * len(test_instances)
        else:
            train_kept_instances = [train_instances[i] for i in kept_idx]
            x_train = np.stack([inst.features for inst in train_kept_instances], axis=0)
            y_train_target = np.stack([np.asarray(train_ipm_results[i].y, dtype=float) for i in kept_idx], axis=0)
            x_mean = x_train.mean(axis=0)
            x_std = np.where(x_train.std(axis=0) < 1e-8, 1.0, x_train.std(axis=0))
            y_mean = y_train_target.mean(axis=0)
            y_std = np.where(y_train_target.std(axis=0) < 1e-8, 1.0, y_train_target.std(axis=0))
            use_interior_labels = bool(int(args.l2ca_interiorize_labels)) and _is_certificate_builder(builder_kind)

            dual_model = DualNet(input_dim=x_train.shape[1], output_dim=y_train_target.shape[1])
            train_dual_net(
                dual_model,
                (x_train - x_mean) / x_std,
                (y_train_target - y_mean) / y_std,
                epochs=int(args.epochs),
                lr=float(args.warm_lr),
                batch_size=int(args.batch_size),
                device=args.device,
                b_targets=np.stack([np.asarray(inst.sdp.b, dtype=float) for inst in train_kept_instances], axis=0),
                lambda_obj=float(args.l2ca_lambda_obj),
                y_targets_unscaled=y_train_target,
                y_mean=y_mean,
                y_std=y_std,
                sdp_instances=[inst.sdp for inst in train_kept_instances],
                lambda_feas=float(args.l2ca_lambda_feas),
                feas_margin=float(args.l2ca_feas_margin),
                feas_loss_mode=str(args.l2ca_feas_loss),
                feas_margin_train=float(args.l2ca_feas_margin_train),
                label_margin_debug=bool(args.l2ca_label_margin_debug),
                interiorize_labels=bool(use_interior_labels),
                interior_delta=float(args.l2ca_interior_delta),
                lambda_y=float(args.l2ca_lambda_y),
                obj_debug=bool(args.l2ca_debug),
            )

            x_train_norm = (x_train - x_mean) / x_std
            train_sdps_kept = [inst.sdp for inst in train_kept_instances]
            global_mean_anchor = np.mean(y_train_target, axis=0)
            cached_feasible_anchor = select_robust_anchor(train_sdps_kept, y_train_target)
            legacy_degenerate = bool(_is_certificate_builder(builder_kind) and not cert_regularization_active)

            if bool(args.l2ca_debug) and _is_certificate_builder(builder_kind):
                print(
                    f"  [l2ca-debug] certificate_regularization={'on' if cert_regularization_active else 'off'} "
                    f"(legacy_degenerate_path={'on' if legacy_degenerate else 'off'})"
                )

            anchor_mode = str(args.l2ca_anchor).strip().lower()
            force_global_anchor = anchor_mode == "global_mean"
            anchor_k = 1 if anchor_mode == "knn1" else 5
            tier_auto_enabled = str(args.l2ca_tier_auto).strip().lower() == "on"
            tier0_fallback_mode = str(args.l2ca_tier0_fallback).strip().lower()
            tier_level = 0
            tier_j_idx: int | None = None
            if tier_auto_enabled:
                tier_level, tier_j_idx = detect_tier(
                    train_kept_instances[0].sdp,
                    tol=float(args.l2ca_feas_margin),
                )
                if bool(args.l2ca_debug):
                    print(
                        f"  [l2ca-debug] tier_auto=on tier={int(tier_level)} "
                        f"tier2_j={tier_j_idx}"
                    )
            elif bool(args.l2ca_debug):
                print("  [l2ca-debug] tier_auto=off (legacy anchor path)")

            for inst in test_instances:
                x = (np.asarray(inst.features, dtype=float) - x_mean) / x_std
                xt = torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
                t0 = time.perf_counter()
                with torch.no_grad():
                    y_scaled = dual_model(xt).cpu().numpy().reshape(-1)
                y_pred = y_scaled * y_std + y_mean

                l2ca_result = run_l2ca_inference(
                    sdp=inst.sdp,
                    y_pred=y_pred,
                    x_feat=x,
                    x_train_norm=x_train_norm,
                    y_train=y_train_target,
                    cached_feasible_anchor=cached_feasible_anchor,
                    feas_margin=float(args.l2ca_feas_margin),
                    bisect_iters=int(args.l2ca_bisect_iters),
                    tier_auto=bool(tier_auto_enabled),
                    tier_level=int(tier_level),
                    tier_j_idx=tier_j_idx,
                    tier0_fallback=str(tier0_fallback_mode),
                    force_global_anchor=bool(force_global_anchor),
                    global_mean_anchor=global_mean_anchor,
                    anchor_k=int(anchor_k),
                    skip_lift=builder_kind == "lqr_feas",
                    refine_solver=ipm_solver,
                    refine_iters=int(args.l2ca_refine_iters),
                    enable_refine=(not bool(args.l2ca_refine_only_degenerate)) or bool(legacy_degenerate),
                    debug=bool(args.l2ca_debug),
                )
                y_out = np.asarray(l2ca_result["y_out"], dtype=float).reshape(-1)
                l2_ok = bool(l2ca_result["final_ok"])
                fast_path_accept = bool(l2ca_result["fast_path_accept"])
                repair_ok = bool(l2ca_result["repair_ok"])
                anchor_info = dict(l2ca_result["anchor_info"])
                repair_stats = dict(l2ca_result["repair_stats"])
                lift_stats = dict(l2ca_result["lift_stats"])
                tier0_stats = dict(l2ca_result["tier0_stats"])
                instance_stats = dict(l2ca_result["instance_stats"])
                anchor_feasible = bool(l2ca_result["anchor_feasible"])
                refine_ran = bool(l2ca_result["refine_ran"])
                refine_accepted = bool(l2ca_result["refine_accepted"])
                refine_used_iters = int(l2ca_result["refine_used_iters"])

                t_ms = (time.perf_counter() - t0) * 1e3
                l2ca_times_ms.append(t_ms)
                l2ca_feas.append(bool(l2_ok))
                l2ca_values.append(float(np.dot(inst.sdp.b, y_out)))
                if builder_kind == "l2_gain":
                    l2ca_gamma_inits.append(extract_gamma_from_dual(y_out, inst.sdp.b))
                else:
                    l2ca_gamma_inits.append(float("nan"))

                if bool(args.l2ca_debug):
                    total_chol_checks = (
                        int(anchor_info.get("chol_checks", 0))
                        + int(repair_stats.get("cholesky_checks", 0))
                        + int(lift_stats.get("cholesky_checks", 0))
                        + 2
                    )
                    total_apply_at = (
                        int(instance_stats.get("apply_at_calls", 0))
                        + int(anchor_info.get("apply_at_calls", 0))
                        + int(repair_stats.get("apply_at_calls", 0))
                        + int(lift_stats.get("apply_at_calls", 0))
                    )
                    print(
                        f"  [l2ca-debug] inst={inst.index} fast_path={bool(fast_path_accept)} "
                        f"tier_auto={bool(tier_auto_enabled)} tier={int(tier_level)} "
                        f"anchor_mode={anchor_info.get('mode', 'none')} anchor_feas={bool(anchor_feasible)} "
                        f"repair_ok={bool(repair_ok)} repair_bisect={int(repair_stats.get('bisect_steps', 0))} "
                        f"tier0_used={bool(tier0_stats.get('used', False))} "
                        f"tier0_step={tier0_stats.get('step', 'none')} "
                        f"lift_ran={bool(lift_stats.get('ran', False))} lift_bisect={int(lift_stats.get('bisect_steps', 0))} "
                        f"refine_ran={bool(refine_ran)} refine_accept={bool(refine_accepted)} "
                        f"refine_iters={int(refine_used_iters)} "
                        f"total_applyAT={int(total_apply_at)} total_chol={int(total_chol_checks)}"
                    )

    # L2CA-H for L2 gain only.
    l2cah_values: list[float] = []
    l2cah_feas: list[bool] = []
    l2cah_times_ms: list[float] = []
    if use_l2cah:
        for idx, inst in enumerate(test_instances):
            init_candidates: list[float] = [1e-6]
            if idx < len(l2a_values) and np.isfinite(l2a_values[idx]):
                init_candidates.append(float(l2a_values[idx]))
            if idx < len(l2ca_gamma_inits) and np.isfinite(l2ca_gamma_inits[idx]):
                init_candidates.append(float(l2ca_gamma_inits[idx]))
            gamma_init = max(init_candidates)
            t0 = time.perf_counter()
            gamma_h, _ = hamiltonian_bisection(inst.system.A, inst.system.Bw, inst.system.Cz, gamma_init=gamma_init)
            t_ms = (time.perf_counter() - t0) * 1e3
            l2cah_times_ms.append(t_ms)
            l2cah_values.append(float(gamma_h))
            l2cah_feas.append(_hamiltonian_feasible(inst.system.A, inst.system.Bw, inst.system.Cz, max(float(gamma_h), 1e-8), 1e-7))

    summaries: list[MethodSummary] = []

    # Reference and external rows.
    if use_mosek and "MOSEK" in ext_rows:
        ref_mosek = MethodSummary(
            method="MOSEK",
            feas_rate=100.0 * float(np.mean(np.asarray(ext_rows["MOSEK"]["feas"], dtype=float))),
            mean_err=float("nan"),
            std_err=float("nan"),
            max_err=float("nan"),
            mean_time_ms=float(np.mean(np.asarray(ext_rows["MOSEK"]["times"], dtype=float))),
            total_time_ms=float(np.sum(np.asarray(ext_rows["MOSEK"]["times"], dtype=float))),
            speedup=1.0,
        )
        summaries.append(ref_mosek)

    if use_scs and "SCS" in ext_rows:
        summaries.append(
            _build_solver_row(
                ext_rows["SCS"]["values"],
                ext_rows["SCS"]["feas"],
                ext_rows["SCS"]["times"],
                ref_values,
                ref_total,
                "SCS",
            )
        )

    if use_ipm and ipm_values:
        summaries.append(_build_solver_row(ipm_values, ipm_feas, ipm_times_ms, ref_values, ref_total, "IPM (cold)"))
    if use_l2ws and warm_values:
        summaries.append(_build_solver_row(warm_values, warm_feas, warm_times_ms, ref_values, ref_total, "L2WS"))
    if use_l2a and l2a_values:
        summaries.append(_build_solver_row(l2a_values, l2a_feas, l2a_times_ms, ref_values, ref_total, "L2A"))
    if use_l2ca and l2ca_values:
        l2ca_ref_values = ref_values
        if builder_kind == "l2_gain":
            # MOSEK/IPM references are reported as gamma for L2 gain, while L2CA now
            # reports dual objective b^T y. Convert gamma -> -gamma^2 for fair error.
            l2ca_ref_values = -np.square(ref_values)
        summaries.append(_build_solver_row(l2ca_values, l2ca_feas, l2ca_times_ms, l2ca_ref_values, ref_total, "L2CA"))
    if use_l2cah and l2cah_values:
        summaries.append(_build_solver_row(l2cah_values, l2cah_feas, l2cah_times_ms, ref_values, ref_total, "L2CA-H"))

    if not summaries:
        print(f"[warning] No selected algorithms produced results for {system_name}.")
        return []

    title = f"{title_description} — {system_name}"
    _print_table(title, summaries, n_states=system.A.shape[0], sdp_dim=test_instances[0].sdp.dim)

    if args.save_results:
        out = Path("results") / f"{study_key}_{system_name}.csv"
        _save_csv(out, summaries)
        print(f"Saved: {out}")

    return summaries


def _entry_selection_tokens(study_key: str, entry: tuple[Any, ...]) -> set[str]:
    name = str(entry[0]).strip().lower()
    tokens = {name}
    if study_key.startswith("power_") and len(entry) >= 3:
        bus = str(entry[2]).strip().lower()
        digits = "".join(ch for ch in bus if ch.isdigit())
        if digits:
            tokens.update({digits, f"{digits}bus", f"bus{digits}"})
    return tokens


def run_study(study_key: str, args: argparse.Namespace) -> None:
    cfg = CASE_STUDIES[study_key]
    description = cfg["description"]
    builder_kind = cfg["sdp_builder"]
    perturb = tuple(cfg["perturb"])
    selected_systems = {s.strip().lower() for s in args.systems if s.strip()}
    run_all_systems = ("all" in selected_systems) or (not selected_systems)

    if not run_all_systems:
        print(f"[info] {study_key}: filtering systems to {sorted(selected_systems)}")

    matched = 0
    for entry in cfg["systems"]:
        if not run_all_systems:
            entry_tokens = _entry_selection_tokens(study_key, entry)
            if entry_tokens.isdisjoint(selected_systems):
                continue
        matched += 1
        loaded = _load_system_for_entry(study_key, entry, args)
        if loaded is None:
            continue
        name, system = loaded
        _evaluate_system(study_key, description, name, system, builder_kind, perturb, args)
    if matched == 0:
        print(f"[warning] No systems matched --systems {args.systems} for study {study_key}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run unified L2WS/L2A/L2CA case studies.")
    parser.add_argument("--study", choices=[*CASE_STUDIES.keys(), "all"], required=True)
    parser.add_argument("--num-train", type=int, default=200)
    parser.add_argument("--num-test", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--power-mat", default="data/PowerSystemData_Consolidated.mat")
    parser.add_argument("--compleib-dir", default="COMPlib_r1_1")
    parser.add_argument("--save-results", action="store_true")
    parser.add_argument("--backbone", choices=["mlp", "gnn"], default="mlp")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 64, 32, 16])
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warm-lr", type=float, default=5e-4)
    parser.add_argument("--episode-size", type=int, default=20)
    parser.add_argument(
        "--l2ca-anchor",
        choices=["knn1", "knn5_best", "global_mean"],
        default="knn5_best",
        help="Training-based anchor mode for L2CA feasibility repair.",
    )
    parser.add_argument(
        "--l2ca-feas-margin",
        type=float,
        default=1e-8,
        help="Margin used by Cholesky feasibility checks in L2CA.",
    )
    parser.add_argument(
        "--l2ca-bisect-iters",
        type=int,
        default=20,
        help="Maximum feasibility-bisection iterations for L2CA.",
    )
    parser.add_argument(
        "--l2ca-tier-auto",
        choices=["on", "off"],
        default="on",
        help="Enable automatic tiered anchor/repair for L2CA.",
    )
    parser.add_argument(
        "--l2ca-tier0-fallback",
        choices=["off", "fi", "short_ipm"],
        default="off",
        help="Tier-0 guaranteed fallback mode for L2CA.",
    )
    parser.add_argument(
        "--l2ca-lambda-obj",
        type=float,
        default=0.0,
        help="Weight on objective-aware dual loss term for L2CA/L2CA-FI dual-net training.",
    )
    parser.add_argument(
        "--l2ca-lambda-feas",
        type=float,
        default=0.0,
        help="Weight on feasibility-aware Cholesky-margin dual loss for L2CA/L2CA-FI training.",
    )
    parser.add_argument(
        "--l2ca-feas-loss",
        choices=["shift", "eig"],
        default="shift",
        help="Feasibility-aware dual loss mode for L2CA/L2CA-FI training.",
    )
    parser.add_argument(
        "--l2ca-feas-margin-train",
        type=float,
        default=0.0,
        help="Training-time feasibility margin used by eig-based feasibility loss.",
    )
    parser.add_argument(
        "--l2ca-label-margin-debug",
        action="store_true",
        default=False,
        help="Print label slack-margin diagnostics for L2CA/L2CA-FI dual training targets.",
    )
    parser.add_argument(
        "--l2ca-interiorize-labels",
        type=int,
        choices=[0, 1],
        default=1,
        help="Enable interiorized dual labels for certificate L2CA/L2CA-FI training (1=yes, 0=no).",
    )
    parser.add_argument(
        "--l2ca-interior-delta",
        type=float,
        default=1e-3,
        help="Target slack margin delta for interiorized dual labels.",
    )
    parser.add_argument(
        "--l2ca-lambda-y",
        type=float,
        default=1.0,
        help="Weight on supervised dual MSE term for L2CA/L2CA-FI training.",
    )
    parser.add_argument(
        "--l2ca-refine-iters",
        type=int,
        default=0,
        help="Optional tiny post-correction IPM refinement iterations for L2CA.",
    )
    parser.add_argument(
        "--l2ca-refine-only-degenerate",
        dest="l2ca_refine_only_degenerate",
        action="store_true",
        default=True,
        help="Apply L2CA refinement only for legacy (non-regularized) certificate cases.",
    )
    parser.add_argument(
        "--no-l2ca-refine-only-degenerate",
        dest="l2ca_refine_only_degenerate",
        action="store_false",
        help="Allow L2CA refinement on non-degenerate datasets as well.",
    )
    parser.add_argument(
        "--degenerate-mode",
        choices=["auto", "on", "off", "force"],
        default="auto",
        help="Controls auto certificate regularization trigger in --cert-regularize auto mode.",
    )
    parser.add_argument(
        "--cert-regularize",
        choices=["auto", "on", "off"],
        default="auto",
        help="Regularize certificate-feasibility SDPs with a canonical linear objective.",
    )
    parser.add_argument(
        "--cert-eps",
        type=float,
        default=1e-6,
        help="Optional strict Lyapunov margin eps in A'P + PA + eps I <= 0.",
    )
    parser.add_argument(
        "--cert-Q",
        choices=["identity", "diag_from_A", "custom_diag"],
        default="identity",
        help="Q construction mode for certificate regularization objective tr(QP).",
    )
    parser.add_argument(
        "--cert-Q-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to Q in tr(QP).",
    )
    parser.add_argument(
        "--l2ca-debug",
        action="store_true",
        help="Print L2CA debug diagnostics and counters.",
    )
    parser.add_argument(
        "--compleib-decay-margin",
        type=float,
        default=5.0,
        help="For COMPlib l2_gain/lyapunov studies, shift A to enforce Re(lambda)<=-margin for IPM conditioning.",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        default=["all"],
        help="Which systems to run inside the chosen study (e.g., all, 9bus, 14, MSD-8).",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["all"],
        help="Algorithms to run: all, MOSEK, SCS, IPM, L2WS, L2A, L2CA, L2CA-H.",
    )
    args = parser.parse_args()
    try:
        args._algorithm_set = _normalize_algorithm_selection(args.algorithms)
    except ValueError as exc:
        parser.error(str(exc))

    # Rebase configured paths for convenience.
    if args.study.startswith("compleib") and not Path(args.compleib_dir).exists():
        print(f"[warning] COMPlib directory not found: {args.compleib_dir}; skipping COMPleib studies.")
        if args.study != "all":
            return

    if args.study.startswith("power_") and not Path(args.power_mat).exists():
        print(f"[warning] Power data file not found: {args.power_mat}; skipping {args.study}.")
        return

    studies = list(CASE_STUDIES.keys()) if args.study == "all" else [args.study]
    for key in studies:
        if key.startswith("compleib") and not Path(args.compleib_dir).exists():
            print(f"[warning] Skipping {key}: missing COMPleib directory.")
            continue
        if key.startswith("power_") and not Path(args.power_mat).exists():
            print(f"[warning] Skipping {key}: missing power data file.")
            continue
        run_study(key, args)


if __name__ == "__main__":
    main()
