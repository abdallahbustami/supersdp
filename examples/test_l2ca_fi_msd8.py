"""Configurable MSD-8 benchmark including L2CA-FI (standalone prototype).

Run:
  PYTHONPATH=src ./l2ws_env/bin/python examples/test_l2ca_fi_msd8.py \
    --num-train 100 --num-test 20 --epochs 50 --seed 44 \
    --algorithms MOSEK SCS IPM L2WS L2A L2CA L2CA-H L2CA-QP L2CA-FI
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from l2ws.benchmarks import installed_solvers, solve_gamma_with_solver, solve_tridiagonal_mass_spring_chain
from l2ws.data import LTISystem, build_l2_gain_sdp
from l2ws.ipm import IPMSettings, InfeasibleIPMSolver
from l2ws.l2ca import (
    DualNet,
    ScalarL2ANet,
    _hamiltonian_feasible,
    bisection_to_feasible,
    check_c_psd,
    choose_anchor_best_knn,
    extract_gamma_from_dual,
    get_anchor_y,
    hamiltonian_bisection,
    is_psd_cholesky,
    is_psd_with_margin,
    select_robust_anchor,
    slack_from_y,
    train_dual_net,
    train_scalar_l2a,
)
from l2ws.l2ca_fi import l2ca_fi_predict
import l2ws.l2ca_fi as fi
from l2ws.l2ws import ProblemConfig, SolverConfig, SuperSDP, TrainingConfig
from l2ws.perturbations import PerturbationSpec, perturb_system

try:
    from l2ws.l2ca_qp import l2ca_qp_predict
except Exception:
    def _proj_simplex(v: np.ndarray) -> np.ndarray:
        z = np.asarray(v, dtype=float).reshape(-1)
        if z.size == 0:
            return z
        u = np.sort(z)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * (np.arange(1, z.size + 1)) > (cssv - 1.0))[0]
        if rho.size == 0:
            theta = 0.0
        else:
            r = int(rho[-1])
            theta = (cssv[r] - 1.0) / float(r + 1)
        w = np.maximum(z - theta, 0.0)
        s = float(np.sum(w))
        if s > 0:
            w /= s
        else:
            w = np.ones_like(w) / float(w.size)
        return w

    def _solve_simplex_qp(Y: np.ndarray, y_hat: np.ndarray, iters: int = 50) -> np.ndarray:
        Y = np.asarray(Y, dtype=float)
        y_hat = np.asarray(y_hat, dtype=float).reshape(-1)
        k = Y.shape[1]
        alpha = np.ones(k, dtype=float) / float(k)
        gram = Y.T @ Y
        L = 2.0 * float(np.linalg.norm(gram, 2))
        eta = 1.0 / (L + 1e-12)
        for _ in range(int(iters)):
            grad = 2.0 * (Y.T @ (Y @ alpha - y_hat))
            alpha = _proj_simplex(alpha - eta * grad)
        return alpha

    def l2ca_qp_predict(
        test_feat: np.ndarray,
        train_feats: np.ndarray,
        train_y: np.ndarray,
        k: int = 5,
        qp_iters: int = 50,
    ) -> np.ndarray:
        x = np.asarray(test_feat, dtype=float).reshape(1, -1)
        X = np.asarray(train_feats, dtype=float)
        Yall = np.asarray(train_y, dtype=float)
        d = np.linalg.norm(X - x, axis=1)
        k_eff = max(1, min(int(k), X.shape[0]))
        idx = np.argsort(d, kind="mergesort")[:k_eff]
        Y = Yall[idx].T
        y_hat = np.mean(Yall[idx], axis=0)
        alpha = _solve_simplex_qp(Y, y_hat, iters=int(qp_iters))
        return Y @ alpha


ALGORITHM_CANONICAL = (
    "MOSEK",
    "SCS",
    "IPM",
    "L2WS",
    "L2A",
    "L2CA",
    "L2CA-H",
    "L2CA-QP",
    "L2CA-FI",
)
ALGORITHM_ALIASES = {
    "mosek": "MOSEK",
    "scs": "SCS",
    "ipm": "IPM",
    "l2ws": "L2WS",
    "l2a": "L2A",
    "l2ca": "L2CA",
    "l2ca-h": "L2CA-H",
    "l2cah": "L2CA-H",
    "l2ca_qp": "L2CA-QP",
    "l2ca-qp": "L2CA-QP",
    "l2caqp": "L2CA-QP",
    "l2ca_fi": "L2CA-FI",
    "l2ca-fi": "L2CA-FI",
    "l2cafi": "L2CA-FI",
}

_FI_FEAS_SHIFTS = np.asarray([0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1], dtype=float)


@dataclass
class InstanceRecord:
    index: int
    system: LTISystem
    sdp: any
    feature: np.ndarray


@dataclass
class MethodData:
    name: str
    values: List[float]
    feas: List[bool] | None
    times_ms: List[float]
    value_kind: str  # "gamma" or "dual"


@dataclass
class MethodSummary:
    name: str
    feas_rate: float | None
    mean_err: float | None
    std_err: float | None
    max_err: float | None
    mean_time_ms: float | None
    total_time_ms: float | None
    speedup: float | None


def _build_feature(system: LTISystem) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(system.A, dtype=float).reshape(-1),
            np.asarray(system.Bw, dtype=float).reshape(-1),
        ]
    )


def _build_instances(base_system: LTISystem, num: int, spec: PerturbationSpec) -> List[InstanceRecord]:
    out: List[InstanceRecord] = []
    for i in range(int(num)):
        sys_i = perturb_system(base_system, spec, i)
        sdp_i = build_l2_gain_sdp(sys_i.A, sys_i.Bw, sys_i.Cz, name=f"msd8_l2gain_{i}")
        out.append(
            InstanceRecord(
                index=i,
                system=sys_i,
                sdp=sdp_i,
                feature=_build_feature(sys_i),
            )
        )
    return out


def _normalize_algorithms(algs: Sequence[str]) -> List[str]:
    out: List[str] = []
    for a in algs:
        key = str(a).strip().lower()
        if key not in ALGORITHM_ALIASES:
            raise ValueError(
                f"Unknown algorithm '{a}'. Supported: {', '.join(ALGORITHM_CANONICAL)}"
            )
        canon = ALGORITHM_ALIASES[key]
        if canon not in out:
            out.append(canon)
    return out


def _fmt_err(mean_e: float | None, std_e: float | None) -> str:
    if mean_e is None or std_e is None:
        return "--"
    if np.isfinite(mean_e) and np.isfinite(std_e):
        return f"{100.0 * mean_e:.3f}% ± {100.0 * std_e:.3f}%"
    return "--"


def _dual_feasible_chol(sdp, y: np.ndarray, tol: float = 1e-7, jitter: float = 1e-9) -> bool:
    S = slack_from_y(sdp, np.asarray(y, dtype=float).reshape(-1))
    n = S.shape[0]
    return is_psd_cholesky(S + (float(tol) + float(jitter)) * np.eye(n, dtype=float), jitter=jitter)


def _fi_feas_penalty_from_slack(
    S: np.ndarray,
    mode: str = "shift",
    margin_train: float = 0.0,
) -> float:
    S = 0.5 * (np.asarray(S, dtype=float) + np.asarray(S, dtype=float).T)
    mode_norm = str(mode).strip().lower()
    if mode_norm == "eig":
        lam = float(np.linalg.eigvalsh(S)[0])
        return float(max(float(margin_train) - lam, 0.0) ** 2)

    n = S.shape[0]
    I = np.eye(n, dtype=float)
    for delta in _FI_FEAS_SHIFTS:
        try:
            np.linalg.cholesky(S + float(delta) * I)
            return float(delta)
        except np.linalg.LinAlgError:
            continue
    return float(_FI_FEAS_SHIFTS[-1] * 10.0)


def _relative_error(
    values: List[float],
    ref_gamma: np.ndarray | None,
    value_kind: str,
) -> tuple[float | None, float | None, float | None]:
    if ref_gamma is None:
        return None, None, None
    vals = np.asarray(values, dtype=float)
    ref = np.asarray(ref_gamma, dtype=float)
    if value_kind == "gamma":
        tgt = ref
    elif value_kind == "dual":
        tgt = -np.square(ref)
    else:
        return None, None, None
    mask = np.isfinite(vals) & np.isfinite(tgt)
    if not np.any(mask):
        return None, None, None
    rel = np.abs(vals[mask] - tgt[mask]) / np.maximum(np.abs(tgt[mask]), 1e-12)
    return float(np.mean(rel)), float(np.std(rel)), float(np.max(rel))


def _summarize(
    method: MethodData,
    ref_gamma: np.ndarray | None,
    hide_error: bool,
    baseline_time_ms: float | None,
) -> MethodSummary:
    tms = np.asarray(method.times_ms, dtype=float)
    finite_t = tms[np.isfinite(tms)]
    mean_t = float(np.mean(finite_t)) if finite_t.size else None
    total_t = float(np.sum(finite_t)) if finite_t.size else None

    if method.feas is None:
        feas_rate = None
    else:
        feas_arr = np.asarray(method.feas, dtype=bool)
        feas_rate = float(np.mean(feas_arr.astype(float))) if feas_arr.size else None

    if hide_error:
        mean_e = std_e = max_e = None
    else:
        mean_e, std_e, max_e = _relative_error(method.values, ref_gamma, method.value_kind)

    speed = None
    if mean_t is not None and baseline_time_ms is not None and mean_t > 0.0 and np.isfinite(mean_t) and np.isfinite(baseline_time_ms):
        speed = float(baseline_time_ms / mean_t)

    return MethodSummary(
        name=method.name,
        feas_rate=feas_rate,
        mean_err=mean_e,
        std_err=std_e,
        max_err=max_e,
        mean_time_ms=mean_t,
        total_time_ms=total_t,
        speedup=speed,
    )


def _print_table(
    summaries: List[MethodSummary],
    n_states: int,
    sdp_dim: int,
    args: argparse.Namespace,
    ref_name: str | None,
    kept_train: int,
    skipped_train: int,
) -> None:
    print("")
    print(f"=== MSD-8 L2 Gain Benchmark + L2CA-FI (n={n_states}, SDP dim={sdp_dim}) ===")
    print(
        f"[info] train={args.num_train}, test={args.num_test}, epochs={args.epochs}, "
        f"seed={args.seed}, kept_train={kept_train}, skipped_train={skipped_train}"
    )
    print(f"[info] algorithms={', '.join(_normalize_algorithms(args.algorithms))}")
    print(f"[info] reference objective row={ref_name if ref_name is not None else 'none'}")
    print(
        f"{'Algorithm':<10} | {'Feas %':>6} | {'Mean Err ± Std':>18} | {'Max Err':>8} | "
        f"{'Mean Time':>9} | {'Total Time':>10} | {'Speedup':>8}"
    )
    print("-" * 96)
    for row in summaries:
        feas_txt = "--" if row.feas_rate is None else f"{100.0 * row.feas_rate:6.1f}"
        mean_t = "nan" if row.mean_time_ms is None else f"{row.mean_time_ms:.3f}ms"
        total_t = "nan" if row.total_time_ms is None else f"{row.total_time_ms:.3f}ms"
        speed = "--" if row.speedup is None else f"{row.speedup:.1f}x"
        max_err_txt = "--" if row.max_err is None else f"{100.0 * row.max_err:.3f}%"
        print(
            f"{row.name:<10} | {feas_txt:>6} | {_fmt_err(row.mean_err, row.std_err):>18} | "
            f"{max_err_txt:>8} | {mean_t:>9} | {total_t:>10} | {speed:>8}"
        )


def _print_fi_diagnostics(diag: Dict[str, Any]) -> None:
    print("")
    print("=== L2CA-FI diagnostics ===")
    n_test = int(diag.get("n_test", 0))
    pred_count = int(diag.get("pred_count", 0))
    pred_feas_count = int(diag.get("pred_feas_count", 0))
    accept_rate = float(pred_feas_count) / float(n_test) if n_test > 0 else float("nan")
    print(
        f"[fi-debug] raw DualNet accept rate: {100.0 * accept_rate:.1f}% "
        f"({pred_feas_count}/{n_test}); y_pred available for {pred_count}/{n_test}"
    )

    lam = np.asarray(diag.get("lam_min_values", []), dtype=float)
    if lam.size > 0:
        print(
            "[fi-debug] lam_min(y_pred slack): "
            f"mean={float(np.mean(lam)):.3e}, "
            f"median={float(np.median(lam)):.3e}, "
            f"min={float(np.min(lam)):.3e}"
        )
    else:
        print("[fi-debug] lam_min(y_pred slack): unavailable")

    pen = np.asarray(diag.get("feas_penalties", []), dtype=float)
    if pen.size > 0:
        print(
            f"[fi-debug] mean feasibility penalty(y_pred): {float(np.mean(pen)):.6e} "
            f"(mode={diag.get('feas_loss_mode', 'shift')}, margin_train={float(diag.get('feas_margin_train', 0.0)):.1e})"
        )
    else:
        print("[fi-debug] mean feasibility penalty(y_pred): unavailable")

    counts = diag.get("stage_counts", {})
    order = ["accept_pred", "spectral_cut", "subgradient", "short_ipm", "full_ipm"]
    print("[fi-debug] stage counts:")
    for key in order:
        c = int(counts.get(key, 0))
        pct = 100.0 * float(c) / float(n_test) if n_test > 0 else float("nan")
        print(f"  {key}: {c} ({pct:.1f}%)")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Configurable MSD-8 benchmark with L2CA-FI.")
    p.add_argument("--num-train", type=int, default=80)
    p.add_argument("--num-test", type=int, default=10)
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-sizes", type=str, default="128,64,32")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=118)
    p.add_argument("--msd-size", type=int, default=8)
    p.add_argument("--A-scale", type=float, default=0.03)
    p.add_argument("--B-scale", type=float, default=0.02)
    p.add_argument("--k", type=int, default=5, help="k for kNN-based methods.")
    p.add_argument("--l2ca-qp-iters", type=int, default=50)
    p.add_argument("--l2ca-anchor", choices=["knn1", "knn5_best", "global_mean"], default="knn5_best")
    p.add_argument("--l2ca-feas-margin", type=float, default=1e-8)
    p.add_argument("--l2ca-bisect-iters", type=int, default=20)
    p.add_argument("--l2ca-lambda-obj", type=float, default=0.0)
    p.add_argument("--l2ca-lambda-feas", type=float, default=0.0)
    p.add_argument("--l2ca-feas-loss", choices=["shift", "eig"], default="shift")
    p.add_argument("--l2ca-feas-margin-train", type=float, default=0.0)
    p.add_argument("--l2ca-label-margin-debug", action="store_true", default=False)
    p.add_argument("--l2ca-interiorize-labels", type=int, choices=[0, 1], default=0)
    p.add_argument("--l2ca-interior-delta", type=float, default=1e-3)
    p.add_argument("--l2ca-lambda-y", type=float, default=1.0)
    p.add_argument("--l2ca-fi-phase1-iters", type=int, default=30)
    p.add_argument("--l2ca-fi-jitter", type=float, default=1e-10)
    p.add_argument("--l2ca-fi-objective-lift", type=int, choices=[0, 1], default=0)
    p.add_argument("--fi-debug", action="store_true", default=False)
    p.add_argument(
        "--algorithms",
        nargs="+",
        default=["MOSEK", "SCS", "IPM", "L2WS", "L2A", "L2CA", "L2CA-H", "L2CA-QP", "L2CA-FI"],
        help="Any subset of: MOSEK SCS IPM L2WS L2A L2CA L2CA-H L2CA-QP L2CA-FI",
    )
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    algorithms = _normalize_algorithms(args.algorithms)

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    hidden_sizes = tuple(int(v.strip()) for v in str(args.hidden_sizes).split(",") if v.strip())
    if not hidden_sizes:
        raise ValueError("--hidden-sizes must contain at least one integer.")

    base_system = solve_tridiagonal_mass_spring_chain(int(args.msd_size))
    spec = PerturbationSpec(
        kind="entrywise_uniform",
        A_scale=float(args.A_scale),
        B_scale=float(args.B_scale),
        seed=int(args.seed),
    )
    instances = _build_instances(base_system, int(args.num_train) + int(args.num_test), spec)
    train_instances = instances[: int(args.num_train)]
    test_instances = instances[int(args.num_train) :]
    if not train_instances or not test_instances:
        raise RuntimeError("Empty train/test split.")

    ipm = InfeasibleIPMSolver(
        IPMSettings(
            max_iters=120,
            tol_abs=1e-6,
            tol_rel=1e-5,
            linear_solve="sylvester",
        )
    )

    # External solver rows.
    ext_rows: Dict[str, MethodData] = {}
    requested_external = [a for a in algorithms if a in {"MOSEK", "SCS"}]
    installed: set[str] = set()
    if requested_external:
        try:
            installed = installed_solvers()
        except Exception:
            installed = set()
        for solver_name in requested_external:
            if solver_name not in installed:
                print(f"[warning] {solver_name} requested but not installed; skipping.")
                continue
            vals: List[float] = []
            feas: List[bool] = []
            tms: List[float] = []
            for inst in test_instances:
                row = solve_gamma_with_solver(inst.system, solver_name)
                vals.append(float(row.value_gamma))
                feas.append(bool(row.feasible))
                tms.append(1000.0 * float(row.time_sec))
            ext_rows[solver_name] = MethodData(
                name=solver_name,
                values=vals,
                feas=feas,
                times_ms=tms,
                value_kind="gamma",
            )

    need_train_ipm = any(a in algorithms for a in {"L2WS", "L2A", "L2CA", "L2CA-H", "L2CA-QP", "L2CA-FI"})
    train_ipm_results = []
    if need_train_ipm:
        for inst in train_instances:
            train_ipm_results.append(ipm.solve(inst.sdp))

    # IPM test baseline / fallback reference.
    need_ipm_test = ("IPM" in algorithms) or (len(ext_rows) == 0)
    ipm_row: MethodData | None = None
    if need_ipm_test:
        vals: List[float] = []
        feas: List[bool] = []
        tms: List[float] = []
        for inst in test_instances:
            t0 = time.perf_counter()
            res = ipm.solve(inst.sdp)
            tms.append((time.perf_counter() - t0) * 1e3)
            feas.append(bool(res.converged))
            if res.converged:
                vals.append(extract_gamma_from_dual(res.y, inst.sdp.b))
            else:
                vals.append(float("nan"))
        ipm_row = MethodData(name="IPM", values=vals, feas=feas, times_ms=tms, value_kind="gamma")

    # Reference gamma for error reporting.
    ref_name: str | None = None
    ref_gamma: np.ndarray | None = None
    if "MOSEK" in ext_rows:
        ref_name = "MOSEK"
        ref_gamma = np.asarray(ext_rows["MOSEK"].values, dtype=float)
    elif "SCS" in ext_rows:
        ref_name = "SCS"
        ref_gamma = np.asarray(ext_rows["SCS"].values, dtype=float)
    elif ipm_row is not None:
        ref_name = "IPM"
        ref_gamma = np.asarray(ipm_row.values, dtype=float)

    method_rows: Dict[str, MethodData] = {}
    method_rows.update(ext_rows)
    if ipm_row is not None and "IPM" in algorithms:
        method_rows["IPM"] = ipm_row

    kept_idx = [i for i, res in enumerate(train_ipm_results) if getattr(res, "converged", False)]
    kept_train = len(kept_idx)
    skipped_train = len(train_instances) - kept_train

    l2a_values: List[float] = [float("nan")] * len(test_instances)
    l2ca_gamma_inits: List[float] = [float("nan")] * len(test_instances)
    train_gamma_mean = float("nan")
    if kept_train > 0:
        y_train_kept = np.stack([np.asarray(train_ipm_results[i].y, dtype=float) for i in kept_idx], axis=0)
        train_gamma_mean = float(
            np.mean(
                [
                    extract_gamma_from_dual(y_train_kept[j], train_instances[kept_idx[j]].sdp.b)
                    for j in range(y_train_kept.shape[0])
                ]
            )
        )

    # L2WS
    if "L2WS" in algorithms:
        vals: List[float] = []
        feas: List[bool] = []
        tms: List[float] = []
        if kept_train < 5:
            print(f"[warning] too few converged training solves for L2WS ({kept_train}).")
            vals = [float("nan")] * len(test_instances)
            feas = [False] * len(test_instances)
            tms = [float("nan")] * len(test_instances)
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
                    epochs=int(args.epochs),
                    batch_size=int(args.batch_size),
                    lr=float(args.lr),
                    hidden_dims=tuple(hidden_sizes),
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
                if res.converged and res.y is not None:
                    vals.append(extract_gamma_from_dual(np.asarray(res.y, dtype=float), inst.sdp.b))
                else:
                    vals.append(float("nan"))
        method_rows["L2WS"] = MethodData(name="L2WS", values=vals, feas=feas, times_ms=tms, value_kind="gamma")

    # L2A
    if "L2A" in algorithms:
        vals: List[float] = []
        tms: List[float] = []
        if kept_train < 5:
            print(f"[warning] too few converged training solves for L2A ({kept_train}).")
            vals = [float("nan")] * len(test_instances)
            tms = [float("nan")] * len(test_instances)
        else:
            x_train = np.stack([train_instances[i].feature for i in kept_idx], axis=0)
            y_train = np.asarray(
                [extract_gamma_from_dual(np.asarray(train_ipm_results[i].y, dtype=float), train_instances[i].sdp.b) for i in kept_idx],
                dtype=float,
            )
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
                epochs=int(args.epochs),
                lr=float(args.lr),
                batch_size=int(args.batch_size),
                device="cpu",
            )
            dev = torch.device("cpu")
            for inst in test_instances:
                x = (np.asarray(inst.feature, dtype=float) - x_mean) / x_std
                xt = torch.as_tensor(x, dtype=torch.float32, device=dev).unsqueeze(0)
                t0 = time.perf_counter()
                with torch.no_grad():
                    y_hat = float(model(xt).cpu().numpy().reshape(-1)[0])
                tms.append((time.perf_counter() - t0) * 1e3)
                vals.append(y_hat * y_std + y_mean)
        l2a_values = vals
        method_rows["L2A"] = MethodData(name="L2A", values=vals, feas=None, times_ms=tms, value_kind="gamma")

    # Shared DualNet predictor for L2CA and L2CA-FI.
    dual_model = None
    dual_x_mean = None
    dual_x_std = None
    dual_y_mean = None
    dual_y_std = None
    need_dual_predictor = any(a in algorithms for a in {"L2CA", "L2CA-FI"})
    if need_dual_predictor and kept_train >= 5:
        x_train = np.stack([train_instances[i].feature for i in kept_idx], axis=0)
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
            epochs=int(args.epochs),
            lr=float(args.lr),
            batch_size=int(args.batch_size),
            device="cpu",
            b_targets=np.stack([np.asarray(train_instances[i].sdp.b, dtype=float) for i in kept_idx], axis=0),
            lambda_obj=float(args.l2ca_lambda_obj),
            y_targets_unscaled=y_train,
            y_mean=y_mean,
            y_std=y_std,
            sdp_instances=[train_instances[i].sdp for i in kept_idx],
            lambda_feas=float(args.l2ca_lambda_feas),
            feas_margin=float(args.l2ca_feas_margin),
            feas_loss_mode=str(args.l2ca_feas_loss),
            feas_margin_train=float(args.l2ca_feas_margin_train),
            label_margin_debug=bool(args.l2ca_label_margin_debug),
            interiorize_labels=bool(int(args.l2ca_interiorize_labels)),
            interior_delta=float(args.l2ca_interior_delta),
            lambda_y=float(args.l2ca_lambda_y),
            obj_debug=False,
        )
        dual_model = model
        dual_x_mean = x_mean
        dual_x_std = x_std
        dual_y_mean = y_mean
        dual_y_std = y_std

    # L2CA
    if "L2CA" in algorithms:
        vals: List[float] = []
        feas: List[bool] = []
        tms: List[float] = []
        if kept_train < 5 or dual_model is None:
            print(f"[warning] too few converged training solves for L2CA ({kept_train}).")
            vals = [float("nan")] * len(test_instances)
            feas = [False] * len(test_instances)
            tms = [float("nan")] * len(test_instances)
        else:
            assert dual_x_mean is not None and dual_x_std is not None
            assert dual_y_mean is not None and dual_y_std is not None
            x_train = np.stack([train_instances[i].feature for i in kept_idx], axis=0)
            y_train = np.stack([np.asarray(train_ipm_results[i].y, dtype=float) for i in kept_idx], axis=0)
            x_train_norm = (x_train - dual_x_mean) / dual_x_std
            train_sdps = [train_instances[i].sdp for i in kept_idx]
            robust_anchor = select_robust_anchor(train_sdps, y_train)
            anchor_mode = str(args.l2ca_anchor).strip().lower()

            dev = torch.device("cpu")
            for t_idx, inst in enumerate(test_instances):
                x = (np.asarray(inst.feature, dtype=float) - dual_x_mean) / dual_x_std
                xt = torch.as_tensor(x, dtype=torch.float32, device=dev).unsqueeze(0)
                t0 = time.perf_counter()
                with torch.no_grad():
                    y_scaled = dual_model(xt).cpu().numpy().reshape(-1)
                y_pred = y_scaled * dual_y_std + dual_y_mean

                S_pred = slack_from_y(inst.sdp, y_pred)
                n_s = S_pred.shape[0]
                if is_psd_with_margin(S_pred, margin=float(args.l2ca_feas_margin), jitter=1e-9):
                    y_out = y_pred
                else:
                    if anchor_mode == "global_mean":
                        y_anchor = np.mean(y_train, axis=0)
                    elif anchor_mode == "knn1":
                        y_anchor = get_anchor_y(x, x_train_norm, y_train, mode="knn1")
                    elif anchor_mode == "knn5_best":
                        y_anchor, _, _ = choose_anchor_best_knn(
                            x,
                            x_train_norm,
                            y_train,
                            inst.sdp,
                            k=max(1, int(args.k)),
                            global_mean=np.mean(y_train, axis=0),
                            cached_feasible_anchor=robust_anchor,
                            feas_margin=float(args.l2ca_feas_margin),
                            jitter=1e-9,
                            robust=False,
                            allow_fullscan=True,
                        )
                    else:
                        raise ValueError(f"Unsupported L2CA anchor mode '{anchor_mode}'.")

                    S_anchor = slack_from_y(inst.sdp, y_anchor)
                    anchor_ok = is_psd_cholesky(
                        S_anchor + float(args.l2ca_feas_margin) * np.eye(n_s, dtype=float),
                        jitter=1e-9,
                    )
                    if not anchor_ok:
                        y_anchor = robust_anchor
                        S_anchor = slack_from_y(inst.sdp, y_anchor)
                        anchor_ok = is_psd_cholesky(
                            S_anchor + float(args.l2ca_feas_margin) * np.eye(n_s, dtype=float),
                            jitter=1e-9,
                        )
                    if not anchor_ok and check_c_psd(inst.sdp):
                        y_anchor = np.zeros_like(y_pred)
                        anchor_ok = True
                    if anchor_ok:
                        y_corr, _, repair_ok, _ = bisection_to_feasible(
                            inst.sdp,
                            y_pred,
                            y_anchor,
                            max_iters=int(args.l2ca_bisect_iters),
                            tol=float(args.l2ca_feas_margin),
                            jitter=1e-9,
                            debug=False,
                        )
                        y_out = y_corr if repair_ok else y_pred
                    else:
                        y_out = y_pred

                S_out = slack_from_y(inst.sdp, y_out)
                ok = is_psd_cholesky(
                    S_out + float(args.l2ca_feas_margin) * np.eye(n_s, dtype=float),
                    jitter=1e-9,
                )
                tms.append((time.perf_counter() - t0) * 1e3)
                feas.append(bool(ok))
                vals.append(float(np.dot(inst.sdp.b, y_out)))
                l2ca_gamma_inits[t_idx] = extract_gamma_from_dual(y_out, inst.sdp.b)

        method_rows["L2CA"] = MethodData(name="L2CA", values=vals, feas=feas, times_ms=tms, value_kind="dual")

    # L2CA-H
    if "L2CA-H" in algorithms:
        vals: List[float] = []
        feas: List[bool] = []
        tms: List[float] = []
        base_init = float(train_gamma_mean) if np.isfinite(train_gamma_mean) else 1.0
        base_init = max(base_init, 1e-6)
        for i, inst in enumerate(test_instances):
            init_candidates = [base_init]
            if i < len(l2a_values) and np.isfinite(l2a_values[i]):
                init_candidates.append(float(l2a_values[i]))
            if i < len(l2ca_gamma_inits) and np.isfinite(l2ca_gamma_inits[i]):
                init_candidates.append(float(l2ca_gamma_inits[i]))
            gamma_init = max(init_candidates)
            t0 = time.perf_counter()
            gamma_h, _ = hamiltonian_bisection(
                inst.system.A,
                inst.system.Bw,
                inst.system.Cz,
                gamma_init=gamma_init,
            )
            tms.append((time.perf_counter() - t0) * 1e3)
            vals.append(float(gamma_h))
            feas.append(
                _hamiltonian_feasible(
                    inst.system.A,
                    inst.system.Bw,
                    inst.system.Cz,
                    max(float(gamma_h), 1e-8),
                    1e-7,
                )
            )
        method_rows["L2CA-H"] = MethodData(name="L2CA-H", values=vals, feas=feas, times_ms=tms, value_kind="gamma")

    # L2CA-QP
    if "L2CA-QP" in algorithms:
        vals: List[float] = []
        feas: List[bool] = []
        tms: List[float] = []
        if kept_train < max(5, int(args.k)):
            print(
                f"[warning] too few converged training solves for L2CA-QP "
                f"({kept_train}; need at least {max(5, int(args.k))})."
            )
            vals = [float("nan")] * len(test_instances)
            feas = [False] * len(test_instances)
            tms = [float("nan")] * len(test_instances)
        else:
            train_feats = np.stack([np.asarray(train_instances[i].feature, dtype=float) for i in kept_idx], axis=0)
            train_y = np.stack([np.asarray(train_ipm_results[i].y, dtype=float) for i in kept_idx], axis=0)
            for inst in test_instances:
                t0 = time.perf_counter()
                y_qp = l2ca_qp_predict(
                    test_feat=np.asarray(inst.feature, dtype=float),
                    train_feats=train_feats,
                    train_y=train_y,
                    k=int(args.k),
                    qp_iters=int(args.l2ca_qp_iters),
                )
                tms.append((time.perf_counter() - t0) * 1e3)
                vals.append(float(np.dot(inst.sdp.b, np.asarray(y_qp, dtype=float))))
                feas.append(_dual_feasible_chol(inst.sdp, np.asarray(y_qp, dtype=float), tol=1e-7))
        method_rows["L2CA-QP"] = MethodData(name="L2CA-QP", values=vals, feas=feas, times_ms=tms, value_kind="dual")

    # L2CA-FI
    fi_diag: Dict[str, Any] | None = None
    if "L2CA-FI" in algorithms:
        vals: List[float] = []
        feas: List[bool] = []
        tms: List[float] = []
        fi_diag = {
            "n_test": len(test_instances),
            "pred_count": 0,
            "pred_feas_count": 0,
            "lam_min_values": [],
            "feas_penalties": [],
            "feas_loss_mode": str(args.l2ca_feas_loss),
            "feas_margin_train": float(args.l2ca_feas_margin_train),
            "stage_counts": {
                "accept_pred": 0,
                "spectral_cut": 0,
                "subgradient": 0,
                "short_ipm": 0,
                "full_ipm": 0,
            },
        }

        dev = torch.device("cpu")
        for inst in test_instances:
            y_pred = None
            if dual_model is not None and dual_x_mean is not None and dual_x_std is not None and dual_y_mean is not None and dual_y_std is not None:
                x = (np.asarray(inst.feature, dtype=float) - dual_x_mean) / dual_x_std
                xt = torch.as_tensor(x, dtype=torch.float32, device=dev).unsqueeze(0)
                with torch.no_grad():
                    y_scaled = dual_model(xt).cpu().numpy().reshape(-1)
                y_pred = y_scaled * dual_y_std + dual_y_mean
                if bool(args.fi_debug):
                    fi_diag["pred_count"] += 1
                    S_pred = slack_from_y(inst.sdp, y_pred)
                    S_pred = 0.5 * (S_pred + S_pred.T)
                    lam_min = float(np.linalg.eigvalsh(S_pred)[0])
                    fi_diag["lam_min_values"].append(lam_min)
                    fi_diag["feas_penalties"].append(
                        _fi_feas_penalty_from_slack(
                            S_pred,
                            mode=str(args.l2ca_feas_loss),
                            margin_train=float(args.l2ca_feas_margin_train),
                        )
                    )
                    if _dual_feasible_chol(
                        inst.sdp,
                        np.asarray(y_pred, dtype=float),
                        tol=1e-7,
                        jitter=float(args.l2ca_fi_jitter),
                    ):
                        fi_diag["pred_feas_count"] += 1

            if bool(args.fi_debug):
                stage_flags = {
                    "spectral_called": False,
                    "subgradient_called": False,
                    "short_called": False,
                    "short_success": False,
                    "full_called": False,
                    "accept_pred": bool(
                        y_pred is not None
                        and _dual_feasible_chol(
                            inst.sdp,
                            np.asarray(y_pred, dtype=float),
                            tol=1e-7,
                            jitter=float(args.l2ca_fi_jitter),
                        )
                    ),
                }

                orig_spec = fi._spectral_cut_correction
                orig_subg = fi._subgradient_repair
                orig_short = fi._short_ipm_feasible_dual
                orig_ipm_solve = InfeasibleIPMSolver.solve

                def _spec_wrap(*w_args, **w_kwargs):
                    stage_flags["spectral_called"] = True
                    return orig_spec(*w_args, **w_kwargs)

                def _subg_wrap(*w_args, **w_kwargs):
                    stage_flags["subgradient_called"] = True
                    return orig_subg(*w_args, **w_kwargs)

                def _short_wrap(*w_args, **w_kwargs):
                    stage_flags["short_called"] = True
                    out = orig_short(*w_args, **w_kwargs)
                    if out is not None:
                        stage_flags["short_success"] = True
                    return out

                def _ipm_solve_wrap(self, *w_args, **w_kwargs):
                    if int(getattr(self.settings, "max_iters", -1)) == 180:
                        stage_flags["full_called"] = True
                    return orig_ipm_solve(self, *w_args, **w_kwargs)

                fi._spectral_cut_correction = _spec_wrap
                fi._subgradient_repair = _subg_wrap
                fi._short_ipm_feasible_dual = _short_wrap
                InfeasibleIPMSolver.solve = _ipm_solve_wrap

                try:
                    t0 = time.perf_counter()
                    y_out = l2ca_fi_predict(
                        sdp=inst.sdp,
                        y_pred=y_pred,
                        max_phase1_iters=int(args.l2ca_fi_phase1_iters),
                        jitter=float(args.l2ca_fi_jitter),
                        objective_lift=bool(int(args.l2ca_fi_objective_lift)),
                    )
                    tms.append((time.perf_counter() - t0) * 1e3)
                finally:
                    fi._spectral_cut_correction = orig_spec
                    fi._subgradient_repair = orig_subg
                    fi._short_ipm_feasible_dual = orig_short
                    InfeasibleIPMSolver.solve = orig_ipm_solve

                if stage_flags["accept_pred"]:
                    fi_diag["stage_counts"]["accept_pred"] += 1
                elif stage_flags["full_called"]:
                    fi_diag["stage_counts"]["full_ipm"] += 1
                elif stage_flags["short_success"]:
                    fi_diag["stage_counts"]["short_ipm"] += 1
                elif stage_flags["subgradient_called"]:
                    fi_diag["stage_counts"]["subgradient"] += 1
                elif stage_flags["spectral_called"]:
                    fi_diag["stage_counts"]["spectral_cut"] += 1
                else:
                    fi_diag["stage_counts"]["spectral_cut"] += 1
            else:
                t0 = time.perf_counter()
                y_out = l2ca_fi_predict(
                    sdp=inst.sdp,
                    y_pred=y_pred,
                    max_phase1_iters=int(args.l2ca_fi_phase1_iters),
                    jitter=float(args.l2ca_fi_jitter),
                    objective_lift=bool(int(args.l2ca_fi_objective_lift)),
                )
                tms.append((time.perf_counter() - t0) * 1e3)
            vals.append(float(np.dot(inst.sdp.b, np.asarray(y_out, dtype=float))))
            feas.append(_dual_feasible_chol(inst.sdp, np.asarray(y_out, dtype=float), tol=1e-7))

        method_rows["L2CA-FI"] = MethodData(
            name="L2CA-FI",
            values=vals,
            feas=feas,
            times_ms=tms,
            value_kind="dual",
        )

    available_names = [a for a in algorithms if a in method_rows]
    if not available_names:
        raise RuntimeError("No algorithm rows were produced. Check solver availability and options.")

    baseline_row = ref_name if ref_name is not None else None
    baseline_time = None
    if baseline_row == "MOSEK" and "MOSEK" in ext_rows:
        bt = np.nanmean(np.asarray(ext_rows["MOSEK"].times_ms, dtype=float))
        baseline_time = float(bt) if np.isfinite(bt) else None
    elif baseline_row == "SCS" and "SCS" in ext_rows:
        bt = np.nanmean(np.asarray(ext_rows["SCS"].times_ms, dtype=float))
        baseline_time = float(bt) if np.isfinite(bt) else None
    elif baseline_row == "IPM" and ipm_row is not None:
        bt = np.nanmean(np.asarray(ipm_row.times_ms, dtype=float))
        baseline_time = float(bt) if np.isfinite(bt) else None
    if baseline_time is None:
        for name in available_names:
            bt = np.nanmean(np.asarray(method_rows[name].times_ms, dtype=float))
            if np.isfinite(bt):
                baseline_row = name
                baseline_time = float(bt)
                break

    summaries: List[MethodSummary] = []
    for name in available_names:
        summaries.append(
            _summarize(
                method_rows[name],
                ref_gamma=ref_gamma,
                hide_error=(name == ref_name),
                baseline_time_ms=baseline_time,
            )
        )

    _print_table(
        summaries=summaries,
        n_states=int(base_system.A.shape[0]),
        sdp_dim=int(test_instances[0].sdp.dim),
        args=args,
        ref_name=ref_name,
        kept_train=kept_train,
        skipped_train=skipped_train,
    )
    if bool(args.fi_debug) and fi_diag is not None:
        _print_fi_diagnostics(fi_diag)


if __name__ == "__main__":
    main()
