"""Standalone L2CA-FI prototype.

L2CA-FI (Feasibility-Initialized L2CA) computes an instance-specific
dual-feasible point for a single SDP instance:

    S(y) = C - A^T(y) ⪰ 0

without relying on anchors from other instances.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .ipm import IPMSettings, IPMState, InfeasibleIPMSolver
from .problem import SDPInstance


def slack_from_y(sdp: SDPInstance, y: np.ndarray) -> np.ndarray:
    """Return symmetric dual slack S(y) = C - A^T(y)."""
    y = np.asarray(y, dtype=float).reshape(-1)
    S = sdp.C - sdp.apply_AT(y)
    return 0.5 * (S + S.T)


def is_psd_cholesky(M: np.ndarray, jitter: float = 1e-10) -> bool:
    """PSD check via Cholesky on M + jitter*I."""
    M = 0.5 * (np.asarray(M, dtype=float) + np.asarray(M, dtype=float).T)
    n = M.shape[0]
    try:
        np.linalg.cholesky(M + float(jitter) * np.eye(n, dtype=float))
        return True
    except np.linalg.LinAlgError:
        return False


def check_c_psd(sdp: SDPInstance, tol: float = 1e-10) -> bool:
    """Return True when objective matrix C is PSD (up to tolerance)."""
    eigs = np.linalg.eigvalsh(0.5 * (sdp.C + sdp.C.T))
    return bool(float(eigs[0]) >= -float(tol))


def _lam_min_and_vec(M: np.ndarray) -> tuple[float, np.ndarray]:
    vals, vecs = np.linalg.eigh(0.5 * (M + M.T))
    idx = int(np.argmin(vals))
    return float(vals[idx]), np.asarray(vecs[:, idx], dtype=float).reshape(-1)


def _is_dual_feasible(
    S: np.ndarray,
    tol: float = 1e-7,
    jitter: float = 1e-10,
) -> bool:
    """Numerically relaxed dual feasibility check: min eig(S) >= -tol."""
    n = S.shape[0]
    return is_psd_cholesky(S + float(tol) * np.eye(n, dtype=float), jitter=jitter)


def _spectral_cut_correction(
    sdp: SDPInstance,
    y_start: np.ndarray,
    max_iters: int,
    jitter: float,
) -> np.ndarray:
    """Simple cut-based correction using violated minimum-eigenvector cuts."""
    y = np.asarray(y_start, dtype=float).reshape(-1).copy()
    m = sdp.num_constraints
    if y.shape[0] != m:
        y = np.zeros(m, dtype=float)

    A_t = np.asarray(sdp.A, dtype=float)
    S = slack_from_y(sdp, y)
    for _ in range(int(max_iters)):
        if _is_dual_feasible(S, tol=1e-7, jitter=jitter):
            return y

        lam_min, v = _lam_min_and_vec(S)
        if lam_min >= 0.0:
            return y

        # Constraint cut: v^T(C - A^T y)v >= 0  ->  c - a^T y >= 0
        Av = np.einsum("mij,j->mi", A_t, v)
        a_vec = np.einsum("i,mi->m", v, Av)
        c_val = float(v.T @ sdp.C @ v)
        g_val = c_val - float(np.dot(a_vec, y))
        viol = max(0.0, -g_val)
        denom = float(np.dot(a_vec, a_vec)) + 1e-12

        # Move y to satisfy this linearized cut.
        step = viol / denom
        if not np.isfinite(step) or step <= 0.0:
            step = 1e-3 / np.sqrt(denom)
        y = y - step * a_vec
        S = S + float(step) * sdp.apply_AT(a_vec)
        S = 0.5 * (S + S.T)

    return y


def _subgradient_repair(
    sdp: SDPInstance,
    y_start: np.ndarray,
    max_iters: int = 200,
    tol: float = 1e-7,
    jitter: float = 1e-10,
) -> np.ndarray:
    """Maximize lambda_min(S(y)) via conservative subgradient ascent."""
    y = np.asarray(y_start, dtype=float).reshape(-1).copy()
    m = sdp.num_constraints
    if y.shape[0] != m:
        y = np.zeros(m, dtype=float)

    # Build a static Gram preconditioner from <Ai, Aj>.
    A_t = np.asarray(sdp.A, dtype=float)
    G = np.empty((m, m), dtype=float)
    for i in range(m):
        Ai = A_t[i]
        for j in range(i, m):
            val = float(np.sum(Ai * A_t[j]))
            G[i, j] = val
            G[j, i] = val
    G = 0.5 * (G + G.T) + 1e-10 * np.eye(m, dtype=float)

    step0 = 1.0
    for it in range(int(max_iters)):
        S = slack_from_y(sdp, y)
        if _is_dual_feasible(S, tol=tol, jitter=jitter):
            return y

        lam, v = _lam_min_and_vec(S)
        if lam >= -tol:
            return y

        Av = np.einsum("mij,j->mi", A_t, v)
        a_vec = np.einsum("i,mi->m", v, Av)

        g = -a_vec  # subgradient of lambda_min(S(y))
        try:
            d = np.linalg.solve(G, g)
        except np.linalg.LinAlgError:
            d = g

        A_d = sdp.apply_AT(d)
        A_d = 0.5 * (A_d + A_d.T)
        step = step0 / np.sqrt(float(it) + 1.0)

        accepted = False
        for _ in range(12):
            S_try = S - float(step) * A_d
            S_try = 0.5 * (S_try + S_try.T)
            lam_try = float(np.linalg.eigvalsh(S_try)[0])
            if lam_try > lam + 1e-10:
                y = y + step * d
                accepted = True
                break
            step *= 0.5

        if not accepted:
            # One-shot cut correction fallback.
            viol = -float(v.T @ S @ v)
            denom = float(np.dot(a_vec, a_vec)) + 1e-12
            y = y - (viol / denom) * a_vec

    return y


def _objective_lift_along_b(
    sdp: SDPInstance,
    y_feas: np.ndarray,
    bisect_iters: int = 8,
    max_expand: int = 6,
    jitter: float = 1e-10,
) -> np.ndarray:
    """Try to increase b^T y along direction b while maintaining feasibility."""
    y0 = np.asarray(y_feas, dtype=float).reshape(-1)
    b = np.asarray(sdp.b, dtype=float).reshape(-1)
    if y0.shape != b.shape:
        return y0

    S0 = slack_from_y(sdp, y0)
    if not _is_dual_feasible(S0, tol=1e-7, jitter=jitter):
        return y0

    norm_b = float(np.linalg.norm(b))
    if norm_b <= 1e-14:
        return y0
    d = b / norm_b
    D = 0.5 * (sdp.apply_AT(d) + sdp.apply_AT(d).T)

    def feasible_t(t: float) -> bool:
        S = S0 - float(t) * D
        S = 0.5 * (S + S.T)
        return _is_dual_feasible(S, tol=1e-7, jitter=jitter)

    t_lo = 0.0
    t_hi = 1.0
    if feasible_t(t_hi):
        for _ in range(int(max_expand)):
            t_next = 2.0 * t_hi
            if feasible_t(t_next):
                t_lo = t_hi
                t_hi = t_next
            else:
                break
        else:
            return y0 + t_hi * d
    else:
        # If no progress at t=1, keep [0,1] as bracket.
        pass

    # Ensure upper end is infeasible for bisection.
    if feasible_t(t_hi):
        return y0 + t_hi * d

    for _ in range(int(bisect_iters)):
        t_mid = 0.5 * (t_lo + t_hi)
        if feasible_t(t_mid):
            t_lo = t_mid
        else:
            t_hi = t_mid

    return y0 + t_lo * d


def _short_ipm_feasible_dual(
    sdp: SDPInstance,
    y_start: np.ndarray,
    jitter: float = 1e-10,
    max_iters_schedule: Sequence[int] = (4, 8, 16, 32, 64, 96),
) -> Optional[np.ndarray]:
    """Fallback: run progressively longer IPM solves and return first feasible dual."""
    m = sdp.num_constraints
    n = sdp.dim

    y0 = np.asarray(y_start, dtype=float).reshape(-1)
    if y0.shape[0] != m:
        y0 = np.zeros(m, dtype=float)

    S0 = slack_from_y(sdp, y0)
    lam_min = float(np.min(np.linalg.eigvalsh(S0)))
    shift = max(1e-6, -lam_min + 1e-4)
    S0_pd = S0 + shift * np.eye(n, dtype=float)
    X0 = max(1.0, shift) * np.eye(n, dtype=float)
    init_state = IPMState(X=X0, y=y0, S=S0_pd)

    for k in max_iters_schedule:
        settings = IPMSettings(
            max_iters=int(k),
            tol_abs=1e-6,
            tol_rel=1e-5,
            linear_solve="sylvester",
        )
        solver = InfeasibleIPMSolver(settings)
        try:
            res = solver.solve(sdp, initial_state=init_state)
        except Exception:
            continue
        y_try = np.asarray(res.y, dtype=float).reshape(-1)
        S_try = slack_from_y(sdp, y_try)
        if _is_dual_feasible(S_try, tol=1e-7, jitter=jitter):
            return y_try
        # Warm-start next attempt from latest iterate even if not yet feasible.
        S_try_pd = 0.5 * (S_try + S_try.T)
        vals = np.linalg.eigvalsh(S_try_pd)
        min_eig = float(np.min(vals))
        if min_eig <= 0.0:
            S_try_pd = S_try_pd + (abs(min_eig) + 1e-6) * np.eye(n, dtype=float)
        init_state = IPMState(X=np.eye(n, dtype=float), y=y_try, S=S_try_pd)

    return None


def _bisection_to_anchor_with_cached_slacks(
    y_start: np.ndarray,
    y_anchor: np.ndarray,
    S_start: np.ndarray,
    S_anchor: np.ndarray,
    max_iters: int = 20,
    tol: float = 1e-7,
    jitter: float = 1e-10,
) -> tuple[np.ndarray, bool]:
    """Find feasible point on [y_start, y_anchor] using slack interpolation only."""
    y0 = np.asarray(y_start, dtype=float).reshape(-1)
    ya = np.asarray(y_anchor, dtype=float).reshape(-1)
    S0 = 0.5 * (np.asarray(S_start, dtype=float) + np.asarray(S_start, dtype=float).T)
    Sa = 0.5 * (np.asarray(S_anchor, dtype=float) + np.asarray(S_anchor, dtype=float).T)

    if _is_dual_feasible(S0, tol=tol, jitter=jitter):
        return y0, True
    if not _is_dual_feasible(Sa, tol=tol, jitter=jitter):
        return y0, False

    lo = 0.0
    hi = 1.0
    for _ in range(int(max_iters)):
        mid = 0.5 * (lo + hi)
        S_mid = (1.0 - mid) * S0 + mid * Sa
        if _is_dual_feasible(S_mid, tol=tol, jitter=jitter):
            hi = mid
        else:
            lo = mid
        if (hi - lo) < 1e-10:
            break

    y_out = (1.0 - hi) * y0 + hi * ya
    S_out = (1.0 - hi) * S0 + hi * Sa
    return y_out, _is_dual_feasible(S_out, tol=tol, jitter=jitter)


def phase1_find_feasible_anchor(
    sdp: SDPInstance,
    y0: np.ndarray | None = None,
    max_iters: int = 30,
    jitter: float = 1e-10,
) -> np.ndarray:
    """Find an instance-specific dual-feasible anchor for the test SDP."""
    m = sdp.num_constraints
    if y0 is None:
        y = np.zeros(m, dtype=float)
    else:
        y = np.asarray(y0, dtype=float).reshape(-1)
        if y.shape[0] != m:
            y = np.zeros(m, dtype=float)

    S = slack_from_y(sdp, y)
    if _is_dual_feasible(S, tol=1e-7, jitter=jitter):
        return y

    # Phase-I style local correction.
    y_corr = _spectral_cut_correction(
        sdp=sdp,
        y_start=y,
        max_iters=int(max_iters),
        jitter=float(jitter),
    )
    y_corr = _subgradient_repair(
        sdp=sdp,
        y_start=y_corr,
        max_iters=max(16, int(max_iters)),
        tol=1e-7,
        jitter=float(jitter),
    )
    if _is_dual_feasible(slack_from_y(sdp, y_corr), tol=1e-7, jitter=jitter):
        return y_corr

    # C-PSD fallback: bisect toward y=0 only after local repair fails.
    if check_c_psd(sdp):
        y_zero = np.zeros_like(y_corr)
        S_corr = slack_from_y(sdp, y_corr)
        y_seg, ok_seg = _bisection_to_anchor_with_cached_slacks(
            y_start=y_corr,
            y_anchor=y_zero,
            S_start=S_corr,
            S_anchor=0.5 * (sdp.C + sdp.C.T),
            max_iters=min(20, max(8, int(max_iters))),
            tol=1e-7,
            jitter=float(jitter),
        )
        if ok_seg:
            return y_seg

    # Robust fallback: short IPM schedule, still instance-specific and non-oracle.
    y_ipm = _short_ipm_feasible_dual(sdp=sdp, y_start=y_corr, jitter=jitter)
    if y_ipm is not None:
        y_ipm = _subgradient_repair(
            sdp=sdp,
            y_start=y_ipm,
            max_iters=max(80, int(max_iters) * 2),
            tol=1e-7,
            jitter=float(jitter),
        )
        return y_ipm

    # Last resort: return best attempt (caller will re-check feasibility).
    return y_corr


def l2ca_fi_predict(
    sdp: SDPInstance,
    y_pred: np.ndarray | None = None,
    max_phase1_iters: int = 30,
    jitter: float = 1e-10,
    objective_lift: bool = False,
) -> np.ndarray:
    """Return a dual-feasible y for this specific SDP instance."""
    m = sdp.num_constraints
    if y_pred is not None:
        y0 = np.asarray(y_pred, dtype=float).reshape(-1)
        if y0.shape[0] != m:
            y0 = np.zeros(m, dtype=float)
        S0 = slack_from_y(sdp, y0)
        if _is_dual_feasible(S0, tol=1e-7, jitter=jitter):
            y_feas = y0
        else:
            y_feas = phase1_find_feasible_anchor(
                sdp=sdp,
                y0=y0,
                max_iters=int(max_phase1_iters),
                jitter=float(jitter),
            )
    else:
        y_feas = phase1_find_feasible_anchor(
            sdp=sdp,
            y0=None,
            max_iters=int(max_phase1_iters),
            jitter=float(jitter),
        )

    if not _is_dual_feasible(slack_from_y(sdp, y_feas), tol=1e-7, jitter=jitter):
        # Tight fallback: one more full-schedule attempt from zero.
        y_try = _short_ipm_feasible_dual(
            sdp=sdp,
            y_start=np.zeros(m, dtype=float),
            jitter=jitter,
            max_iters_schedule=(20, 40, 80, 120),
        )
        if y_try is not None and _is_dual_feasible(slack_from_y(sdp, y_try), tol=1e-7, jitter=jitter):
            y_feas = y_try

    if not _is_dual_feasible(slack_from_y(sdp, y_feas), tol=1e-7, jitter=jitter):
        # Last-resort robust fallback: full IPM solve for this instance.
        n = sdp.dim
        S_init = slack_from_y(sdp, y_feas)
        vals = np.linalg.eigvalsh(S_init)
        min_eig = float(np.min(vals))
        if min_eig <= 0.0:
            S_init = S_init + (abs(min_eig) + 1e-6) * np.eye(n, dtype=float)
        state = IPMState(
            X=np.eye(n, dtype=float),
            y=np.asarray(y_feas, dtype=float).reshape(-1),
            S=0.5 * (S_init + S_init.T),
        )
        solver = InfeasibleIPMSolver(
            IPMSettings(
                max_iters=180,
                tol_abs=1e-6,
                tol_rel=1e-5,
                linear_solve="sylvester",
            )
        )
        try:
            res = solver.solve(sdp, initial_state=state)
            y_full = np.asarray(res.y, dtype=float).reshape(-1)
            if _is_dual_feasible(slack_from_y(sdp, y_full), tol=1e-7, jitter=jitter):
                y_feas = y_full
        except Exception:
            pass

    if not _is_dual_feasible(slack_from_y(sdp, y_feas), tol=1e-7, jitter=jitter):
        y_fix = _subgradient_repair(
            sdp=sdp,
            y_start=y_feas,
            max_iters=160,
            tol=1e-7,
            jitter=float(jitter),
        )
        if _is_dual_feasible(slack_from_y(sdp, y_fix), tol=1e-7, jitter=jitter):
            y_feas = y_fix

    if objective_lift and _is_dual_feasible(slack_from_y(sdp, y_feas), tol=1e-7, jitter=jitter):
        y_lift = _objective_lift_along_b(sdp=sdp, y_feas=y_feas, jitter=jitter)
        if _is_dual_feasible(slack_from_y(sdp, y_lift), tol=1e-7, jitter=jitter):
            y_feas = y_lift

    return np.asarray(y_feas, dtype=float).reshape(-1)


__all__ = [
    "slack_from_y",
    "is_psd_cholesky",
    "phase1_find_feasible_anchor",
    "l2ca_fi_predict",
]
