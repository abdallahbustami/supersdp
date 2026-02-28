"""L2CA models and certification utilities."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .data import L2GainInstance
from .ipm import IPMSettings, IPMState, InfeasibleIPMSolver
from .learning import DualNet, ScalarL2ANet, train_dual_net, train_scalar_l2a
from .problem import SDPInstance

def extract_gamma_from_dual(y: np.ndarray, sdp_b: np.ndarray) -> float:
    """For the L2-gain encoding, gamma^2 is the last dual component."""
    _ = sdp_b
    gamma_sq = float(np.asarray(y, dtype=float).reshape(-1)[-1])
    return float(np.sqrt(max(gamma_sq, 0.0)))


def solve_instances_for_duals(
    instances: Sequence[L2GainInstance],
    solver: InfeasibleIPMSolver,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray]], List[int]]:
    """Solve instances with IPM and return converged (y*, X*, S*) plus kept indices."""
    outputs: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    kept_indices: List[int] = []
    skipped = 0
    for local_idx, inst in enumerate(instances):
        result = solver.solve(inst.sdp)
        if not result.converged:
            skipped += 1
            continue
        outputs.append((result.y.copy(), result.X.copy(), result.S.copy()))
        kept_indices.append(local_idx)
    if skipped > 0:
        print(f"[warning] Skipped {skipped} training instances with non-converged IPM solves.")
    return outputs, kept_indices


def certify_dual(
    sdp: SDPInstance,
    y_pred: np.ndarray,
    tol: float = 1e-6,
) -> Tuple[bool, float, float]:
    """Check dual feasibility and return (is_feasible, lambda_min, dual_obj)."""
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    S = sdp.C - sdp.apply_AT(y_pred)
    S = 0.5 * (S + S.T)
    eigs = np.linalg.eigvalsh(S)
    lam_min = float(eigs[0])
    dual_obj = float(np.dot(sdp.b, y_pred))
    return lam_min >= -float(tol), lam_min, dual_obj


def slack_from_y(
    sdp: SDPInstance,
    y: np.ndarray,
    stats: Dict[str, Any] | None = None,
) -> np.ndarray:
    """Return symmetric dual slack S(y) = C - A*(y)."""
    y = np.asarray(y, dtype=float).reshape(-1)
    S = sdp.C - sdp.apply_AT(y)
    if stats is not None:
        stats["apply_at_calls"] = int(stats.get("apply_at_calls", 0)) + 1
    return 0.5 * (S + S.T)


def _slack_from_y(sdp: SDPInstance, y: np.ndarray) -> np.ndarray:
    """Private alias retained for backward compatibility."""
    return slack_from_y(sdp, y, stats=None)


def _lam_min(S: np.ndarray) -> float:
    """Smallest eigenvalue of a symmetric matrix."""
    return float(np.linalg.eigvalsh(S)[0])


def is_psd_cholesky(M: np.ndarray, jitter: float = 1e-9) -> bool:
    """Fast PSD predicate using Cholesky on M + jitter*I."""
    M = 0.5 * (np.asarray(M, dtype=float) + np.asarray(M, dtype=float).T)
    n = M.shape[0]
    try:
        np.linalg.cholesky(M + float(jitter) * np.eye(n, dtype=float))
        return True
    except np.linalg.LinAlgError:
        return False


def is_psd_with_margin(M: np.ndarray, margin: float = 1e-8, jitter: float = 1e-9) -> bool:
    """Return True when M is PSD with a safety margin."""
    M = 0.5 * (np.asarray(M, dtype=float) + np.asarray(M, dtype=float).T)
    n = M.shape[0]
    shifted = M - float(margin) * np.eye(n, dtype=float)
    return is_psd_cholesky(shifted, jitter=jitter)


def knn_indices(test_feat: np.ndarray, train_feats: np.ndarray, k: int) -> np.ndarray:
    """Deterministic nearest-neighbor indices by Euclidean distance."""
    x = np.asarray(test_feat, dtype=float).reshape(1, -1)
    X = np.asarray(train_feats, dtype=float)
    if X.ndim != 2:
        raise ValueError("train_feats must be a 2D array.")
    if X.shape[1] != x.shape[1]:
        raise ValueError("test_feat dimension mismatch with train_feats.")
    if X.shape[0] == 0:
        return np.array([], dtype=int)
    k_eff = max(1, min(int(k), X.shape[0]))
    dists = np.linalg.norm(X - x, axis=1)
    return np.argsort(dists, kind="mergesort")[:k_eff]


def _shift_rank_for_feasibility(
    S: np.ndarray,
    jitter: float,
    shifts: tuple[float, ...] = (0.0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4),
) -> tuple[int, int]:
    """Cheap proxy for distance-to-feasibility via minimal diagonal shift."""
    checks = 0
    for rank, sh in enumerate(shifts):
        checks += 1
        n = S.shape[0]
        if is_psd_cholesky(S + float(sh) * np.eye(n, dtype=float), jitter=jitter):
            return rank, checks
    return len(shifts), checks


def get_anchor_y(
    test_feat: np.ndarray,
    train_feats: np.ndarray,
    train_y: np.ndarray,
    mode: str = "knn5_best",
) -> np.ndarray:
    """Return a deterministic training-based anchor dual vector."""
    x = np.asarray(test_feat, dtype=float).reshape(1, -1)
    X = np.asarray(train_feats, dtype=float)
    Y = np.asarray(train_y, dtype=float)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("train_feats and train_y must be 2D arrays.")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("train_feats and train_y must have matching row counts.")
    if X.shape[1] != x.shape[1]:
        raise ValueError("test_feat dimension mismatch with train_feats.")
    if X.shape[0] == 0:
        raise ValueError("train_feats must be non-empty.")

    mode_norm = str(mode).strip().lower()
    if mode_norm not in {"knn1", "knn5_best", "global_mean"}:
        raise ValueError("mode must be one of: knn1, knn5_best, global_mean.")

    if mode_norm == "global_mean":
        return np.mean(Y, axis=0).astype(float)

    dists = np.linalg.norm(X - x, axis=1)
    order = np.argsort(dists, kind="mergesort")
    if mode_norm == "knn1":
        return np.asarray(Y[int(order[0])], dtype=float).copy()

    k = min(5, X.shape[0])
    nn_idx = order[:k]
    return np.mean(Y[nn_idx], axis=0).astype(float)


def choose_anchor_best_knn(
    test_feat: np.ndarray,
    train_feats: np.ndarray,
    train_y: np.ndarray,
    sdp: SDPInstance,
    k: int = 5,
    global_mean: np.ndarray | None = None,
    cached_feasible_anchor: np.ndarray | None = None,
    feas_margin: float = 1e-8,
    jitter: float = 1e-9,
    robust: bool = True,
    allow_fullscan: bool = True,
) -> tuple[np.ndarray, bool, Dict[str, Any]]:
    """Pick best kNN anchor by objective with feasibility-aware fallback."""
    x = np.asarray(test_feat, dtype=float).reshape(1, -1)
    X = np.asarray(train_feats, dtype=float)
    Y = np.asarray(train_y, dtype=float)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("train_feats and train_y must be 2D arrays.")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("train_feats and train_y must have matching row counts.")
    if X.shape[1] != x.shape[1]:
        raise ValueError("test_feat dimension mismatch with train_feats.")
    if X.shape[0] == 0:
        raise ValueError("train_feats must be non-empty.")

    info: Dict[str, Any] = {
        "mode": "knn_best",
        "k": int(max(1, min(int(k), X.shape[0]))),
        "candidate_count": 0,
        "chol_checks": 0,
        "apply_at_calls": 0,
        "used_fallback": False,
        "allow_fullscan": bool(allow_fullscan),
    }

    idx = knn_indices(x, X, k=int(k))
    info["candidate_count"] = int(idx.shape[0])

    # Check kNN candidates in descending dual objective order; first feasible one
    # is already the best feasible anchor among the k candidates.
    cand_obj = np.asarray([float(np.dot(sdp.b, Y[int(j)])) for j in idx], dtype=float)
    order_local = np.argsort(-cand_obj, kind="mergesort")
    ordered_idx = idx[order_local]

    best_rank_y: np.ndarray | None = None
    best_rank = 10**9
    best_rank_obj = -np.inf

    for j in ordered_idx:
        yj = np.asarray(Y[int(j)], dtype=float).reshape(-1)
        Sj = slack_from_y(sdp, yj, stats=info)
        objj = float(np.dot(sdp.b, yj))

        info["chol_checks"] += 1
        if is_psd_with_margin(Sj, margin=float(feas_margin), jitter=jitter):
            info["mode"] = "knn_best"
            return yj.copy(), True, info
        info["chol_checks"] += 1
        if is_psd_cholesky(Sj, jitter=jitter):
            info["mode"] = "knn_best"
            return yj.copy(), True, info

        if robust:
            rank, checks = _shift_rank_for_feasibility(Sj, jitter=jitter)
            info["chol_checks"] += int(checks)
            if rank < best_rank or (rank == best_rank and objj > best_rank_obj):
                best_rank = rank
                best_rank_obj = objj
                best_rank_y = yj

    # Fallback 1: global mean.
    if global_mean is not None:
        info["used_fallback"] = True
        yg = np.asarray(global_mean, dtype=float).reshape(-1)
        Sg = slack_from_y(sdp, yg, stats=info)
        info["chol_checks"] += 1
        if is_psd_cholesky(Sg, jitter=jitter):
            info["mode"] = "global_mean"
            return yg.copy(), True, info

    # Fallback 2: cached feasible anchor from training solves.
    if cached_feasible_anchor is not None:
        info["used_fallback"] = True
        yc = np.asarray(cached_feasible_anchor, dtype=float).reshape(-1)
        Sc = slack_from_y(sdp, yc, stats=info)
        info["chol_checks"] += 1
        if is_psd_cholesky(Sc, jitter=jitter):
            info["mode"] = "cached_feasible"
            return yc.copy(), True, info

    dists = np.linalg.norm(X - x, axis=1)
    full_order = np.argsort(dists, kind="mergesort")
    if allow_fullscan:
        # Fallback 3: nearest feasible point over all train_y (deterministic scan).
        for j in full_order:
            yj = np.asarray(Y[int(j)], dtype=float).reshape(-1)
            Sj = slack_from_y(sdp, yj, stats=info)
            info["chol_checks"] += 1
            if is_psd_cholesky(Sj, jitter=jitter):
                info["mode"] = "fullscan_feasible"
                info["used_fallback"] = True
                return yj.copy(), True, info

    # No feasible anchor found; return closest-to-feasible candidate if available.
    if robust and best_rank_y is not None:
        info["mode"] = "closest_feas_rank"
        return best_rank_y.copy(), False, info

    # Last resort: nearest neighbor.
    info["mode"] = "nn_last_resort"
    return np.asarray(Y[int(full_order[0])], dtype=float).copy(), False, info


def check_c_psd(sdp: SDPInstance, tol: float = 1e-10) -> bool:
    """Check whether the objective matrix C is positive semidefinite."""
    eigs = np.linalg.eigvalsh(sdp.C)
    return bool(float(eigs[0]) >= -float(tol))


def detect_tier(
    sdp: SDPInstance,
    tol: float = 1e-9,
) -> tuple[int, int | None]:
    """Detect guaranteed-anchor tier for one SDP instance.

    Tier 1: C is PSD -> universal anchor y=0.
    Tier 2: exists j with Aj negative definite (conservative test).
    Tier 0: neither condition available.
    """
    tol = float(max(tol, 0.0))
    C = 0.5 * (np.asarray(sdp.C, dtype=float) + np.asarray(sdp.C, dtype=float).T)
    if float(np.linalg.eigvalsh(C)[0]) >= -tol:
        return 1, None

    best_j: int | None = None
    best_maxeig = np.inf
    strict_tol = max(1e-12, tol)
    for j, Aj in enumerate(np.asarray(sdp.A, dtype=float)):
        Ajs = 0.5 * (Aj + Aj.T)
        eigs = np.linalg.eigvalsh(Ajs)
        max_eig = float(eigs[-1])
        min_eig = float(eigs[0])
        # Conservative: require strictly negative-definite behavior.
        if max_eig < -strict_tol and min_eig < -strict_tol:
            if max_eig < best_maxeig:
                best_maxeig = max_eig
                best_j = int(j)
    if best_j is not None:
        return 2, best_j
    return 0, None


def build_anchor(
    sdp: SDPInstance,
    tier: int,
    j_idx: int | None = None,
    tol: float = 1e-9,
    jitter: float = 1e-9,
    max_doublings: int = 60,
    bisect_iters: int = 30,
) -> tuple[np.ndarray, bool, Dict[str, Any]]:
    """Construct a guaranteed anchor for Tier-1/2 when possible."""
    m = sdp.num_constraints
    n = sdp.dim
    y_anchor = np.zeros(m, dtype=float)
    info: Dict[str, Any] = {
        "tier": int(tier),
        "j_idx": None if j_idx is None else int(j_idx),
        "lambda": 0.0,
        "doublings": 0,
        "mode": "none",
    }
    feas_tol = float(max(tol, 0.0))

    if int(tier) == 1:
        S0 = 0.5 * (np.asarray(sdp.C, dtype=float) + np.asarray(sdp.C, dtype=float).T)
        ok0 = is_psd_cholesky(S0 + feas_tol * np.eye(n, dtype=float), jitter=jitter)
        info["mode"] = "tier1_zero"
        return y_anchor, bool(ok0), info

    if int(tier) != 2 or j_idx is None or int(j_idx) < 0 or int(j_idx) >= m:
        info["mode"] = "tier2_invalid"
        return y_anchor, False, info

    Aj = 0.5 * (np.asarray(sdp.A[int(j_idx)], dtype=float) + np.asarray(sdp.A[int(j_idx)], dtype=float).T)
    C = 0.5 * (np.asarray(sdp.C, dtype=float) + np.asarray(sdp.C, dtype=float).T)
    I = np.eye(n, dtype=float)

    def _feasible_lambda(lam: float) -> bool:
        S_lam = C - float(lam) * Aj
        S_lam = 0.5 * (S_lam + S_lam.T)
        return is_psd_cholesky(S_lam + feas_tol * I, jitter=jitter)

    if _feasible_lambda(0.0):
        info["mode"] = "tier2_zero"
        return y_anchor, True, info

    lam_lo = 0.0
    lam_hi = 1.0
    found = False
    for _ in range(int(max_doublings)):
        info["doublings"] = int(info["doublings"]) + 1
        if _feasible_lambda(lam_hi):
            found = True
            break
        lam_lo = lam_hi
        lam_hi *= 2.0

    if not found:
        info["mode"] = "tier2_failed"
        return y_anchor, False, info

    for _ in range(int(bisect_iters)):
        lam_mid = 0.5 * (lam_lo + lam_hi)
        if _feasible_lambda(lam_mid):
            lam_hi = lam_mid
        else:
            lam_lo = lam_mid
        if (lam_hi - lam_lo) < 1e-12:
            break

    y_anchor[int(j_idx)] = float(lam_hi)
    info["lambda"] = float(lam_hi)
    info["mode"] = "tier2_coord"
    return y_anchor, True, info


def _short_ipm_feasible_dual(
    sdp: SDPInstance,
    y_start: np.ndarray,
    feas_tol: float = 1e-8,
    jitter: float = 1e-9,
    schedule: tuple[int, ...] = (3, 6, 10, 15),
) -> np.ndarray | None:
    """Short-IPM feasibility fallback for Tier-0 cascade."""
    m = sdp.num_constraints
    n = sdp.dim
    y = np.asarray(y_start, dtype=float).reshape(-1)
    if y.shape[0] != m:
        y = np.zeros(m, dtype=float)

    S = slack_from_y(sdp, y)
    vals = np.linalg.eigvalsh(S)
    min_eig = float(np.min(vals))
    if min_eig <= 0.0:
        S = S + (abs(min_eig) + 1e-6) * np.eye(n, dtype=float)
    state = IPMState(X=np.eye(n, dtype=float), y=y.copy(), S=0.5 * (S + S.T))

    for k in schedule:
        solver = InfeasibleIPMSolver(
            IPMSettings(
                max_iters=max(40, int(4 * k)),
                tol_abs=1e-6,
                tol_rel=1e-5,
                linear_solve="sylvester",
            )
        )
        try:
            res = solver.refine(
                sdp,
                y0=state.y,
                X0=state.X,
                S0=state.S,
                num_iters=int(k),
            )
        except Exception:
            continue
        y_try = np.asarray(res.y, dtype=float).reshape(-1)
        S_try = slack_from_y(sdp, y_try)
        if is_psd_cholesky(
            S_try + float(max(feas_tol, 0.0)) * np.eye(n, dtype=float),
            jitter=jitter,
        ):
            return y_try
        S_try = 0.5 * (S_try + S_try.T)
        vals_try = np.linalg.eigvalsh(S_try)
        min_eig_try = float(np.min(vals_try))
        if min_eig_try <= 0.0:
            S_try = S_try + (abs(min_eig_try) + 1e-6) * np.eye(n, dtype=float)
        state = IPMState(X=np.eye(n, dtype=float), y=y_try, S=S_try)
    return None


def tier0_repair_cascade(
    sdp: SDPInstance,
    y_pred: np.ndarray,
    feas_tol: float = 1e-8,
    jitter: float = 1e-9,
    subgrad_steps: int = 3,
    fallback_mode: str = "off",
    debug: bool = False,
) -> tuple[np.ndarray, bool, Dict[str, Any]]:
    """Tier-0 feasibility repair: spectral -> few subgradient cuts -> fallback."""
    y_curr = np.asarray(y_pred, dtype=float).reshape(-1).copy()
    m = sdp.num_constraints
    n = sdp.dim
    if y_curr.shape[0] != m:
        y_curr = np.zeros(m, dtype=float)

    info: Dict[str, Any] = {
        "used": True,
        "step": "none",
        "success": False,
        "fallback": str(fallback_mode).strip().lower(),
        "subgrad_iters": 0,
    }
    tol = float(max(feas_tol, 0.0))
    I = np.eye(n, dtype=float)

    def _check_feas(y: np.ndarray) -> bool:
        S = slack_from_y(sdp, y)
        return bool(is_psd_cholesky(S + tol * I, jitter=jitter))

    # Step 1: spectral slack repair + least squares in dual space.
    try:
        S = slack_from_y(sdp, y_curr)
        vals, vecs = np.linalg.eigh(S)
        vals_clip = np.maximum(vals, 0.0)
        S_proj = vecs @ np.diag(vals_clip) @ vecs.T
        S_proj = 0.5 * (S_proj + S_proj.T)
        rhs = (np.asarray(sdp.C, dtype=float) - S_proj).reshape(-1)
        A_mat = np.asarray(sdp.A, dtype=float).reshape(m, -1).T
        y_ls, *_ = np.linalg.lstsq(A_mat, rhs, rcond=None)
        y_ls = np.asarray(y_ls, dtype=float).reshape(-1)
        if y_ls.shape[0] == m and np.all(np.isfinite(y_ls)):
            y_curr = y_ls
            if _check_feas(y_curr):
                info["step"] = "spectral"
                info["success"] = True
                return y_curr, True, info
    except Exception:
        if debug:
            info["spectral_error"] = True

    # Step 2: a few minimum-eigenvector cut updates.
    A_t = np.asarray(sdp.A, dtype=float)
    for k in range(int(max(1, subgrad_steps))):
        S = slack_from_y(sdp, y_curr)
        vals, vecs = np.linalg.eigh(S)
        lam_min = float(vals[0])
        if lam_min >= -tol:
            info["step"] = "subgrad"
            info["success"] = True
            info["subgrad_iters"] = int(k)
            return y_curr, True, info
        v = np.asarray(vecs[:, 0], dtype=float).reshape(-1)
        Av = np.einsum("mij,j->mi", A_t, v)
        g = -np.einsum("i,mi->m", v, Av)
        denom = float(np.dot(g, g)) + 1e-12
        step = abs(lam_min) / denom
        y_curr = y_curr + float(step) * g
        info["subgrad_iters"] = int(k + 1)
        if _check_feas(y_curr):
            info["step"] = "subgrad"
            info["success"] = True
            return y_curr, True, info

    # Step 3: guaranteed fallback (optional).
    mode = str(fallback_mode).strip().lower()
    if mode == "fi":
        try:
            from .l2ca_fi import l2ca_fi_predict

            y_fb = l2ca_fi_predict(
                sdp=sdp,
                y_pred=y_curr,
                max_phase1_iters=30,
                jitter=jitter,
                objective_lift=False,
            )
            if _check_feas(y_fb):
                info["step"] = "fallback_fi"
                info["success"] = True
                return np.asarray(y_fb, dtype=float).reshape(-1), True, info
        except Exception:
            if debug:
                info["fallback_error"] = "fi"
    elif mode == "short_ipm":
        y_fb = _short_ipm_feasible_dual(
            sdp=sdp,
            y_start=y_curr,
            feas_tol=tol,
            jitter=jitter,
        )
        if y_fb is not None and _check_feas(y_fb):
            info["step"] = "fallback_short_ipm"
            info["success"] = True
            return np.asarray(y_fb, dtype=float).reshape(-1), True, info

    info["step"] = "failed"
    info["success"] = False
    return y_curr, False, info


def find_knn_anchor(
    sdp: SDPInstance,
    test_features: np.ndarray,
    train_features: np.ndarray,
    train_duals: np.ndarray,
    k: int = 5,
    tol: float = 1e-8,
) -> Tuple[np.ndarray | None, bool]:
    """Find nearest training dual that is feasible for the given test SDP."""
    test_features = np.asarray(test_features, dtype=float).reshape(1, -1)
    train_features = np.asarray(train_features, dtype=float)
    train_duals = np.asarray(train_duals, dtype=float)

    if train_features.ndim != 2 or train_duals.ndim != 2:
        raise ValueError("train_features and train_duals must be 2D arrays.")
    if train_features.shape[0] != train_duals.shape[0]:
        raise ValueError("train_features and train_duals must have matching row counts.")
    if train_features.shape[1] != test_features.shape[1]:
        raise ValueError("test_features dimension mismatch with train_features.")

    if train_features.shape[0] == 0:
        return None, False

    nearest_idx = knn_indices(test_features, train_features, k=int(k))

    for idx in nearest_idx:
        y_candidate = np.asarray(train_duals[idx], dtype=float).reshape(-1)
        S = slack_from_y(sdp, y_candidate)
        # Relaxed tolerance: min eig(S) >= -tol  <=>  S + tol*I is PSD.
        n = S.shape[0]
        if is_psd_cholesky(S + float(tol) * np.eye(n, dtype=float), jitter=1e-9):
            return y_candidate.copy(), True

    return None, False

def select_robust_anchor(
    train_sdps: Sequence[SDPInstance],
    train_duals: np.ndarray,
    sample_size: int = 50,
    tol: float = 1e-8,
) -> np.ndarray:
    """Select the training dual most likely to be feasible for unseen SDPs."""
    train_duals = np.asarray(train_duals, dtype=float)
    if train_duals.ndim != 2:
        raise ValueError("train_duals must be a 2D array.")
    if len(train_sdps) != train_duals.shape[0]:
        raise ValueError("train_sdps and train_duals must have matching lengths.")
    n_train = len(train_sdps)
    if n_train == 0:
        raise ValueError("train_sdps must be non-empty.")

    n_check = min(int(sample_size), n_train)
    rng = np.random.default_rng(0)
    check_indices = rng.choice(n_train, size=n_check, replace=False)

    best_idx = 0
    best_count = -1
    for i in range(n_train):
        y_i = train_duals[i]
        count = 0
        for j in check_indices:
            sdp = train_sdps[j]
            S = slack_from_y(sdp, y_i)
            n = S.shape[0]
            if is_psd_cholesky(S + float(tol) * np.eye(n, dtype=float), jitter=1e-9):
                count += 1
        if count > best_count:
            best_count = count
            best_idx = i

    return np.asarray(train_duals[best_idx], dtype=float).copy()


def correct_dual(
    sdp: SDPInstance,
    y_pred: np.ndarray,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Project to dual feasibility via eigenvalue clipping + least-squares."""
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    S = sdp.C - sdp.apply_AT(y_pred)
    S = 0.5 * (S + S.T)
    eigs, vecs = np.linalg.eigh(S)
    if float(eigs.min()) >= -1e-8:
        return y_pred, S, True

    eigs_clipped = np.maximum(eigs, eps)
    S_proj = vecs @ np.diag(eigs_clipped) @ vecs.T
    S_proj = 0.5 * (S_proj + S_proj.T)

    n = sdp.dim
    m = sdp.num_constraints
    A_mat = np.column_stack([sdp.A[i].ravel() for i in range(m)])
    rhs = (sdp.C - S_proj).ravel()
    y_new, _, _, _ = np.linalg.lstsq(A_mat, rhs, rcond=None)
    y_new = np.asarray(y_new, dtype=float).reshape(-1)
    S_new = sdp.C - sdp.apply_AT(y_new)
    S_new = 0.5 * (S_new + S_new.T)
    lam_min = float(np.linalg.eigvalsh(S_new)[0])
    ok = lam_min >= -1e-8
    if ok:
        return y_new, S_new, True

    e_last = np.zeros_like(y_new)
    e_last[-1] = 1.0
    low = 0.0
    high = 1.0
    for _ in range(40):
        y_try = y_new + high * e_last
        S_try = sdp.C - sdp.apply_AT(y_try)
        S_try = 0.5 * (S_try + S_try.T)
        if float(np.linalg.eigvalsh(S_try)[0]) >= -1e-8:
            break
        high *= 2.0
    else:
        return y_new, S_new, False

    for _ in range(40):
        mid = 0.5 * (low + high)
        y_mid = y_new + mid * e_last
        S_mid = sdp.C - sdp.apply_AT(y_mid)
        S_mid = 0.5 * (S_mid + S_mid.T)
        lam_mid = float(np.linalg.eigvalsh(S_mid)[0])
        if lam_mid >= -1e-8:
            high = mid
            S_new = S_mid
        else:
            low = mid

    y_new = y_new + high * e_last
    S_new = sdp.C - sdp.apply_AT(y_new)
    S_new = 0.5 * (S_new + S_new.T)
    ok = float(np.linalg.eigvalsh(S_new)[0]) >= -1e-8
    return y_new, S_new, ok


def correct_dual_bisection(
    sdp: SDPInstance,
    y_pred: np.ndarray,
    y_anchor: np.ndarray,
    tol: float = 1e-8,
    max_bisect: int = 25,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Backward-compatible wrapper around bisection_to_feasible."""
    y_out, S_out, ok, _ = bisection_to_feasible(
        sdp,
        y_pred,
        y_anchor,
        max_iters=max_bisect,
        tol=tol,
    )
    return y_out, S_out, ok


def _is_slack_feasible_relaxed(
    S: np.ndarray,
    tol: float,
    jitter: float,
    stats: Dict[str, Any] | None = None,
    debug: bool = False,
    key_prefix: str = "",
) -> bool:
    """Check min eig(S) >= -tol using only Cholesky in hot path."""
    n = S.shape[0]
    ok = is_psd_cholesky(S + float(tol) * np.eye(n, dtype=float), jitter=jitter)
    if stats is not None:
        stats["cholesky_checks"] = int(stats.get("cholesky_checks", 0)) + 1
        if debug:
            stats[f"{key_prefix}lam_min"] = float(_lam_min(S))
    return ok


def bisection_to_feasible(
    sdp: SDPInstance,
    y_pred: np.ndarray,
    y_anchor: np.ndarray,
    max_iters: int = 20,
    tol: float = 1e-8,
    jitter: float = 1e-9,
    debug: bool = False,
) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, Any]]:
    """Find a feasible point on segment [y_pred, y_anchor] by bisection."""
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    y_anchor = np.asarray(y_anchor, dtype=float).reshape(-1)
    if y_pred.shape != y_anchor.shape:
        raise ValueError("y_pred and y_anchor must have matching shape.")

    stats: Dict[str, Any] = {
        "cholesky_checks": 0,
        "apply_at_calls": 0,
        "pred_feasible": False,
        "anchor_feasible": False,
        "bisect_steps": 0,
    }

    # Precompute both slacks once.
    S_pred = slack_from_y(sdp, y_pred, stats=stats)
    if _is_slack_feasible_relaxed(S_pred, tol=tol, jitter=jitter, stats=stats, debug=debug, key_prefix="pred_"):
        stats["pred_feasible"] = True
        return y_pred, S_pred, True, stats

    S_anchor = slack_from_y(sdp, y_anchor, stats=stats)
    if not _is_slack_feasible_relaxed(S_anchor, tol=tol, jitter=jitter, stats=stats, debug=debug, key_prefix="anchor_"):
        return y_pred, S_pred, False, stats
    stats["anchor_feasible"] = True

    # By linearity of apply_AT, slack along segment is linear interpolation.
    low, high = 0.0, 1.0
    S_best = S_anchor
    for _ in range(max_iters):
        mid = 0.5 * (low + high)
        S_mid = (1.0 - mid) * S_pred + mid * S_anchor
        stats["bisect_steps"] += 1
        if _is_slack_feasible_relaxed(S_mid, tol=tol, jitter=jitter, stats=stats):
            high = mid
            S_best = S_mid
        else:
            low = mid
        if (high - low) < 1e-10:
            break

    y_out = (1.0 - high) * y_pred + high * y_anchor
    S_out = (1.0 - high) * S_pred + high * S_anchor
    S_out = 0.5 * (S_out + S_out.T)
    ok = _is_slack_feasible_relaxed(S_out, tol=tol, jitter=jitter, stats=stats, debug=debug, key_prefix="out_")
    if ok:
        return y_out, S_out, True, stats
    return y_out, S_best, False, stats


def maximize_dual_along_b(
    sdp: SDPInstance,
    y_start: np.ndarray,
    max_expand: int = 20,
    bisect_iters: int = 40,
    feas_tol: float = 1e-8,
    jitter: float = 1e-9,
    debug: bool = False,
) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, Any]]:
    """Increase b^T y along direction b while preserving dual feasibility."""
    y_start = np.asarray(y_start, dtype=float).reshape(-1)
    b_dir = np.asarray(sdp.b, dtype=float).reshape(-1)
    if y_start.shape != b_dir.shape:
        raise ValueError("y_start must have same dimension as sdp.b.")

    stats: Dict[str, Any] = {
        "ran": True,
        "cholesky_checks": 0,
        "apply_at_calls": 0,
        "expand_steps": 0,
        "bisect_steps": 0,
        "start_feasible": False,
        "expand_limit_hit": False,
    }

    norm_b = float(np.linalg.norm(b_dir))
    if norm_b <= 1e-14:
        S0 = slack_from_y(sdp, y_start, stats=stats)
        ok0 = _is_slack_feasible_relaxed(S0, tol=feas_tol, jitter=jitter, stats=stats, debug=debug, key_prefix="start_")
        stats["start_feasible"] = bool(ok0)
        return y_start, S0, bool(ok0), stats
    b_dir = b_dir / norm_b

    S_start = slack_from_y(sdp, y_start, stats=stats)
    # Along b-direction: S(y_start + t*b_dir) = S_start - t * A*(b_dir).
    S_step = sdp.apply_AT(b_dir)
    stats["apply_at_calls"] = int(stats.get("apply_at_calls", 0)) + 1
    S_step = 0.5 * (S_step + S_step.T)

    def _check_t(t: float) -> Tuple[bool, np.ndarray]:
        S_t = S_start - float(t) * S_step
        S_t = 0.5 * (S_t + S_t.T)
        ok_t = _is_slack_feasible_relaxed(S_t, tol=feas_tol, jitter=jitter, stats=stats)
        return bool(ok_t), S_t

    ok0, S0 = _check_t(0.0)
    stats["start_feasible"] = bool(ok0)
    if not ok0:
        return y_start, S0, False, stats

    t_lo = 0.0
    S_lo = S0
    t_hi = 1.0
    ok_hi, S_hi = _check_t(t_hi)
    while ok_hi and stats["expand_steps"] < int(max_expand):
        t_lo = t_hi
        S_lo = S_hi
        t_hi *= 2.0
        stats["expand_steps"] += 1
        ok_hi, S_hi = _check_t(t_hi)

    if ok_hi:
        # Still feasible at expansion cap; return best certified feasible point.
        stats["expand_limit_hit"] = True
        y_out = y_start + t_lo * b_dir
        return y_out, S_lo, True, stats

    # Bracketed: t_lo feasible, t_hi infeasible.
    for _ in range(int(bisect_iters)):
        t_mid = 0.5 * (t_lo + t_hi)
        ok_mid, S_mid = _check_t(t_mid)
        stats["bisect_steps"] += 1
        if ok_mid:
            t_lo = t_mid
            S_lo = S_mid
        else:
            t_hi = t_mid
        if (t_hi - t_lo) < 1e-10:
            break

    y_out = y_start + t_lo * b_dir
    return y_out, S_lo, True, stats


def run_l2ca_inference(
    sdp: SDPInstance,
    y_pred: np.ndarray,
    x_feat: np.ndarray,
    x_train_norm: np.ndarray,
    y_train: np.ndarray,
    cached_feasible_anchor: np.ndarray | None,
    feas_margin: float = 1e-8,
    bisect_iters: int = 20,
    tier_auto: bool = True,
    tier_level: int = 0,
    tier_j_idx: int | None = None,
    tier0_fallback: str = "off",
    force_global_anchor: bool = False,
    global_mean_anchor: np.ndarray | None = None,
    anchor_k: int = 5,
    skip_lift: bool = False,
    refine_solver: InfeasibleIPMSolver | None = None,
    refine_iters: int = 0,
    enable_refine: bool = False,
    debug: bool = False,
) -> Dict[str, Any]:
    """Run the full L2CA certify/repair path used by the paper-facing pipeline."""
    x = np.asarray(x_feat, dtype=float).reshape(-1)
    X_train = np.asarray(x_train_norm, dtype=float)
    Y_train = np.asarray(y_train, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if X_train.ndim != 2 or Y_train.ndim != 2:
        raise ValueError("x_train_norm and y_train must be 2D arrays.")
    if X_train.shape[0] != Y_train.shape[0]:
        raise ValueError("x_train_norm and y_train must have matching row counts.")
    if X_train.shape[0] == 0:
        raise ValueError("x_train_norm must be non-empty.")
    if X_train.shape[1] != x.shape[0]:
        raise ValueError("x_feat dimension mismatch with x_train_norm.")

    if global_mean_anchor is None:
        global_mean_anchor = np.mean(Y_train, axis=0)
    if cached_feasible_anchor is not None:
        cached_feasible_anchor = np.asarray(cached_feasible_anchor, dtype=float).reshape(-1)

    instance_stats: Dict[str, Any] = {"apply_at_calls": 0}
    S_pred = slack_from_y(sdp, y_pred, stats=instance_stats)
    n_s = S_pred.shape[0]
    fast_path_accept = is_psd_with_margin(
        S_pred,
        margin=float(feas_margin),
        jitter=1e-9,
    )

    anchor_info: Dict[str, Any] = {"mode": "none", "chol_checks": 1}
    repair_stats: Dict[str, Any] = {"cholesky_checks": 0, "bisect_steps": 0}
    lift_stats: Dict[str, Any] = {"cholesky_checks": 0, "bisect_steps": 0, "ran": False}
    tier0_stats: Dict[str, Any] = {"used": False, "step": "none", "success": False}
    anchor_feasible = False
    y_out = y_pred.copy()
    repair_ok = bool(fast_path_accept)

    if not fast_path_accept:
        if bool(tier_auto):
            if int(tier_level) in {1, 2}:
                y_anchor = np.zeros_like(y_pred)
                if cached_feasible_anchor is not None:
                    S_robust = slack_from_y(
                        sdp,
                        cached_feasible_anchor,
                        stats=instance_stats,
                    )
                    robust_ok = is_psd_with_margin(
                        S_robust,
                        margin=float(feas_margin),
                        jitter=1e-9,
                    )
                else:
                    robust_ok = False

                if robust_ok and cached_feasible_anchor is not None:
                    y_anchor = cached_feasible_anchor.copy()
                    anchor_feasible = True
                    tier_info = {
                        "mode": f"tier{int(tier_level)}_robust_preferred",
                        "doublings": 0,
                        "lambda": 0.0,
                    }
                else:
                    y_anchor, anchor_feasible, tier_info = build_anchor(
                        sdp,
                        tier=int(tier_level),
                        j_idx=tier_j_idx,
                        tol=float(feas_margin),
                        jitter=1e-9,
                    )
                anchor_info = {
                    "mode": str(tier_info.get("mode", f"tier{int(tier_level)}")),
                    "chol_checks": int(anchor_info.get("chol_checks", 0))
                    + int(tier_info.get("doublings", 0))
                    + 1,
                    "apply_at_calls": 0,
                    "tier": int(tier_level),
                    "tier_j": tier_j_idx,
                    "tier_lambda": float(tier_info.get("lambda", 0.0)),
                }
                if anchor_feasible:
                    y_feas, _, repair_ok, repair_stats = bisection_to_feasible(
                        sdp,
                        y_pred,
                        y_anchor,
                        max_iters=int(bisect_iters),
                        tol=float(feas_margin),
                        jitter=1e-9,
                        debug=bool(debug),
                    )
                    y_out = y_feas
                    if repair_ok:
                        out_obj = float(np.dot(sdp.b, y_out))
                        anc_obj = float(np.dot(sdp.b, y_anchor))
                        if anc_obj > out_obj:
                            y_out = y_anchor.copy()
                else:
                    y_out, repair_ok, tier0_stats = tier0_repair_cascade(
                        sdp,
                        y_pred,
                        feas_tol=float(feas_margin),
                        jitter=1e-9,
                        subgrad_steps=3,
                        fallback_mode=str(tier0_fallback),
                        debug=bool(debug),
                    )
            else:
                anchor_info = {
                    "mode": "tier0_robust_anchor",
                    "chol_checks": int(anchor_info.get("chol_checks", 0)),
                    "apply_at_calls": 0,
                    "tier": 0,
                }
                robust_ok = False
                if cached_feasible_anchor is not None:
                    S_robust = slack_from_y(
                        sdp,
                        cached_feasible_anchor,
                        stats=instance_stats,
                    )
                    robust_ok = is_psd_with_margin(
                        S_robust,
                        margin=float(feas_margin),
                        jitter=1e-9,
                    )
                    anchor_info["chol_checks"] = int(anchor_info.get("chol_checks", 0)) + 1

                if robust_ok and cached_feasible_anchor is not None:
                    anchor_feasible = True
                    y_feas, _, repair_ok, repair_stats = bisection_to_feasible(
                        sdp,
                        y_pred,
                        cached_feasible_anchor,
                        max_iters=int(bisect_iters),
                        tol=float(feas_margin),
                        jitter=1e-9,
                        debug=bool(debug),
                    )
                    y_out = y_feas
                    anchor_info["mode"] = "tier0_robust_bisect"
                    if repair_ok:
                        out_obj = float(np.dot(sdp.b, y_out))
                        anc_obj = float(np.dot(sdp.b, cached_feasible_anchor))
                        if anc_obj > out_obj:
                            y_out = cached_feasible_anchor.copy()
                    else:
                        anchor_info["mode"] = "tier0_cascade_after_robust"
                        y_out, repair_ok, tier0_stats = tier0_repair_cascade(
                            sdp,
                            y_pred,
                            feas_tol=float(feas_margin),
                            jitter=1e-9,
                            subgrad_steps=3,
                            fallback_mode=str(tier0_fallback),
                            debug=bool(debug),
                        )
                else:
                    dists = np.linalg.norm(X_train - x.reshape(1, -1), axis=1)
                    nn_idx = int(np.argmin(dists))
                    y_nn = np.asarray(Y_train[nn_idx], dtype=float).reshape(-1)
                    S_nn = slack_from_y(sdp, y_nn, stats=instance_stats)
                    nn_ok = is_psd_with_margin(
                        S_nn,
                        margin=float(feas_margin),
                        jitter=1e-9,
                    )
                    anchor_info["chol_checks"] = int(anchor_info.get("chol_checks", 0)) + 1

                    if nn_ok:
                        anchor_feasible = True
                        y_feas, _, repair_ok, repair_stats = bisection_to_feasible(
                            sdp,
                            y_pred,
                            y_nn,
                            max_iters=int(bisect_iters),
                            tol=float(feas_margin),
                            jitter=1e-9,
                            debug=bool(debug),
                        )
                        y_out = y_feas
                        anchor_info["mode"] = "tier0_nn_bisect"
                        if repair_ok:
                            out_obj = float(np.dot(sdp.b, y_out))
                            anc_obj = float(np.dot(sdp.b, y_nn))
                            if anc_obj > out_obj:
                                y_out = y_nn.copy()
                        else:
                            anchor_info["mode"] = "tier0_cascade_after_nn"
                            y_out, repair_ok, tier0_stats = tier0_repair_cascade(
                                sdp,
                                y_pred,
                                feas_tol=float(feas_margin),
                                jitter=1e-9,
                                subgrad_steps=3,
                                fallback_mode=str(tier0_fallback),
                                debug=bool(debug),
                            )
                    else:
                        anchor_info["mode"] = "tier0_cascade"
                        y_out, repair_ok, tier0_stats = tier0_repair_cascade(
                            sdp,
                            y_pred,
                            feas_tol=float(feas_margin),
                            jitter=1e-9,
                            subgrad_steps=3,
                            fallback_mode=str(tier0_fallback),
                            debug=bool(debug),
                        )
        else:
            if bool(force_global_anchor):
                y_anchor = np.asarray(global_mean_anchor, dtype=float).reshape(-1).copy()
                S_anchor = slack_from_y(sdp, y_anchor, stats=instance_stats)
                anchor_feasible = is_psd_cholesky(
                    S_anchor + float(feas_margin) * np.eye(S_anchor.shape[0], dtype=float),
                    jitter=1e-9,
                )
                anchor_info = {"mode": "global_mean", "chol_checks": 1, "apply_at_calls": 0}
            else:
                y_anchor, anchor_feasible, anchor_info = choose_anchor_best_knn(
                    x,
                    X_train,
                    Y_train,
                    sdp,
                    k=int(anchor_k),
                    global_mean=np.asarray(global_mean_anchor, dtype=float),
                    cached_feasible_anchor=cached_feasible_anchor,
                    feas_margin=float(feas_margin),
                    jitter=1e-9,
                    robust=False,
                    allow_fullscan=True,
                )

            if not anchor_feasible:
                zero_anchor = np.zeros_like(y_pred)
                S_zero = slack_from_y(sdp, zero_anchor, stats=instance_stats)
                zero_ok = is_psd_cholesky(
                    S_zero + float(feas_margin) * np.eye(S_zero.shape[0], dtype=float),
                    jitter=1e-9,
                )
                anchor_info["chol_checks"] = int(anchor_info.get("chol_checks", 0)) + 1
                if zero_ok:
                    anchor_repaired, _, repair_anchor_ok, repair_anchor_stats = bisection_to_feasible(
                        sdp,
                        np.asarray(y_anchor, dtype=float).reshape(-1),
                        zero_anchor,
                        max_iters=int(bisect_iters),
                        tol=float(feas_margin),
                        jitter=1e-9,
                        debug=False,
                    )
                    anchor_info["chol_checks"] = int(anchor_info.get("chol_checks", 0)) + int(
                        repair_anchor_stats.get("cholesky_checks", 0)
                    )
                    anchor_info["apply_at_calls"] = int(anchor_info.get("apply_at_calls", 0)) + int(
                        repair_anchor_stats.get("apply_at_calls", 0)
                    )
                    if repair_anchor_ok:
                        y_anchor = anchor_repaired
                        anchor_feasible = True
                        anchor_info["mode"] = f"{anchor_info.get('mode', 'anchor')}_repaired"
                    else:
                        y_anchor = zero_anchor
                        anchor_feasible = True
                        anchor_info["mode"] = "zero_certified"

            if anchor_feasible:
                y_feas, _, repair_ok, repair_stats = bisection_to_feasible(
                    sdp,
                    y_pred,
                    y_anchor,
                    max_iters=int(bisect_iters),
                    tol=float(feas_margin),
                    jitter=1e-9,
                    debug=bool(debug),
                )
                y_out = y_feas
                if repair_ok:
                    out_obj = float(np.dot(sdp.b, y_out))
                    anc_obj = float(np.dot(sdp.b, y_anchor))
                    if anc_obj > out_obj:
                        y_out = y_anchor.copy()
            else:
                y_out = y_pred.copy()
                repair_ok = False

    if repair_ok and (not fast_path_accept) and not bool(skip_lift):
        y_lift, _, lift_ok, lift_stats = maximize_dual_along_b(
            sdp,
            y_out,
            max_expand=6,
            bisect_iters=8,
            feas_tol=float(feas_margin),
            jitter=1e-9,
            debug=False,
        )
        if lift_ok:
            lift_obj = float(np.dot(sdp.b, y_lift))
            curr_obj = float(np.dot(sdp.b, y_out))
            if lift_obj > curr_obj:
                y_out = y_lift

    refine_ran = False
    refine_accepted = False
    refine_used_iters = 0
    if bool(enable_refine) and int(refine_iters) > 0 and refine_solver is not None:
        try:
            base_obj = float(np.dot(sdp.b, y_out))
            res_ref = refine_solver.refine(
                sdp,
                y0=y_out,
                num_iters=int(refine_iters),
            )
            y_ref = np.asarray(res_ref.y, dtype=float).reshape(-1)
            S_ref = slack_from_y(sdp, y_ref, stats=instance_stats)
            ref_ok = is_psd_cholesky(
                S_ref + float(feas_margin) * np.eye(n_s, dtype=float),
                jitter=1e-9,
            )
            ref_obj = float(np.dot(sdp.b, y_ref))
            if ref_ok and ref_obj >= base_obj:
                y_out = y_ref
                refine_accepted = True
            refine_ran = True
            refine_used_iters = int(res_ref.iterations)
        except Exception as exc:
            if bool(debug):
                print(f"[l2ca-debug] refine_failed={exc}")

    S_out = slack_from_y(sdp, y_out, stats=instance_stats)
    final_ok = is_psd_cholesky(
        S_out + float(feas_margin) * np.eye(n_s, dtype=float),
        jitter=1e-9,
    )

    return {
        "y_out": y_out,
        "S_out": S_out,
        "final_ok": bool(final_ok),
        "fast_path_accept": bool(fast_path_accept),
        "repair_ok": bool(repair_ok),
        "anchor_info": anchor_info,
        "repair_stats": repair_stats,
        "lift_stats": lift_stats,
        "tier0_stats": tier0_stats,
        "instance_stats": instance_stats,
        "anchor_feasible": bool(anchor_feasible),
        "refine_ran": bool(refine_ran),
        "refine_accepted": bool(refine_accepted),
        "refine_used_iters": int(refine_used_iters),
    }


def maximize_dual_toward_direction(
    sdp: SDPInstance,
    y_start: np.ndarray,
    direction: np.ndarray,
    bisect_iters: int = 15,
    feas_tol: float = 1e-8,
    jitter: float = 1e-9,
    debug: bool = False,
) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, Any]]:
    """Legacy utility: maximize dual objective along y_start + t*direction, t in [0,1]."""
    y_start = np.asarray(y_start, dtype=float).reshape(-1)
    d = np.asarray(direction, dtype=float).reshape(-1)
    if y_start.shape != d.shape:
        raise ValueError("direction must match y_start shape.")

    stats: Dict[str, Any] = {
        "cholesky_checks": 0,
        "apply_at_calls": 0,
        "bisect_steps": 0,
        "ran": False,
        "gain_coeff": float(np.dot(sdp.b, d)),
    }

    S0 = slack_from_y(sdp, y_start, stats=stats)
    ok0 = _is_slack_feasible_relaxed(S0, tol=feas_tol, jitter=jitter, stats=stats, debug=debug, key_prefix="start_")
    if not ok0:
        return y_start, S0, False, stats

    gain = float(np.dot(sdp.b, d))
    if not np.isfinite(gain) or gain <= 1e-16:
        return y_start, S0, True, stats

    stats["ran"] = True
    S_dir = sdp.apply_AT(d)
    stats["apply_at_calls"] = int(stats.get("apply_at_calls", 0)) + 1
    S_dir = 0.5 * (S_dir + S_dir.T)

    def _check_t(t: float) -> Tuple[bool, np.ndarray]:
        S_t = S0 - float(t) * S_dir
        S_t = 0.5 * (S_t + S_t.T)
        ok_t = _is_slack_feasible_relaxed(S_t, tol=feas_tol, jitter=jitter, stats=stats)
        return bool(ok_t), S_t

    ok1, S1 = _check_t(1.0)
    if ok1:
        return y_start + d, S1, True, stats

    lo, hi = 0.0, 1.0
    S_lo = S0
    for _ in range(int(bisect_iters)):
        mid = 0.5 * (lo + hi)
        ok_mid, S_mid = _check_t(mid)
        stats["bisect_steps"] += 1
        if ok_mid:
            lo = mid
            S_lo = S_mid
        else:
            hi = mid
        if (hi - lo) < 1e-10:
            break

    y_out = y_start + lo * d
    return y_out, S_lo, True, stats


def _hamiltonian_matrix(A: np.ndarray, Bw: np.ndarray, Cz: np.ndarray, gamma: float) -> np.ndarray:
    inv_gamma_sq = 1.0 / (gamma * gamma)
    top_right = inv_gamma_sq * (Bw @ Bw.T)
    bottom_left = -(Cz.T @ Cz)
    return np.block([[A, top_right], [bottom_left, -A.T]])


def _hamiltonian_feasible(A: np.ndarray, Bw: np.ndarray, Cz: np.ndarray, gamma: float, tol: float) -> bool:
    if not np.isfinite(gamma) or gamma <= 0:
        return False
    H = _hamiltonian_matrix(A, Bw, Cz, gamma)
    eigs = np.linalg.eigvals(H)
    min_abs_real = float(np.min(np.abs(np.real(eigs))))
    return min_abs_real > tol


def hamiltonian_bisection(
    A: np.ndarray,
    Bw: np.ndarray,
    Cz: np.ndarray,
    gamma_init: float,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> Tuple[float, int]:
    """Bisect gamma using Hamiltonian imaginary-axis crossing criterion."""
    gamma_init = float(max(gamma_init, 1e-6))
    low = gamma_init * 0.5
    high = gamma_init * 1.5
    low = max(low, 1e-8)
    high = max(high, low * 2.0)

    for _ in range(max_iter):
        if not _hamiltonian_feasible(A, Bw, Cz, low, tol):
            break
        low *= 0.5
        if low <= 1e-12:
            break
    for _ in range(max_iter):
        if _hamiltonian_feasible(A, Bw, Cz, high, tol):
            break
        high *= 2.0

    if not _hamiltonian_feasible(A, Bw, Cz, high, tol):
        return float(high), 0

    n_bisect = 0
    for _ in range(max_iter):
        n_bisect += 1
        mid = 0.5 * (low + high)
        if _hamiltonian_feasible(A, Bw, Cz, mid, tol):
            high = mid
        else:
            low = mid
        if (high - low) <= tol * max(1.0, high):
            break
    return float(high), n_bisect


__all__ = [
    "DualNet",
    "ScalarL2ANet",
    "extract_gamma_from_dual",
    "solve_instances_for_duals",
    "train_dual_net",
    "train_scalar_l2a",
    "certify_dual",
    "slack_from_y",
    "is_psd_cholesky",
    "is_psd_with_margin",
    "knn_indices",
    "choose_anchor_best_knn",
    "get_anchor_y",
    "check_c_psd",
    "detect_tier",
    "build_anchor",
    "tier0_repair_cascade",
    "find_knn_anchor",
    "select_robust_anchor",
    "correct_dual",
    "bisection_to_feasible",
    "correct_dual_bisection",
    "maximize_dual_along_b",
    "run_l2ca_inference",
    "maximize_dual_toward_direction",
    "_hamiltonian_matrix",
    "_hamiltonian_feasible",
    "hamiltonian_bisection",
]
