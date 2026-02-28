"""Infeasible predictor-corrector IPM (Potra–Sheng) for standard-form SDPs.
This module implements Algorithm 2 in the paper: 
"Learning to Warm-Start an Interior Point Method for Adaptive Semidefinite
Programming" (Zhu, Taha).
The solver is designed for warm starting: the initial point only needs to be
positive definite and within the wide neighborhood \\tilde{N}_F(\\gamma).
"""

from __future__ import annotations
from dataclasses import dataclass, replace
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.linalg import solve_sylvester
from .problem import SDPInstance

Array = np.ndarray

def symmetrize(mat: Array) -> Array:
    return 0.5 * (mat + mat.T)

def _eigvals_xs(X: Array, S: Array) -> np.ndarray:
    """Eigenvalues of X S (guaranteed real/positive for SPD X,S)."""
    # Use the symmetric square root X^{1/2} so we can always form a symmetric
    # similarity transform even when X is slightly indefinite numerically.
    X_sqrt, _ = _sym_sqrt_and_invsqrt(X, eps=1e-14)
    mid = symmetrize(X_sqrt @ S @ X_sqrt)
    return np.linalg.eigvalsh(mid)


def nu_value(X: Array, S: Array) -> float:
    n = X.shape[0]
    return float(np.trace(X @ S) / n)


def dF_value(X: Array, S: Array) -> float:
    nu = nu_value(X, S)
    eigs = _eigvals_xs(X, S)
    return float(np.linalg.norm(eigs - nu))


def in_neighborhood(X: Array, S: Array, gamma: float) -> bool:
    nu = nu_value(X, S)
    return dF_value(X, S) <= gamma * nu


def _sym_sqrt_and_invsqrt(X: Array, eps: float = 1e-12) -> tuple[Array, Array]:
    vals, vecs = np.linalg.eigh(symmetrize(X))
    vals = np.clip(vals, eps, None)
    sqrt_vals = np.sqrt(vals)
    inv_sqrt_vals = 1.0 / sqrt_vals
    X_sqrt = (vecs * sqrt_vals) @ vecs.T
    X_inv_sqrt = (vecs * inv_sqrt_vals) @ vecs.T
    return X_sqrt, X_inv_sqrt


@dataclass
class IPMSettings:
    """Solver parameters matching Algorithm 2."""

    max_iters: int = 120
    tol_abs: float = 1e-6  # Conservative tolerance that reliably converges (1e-7/1e-8 can fail on some instances)
    tol_rel: float = 1e-5  # Relative tolerance
    gamma: float = 0.3  # neighborhood parameter (\\gamma)
    beta: float = 0.7  # parameter (\\beta) in the closed-form \\bar{\\theta}
    sym_eps: float = 1e-12
    schur_reg: float = 1e-8
    max_backtracks: int = 12
    backtrack_ratio: float = 0.8
    linear_solve: str = "kron"  # "kron" matches paper derivation; "sylvester" is a stable equivalent.
    verbose: bool = False

    def validate(self) -> None:
        if not (0.0 < self.gamma < 1.0):
            raise ValueError("gamma must be in (0, 1).")
        if not (0.0 < self.beta < 1.0):
            raise ValueError("beta must be in (0, 1).")
        if self.beta <= self.gamma:
            raise ValueError("beta must be strictly larger than gamma.")
        if self.linear_solve not in {"sylvester", "kron"}:
            raise ValueError("linear_solve must be 'sylvester' or 'kron'.")


@dataclass
class IPMState:
    X: Array
    y: Array
    S: Array

    def copy(self) -> "IPMState":
        return IPMState(self.X.copy(), self.y.copy(), self.S.copy())


@dataclass
class IPMResult:
    X: Array
    y: Array
    S: Array
    converged: bool
    iterations: int
    history: List[Dict[str, float]]
    iterates: List[IPMState] | None = None

    def state(self) -> IPMState:
        return IPMState(self.X, self.y, self.S)


class InfeasibleIPMSolver:
    """Potra–Sheng infeasible IPM (Algorithm 2) for SDPs."""

    def __init__(self, settings: Optional[IPMSettings] = None) -> None:
        self.settings = settings or IPMSettings()
        self.settings.validate()

    def solve(
        self,
        instance: SDPInstance,
        initial_state: Optional[IPMState] = None,
        time_budget: float | None = None,
        capture_iterates: bool = False,
        max_captured_iters: int = 10,
    ) -> IPMResult:
        X, y, S = self._initialize(instance, initial_state)
        n = instance.dim
        nu = nu_value(X, S)
        history: List[Dict[str, float]] = []
        iterates: List[IPMState] | None = [] if capture_iterates else None
        if iterates is not None:
            iterates.append(IPMState(X.copy(), y.copy(), S.copy()))
        start_time = time.perf_counter()

        if time_budget is not None and time_budget <= 0:
            return IPMResult(X, y, S, False, 0, history, iterates=iterates)

        for k in range(self.settings.max_iters):
            if time_budget is not None and (time.perf_counter() - start_time) >= time_budget:
                return IPMResult(X, y, S, False, k, history, iterates=iterates)
            r, Rd = instance.residuals(X, y, S)
            gap = float(np.trace(X @ S))
            metric = max(abs(gap), float(np.linalg.norm(Rd, ord="fro")), float(np.max(np.abs(r))))

            history.append(
                {
                    "iter": float(k),
                    "gap": gap,
                    "nu": float(nu),
                    "primal_residual": float(np.max(np.abs(r))),
                    "dual_residual": float(np.linalg.norm(Rd, ord="fro")),
                    "dF": float(dF_value(X, S)),
                }
            )

            tol_threshold = max(self.settings.tol_abs, self.settings.tol_rel * max(nu, 1.0))
            if metric <= tol_threshold:
                return IPMResult(X, y, S, True, k, history, iterates=iterates)

            # Predictor step: solve (10).
            U, w, V = self._solve_linear_system_predictor(instance, X, S, r, Rd)

            theta0 = self._theta_bar(X, U, V, nu)
            step = self._line_search_step(instance, X, y, S, U, w, V, nu, theta0)
            if step is None:
                return IPMResult(X, y, S, False, k, history, iterates=iterates)
            X, y, S, theta = step
            if iterates is not None and len(iterates) < int(max_captured_iters):
                iterates.append(IPMState(X.copy(), y.copy(), S.copy()))

            # Algorithm 2 update.
            nu = (1.0 - theta) * nu

        return IPMResult(X, y, S, False, self.settings.max_iters, history, iterates=iterates)

    def refine(
        self,
        instance: SDPInstance,
        y0: Array | None = None,
        X0: Array | None = None,
        S0: Array | None = None,
        num_iters: int = 3,
        time_budget: float | None = None,
    ) -> IPMResult:
        """Run a tiny number of IPM iterations from a warm start."""
        n = instance.dim
        m = instance.num_constraints
        k = int(num_iters)
        if k <= 0:
            raise ValueError("num_iters must be positive.")

        if y0 is None:
            y_init = np.zeros(m, dtype=float)
        else:
            y_init = np.asarray(y0, dtype=float).reshape(-1)
            if y_init.shape[0] != m:
                raise ValueError("y0 dimension mismatch with SDP constraints.")

        if S0 is None:
            S_init = symmetrize(instance.C - instance.apply_AT(y_init))
        else:
            S_init = symmetrize(np.asarray(S0, dtype=float))
            if S_init.shape != (n, n):
                raise ValueError("S0 must have shape (n, n).")

        if X0 is None:
            c_norm = float(np.linalg.norm(instance.C, ord="fro"))
            s_trace = float(np.trace(S_init)) if np.all(np.isfinite(S_init)) else 0.0
            scale = max(1.0, c_norm / max(n, 1), abs(s_trace) / max(n, 1))
            S_pd = symmetrize(S_init)
            vals = np.linalg.eigvalsh(S_pd)
            min_eig = float(np.min(vals))
            if min_eig <= self.settings.sym_eps:
                shift = (self.settings.sym_eps - min_eig) + self.settings.sym_eps
                S_pd = S_pd + shift * np.eye(n, dtype=float)
            try:
                S_inv = np.linalg.inv(S_pd)
                X_init = scale * symmetrize(S_inv)
            except np.linalg.LinAlgError:
                X_init = scale * np.eye(n, dtype=float)
        else:
            X_init = symmetrize(np.asarray(X0, dtype=float))
            if X_init.shape != (n, n):
                raise ValueError("X0 must have shape (n, n).")

        state = IPMState(X=X_init, y=y_init, S=S_init)
        # Keep refinement cheap: short solve horizon and shallow backtracking.
        tmp_settings = replace(
            self.settings,
            max_iters=k,
            max_backtracks=min(3, int(self.settings.max_backtracks)),
            backtrack_ratio=min(float(self.settings.backtrack_ratio), 0.7),
        )
        tmp_solver = InfeasibleIPMSolver(tmp_settings)
        return tmp_solver.solve(
            instance,
            initial_state=state,
            time_budget=time_budget,
            capture_iterates=False,
        )

    def _initialize(self, instance: SDPInstance, state: Optional[IPMState]) -> tuple[Array, Array, Array]:
        n = instance.dim
        m = instance.num_constraints
        # Heuristic scaling so that the initial point roughly matches the problem magnitude.
        c_norm = np.linalg.norm(instance.C, ord="fro")
        scale = max(1.0, c_norm / max(n, 1))
        if state is None:
            X = scale * np.eye(n, dtype=float)
            S = scale * np.eye(n, dtype=float)
            y = np.zeros(m, dtype=float)
        else:
            X, y, S = state.X.copy(), state.y.copy(), state.S.copy()

        X = self._project_spd(symmetrize(X))
        S = self._project_spd(symmetrize(S))

        if not (self._is_spd(X) and self._is_spd(S) and np.all(np.isfinite(X)) and np.all(np.isfinite(S))):
            # Safety fallback for pathological warm-starts.
            X = np.eye(n, dtype=float)
            S = np.eye(n, dtype=float)
            y = np.zeros(m, dtype=float)

        return X, y, S

    def _project_spd(self, M: Array) -> Array:
        """Project to SPD by shifting the spectrum if needed (tiny numerical safeguard)."""
        M = symmetrize(M)
        vals = np.linalg.eigvalsh(M)
        min_eig = float(vals.min())
        if min_eig > self.settings.sym_eps:
            return M
        shift = (self.settings.sym_eps - min_eig) + self.settings.sym_eps
        return M + shift * np.eye(M.shape[0])

    def _is_spd(self, M: Array) -> bool:
        vals = np.linalg.eigvalsh(symmetrize(M))
        return bool(np.all(vals > self.settings.sym_eps))

    def _line_search_step(
        self,
        instance: SDPInstance,
        X: Array,
        y: Array,
        S: Array,
        U: Array,
        w: Array,
        V: Array,
        nu: float,
        theta0: float,
    ) -> tuple[Array, Array, Array, float] | None:
        """Backtracking line search to enforce PD and neighborhood membership."""
        theta = theta0
        for _ in range(self.settings.max_backtracks):
            X_bar = symmetrize(X + theta * U)
            y_bar = y + theta * w
            S_bar = symmetrize(S + theta * V)
            if not (self._is_spd(X_bar) and self._is_spd(S_bar) and in_neighborhood(X_bar, S_bar, self.settings.gamma)):
                theta *= self.settings.backtrack_ratio
                continue

            Uc, wc, Vc = self._solve_linear_system_corrector(instance, X_bar, S_bar, nu, theta)
            X_new = symmetrize(X_bar + Uc)
            y_new = y_bar + wc
            S_new = symmetrize(S_bar + Vc)
            if self._is_spd(X_new) and self._is_spd(S_new) and in_neighborhood(X_new, S_new, self.settings.gamma):
                return X_new, y_new, S_new, theta
            theta *= self.settings.backtrack_ratio
        return None

    def _theta_bar(self, X: Array, U: Array, V: Array, nu: float) -> float:
        """Closed-form step size \\bar{\\theta} from the paper (no line search)."""
        if nu <= 0:
            return 1e-3
        X_sqrt, X_inv_sqrt = _sym_sqrt_and_invsqrt(X, eps=self.settings.sym_eps)
        delta = float(np.linalg.norm(X_inv_sqrt @ U @ V @ X_sqrt, ord="fro") / nu)
        denom = np.sqrt(1.0 + 4.0 * delta / (self.settings.beta - self.settings.gamma)) + 1.0
        theta = float(2.0 / denom)
        return min(max(theta, 1e-6), 1.0)

    def _solve_linear_system_predictor(
        self, instance: SDPInstance, X: Array, S: Array, r: Array, Rd: Array
    ) -> tuple[Array, Array, Array]:
        """Solve (10) for (U, w, V) using Schur-complement elimination."""
        if self.settings.linear_solve == "kron":
            return self._solve_linear_system_predictor_kron(instance, X, S, r, Rd)
        m = instance.num_constraints
        XS = X @ S
        SX = S @ X

        C0 = -2.0 * (X @ S @ X + X @ Rd @ X)
        U0 = symmetrize(solve_sylvester(XS, SX, C0))

        Ui = np.empty((m, X.shape[0], X.shape[0]), dtype=float)
        for i in range(m):
            Ci = 2.0 * (X @ instance.A[i] @ X)
            Ui[i] = symmetrize(solve_sylvester(XS, SX, Ci))

        schur = np.empty((m, m), dtype=float)
        for i in range(m):
            schur[:, i] = instance.apply_A(Ui[i])

        rhs = r - instance.apply_A(U0)
        schur = symmetrize(schur) + self.settings.schur_reg * np.eye(m)

        try:
            w = np.linalg.solve(schur, rhs)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(schur, rhs, rcond=None)[0]

        U = symmetrize(U0 + np.tensordot(w, Ui, axes=(0, 0)))
        V = symmetrize(Rd - instance.apply_AT(w))
        return U, w, V

    def _solve_linear_system_corrector(
        self, instance: SDPInstance, X: Array, S: Array, nu: float, theta: float
    ) -> tuple[Array, Array, Array]:
        """Solve (12) for (\\bar{U}, \\bar{w}, \\bar{V})."""
        if self.settings.linear_solve == "kron":
            return self._solve_linear_system_corrector_kron(instance, X, S, nu, theta)
        m = instance.num_constraints
        XS = X @ S
        SX = S @ X

        C0 = 2.0 * ((1.0 - theta) * nu * X - X @ S @ X)
        U0 = symmetrize(solve_sylvester(XS, SX, C0))

        Ui = np.empty((m, X.shape[0], X.shape[0]), dtype=float)
        for i in range(m):
            Ci = 2.0 * (X @ instance.A[i] @ X)
            Ui[i] = symmetrize(solve_sylvester(XS, SX, Ci))

        schur = np.empty((m, m), dtype=float)
        for i in range(m):
            schur[:, i] = instance.apply_A(Ui[i])

        rhs = -instance.apply_A(U0)
        schur = symmetrize(schur) + self.settings.schur_reg * np.eye(m)

        try:
            w = np.linalg.solve(schur, rhs)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(schur, rhs, rcond=None)[0]

        U = symmetrize(U0 + np.tensordot(w, Ui, axes=(0, 0)))
        V = symmetrize(-instance.apply_AT(w))
        return U, w, V

    def _vec(self, M: Array) -> Array:
        return np.asarray(M, dtype=float).reshape(-1, order="F")

    def _unvec(self, v: Array, n: int) -> Array:
        return np.asarray(v, dtype=float).reshape((n, n), order="F")

    def _solve_linear_system_predictor_kron(
        self, instance: SDPInstance, X: Array, S: Array, r: Array, Rd: Array
    ) -> tuple[Array, Array, Array]:
        """Solve (10) using the vectorization/Kronecker derivation in Section IV-B."""
        n = instance.dim
        m = instance.num_constraints
        XS = X @ S

        I = np.eye(n, dtype=float)
        M = np.kron(I, XS) + np.kron(XS, I)

        # A_vec stacks vec(Ai)^T as rows.
        A_vec = np.stack([self._vec(Ai) for Ai in instance.A], axis=0)  # (m, n^2)

        X_kron = np.kron(X, X)
        B = 2.0 * (X_kron @ A_vec.T)  # (n^2, m)

        rhs_base = -2.0 * self._vec(X @ S @ X + X @ Rd @ X)  # (n^2,)
        M_inv_rhs = np.linalg.solve(M, rhs_base)
        M_inv_B = np.linalg.solve(M, B)

        schur = A_vec @ M_inv_B  # (m, m)
        rhs = r - (A_vec @ M_inv_rhs)
        schur = symmetrize(schur) + self.settings.schur_reg * np.eye(m)

        try:
            w = np.linalg.solve(schur, rhs)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(schur, rhs, rcond=None)[0]

        vecU = M_inv_rhs + M_inv_B @ w
        U = symmetrize(self._unvec(vecU, n))
        V = symmetrize(Rd - instance.apply_AT(w))
        return U, w, V

    def _solve_linear_system_corrector_kron(
        self, instance: SDPInstance, X: Array, S: Array, nu: float, theta: float
    ) -> tuple[Array, Array, Array]:
        """Solve (12) using the vectorization/Kronecker derivation."""
        n = instance.dim
        m = instance.num_constraints
        XS = X @ S

        I = np.eye(n, dtype=float)
        M = np.kron(I, XS) + np.kron(XS, I)

        A_vec = np.stack([self._vec(Ai) for Ai in instance.A], axis=0)  # (m, n^2)
        X_kron = np.kron(X, X)
        B = 2.0 * (X_kron @ A_vec.T)  # (n^2, m)

        rhs_base = self._vec(2.0 * ((1.0 - theta) * nu * X - X @ S @ X))
        M_inv_rhs = np.linalg.solve(M, rhs_base)
        M_inv_B = np.linalg.solve(M, B)

        schur = A_vec @ M_inv_B
        rhs = -(A_vec @ M_inv_rhs)
        schur = symmetrize(schur) + self.settings.schur_reg * np.eye(m)

        try:
            w = np.linalg.solve(schur, rhs)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(schur, rhs, rcond=None)[0]

        vecU = M_inv_rhs + M_inv_B @ w
        U = symmetrize(self._unvec(vecU, n))
        V = symmetrize(-instance.apply_AT(w))
        return U, w, V
