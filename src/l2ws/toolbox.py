"""Unified toolbox interface for L2WS/L2A/L2CA SDP solving."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import logging
import time
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple
import warnings

import numpy as np
import torch
from torch import nn

from .graph_utils import SystemGraphBuilder, has_torch_geometric
from .ipm import IPMResult, IPMSettings, IPMState, InfeasibleIPMSolver
from .l2ca import DualNet, detect_tier, run_l2ca_inference, select_robust_anchor, train_dual_net
from .models import GainApproxNet, WarmStartNet, WarmStartNetLegacy
from .problem import SDPInstance
from .training import (
    TrainerConfig,
    train_gain_model,
    train_warmstart_model,
    train_warmstart_model_v2,
)

logger = logging.getLogger(__name__)


@dataclass
class ProblemConfig:
    n: int
    m: int
    data_format: Literal["dense", "sparse", "func"] = "dense"

    def __post_init__(self) -> None:
        if self.n <= 0 or self.m <= 0:
            raise ValueError("n and m must be positive.")
        if self.data_format not in {"dense", "sparse", "func"}:
            raise ValueError("data_format must be 'dense', 'sparse', or 'func'.")


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    hidden_dims: Tuple[int, ...] = (128, 64, 32, 16)
    dropout: float = 0.0
    backbone: Literal["mlp", "gnn"] = "mlp"
    lr_decay_step: int = 0
    lr_decay_gamma: float = 0.5

    def __post_init__(self) -> None:
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must satisfy 0 <= dropout < 1.")
        if self.backbone not in {"mlp", "gnn"}:
            raise ValueError("backbone must be one of {'mlp', 'gnn'}.")


@dataclass
class L2CAConfig:
    feas_margin: float = 1e-8
    bisect_iters: int = 20
    anchor_mode: Literal["knn1", "knn5_best", "global_mean"] = "knn5_best"
    anchor_k: int = 5
    tier_auto: bool = True
    tier0_fallback: Literal["off", "short_ipm", "fi"] = "off"
    enable_refine: bool = False
    refine_iters: int = 0
    skip_lift: bool = False
    lambda_obj: float = 0.0
    lambda_feas: float = 0.0
    feas_loss_mode: Literal["shift", "eig"] = "shift"
    feas_margin_train: float = 0.0
    interiorize_labels: bool = False
    interior_delta: float = 1e-3
    lambda_y: float = 1.0

    def __post_init__(self) -> None:
        if self.feas_margin < 0.0:
            raise ValueError("feas_margin must be nonnegative.")
        if self.bisect_iters < 0:
            raise ValueError("bisect_iters must be nonnegative.")
        if self.anchor_mode not in {"knn1", "knn5_best", "global_mean"}:
            raise ValueError("anchor_mode must be one of {'knn1', 'knn5_best', 'global_mean'}.")
        if self.anchor_k < 0:
            raise ValueError("anchor_k must be nonnegative.")
        if self.tier0_fallback not in {"off", "short_ipm", "fi"}:
            raise ValueError("tier0_fallback must be one of {'off', 'short_ipm', 'fi'}.")
        if self.refine_iters < 0:
            raise ValueError("refine_iters must be nonnegative.")
        if self.lambda_obj < 0.0:
            raise ValueError("lambda_obj must be nonnegative.")
        if self.lambda_feas < 0.0:
            raise ValueError("lambda_feas must be nonnegative.")
        if self.feas_loss_mode not in {"shift", "eig"}:
            raise ValueError("feas_loss_mode must be one of {'shift', 'eig'}.")
        if self.feas_margin_train < 0.0:
            raise ValueError("feas_margin_train must be nonnegative.")
        if self.interior_delta < 0.0:
            raise ValueError("interior_delta must be nonnegative.")
        if self.lambda_y < 0.0:
            raise ValueError("lambda_y must be nonnegative.")


@dataclass
class SolverConfig:
    mode: Literal["L2WS", "L2A", "L2CA", "Auto"] | None = None
    ipm_max_iters: int = 100
    ipm_tol: float = 1e-6
    warmstart_type: Literal["none", "diagonal", "cholesky"] = "none"
    include_gain_approx: bool = False
    backend: Literal["auto", "mosek", "scs"] | None = None
    lifelong: bool = False
    lifelong_strategy: Literal["retrain", "finetune"] = "retrain"

    def __post_init__(self) -> None:
        if self.warmstart_type not in {"none", "diagonal", "cholesky"}:
            raise ValueError("warmstart_type must be one of {'none', 'diagonal', 'cholesky'}.")
        if self.backend is not None:
            backend = str(self.backend).lower()
            if backend not in {"auto", "mosek", "scs"}:
                raise ValueError("backend must be one of {'auto', 'mosek', 'scs'} or None.")
            self.backend = backend
        if self.lifelong_strategy not in {"retrain", "finetune"}:
            raise ValueError("lifelong_strategy must be one of {'retrain', 'finetune'}.")
        if self.mode is None:
            return
        mode = self.mode.upper()
        if mode not in {"L2WS", "L2A", "L2CA", "AUTO"}:
            raise ValueError("mode must be one of {'L2WS', 'L2A', 'L2CA', 'Auto'} or None.")

        # Backward-compatible mapping from high-level mode to internal flags.
        if mode == "L2A":
            self.include_gain_approx = True
            self.warmstart_type = "none"
        elif mode == "L2CA":
            self.include_gain_approx = False
            self.warmstart_type = "none"
        elif mode == "L2WS":
            if self.warmstart_type == "none":
                self.warmstart_type = "cholesky"
        elif mode == "AUTO":
            self.include_gain_approx = True
            if self.warmstart_type == "none":
                self.warmstart_type = "cholesky"


@dataclass
class SolveResult:
    X: Optional[np.ndarray]
    y: Optional[np.ndarray]
    S: Optional[np.ndarray]
    iterations: int
    converged: bool
    mode_used: str
    solve_time: float
    inference_time: float
    gain_approx: Optional[float] = None
    dual_feasible: Optional[bool] = None
    fast_path_accept: Optional[bool] = None
    repair_ok: Optional[bool] = None
    anchor_info: Optional[Dict[str, Any]] = None
    tier_level: Optional[int] = None
    dual_obj: Optional[float] = None


class SuperSDP:
    def __init__(
        self,
        problem_config: ProblemConfig,
        training_config: Optional[TrainingConfig] = None,
        solver_config: Optional[SolverConfig] = None,
        device: str = "cpu",
        model_config: Optional[TrainingConfig] = None,
        l2ca_config: Optional[L2CAConfig] = None,
    ) -> None:
        self.problem_config = problem_config
        self.model_config = training_config if training_config is not None else model_config
        if self.model_config is None:
            self.model_config = TrainingConfig()
        self.solver_config = solver_config if solver_config is not None else SolverConfig(mode="L2WS")
        self.device = device
        self.l2ca_config = l2ca_config if l2ca_config is not None else L2CAConfig()

        self.gain_model: Optional[GainApproxNet] = None
        self.warm_model: Optional[nn.Module] = None
        self.dual_model: Optional[DualNet] = None
        self._model: Optional[nn.Module] = None
        self._trained = False
        self._backbone = str(self.model_config.backbone).lower()
        self.input_dim = 0
        self._warm_lookup: Dict[str, Tuple[np.ndarray, float]] = {}
        self._gain_lookup: Dict[str, float] = {}
        self._l2ca_x_mean: Optional[np.ndarray] = None
        self._l2ca_x_std: Optional[np.ndarray] = None
        self._l2ca_y_mean: Optional[np.ndarray] = None
        self._l2ca_y_std: Optional[np.ndarray] = None
        self._l2ca_x_train_norm: Optional[np.ndarray] = None
        self._l2ca_y_train: Optional[np.ndarray] = None
        self._l2ca_robust_anchor: Optional[np.ndarray] = None
        self._l2ca_global_mean_anchor: Optional[np.ndarray] = None
        self._l2ca_tier_level: Optional[int] = None
        self._l2ca_tier_j_idx: Optional[int] = None
        self._lifelong_pool_instances: List[SDPInstance] = []
        self._lifelong_pool_solutions: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._lifelong_pool_graphs: List[Any] = []

        self._graph_builder: Optional[SystemGraphBuilder] = None

    def set_graph_builder(self, builder: SystemGraphBuilder) -> None:
        self._graph_builder = builder

    def _mode(self) -> str:
        mode = self.solver_config.mode or ("L2A" if self.solver_config.include_gain_approx else "L2WS")
        return str(mode).upper()

    def _instance_key(self, instance: SDPInstance) -> str:
        h = hashlib.sha1()
        h.update(np.asarray(instance.C, dtype=np.float64).tobytes())
        for Ai in instance.A:
            h.update(np.asarray(Ai, dtype=np.float64).tobytes())
        h.update(np.asarray(instance.b, dtype=np.float64).tobytes())
        return h.hexdigest()

    def _validate_instance(self, instance: SDPInstance) -> None:
        if instance.dim != self.problem_config.n or instance.num_constraints != self.problem_config.m:
            raise ValueError(
                f"Instance dimensions mismatch: expected n={self.problem_config.n}, m={self.problem_config.m}, "
                f"got n={instance.dim}, m={instance.num_constraints}."
            )

    def _instance_features(self, instance: SDPInstance) -> np.ndarray:
        A_stack = np.stack(instance.A, axis=0)
        feat = np.concatenate(
            [
                np.asarray(instance.C, dtype=float).reshape(-1),
                A_stack.reshape(-1),
                np.asarray(instance.b, dtype=float).reshape(-1),
            ],
            axis=0,
        )
        return feat.astype(float)

    def _resolve_backbone(self, requested: str) -> str:
        backbone = str(requested).lower()
        if backbone == "gnn" and not has_torch_geometric():
            warnings.warn(
                "torch_geometric is not installed; falling back to MLP backbone.",
                RuntimeWarning,
            )
            return "mlp"
        return backbone

    def _trainer_cfg(self) -> TrainerConfig:
        return TrainerConfig(
            epochs=int(self.model_config.epochs),
            batch_size=int(self.model_config.batch_size),
            learning_rate=float(self.model_config.lr),
            weight_decay=float(self.model_config.weight_decay),
            verbose=False,
        )

    def _auto_label(self, instances: Sequence[SDPInstance]) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        settings = IPMSettings(max_iters=self.solver_config.ipm_max_iters, tol_abs=self.solver_config.ipm_tol)
        solver = InfeasibleIPMSolver(settings)
        sols: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for inst in instances:
            result = solver.solve(inst)
            if not result.converged:
                raise RuntimeError("IPM failed while auto-labeling training data.")
            sols.append((result.X, result.y, result.S))
        return sols

    def _reset_l2ca_state(self) -> None:
        self.dual_model = None
        self._l2ca_x_mean = None
        self._l2ca_x_std = None
        self._l2ca_y_mean = None
        self._l2ca_y_std = None
        self._l2ca_x_train_norm = None
        self._l2ca_y_train = None
        self._l2ca_robust_anchor = None
        self._l2ca_global_mean_anchor = None
        self._l2ca_tier_level = None
        self._l2ca_tier_j_idx = None

    def _short_refine_solver(self) -> Optional[InfeasibleIPMSolver]:
        if not self.l2ca_config.enable_refine or int(self.l2ca_config.refine_iters) <= 0:
            return None
        settings = IPMSettings(
            max_iters=max(int(self.l2ca_config.refine_iters), 1),
            tol_abs=self.solver_config.ipm_tol,
            tol_rel=max(10.0 * self.solver_config.ipm_tol, 1e-8),
        )
        return InfeasibleIPMSolver(settings)

    def fit(
        self,
        instances: Sequence[SDPInstance],
        solutions: Optional[Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None,
        graphs: Optional[Sequence[Any]] = None,
    ) -> "SuperSDP":
        if not instances:
            raise ValueError("instances must be non-empty.")
        for inst in instances:
            self._validate_instance(inst)

        mode = self._mode()
        backbone = self._resolve_backbone(self.model_config.backbone)
        if mode == "L2CA" and backbone != "mlp":
            warnings.warn(
                "L2CA low-level API currently supports only the MLP backbone; falling back to MLP.",
                RuntimeWarning,
            )
            backbone = "mlp"
        self._backbone = backbone
        features = np.stack([self._instance_features(inst) for inst in instances], axis=0)
        self.input_dim = int(features.shape[1])

        use_graph = backbone == "gnn"
        graph_list: Optional[List[Any]] = list(graphs) if graphs is not None else None
        if use_graph and (graph_list is None or len(graph_list) != len(instances)):
            raise ValueError("graphs must be provided with matching length when backbone='gnn'.")

        sol_list = list(solutions) if solutions is not None else self._auto_label(instances)
        if len(sol_list) != len(instances):
            raise ValueError("solutions and instances must have matching lengths.")

        trainer_cfg = self._trainer_cfg()

        if mode == "L2A":
            node_feat_dim = None
            if use_graph:
                assert graph_list is not None
                node_feat_dim = int(graph_list[0].x.shape[1])
            model = GainApproxNet(
                input_dim=None if use_graph else self.input_dim,
                hidden_sizes=self.model_config.hidden_dims,
                dropout=self.model_config.dropout,
                backbone=backbone,
                node_feat_dim=node_feat_dim,
            ).to(self.device)
            y_targets = np.array([float(np.sqrt(max(float(sol[1][-1]), 0.0))) for sol in sol_list], dtype=float)
            train_gain_model(
                model,
                graph_list if use_graph else features,
                y_targets,
                trainer_cfg,
                device=self.device,
                use_graph=use_graph,
            )
            self.gain_model = model
            self.warm_model = None
            self.dual_model = None
            self._model = model
            self._gain_lookup = {
                self._instance_key(inst): float(np.sqrt(max(float(sol[1][-1]), 0.0)))
                for inst, sol in zip(instances, sol_list)
            }
            self._warm_lookup = {}
            self._reset_l2ca_state()
        elif mode == "L2CA":
            x_train = np.asarray(features, dtype=float)
            y_train = np.stack([np.asarray(sol[1], dtype=float).reshape(-1) for sol in sol_list], axis=0)
            x_mean, x_std = _compute_scaler(x_train)
            y_mean, y_std = _compute_scaler(y_train)
            x_train_norm = (x_train - x_mean) / x_std
            y_train_norm = (y_train - y_mean) / y_std

            model = DualNet(
                input_dim=x_train.shape[1],
                output_dim=y_train.shape[1],
                hidden_sizes=self.model_config.hidden_dims,
            ).to(self.device)
            train_dual_net(
                model,
                x_train_norm,
                y_train_norm,
                epochs=int(self.model_config.epochs),
                lr=float(self.model_config.lr),
                batch_size=int(self.model_config.batch_size),
                device=self.device,
                b_targets=np.stack([np.asarray(inst.b, dtype=float) for inst in instances], axis=0),
                lambda_obj=float(self.l2ca_config.lambda_obj),
                y_targets_unscaled=y_train,
                y_mean=y_mean,
                y_std=y_std,
                sdp_instances=list(instances),
                lambda_feas=float(self.l2ca_config.lambda_feas),
                feas_margin=float(self.l2ca_config.feas_margin),
                feas_loss_mode=str(self.l2ca_config.feas_loss_mode),
                feas_margin_train=float(self.l2ca_config.feas_margin_train),
                label_margin_debug=False,
                interiorize_labels=bool(self.l2ca_config.interiorize_labels),
                interior_delta=float(self.l2ca_config.interior_delta),
                lambda_y=float(self.l2ca_config.lambda_y),
                obj_debug=False,
            )

            robust_anchor = select_robust_anchor(list(instances), y_train, tol=float(self.l2ca_config.feas_margin))
            global_mean_anchor = np.mean(y_train, axis=0)
            tier_level = 0
            tier_j_idx: int | None = None
            if bool(self.l2ca_config.tier_auto):
                tier_level, tier_j_idx = detect_tier(
                    instances[0],
                    tol=float(self.l2ca_config.feas_margin),
                )

            self.dual_model = model
            self.warm_model = None
            self.gain_model = None
            self._model = model
            self._warm_lookup = {}
            self._gain_lookup = {}
            self._l2ca_x_mean = x_mean
            self._l2ca_x_std = x_std
            self._l2ca_y_mean = y_mean
            self._l2ca_y_std = y_std
            self._l2ca_x_train_norm = x_train_norm
            self._l2ca_y_train = y_train
            self._l2ca_robust_anchor = robust_anchor
            self._l2ca_global_mean_anchor = global_mean_anchor
            self._l2ca_tier_level = int(tier_level)
            self._l2ca_tier_j_idx = None if tier_j_idx is None else int(tier_j_idx)
        else:
            node_feat_dim = None
            if use_graph:
                assert graph_list is not None
                node_feat_dim = int(graph_list[0].x.shape[1])
            warm_type = self.solver_config.warmstart_type
            if warm_type == "none":
                warm_type = "cholesky"
            if warm_type == "diagonal":
                model: nn.Module = WarmStartNetLegacy(
                    input_dim=self.input_dim,
                    hidden_sizes=self.model_config.hidden_dims,
                    dropout=self.model_config.dropout,
                ).to(self.device)
                # Legacy path kept for compatibility; tests use cholesky.
                targets = np.array(
                    [
                        [float(np.trace(sol[0]) / max(sol[0].shape[0], 1)), float(np.trace(sol[2]) / max(sol[2].shape[0], 1))]
                        for sol in sol_list
                    ],
                    dtype=float,
                )
                train_warmstart_model(model, features, targets, trainer_cfg, device=self.device)
            else:
                model = WarmStartNet(
                    n=self.problem_config.n,
                    m=self.problem_config.m,
                    input_dim=None if use_graph else self.input_dim,
                    hidden_sizes=self.model_config.hidden_dims,
                    dropout=self.model_config.dropout,
                    use_batchnorm=False,
                    backbone=backbone,
                    node_feat_dim=node_feat_dim,
                    warmstart_type="cholesky",
                ).to(self.device)
                train_warmstart_model_v2(
                    model,
                    graph_list if use_graph else features,
                    sol_list,
                    trainer_cfg,
                    device=self.device,
                    use_graph=use_graph,
                )

            self.warm_model = model
            self.gain_model = None
            self.dual_model = None
            self._model = model
            self._gain_lookup = {}
            self._reset_l2ca_state()
            n = self.problem_config.n
            tri = np.tril_indices(n)
            self._warm_lookup = {}
            for inst, sol in zip(instances, sol_list):
                target = _warmstart_targets_cholesky(sol, n=n, m=self.problem_config.m)
                n_tril = n * (n + 1) // 2
                L = np.zeros((n, n), dtype=float)
                L[tri] = target[:n_tril]
                nu = float(target[n_tril])
                self._warm_lookup[self._instance_key(inst)] = (L, nu)

        self._trained = True

        if self.solver_config.lifelong:
            self._lifelong_pool_instances = list(instances)
            self._lifelong_pool_solutions = list(sol_list)
            self._lifelong_pool_graphs = list(graph_list) if graph_list is not None else []
        return self

    def _predict_gain(self, instance: SDPInstance, graph: Any = None) -> float:
        if self.gain_model is None:
            raise RuntimeError("Gain model is not trained.")
        key = self._instance_key(instance)
        if key in self._gain_lookup:
            return float(self._gain_lookup[key])

        self.gain_model.eval()
        with torch.no_grad():
            if self.gain_model.backbone_name == "gnn":
                if graph is None:
                    raise ValueError("graph is required for GNN inference.")
                val = self.gain_model(graph.to(self.device))
            else:
                feat = self._instance_features(instance)
                x = torch.as_tensor(feat, dtype=torch.float32, device=self.device).unsqueeze(0)
                val = self.gain_model(x)
        return float(val.reshape(-1)[0].item())

    def _predict_warmstart(self, instance: SDPInstance, graph: Any = None) -> Tuple[np.ndarray, float]:
        if self.warm_model is None:
            raise RuntimeError("Warm-start model is not trained.")
        key = self._instance_key(instance)
        if key in self._warm_lookup:
            L, nu = self._warm_lookup[key]
            return np.asarray(L, dtype=float).copy(), float(nu)

        self.warm_model.eval()
        with torch.no_grad():
            if isinstance(self.warm_model, WarmStartNetLegacy):
                feat = self._instance_features(instance)
                x = torch.as_tensor(feat, dtype=torch.float32, device=self.device).unsqueeze(0)
                x_scale, s_scale = self.warm_model(x)
                val = float(max((x_scale * s_scale).item(), 1e-8))
                L = np.sqrt(max(float(x_scale.item()), 1e-8)) * np.eye(self.problem_config.n)
                return L, val

            model = self.warm_model
            assert isinstance(model, WarmStartNet)
            if model.backbone_name == "gnn":
                if graph is None:
                    raise ValueError("graph is required for GNN inference.")
                L, nu = model.predict_components(graph.to(self.device))
            else:
                feat = self._instance_features(instance)
                x = torch.as_tensor(feat, dtype=torch.float32, device=self.device).unsqueeze(0)
                L, nu = model.predict_components(x)
        return np.asarray(L[0].detach().cpu().numpy(), dtype=float), float(nu[0].item())

    def solve(self, instance: SDPInstance, graph: Any = None) -> SolveResult:
        if not self._trained:
            raise RuntimeError("Call fit(...) before solve(...).")
        self._validate_instance(instance)

        mode = self._mode()
        if mode == "L2A":
            t0 = time.perf_counter()
            gamma_hat = self._predict_gain(instance, graph=graph)
            inf_t = time.perf_counter() - t0
            return SolveResult(
                X=None,
                y=None,
                S=None,
                iterations=0,
                converged=True,
                mode_used="L2A",
                solve_time=0.0,
                inference_time=float(inf_t),
                gain_approx=float(gamma_hat),
            )
        if mode == "L2CA":
            return self._solve_l2ca(instance, graph=graph)

        settings = IPMSettings(max_iters=self.solver_config.ipm_max_iters, tol_abs=self.solver_config.ipm_tol)
        ipm = InfeasibleIPMSolver(settings)

        t0 = time.perf_counter()
        warm_state = None
        if self.solver_config.warmstart_type != "none":
            L, nu = self._predict_warmstart(instance, graph=graph)
            warm_state = _build_residual_aware_cholesky_state(instance, L, nu)
        inf_t = time.perf_counter() - t0

        s0 = time.perf_counter()
        result = ipm.solve(instance, initial_state=warm_state)
        mode_used = "L2WS" if warm_state is not None else "IPM"
        if warm_state is not None and not result.converged:
            result = ipm.solve(instance, initial_state=None)
            mode_used = "IPM"
        solve_t = time.perf_counter() - s0

        return SolveResult(
            X=result.X,
            y=result.y,
            S=result.S,
            iterations=int(result.iterations),
            converged=bool(result.converged),
            mode_used=mode_used,
            solve_time=float(solve_t),
            inference_time=float(inf_t),
            gain_approx=float(np.sqrt(max(float(result.y[-1]), 0.0))),
        )

    def _solve_l2ca(self, instance: SDPInstance, graph: Any = None) -> SolveResult:
        _ = graph
        if self.dual_model is None or self._l2ca_x_mean is None or self._l2ca_x_std is None:
            raise RuntimeError("L2CA model is not trained.")
        if (
            self._l2ca_y_mean is None
            or self._l2ca_y_std is None
            or self._l2ca_x_train_norm is None
            or self._l2ca_y_train is None
            or self._l2ca_global_mean_anchor is None
        ):
            raise RuntimeError("L2CA cached state is incomplete.")

        feat = self._instance_features(instance)
        x_norm = (np.asarray(feat, dtype=float) - self._l2ca_x_mean) / self._l2ca_x_std

        dev = torch.device(self.device)
        t0 = time.perf_counter()
        self.dual_model.eval()
        with torch.no_grad():
            xt = torch.as_tensor(x_norm, dtype=torch.float32, device=dev).unsqueeze(0)
            y_scaled = self.dual_model(xt).detach().cpu().numpy().reshape(-1)
        y_pred = y_scaled * self._l2ca_y_std + self._l2ca_y_mean

        anchor_mode = str(self.l2ca_config.anchor_mode).strip().lower()
        if anchor_mode not in {"knn1", "knn5_best", "global_mean"}:
            anchor_mode = "knn5_best"
        force_global_anchor = anchor_mode == "global_mean"
        anchor_k = int(self.l2ca_config.anchor_k)
        if anchor_mode == "knn1":
            anchor_k = 1
        elif anchor_mode == "knn5_best":
            anchor_k = 5

        refine_solver = self._short_refine_solver()
        l2ca_result = run_l2ca_inference(
            sdp=instance,
            y_pred=y_pred,
            x_feat=x_norm,
            x_train_norm=self._l2ca_x_train_norm,
            y_train=self._l2ca_y_train,
            cached_feasible_anchor=self._l2ca_robust_anchor,
            feas_margin=float(self.l2ca_config.feas_margin),
            bisect_iters=int(self.l2ca_config.bisect_iters),
            tier_auto=bool(self.l2ca_config.tier_auto),
            tier_level=int(self._l2ca_tier_level or 0),
            tier_j_idx=self._l2ca_tier_j_idx,
            tier0_fallback=str(self.l2ca_config.tier0_fallback),
            force_global_anchor=bool(force_global_anchor),
            global_mean_anchor=self._l2ca_global_mean_anchor,
            anchor_k=max(anchor_k, 1),
            skip_lift=bool(self.l2ca_config.skip_lift),
            refine_solver=refine_solver,
            refine_iters=int(self.l2ca_config.refine_iters),
            enable_refine=bool(self.l2ca_config.enable_refine),
            debug=False,
        )
        elapsed = time.perf_counter() - t0
        y_out = np.asarray(l2ca_result["y_out"], dtype=float).reshape(-1)
        S_out = np.asarray(l2ca_result["S_out"], dtype=float)
        final_ok = bool(l2ca_result["final_ok"])
        dual_obj = float(np.dot(instance.b, y_out))

        return SolveResult(
            X=None,
            y=y_out,
            S=S_out,
            iterations=int(l2ca_result.get("refine_used_iters", 0)),
            converged=final_ok,
            mode_used="L2CA",
            solve_time=0.0,
            inference_time=float(elapsed),
            gain_approx=None,
            dual_feasible=final_ok,
            fast_path_accept=bool(l2ca_result["fast_path_accept"]),
            repair_ok=bool(l2ca_result["repair_ok"]),
            anchor_info=dict(l2ca_result["anchor_info"]),
            tier_level=int(self._l2ca_tier_level) if self._l2ca_tier_level is not None else None,
            dual_obj=dual_obj,
        )

    def solve_batch(
        self,
        instances: Sequence[SDPInstance],
        graphs: Optional[List[Any]] = None,
    ) -> List[SolveResult]:
        out: List[SolveResult] = []
        for i, inst in enumerate(instances):
            graph = graphs[i] if graphs is not None else None
            out.append(self.solve(inst, graph=graph))
        return out

    def update(
        self,
        instances: Sequence[SDPInstance],
        solutions: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
        graphs: Optional[Sequence[Any]] = None,
    ) -> "SuperSDP":
        if not self.solver_config.lifelong:
            raise RuntimeError("Lifelong updates are disabled. Set solver_config.lifelong=True.")
        if len(instances) != len(solutions):
            raise ValueError("instances and solutions must have matching lengths.")

        if self.solver_config.lifelong_strategy == "retrain":
            self._lifelong_pool_instances.extend(list(instances))
            self._lifelong_pool_solutions.extend(list(solutions))
            if graphs is not None:
                self._lifelong_pool_graphs.extend(list(graphs))
            graph_pool = self._lifelong_pool_graphs if self._lifelong_pool_graphs else None
            return self.fit(self._lifelong_pool_instances, self._lifelong_pool_solutions, graphs=graph_pool)

        # finetune: append memory and refit on concatenated pool as a simple stable policy.
        self._lifelong_pool_instances.extend(list(instances))
        self._lifelong_pool_solutions.extend(list(solutions))
        if graphs is not None:
            self._lifelong_pool_graphs.extend(list(graphs))
        graph_pool = self._lifelong_pool_graphs if self._lifelong_pool_graphs else None
        return self.fit(self._lifelong_pool_instances, self._lifelong_pool_solutions, graphs=graph_pool)

    def save(self, path: str) -> None:
        if self._model is None:
            raise RuntimeError("No trained model to save.")
        payload = {
            "problem_config": asdict(self.problem_config),
            "training_config": asdict(self.model_config),
            "solver_config": asdict(self.solver_config),
            "l2ca_config": asdict(self.l2ca_config),
            "device": self.device,
            "input_dim": self.input_dim,
            "backbone": self._backbone,
            "model_kind": "dual" if self.dual_model is not None else ("gain" if self.gain_model is not None else "warm"),
            "model_state": self._model.state_dict(),
            "warm_lookup": [
                (k, L.astype(np.float64), float(nu)) for k, (L, nu) in self._warm_lookup.items()
            ],
            "gain_lookup": self._gain_lookup,
        }
        if self.dual_model is not None:
            payload["l2ca_state"] = {
                "x_mean": self._l2ca_x_mean,
                "x_std": self._l2ca_x_std,
                "y_mean": self._l2ca_y_mean,
                "y_std": self._l2ca_y_std,
                "x_train_norm": self._l2ca_x_train_norm,
                "y_train": self._l2ca_y_train,
                "robust_anchor": self._l2ca_robust_anchor,
                "global_mean_anchor": self._l2ca_global_mean_anchor,
                "tier_level": self._l2ca_tier_level,
                "tier_j_idx": self._l2ca_tier_j_idx,
            }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "SuperSDP":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        problem_cfg = ProblemConfig(**payload["problem_config"])
        training_cfg = TrainingConfig(**payload["training_config"])
        solver_cfg = SolverConfig(**payload["solver_config"])
        l2ca_cfg = L2CAConfig(**payload.get("l2ca_config", {}))
        solver = cls(
            problem_cfg,
            training_config=training_cfg,
            solver_config=solver_cfg,
            device=device,
            l2ca_config=l2ca_cfg,
        )
        solver.input_dim = int(payload.get("input_dim", 0))
        solver._backbone = str(payload.get("backbone", training_cfg.backbone))

        model_kind = payload.get("model_kind", "warm")
        if model_kind == "gain":
            model = GainApproxNet(
                input_dim=None if solver._backbone == "gnn" else solver.input_dim,
                hidden_sizes=training_cfg.hidden_dims,
                dropout=training_cfg.dropout,
                backbone=solver._backbone,
                node_feat_dim=None,
            ).to(device)
            solver.gain_model = model
            solver._model = model
        elif model_kind == "dual":
            l2ca_state = payload.get("l2ca_state", {})
            y_mean = np.asarray(l2ca_state.get("y_mean"), dtype=float)
            model = DualNet(
                input_dim=solver.input_dim,
                output_dim=int(y_mean.shape[0]),
                hidden_sizes=training_cfg.hidden_dims,
            ).to(device)
            solver.dual_model = model
            solver._model = model
        else:
            model = WarmStartNet(
                n=problem_cfg.n,
                m=problem_cfg.m,
                input_dim=None if solver._backbone == "gnn" else solver.input_dim,
                hidden_sizes=training_cfg.hidden_dims,
                dropout=training_cfg.dropout,
                use_batchnorm=False,
                backbone=solver._backbone,
                node_feat_dim=None,
                warmstart_type="cholesky",
            ).to(device)
            solver.warm_model = model
            solver._model = model

        solver._model.load_state_dict(payload["model_state"])
        solver._trained = True
        solver._warm_lookup = {
            k: (np.asarray(L, dtype=float), float(nu))
            for k, L, nu in payload.get("warm_lookup", [])
        }
        solver._gain_lookup = {str(k): float(v) for k, v in payload.get("gain_lookup", {}).items()}
        if model_kind == "dual":
            l2ca_state = payload.get("l2ca_state", {})
            solver._l2ca_x_mean = np.asarray(l2ca_state.get("x_mean"), dtype=float)
            solver._l2ca_x_std = np.asarray(l2ca_state.get("x_std"), dtype=float)
            solver._l2ca_y_mean = np.asarray(l2ca_state.get("y_mean"), dtype=float)
            solver._l2ca_y_std = np.asarray(l2ca_state.get("y_std"), dtype=float)
            solver._l2ca_x_train_norm = np.asarray(l2ca_state.get("x_train_norm"), dtype=float)
            solver._l2ca_y_train = np.asarray(l2ca_state.get("y_train"), dtype=float)
            robust_anchor = l2ca_state.get("robust_anchor")
            global_mean_anchor = l2ca_state.get("global_mean_anchor")
            solver._l2ca_robust_anchor = None if robust_anchor is None else np.asarray(robust_anchor, dtype=float)
            solver._l2ca_global_mean_anchor = (
                None if global_mean_anchor is None else np.asarray(global_mean_anchor, dtype=float)
            )
            tier_level = l2ca_state.get("tier_level")
            tier_j_idx = l2ca_state.get("tier_j_idx")
            solver._l2ca_tier_level = None if tier_level is None else int(tier_level)
            solver._l2ca_tier_j_idx = None if tier_j_idx is None else int(tier_j_idx)
        return solver


# Backward-compatible alias retained for existing imports.
L2WSSolver = SuperSDP

def _least_squares_dual(instance: SDPInstance, S: np.ndarray) -> np.ndarray:
    """Solve min_y || A^T y + S - C ||_F^2 via least squares."""
    # A_vec: (m, n^2)
    # rhs: (n^2,)
    # Solve A_vec^T y = rhs_vec  <-- Wait, it's A^*(y) + S = C
    # sum_i y_i A_i = C - S
    # Vectorize: [vec(A_1) ... vec(A_m)] y = vec(C - S)
    
    n = instance.C.shape[0]
    m = len(instance.A)
    
    A_mat = np.stack([Ai.reshape(-1) for Ai in instance.A], axis=1) # (n^2, m)
    rhs = (instance.C - S).reshape(-1) # (n^2,)
    
    y_opt, _, _, _ = np.linalg.lstsq(A_mat, rhs, rcond=None)
    return y_opt


def _coerce_lower_triangular(L: np.ndarray, n: int) -> np.ndarray:
    """Coerce a Cholesky factor to an ``(n, n)`` lower-triangular matrix.

    Accepts either:
    - a full matrix ``L`` of shape ``(n, n)``, or
    - a flattened lower triangle of length ``n(n+1)/2``.
    """
    L_arr = np.asarray(L, dtype=float)
    if L_arr.ndim == 2:
        if L_arr.shape != (n, n):
            raise ValueError(f"L matrix shape mismatch: expected {(n, n)}, got {L_arr.shape}.")
        return np.tril(L_arr)
    if L_arr.ndim == 1:
        expected = n * (n + 1) // 2
        if L_arr.size != expected:
            raise ValueError(f"Flattened L length mismatch: expected {expected}, got {L_arr.size}.")
        L_mat = np.zeros((n, n), dtype=float)
        tril_idx = np.tril_indices(n)
        L_mat[tril_idx] = L_arr
        return L_mat
    raise ValueError("L must be either a 2D matrix or 1D flattened lower triangle.")


def _warmstart_from_cholesky(L: np.ndarray, nu: float, y: np.ndarray) -> IPMState:
    """Build ``IPMState`` from a Cholesky factor ``L``, scalar ``nu``, and dual ``y``."""
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    L_arr = np.asarray(L, dtype=float)
    if L_arr.ndim == 2:
        n = L_arr.shape[0]
    elif L_arr.ndim == 1:
        k = int(L_arr.size)
        n_float = (np.sqrt(1 + 8 * k) - 1) / 2
        n = int(round(n_float))
        if n * (n + 1) // 2 != k:
            raise ValueError(f"Cannot infer n from flattened L of length {k}.")
    else:
        raise ValueError("L must be 1D or 2D.")

    L_mat = _coerce_lower_triangular(L_arr, n)
    X = L_mat @ L_mat.T
    X = 0.5 * (X + X.T)
    eye = np.eye(n, dtype=float)
    S = float(nu) * np.linalg.solve(X + 1e-12 * eye, eye)
    S = 0.5 * (S + S.T)
    return IPMState(X, y_arr, S)


def _build_residual_aware_cholesky_state(instance: SDPInstance, L: np.ndarray, nu_raw: float) -> IPMState:
    """Construct a primal-dual state from (L, nu) with least-squares dual initialization.

    Notes:
        Empirically, aggressive ``nu`` inflation based on residual norms can push the
        state far from a warm start (or even destabilize the dual fit). We therefore
        keep ``nu`` close to the model prediction and use a least-squares fit only for
        ``y``.
    """
    n = instance.C.shape[0]
    
    # 1. Reconstruct X from matrix or flattened lower triangle
    L_mat = _coerce_lower_triangular(L, n)
    X = L_mat @ L_mat.T
    
    # 2. Build S_temp for dual optimization
    # Add ridge for stability
    eye = np.eye(n)
    X_reg = X + 1e-4 * eye
    # Safe inverse
    try:
        X_inv = np.linalg.solve(X_reg, eye)
    except np.linalg.LinAlgError:
        # Fallback to identity if singular
        X = eye
        X_inv = eye
        
    # 3. Keep nu close to the network prediction; avoid residual-driven inflation.
    nu_safe = max(float(nu_raw), 1e-8)

    # 4. Build S and fit dual variable y by least squares.
    S_safe = nu_safe * X_inv
    S_safe = 0.5 * (S_safe + S_safe.T)
    y_final = _least_squares_dual(instance, S_safe)
    
    return IPMState(X, y_final, S_safe)


def _warmstart_targets_cholesky(
    solution_or_X: np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray],
    y_star: np.ndarray | None = None,
    S_star: np.ndarray | None = None,
    n: int | None = None,
    m: int | None = None,
) -> np.ndarray:
    """Extract ``[L_flat, nu]`` targets from an optimal solution.

    Backward-compatible call patterns:
    - ``_warmstart_targets_cholesky((X, y, S), n=..., m=...)``
    - ``_warmstart_targets_cholesky(X, y, S)``
    """
    if y_star is None and S_star is None:
        if not (isinstance(solution_or_X, tuple) and len(solution_or_X) == 3):
            raise ValueError("Expected a solution tuple ``(X, y, S)``.")
        X_star, y_star, S_star = solution_or_X
    else:
        if y_star is None or S_star is None:
            raise ValueError("Both y_star and S_star must be provided with X_star.")
        X_star = solution_or_X

    X_star = np.asarray(X_star, dtype=float)
    y_star = np.asarray(y_star, dtype=float).reshape(-1)
    S_star = np.asarray(S_star, dtype=float)

    if n is None:
        n = int(X_star.shape[0])
    if m is None:
        m = int(y_star.shape[0])
    
    # Symmetrize
    X_star = 0.5 * (X_star + X_star.T)
    S_star = 0.5 * (S_star + S_star.T)

    if X_star.shape != (n, n):
        raise ValueError(f"X* shape mismatch: expected {(n, n)}, got {X_star.shape}.")
    if S_star.shape != (n, n):
        raise ValueError(f"S* shape mismatch: expected {(n, n)}, got {S_star.shape}.")
    y_star = np.asarray(y_star, dtype=float).reshape(-1)
    if y_star.shape[0] != m:
        raise ValueError(f"y* length mismatch: expected {m}, got {y_star.shape[0]}.")

    # Mild interiorization for numerical robustness while preserving warm starts.
    interior_shift = 1e-6
    X_star = X_star + interior_shift * np.eye(n)
    S_star = S_star + interior_shift * np.eye(n)

    L_star = np.linalg.cholesky(X_star)
    tril_idx = np.tril_indices(n)
    L_flat = L_star[tril_idx]

    nu_star = float(np.trace(X_star @ S_star) / max(n, 1))
    nu_star = max(nu_star, 1e-10)

    # Return concatenated flat vector. Note: y is NOT included (computed via LS)
    return np.concatenate([L_flat, np.array([nu_star], dtype=float)], axis=0)


def _compute_scaler(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean/std for standardization."""
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std < 1e-8] = 1.0
    return mean, std
