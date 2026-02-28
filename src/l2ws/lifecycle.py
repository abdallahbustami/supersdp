"""L2WS lifelong-learning lifecycle utilities."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from .data import L2GainInstance, LTISystem, build_system_graph, generate_l2_gain_instances
from .graph_utils import has_torch_geometric
from .ipm import IPMResult, InfeasibleIPMSolver
from .models import GainApproxNet, WarmStartNet
from .training import TrainerConfig, train_gain_model, train_warmstart_model_v2
from .toolbox import _build_residual_aware_cholesky_state, _warmstart_targets_cholesky

logger = logging.getLogger(__name__)


def default_system(cz_scale: float = 1.0) -> LTISystem:
    """Return the default 2-state LTI system used in the case study."""
    A = np.array([[-3.0, -2.0], [1.0, 0.0]])
    Bw = np.array([[1.0], [0.0]])
    Cz = cz_scale * np.array([[2.0, 1.0]])
    return LTISystem(A=A, Bw=Bw, Cz=Cz)


def gamma_from_result(result: IPMResult) -> float:
    """Extract L2 gain from an IPM result (using ``y[-1] = gamma^2``)."""
    gamma_squared = float(result.y[-1])
    return float(np.sqrt(max(gamma_squared, 0.0)))


def warmstart_targets(result: IPMResult) -> np.ndarray:
    """Return cholesky warm-start target vector ``[L_flat, nu]``."""
    n = result.X.shape[0]
    m = result.y.shape[0]
    return _warmstart_targets_cholesky((result.X, result.y, result.S), n=n, m=m)


def warmstart_targets_from_solution(
    solution: tuple[np.ndarray, np.ndarray, np.ndarray],
    sdp_n: int,
    sdp_m: int,
) -> np.ndarray:
    """Compute ``[L_flat, nu]`` from a solution tuple."""
    return _warmstart_targets_cholesky(solution, n=sdp_n, m=sdp_m)


def predict_warmstart(
    model: WarmStartNet,
    instance: L2GainInstance,
    device: torch.device,
    backbone: str,
    feature_mean: np.ndarray | None = None,
    feature_std: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Predict ``(L, nu)`` for warm-starting the IPM."""
    model.eval()
    with torch.no_grad():
        if backbone == "gnn":
            graph = instance.graph if instance.graph is not None else build_system_graph(instance.system)
            graph = graph.to(device) if hasattr(graph, "to") else graph
            L, nu = model.predict_components(graph)
        else:
            feature = np.asarray(instance.features, dtype=np.float32).reshape(-1)
            if feature_mean is not None and feature_std is not None:
                feature = (feature - feature_mean) / feature_std
            tensor = torch.as_tensor(feature, dtype=torch.float32, device=device).unsqueeze(0)
            L, nu = model.predict_components(tensor)

    return (
        L.squeeze(0).cpu().numpy(),
        float(nu.squeeze(0).item()),
    )


def build_warm_state(instance: L2GainInstance, prediction: tuple[np.ndarray, float]):
    """Build a Cholesky-based warm-start state for the IPM."""
    L, nu = prediction
    return _build_residual_aware_cholesky_state(instance.sdp, L, nu)


def solve_baseline(instances: Sequence[L2GainInstance], solver: InfeasibleIPMSolver) -> None:
    """Solve each instance without learning to obtain reference stats."""
    for inst in instances:
        result = solver.solve(inst.sdp)
        if not result.converged:
            raise RuntimeError(f"Baseline IPM failed to converge for instance {inst.index}")
        inst.true_gamma = gamma_from_result(result)
        inst.baseline_iterations = result.iterations


def run_l2ws_lifecycle(
    instances: Sequence[L2GainInstance],
    solver: InfeasibleIPMSolver,
    model: WarmStartNet,
    trainer_cfg: TrainerConfig,
    episode_size: int,
    device: torch.device,
    backbone: str = "mlp",
    retrain_strategy: str = "full_retrain",
    replay_capacity: int | None = None,
    model_builder=None,
    standardize_inputs: bool = False,
) -> List[Dict[str, float]]:
    """Simulate the lifelong-learning process for L2WS warm starts."""
    if retrain_strategy not in {"full_retrain", "finetune"}:
        raise ValueError("retrain_strategy must be 'full_retrain' or 'finetune'.")
    if retrain_strategy == "full_retrain" and model_builder is None:
        raise ValueError("model_builder is required when retrain_strategy='full_retrain'.")

    feature_memory: List[np.ndarray] = []
    graph_memory: List[Any] = []
    solution_memory: List[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    rng = np.random.default_rng(0)
    metrics: List[Dict[str, float]] = []
    model_ready = False

    feature_mean = None
    feature_std = None

    for episode_idx, start in enumerate(range(0, len(instances), episode_size), start=1):
        batch = instances[start : start + episode_size]
        iter_counts: List[int] = []
        baseline_counts: List[int] = []
        raw_warm_iters: List[int] = []
        warm_converged = 0
        fallback_count = 0

        for inst in batch:
            if model_ready:
                model.eval()
                pred = predict_warmstart(
                    model,
                    inst,
                    device,
                    backbone=backbone,
                    feature_mean=feature_mean,
                    feature_std=feature_std,
                )
                warm_state = build_warm_state(inst, pred)
                warm_result = solver.solve(inst.sdp, warm_state)
                raw_warm_iters.append(warm_result.iterations)
                if warm_result.converged:
                    warm_converged += 1
                    result = warm_result
                else:
                    fallback_count += 1
                    result = solver.solve(inst.sdp, None)
            else:
                result = solver.solve(inst.sdp, None)

            if not result.converged:
                raise RuntimeError(f"IPM failed to converge for instance {inst.index}")

            inst.warmstart_iterations = result.iterations
            iter_counts.append(result.iterations)
            baseline_counts.append(inst.baseline_iterations or 0)

            feature_memory.append(np.asarray(inst.features, dtype=float))
            if backbone == "gnn":
                graph = inst.graph if inst.graph is not None else build_system_graph(inst.system)
                graph_memory.append(graph)
            solution_memory.append((result.X, result.y, result.S))

            if replay_capacity is not None and len(solution_memory) > replay_capacity:
                j = int(rng.integers(0, len(solution_memory)))
                if j < replay_capacity:
                    feature_memory[j] = feature_memory[-1]
                    solution_memory[j] = solution_memory[-1]
                    if backbone == "gnn":
                        graph_memory[j] = graph_memory[-1]
                feature_memory.pop()
                solution_memory.pop()
                if backbone == "gnn":
                    graph_memory.pop()

        all_targets = np.stack(
            [warmstart_targets_from_solution(sol, sdp_n=model.n, sdp_m=model.m) for sol in solution_memory],
            axis=0,
        )
        target_mean = all_targets.mean(axis=0)
        target_std = all_targets.std(axis=0)
        target_std = np.where(target_std < 1e-8, 1.0, target_std)

        if retrain_strategy == "full_retrain":
            model = model_builder()

        model.set_output_scaler(
            torch.as_tensor(target_mean, dtype=torch.float32),
            torch.as_tensor(target_std, dtype=torch.float32),
        )

        if backbone == "gnn":
            train_warmstart_model_v2(
                model,
                graph_memory,
                solution_memory,
                trainer_cfg,
                device=device,
                use_graph=True,
            )
        else:
            train_inputs = np.stack(feature_memory)
            if standardize_inputs:
                feature_mean = train_inputs.mean(axis=0)
                feature_std = train_inputs.std(axis=0)
                feature_std = np.where(feature_std < 1e-8, 1.0, feature_std)
                train_inputs = (train_inputs - feature_mean) / feature_std

            train_warmstart_model_v2(
                model,
                train_inputs,
                solution_memory,
                trainer_cfg,
                device=device,
                use_graph=False,
            )
            if standardize_inputs and feature_mean is not None and feature_std is not None:
                model._feature_mean = np.asarray(feature_mean, dtype=np.float32).copy()
                model._feature_std = np.asarray(feature_std, dtype=np.float32).copy()

        model_ready = True

        metrics.append(
            {
                "episode": episode_idx,
                "avg_iterations": float(np.mean(iter_counts)),
                "avg_baseline_iterations": float(np.mean(baseline_counts)),
                "raw_warm_avg_iterations": float(np.mean(raw_warm_iters)) if raw_warm_iters else float("nan"),
                "warm_converged": int(warm_converged),
                "warm_attempts": int(len(raw_warm_iters)),
                "fallback_count": int(fallback_count),
            }
        )

    return metrics


def run_l2a_training(
    instances: Sequence[L2GainInstance],
    model: GainApproxNet,
    trainer_cfg: TrainerConfig,
    splits: Sequence[float] = (0.8, 0.1, 0.1),
    device: torch.device | str = "cpu",
) -> Dict[str, float]:
    """Train/evaluate the L2A approximation model."""
    features = np.stack([inst.features for inst in instances])
    labels = np.array([inst.true_gamma for inst in instances], dtype=float)
    use_graph = getattr(model, "backbone_name", "mlp") == "gnn"
    graph_data = None
    if use_graph:
        graph_data = [inst.graph if inst.graph is not None else build_system_graph(inst.system) for inst in instances]

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(instances))
    features = features[perm]
    labels = labels[perm]
    if use_graph:
        assert graph_data is not None
        graph_data = [graph_data[int(i)] for i in perm]

    n = len(instances)
    n_train = int(splits[0] * n)
    n_val = int(splits[1] * n)
    n_test = n - n_train - n_val

    train_inputs = features[:n_train]
    train_targets = labels[:n_train]
    val_inputs = features[n_train : n_train + n_val]
    val_targets = labels[n_train : n_train + n_val]
    test_inputs = features[-n_test:]
    test_targets = labels[-n_test:]
    if use_graph:
        assert graph_data is not None
        train_graphs = graph_data[:n_train]
        val_graphs = graph_data[n_train : n_train + n_val]
        test_graphs = graph_data[-n_test:]
    else:
        train_graphs = None
        val_graphs = None
        test_graphs = None

    train_gain_model(
        model,
        train_graphs if use_graph else train_inputs,
        train_targets,
        trainer_cfg,
        device=device,
        use_graph=use_graph,
    )

    device = torch.device(device)
    model.eval()

    def predict(batch_inputs: np.ndarray, batch_graphs=None) -> np.ndarray:
        if not use_graph and batch_inputs.size == 0:
            return np.array([], dtype=float)
        if use_graph and (batch_graphs is None or len(batch_graphs) == 0):
            return np.array([], dtype=float)

        if use_graph:
            from torch_geometric.data import Batch

            preds: List[np.ndarray] = []
            step = max(int(trainer_cfg.batch_size), 1)
            with torch.no_grad():
                for start in range(0, len(batch_graphs), step):
                    batch = Batch.from_data_list(batch_graphs[start : start + step]).to(device)
                    preds.append(model(batch).cpu().numpy().reshape(-1))
            return np.concatenate(preds) if preds else np.array([], dtype=float)

        with torch.no_grad():
            return model(torch.as_tensor(batch_inputs, dtype=torch.float32, device=device)).cpu().numpy().reshape(-1)

    train_pred = predict(train_inputs, train_graphs)
    val_pred = predict(val_inputs, val_graphs)

    def evaluate(inputs: np.ndarray, targets: np.ndarray, eval_graphs=None) -> Dict[str, float]:
        if targets.size == 0:
            return {"rmse": float("nan"), "mae": float("nan"), "avg_time_per_batch": 0.0}
        start = time.perf_counter()
        preds = predict(inputs, eval_graphs)
        elapsed = time.perf_counter() - start
        errors = preds - targets
        rmse = float(np.sqrt(np.mean(errors**2)))
        mae = float(np.mean(np.abs(errors)))
        return {"rmse": rmse, "mae": mae, "avg_time_per_batch": elapsed}

    test_metrics = evaluate(test_inputs, test_targets, test_graphs)
    return {
        "train_mae": float(np.mean(np.abs(train_pred - train_targets))) if train_targets.size else float("nan"),
        "val_mae": float(np.mean(np.abs(val_pred - val_targets))) if val_targets.size else float("nan"),
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
    }


def run_case_study(
    num_instances: int = 1000,
    episode_size: int = 50,
    perturb_range: tuple[float, float] = (-0.2, 0.2),
    seed: int = 1,
    device: str = "cpu",
    cz_scale: float = 1.0,
    feature_mode: str = "A",
    base_system: LTISystem | None = None,
    hidden_dims: tuple[int, ...] = (128, 64, 32, 16),
    dropout: float = 0.2,
    epochs_per_episode: int = 200,
    batch_size: int = 64,
    learning_rate: float = 5e-4,
    standardize_inputs: bool = True,
    use_batchnorm: bool = False,
    backbone: str = "mlp",
    retrain_strategy: str = "finetune",
    replay_capacity: int | None = None,
) -> Dict[str, Any]:
    """Run full L2WS + L2A lifecycle for the L2 gain task."""
    if base_system is None:
        base_system = default_system(cz_scale=cz_scale)

    effective_backbone = backbone
    if backbone == "gnn" and not has_torch_geometric():
        logger.warning("torch_geometric not available; falling back to MLP backbone for L2WS and L2A.")
        effective_backbone = "mlp"

    instances = generate_l2_gain_instances(
        base_system,
        num_instances,
        perturb_range=perturb_range,
        seed=seed,
        feature_mode=feature_mode,
        cache_graph=(effective_backbone == "gnn"),
    )

    graph_mean_np = None
    graph_std_np = None
    if effective_backbone == "gnn":
        n_train_graph = max(int(0.8 * len(instances)), 1)
        train_graphs = []
        for inst in instances[:n_train_graph]:
            graph = inst.graph if inst.graph is not None else build_system_graph(inst.system)
            inst.graph = graph
            train_graphs.append(graph)

        train_x_all = torch.cat([g.x for g in train_graphs], dim=0)
        graph_mean = train_x_all.mean(dim=0, keepdim=True)
        graph_std = train_x_all.std(dim=0, keepdim=True)
        graph_std = torch.where(graph_std < 1e-8, torch.ones_like(graph_std), graph_std)
        graph_mean_np = graph_mean.cpu().numpy()
        graph_std_np = graph_std.cpu().numpy()

        for inst in instances:
            graph = inst.graph if inst.graph is not None else build_system_graph(inst.system)
            graph.x = (graph.x - graph_mean) / graph_std
            inst.graph = graph

    from .ipm import IPMSettings

    solver = InfeasibleIPMSolver(IPMSettings(max_iters=120))
    solve_baseline(instances, solver)

    sdp_n = instances[0].sdp.dim
    sdp_m = instances[0].sdp.num_constraints
    feature_dim = instances[0].features.shape[0]
    node_feat_dim = None
    if effective_backbone == "gnn":
        first_graph = instances[0].graph if instances[0].graph is not None else build_system_graph(instances[0].system)
        node_feat_dim = int(first_graph.x.shape[1])

    warm_model = WarmStartNet(
        n=sdp_n,
        m=sdp_m,
        input_dim=feature_dim,
        hidden_sizes=hidden_dims,
        dropout=dropout,
        activation="relu",
        use_batchnorm=use_batchnorm,
        backbone=effective_backbone,
        node_feat_dim=node_feat_dim,
        warmstart_type="cholesky",
    )

    warm_cfg = TrainerConfig(
        epochs=epochs_per_episode,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=1e-5,
    )

    device_obj = torch.device(device)
    l2ws_metrics = run_l2ws_lifecycle(
        instances,
        solver,
        warm_model,
        warm_cfg,
        episode_size,
        device_obj,
        backbone=effective_backbone,
        retrain_strategy=retrain_strategy,
        replay_capacity=replay_capacity,
        model_builder=lambda: WarmStartNet(
            n=sdp_n,
            m=sdp_m,
            input_dim=feature_dim,
            hidden_sizes=hidden_dims,
            dropout=dropout,
            activation="relu",
            use_batchnorm=use_batchnorm,
            backbone=effective_backbone,
            node_feat_dim=node_feat_dim,
            warmstart_type="cholesky",
        ) if retrain_strategy == "full_retrain" else None,
        standardize_inputs=standardize_inputs,
    )

    approx_model = GainApproxNet(
        input_dim=feature_dim,
        backbone=effective_backbone,
        node_feat_dim=node_feat_dim,
    )
    approx_cfg = TrainerConfig(epochs=80, batch_size=32, learning_rate=1e-3, weight_decay=1e-5)
    l2a_metrics = run_l2a_training(instances, approx_model, approx_cfg, device=device_obj)

    return {
        "l2ws": l2ws_metrics,
        "l2a": l2a_metrics,
        "graph_feature_mean": graph_mean_np,
        "graph_feature_std": graph_std_np,
    }


def format_metrics(metrics: Dict[str, Any]) -> str:
    """Pretty-print workflow metrics."""
    lines = ["L2WS episodic iteration counts:"]
    for item in metrics["l2ws"]:
        lines.append(
            f"  Episode {item['episode']:02d}: "
            f"warm-start avg iter={item['avg_iterations']:.2f}, "
            f"baseline avg iter={item['avg_baseline_iterations']:.2f}"
        )
        if item.get("warm_attempts", 0) > 0:
            lines.append(
                f"    raw warm avg iter={item['raw_warm_avg_iterations']:.2f}, "
                f"warm converged={item['warm_converged']}/{item['warm_attempts']}, "
                f"fallback={item['fallback_count']}"
            )
    l2a = metrics["l2a"]
    lines.append(
        "L2A approximation metrics: "
        f"train MAE={l2a['train_mae']:.4f}, "
        f"val MAE={l2a['val_mae']:.4f}, "
        f"test RMSE={l2a['test_rmse']:.4f}, "
        f"test MAE={l2a['test_mae']:.4f}"
    )
    return "\n".join(lines)


__all__ = [
    "default_system",
    "gamma_from_result",
    "warmstart_targets",
    "warmstart_targets_from_solution",
    "predict_warmstart",
    "build_warm_state",
    "solve_baseline",
    "run_l2ws_lifecycle",
    "run_l2a_training",
    "run_case_study",
    "format_metrics",
]
