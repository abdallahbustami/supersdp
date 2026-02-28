"""Training loops for L2WS (warm-start) and L2A (approximation)."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm.auto import tqdm

from .problem import SDPInstance

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Generic trainer hyperparameters."""

    epochs: int = 80
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    verbose: bool = False


def _ensure_tensor(data: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(data, dtype=torch.float32)


def _train_model(
    model: nn.Module,
    inputs: np.ndarray,
    targets: np.ndarray,
    config: TrainerConfig,
    device: torch.device | str = "cpu",
) -> List[Dict[str, float]]:
    if inputs.ndim != 2:
        raise ValueError("Inputs must be a 2D array of shape (N, D).")
    if targets.ndim == 1:
        targets = targets[:, None]

    dataset = TensorDataset(_ensure_tensor(inputs), _ensure_tensor(targets))
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    device = torch.device(device)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_fn = nn.MSELoss()
    history: List[Dict[str, float]] = []

    epoch_iter = tqdm(
        range(1, config.epochs + 1),
        disable=not config.verbose,
        desc="train",
    )
    for epoch in epoch_iter:
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = loss_fn(preds, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)

        epoch_loss = running_loss / max(len(dataset), 1)
        history.append({"epoch": epoch, "loss": epoch_loss})
        if config.verbose:
            epoch_iter.set_postfix(loss=f"{epoch_loss:.4e}")

    return history


def _stack_solution_targets(
    solutions: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not solutions:
        raise ValueError("solutions must be non-empty.")

    Xs = np.stack([np.asarray(sol[0], dtype=float) for sol in solutions], axis=0)
    ys = np.stack([np.asarray(sol[1], dtype=float).reshape(-1) for sol in solutions], axis=0)
    Ss = np.stack([np.asarray(sol[2], dtype=float) for sol in solutions], axis=0)
    return Xs, ys, Ss


def _stack_warmstart_component_targets(
    solutions: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    interior_shift: float = 1e-2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build Cholesky warm-start targets from SDP solutions.

    Returns:
        flat_targets: ``[L_flat, nu]`` vectors of shape ``(N, n_tril+1)``.
        L_targets: Lower-triangular factors of shape ``(N, n, n)``.
        nu_targets: Scalars of shape ``(N, 1)``.
    """
    if not solutions:
        raise ValueError("solutions must be non-empty.")

    flat_target_list: List[np.ndarray] = []
    L_target_list: List[np.ndarray] = []
    nu_target_list: List[float] = []
    expected_n: int | None = None

    for X_star, _, S_star in solutions:
        X_star = 0.5 * (np.asarray(X_star, dtype=float) + np.asarray(X_star, dtype=float).T)
        S_star = 0.5 * (np.asarray(S_star, dtype=float) + np.asarray(S_star, dtype=float).T)

        n = X_star.shape[0]
        if expected_n is None:
            expected_n = n
        elif n != expected_n:
            raise ValueError("All solutions must share the same matrix dimension n.")

        X_int = X_star + interior_shift * np.eye(n)
        S_int = S_star + interior_shift * np.eye(n)

        L_star = np.linalg.cholesky(X_int)
        tril = np.tril_indices(n)
        L_flat = L_star[tril]
        nu_star = max(float(np.trace(X_int @ S_int) / max(n, 1)), 1e-10)
        flat_target_list.append(np.concatenate([L_flat, np.array([nu_star], dtype=float)], axis=0))
        L_target_list.append(L_star)
        nu_target_list.append(nu_star)

    flat_targets = np.stack(flat_target_list, axis=0)
    L_targets = np.stack(L_target_list, axis=0)
    nu_targets = np.asarray(nu_target_list, dtype=float).reshape(-1, 1)
    return flat_targets, L_targets, nu_targets


def _warmstart_loss(
    pred_raw: torch.Tensor,
    target_raw: torch.Tensor,
) -> torch.Tensor:
    """Standardized loss with explicit ``nu`` upweighting."""
    pred_L = pred_raw[:, :-1]
    target_L = target_raw[:, :-1]
    pred_nu = pred_raw[:, -1]
    target_nu = target_raw[:, -1]

    loss_L = F.smooth_l1_loss(pred_L, target_L)
    loss_nu = F.smooth_l1_loss(pred_nu, target_nu)
    return loss_L + (10.0 * loss_nu)


def _resolve_target_scaler(
    model: nn.Module,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return target scaler; initialize it on the model when missing."""
    output_dim = targets.shape[1]
    model_mean = getattr(model, "_target_mean", None)
    model_std = getattr(model, "_target_std", None)
    model_has_scaler = bool(getattr(model, "_use_output_scaler", False))

    if (
        model_has_scaler
        and torch.is_tensor(model_mean)
        and torch.is_tensor(model_std)
        and model_mean.numel() == output_dim
        and model_std.numel() == output_dim
    ):
        mean = model_mean.detach().cpu().numpy().astype(float)
        std = model_std.detach().cpu().numpy().astype(float)
        std = np.where(np.abs(std) < 1e-8, 1.0, std)
        return mean, std

    mean = targets.mean(axis=0)
    std = targets.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    if hasattr(model, "set_output_scaler"):
        model.set_output_scaler(
            torch.as_tensor(mean, dtype=torch.float32),
            torch.as_tensor(std, dtype=torch.float32),
        )
    return mean, std


def _train_model_warmstart_mlp(
    model: nn.Module,
    inputs: np.ndarray,
    solutions: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    config: TrainerConfig,
    device: torch.device | str = "cpu",
) -> List[Dict[str, float]]:
    if inputs.ndim != 2:
        raise ValueError("Inputs must be a 2D array of shape (N, D).")
    if getattr(model, "warmstart_type", None) != "cholesky":
        raise ValueError("train_warmstart_model_v2 expects WarmStartNet in cholesky mode.")

    targets_flat, _, _ = _stack_warmstart_component_targets(solutions)
    if inputs.shape[0] != targets_flat.shape[0]:
        raise ValueError("inputs and solutions must have matching sample counts.")

    _resolve_target_scaler(model, targets_flat)

    dataset = TensorDataset(
        _ensure_tensor(inputs),
        _ensure_tensor(targets_flat),
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    device = torch.device(device)
    model.to(device)
    mean_t = getattr(model, "_target_mean", None)
    std_t = getattr(model, "_target_std", None)
    if not torch.is_tensor(mean_t) or not torch.is_tensor(std_t):
        raise ValueError("WarmStartNet target scaler is not set.")
    mean_t = mean_t.to(device)
    std_t = std_t.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history: List[Dict[str, float]] = []

    epoch_iter = tqdm(
        range(1, config.epochs + 1),
        disable=not config.verbose,
        desc="warmstart-mlp",
    )
    for epoch in epoch_iter:
        model.train()
        running_loss = 0.0
        seen = 0
        for batch_x, batch_targets in loader:
            batch_x = batch_x.to(device)
            batch_targets = batch_targets.to(device)
            batch_size = batch_x.size(0)
            batch_targets_scaled = (batch_targets - mean_t.unsqueeze(0)) / std_t.unsqueeze(0)

            optimizer.zero_grad()
            pred_raw = model.raw_forward(batch_x, apply_scaler=False)
            loss = _warmstart_loss(pred_raw, batch_targets_scaled)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += float(loss.item()) * batch_size
            seen += batch_size

        epoch_loss = running_loss / max(seen, 1)
        history.append({"epoch": epoch, "loss": epoch_loss})
        if config.verbose:
            epoch_iter.set_postfix(loss=f"{epoch_loss:.4e}")

    return history


def _train_model_graph(
    model: nn.Module,
    graphs: Sequence,
    solutions: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    config: TrainerConfig,
    device: torch.device | str = "cpu",
) -> List[Dict[str, float]]:
    """Train a warm-start model with graph inputs using PyG batching."""
    if getattr(model, "warmstart_type", None) != "cholesky":
        raise ValueError("train_warmstart_model_v2 expects WarmStartNet in cholesky mode.")
    try:
        from torch_geometric.data import Batch
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "torch_geometric is required for graph-based warm-start training. "
            "Install with: pip install torch-geometric"
        ) from exc

    targets_flat, _, _ = _stack_warmstart_component_targets(solutions)
    if len(graphs) != targets_flat.shape[0]:
        raise ValueError("graphs and solutions must have matching sample counts.")

    _resolve_target_scaler(model, targets_flat)

    device = torch.device(device)
    model.to(device)
    mean_t = getattr(model, "_target_mean", None)
    std_t = getattr(model, "_target_std", None)
    if not torch.is_tensor(mean_t) or not torch.is_tensor(std_t):
        raise ValueError("WarmStartNet target scaler is not set.")
    mean_t = mean_t.to(device)
    std_t = std_t.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    n_samples = len(graphs)
    history: List[Dict[str, float]] = []
    epoch_iter = tqdm(
        range(1, config.epochs + 1),
        disable=not config.verbose,
        desc="warmstart-gnn",
    )

    for epoch in epoch_iter:
        model.train()
        perm = np.random.permutation(n_samples)
        running_loss = 0.0
        seen = 0

        for start in range(0, n_samples, config.batch_size):
            idx = perm[start : start + config.batch_size]
            batch_graphs = [graphs[int(i)] for i in idx]
            batch_data = Batch.from_data_list(batch_graphs).to(device)
            batch_targets = torch.as_tensor(targets_flat[idx], dtype=torch.float32, device=device)
            batch_targets_scaled = (batch_targets - mean_t.unsqueeze(0)) / std_t.unsqueeze(0)

            optimizer.zero_grad()
            pred_raw = model.raw_forward(batch_data, apply_scaler=False)
            loss = _warmstart_loss(pred_raw, batch_targets_scaled)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = len(idx)
            running_loss += float(loss.item()) * batch_size
            seen += batch_size

        epoch_loss = running_loss / max(seen, 1)
        history.append({"epoch": epoch, "loss": epoch_loss})
        if config.verbose:
            epoch_iter.set_postfix(loss=f"{epoch_loss:.4e}")

    return history


def _train_model_graph_mse(
    model: nn.Module,
    graphs: Sequence,
    targets: np.ndarray,
    config: TrainerConfig,
    device: torch.device | str = "cpu",
) -> List[Dict[str, float]]:
    """Train a regression model with graph inputs and MSE loss."""
    try:
        from torch_geometric.data import Batch
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "torch_geometric is required for graph-based training. "
            "Install with: pip install torch-geometric"
        ) from exc

    if targets.ndim == 1:
        targets = targets[:, None]
    if len(graphs) != targets.shape[0]:
        raise ValueError("graphs and targets must have matching sample counts.")

    device = torch.device(device)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_fn = nn.MSELoss()

    n_samples = len(graphs)
    history: List[Dict[str, float]] = []
    epoch_iter = tqdm(
        range(1, config.epochs + 1),
        disable=not config.verbose,
        desc="gain-gnn",
    )

    for epoch in epoch_iter:
        model.train()
        perm = np.random.permutation(n_samples)
        running_loss = 0.0
        seen = 0

        for start in range(0, n_samples, config.batch_size):
            idx = perm[start : start + config.batch_size]
            batch_graphs = [graphs[int(i)] for i in idx]
            batch_data = Batch.from_data_list(batch_graphs).to(device)
            batch_targets = torch.as_tensor(targets[idx], dtype=torch.float32, device=device)

            optimizer.zero_grad()
            preds = model(batch_data)
            loss = loss_fn(preds, batch_targets)
            loss.backward()
            optimizer.step()

            batch_size = len(idx)
            running_loss += float(loss.item()) * batch_size
            seen += batch_size

        epoch_loss = running_loss / max(seen, 1)
        history.append({"epoch": epoch, "loss": epoch_loss})
        if config.verbose:
            epoch_iter.set_postfix(loss=f"{epoch_loss:.4e}")

    return history


def train_warmstart_model(
    model: nn.Module,
    inputs: np.ndarray,
    targets: np.ndarray,
    config: TrainerConfig,
    device: torch.device | str = "cpu",
) -> List[Dict[str, float]]:
    """Train the legacy warm-start predictor using raw MSE targets."""
    return _train_model(model, inputs, targets, config, device)


def train_warmstart_model_v2(
    model: nn.Module,
    inputs,
    solutions: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    config: TrainerConfig,
    device: torch.device | str = "cpu",
    use_graph: bool = False,
) -> List[Dict[str, float]]:
    """Train warm-start models in standardized raw-output space.

    Loss: SmoothL1 between raw model outputs and standardized flat targets
    ``[L_flat, nu]``. Standardization balances gradient magnitudes across
    all Cholesky components and ``nu``.

    Args:
        model: Warm-start network.
        inputs: Flat features (MLP path) or graph objects (GNN path).
        solutions: Ground-truth tuples ``(X*, y*, S*)``.
        config: Trainer configuration.
        device: Torch device.
        use_graph: Whether to run graph batching logic.
    """
    if use_graph:
        return _train_model_graph(model, inputs, solutions, config, device)
    return _train_model_warmstart_mlp(model, np.asarray(inputs, dtype=float), solutions, config, device)


def train_gain_model(
    model: nn.Module,
    inputs,
    targets: np.ndarray,
    config: TrainerConfig,
    device: torch.device | str = "cpu",
    use_graph: bool = False,
) -> List[Dict[str, float]]:
    """Train the L2 gain approximation network."""
    if use_graph:
        return _train_model_graph_mse(model, inputs, np.asarray(targets, dtype=float), config, device)
    return _train_model(
        model,
        np.asarray(inputs, dtype=float),
        np.asarray(targets, dtype=float),
        config,
        device,
    )


def train_unsupervised_central_path(
    model: nn.Module,
    features: np.ndarray,
    instances: Sequence[SDPInstance],
    nu: float,
    config: TrainerConfig,
    device: torch.device | str = "cpu",
) -> List[Dict[str, float]]:
    """Unsupervised L2A training using the central-path residual (Eq. (23)).

    Loss: ``||A(X)-b||^2 + ||A^*(y)+S-C||_F^2 + ||X S - nu I||_F^2``.

    The model must output ``(X, y, S)`` for each input feature vector.
    """

    if nu <= 0:
        raise ValueError("nu must be positive for the central-path residual loss.")
    if features.ndim != 2:
        raise ValueError("features must be a 2D array (N, D).")
    if not instances:
        raise ValueError("instances must be a non-empty list.")
    if len(instances) != features.shape[0]:
        raise ValueError("features and instances must have the same length.")

    n = instances[0].dim
    m = instances[0].num_constraints
    for inst in instances:
        if inst.dim != n or inst.num_constraints != m:
            raise ValueError("All instances must share the same n and m for batching.")

    class _CentralPathDataset(Dataset):
        def __init__(self, feats: np.ndarray, sdp_list: Sequence[SDPInstance]):
            self.x = torch.as_tensor(feats, dtype=torch.float32)
            self.A = torch.as_tensor(np.stack([inst.A for inst in sdp_list], axis=0), dtype=torch.float32)
            self.b = torch.as_tensor(np.stack([inst.b for inst in sdp_list], axis=0), dtype=torch.float32)
            self.C = torch.as_tensor(np.stack([inst.C for inst in sdp_list], axis=0), dtype=torch.float32)

        def __len__(self) -> int:
            return self.x.shape[0]

        def __getitem__(self, idx: int):
            return self.x[idx], self.A[idx], self.b[idx], self.C[idx]

    dataset = _CentralPathDataset(features, instances)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    device = torch.device(device)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history: List[Dict[str, float]] = []

    eye_n = torch.eye(n, device=device).unsqueeze(0)

    epoch_iter = tqdm(
        range(1, config.epochs + 1),
        disable=not config.verbose,
        desc="central-path",
    )
    for epoch in epoch_iter:
        model.train()
        running = 0.0
        seen = 0
        for batch_x, batch_A, batch_b, batch_C in loader:
            batch_x = batch_x.to(device)
            batch_A = batch_A.to(device)
            batch_b = batch_b.to(device)
            batch_C = batch_C.to(device)
            batch_size = batch_x.size(0)
            optimizer.zero_grad()

            X, y, S = model(batch_x)

            AX = torch.einsum("bmij,bij->bm", batch_A, X)
            rp = AX - batch_b
            ATy = torch.einsum("bmij,bm->bij", batch_A, y)
            rd = ATy + S - batch_C
            rc = X @ S - nu * eye_n.expand(batch_size, -1, -1)

            loss = rp.square().mean() + rd.square().mean() + rc.square().mean()

            loss.backward()
            optimizer.step()
            running += float(loss.item()) * batch_size
            seen += batch_size

        epoch_loss = running / max(seen, 1)
        history.append({"epoch": epoch, "loss": epoch_loss})
        if config.verbose:
            epoch_iter.set_postfix(loss=f"{epoch_loss:.4e}")

    return history
