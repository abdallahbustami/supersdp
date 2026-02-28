"""Shared learning primitives used by L2A/L2CA pipelines.

This module centralizes common MLP heads and MSE training loops so algorithm
modules stay focused on optimization logic.
"""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class DualNet(nn.Module):
    """MLP that predicts SDP dual variable ``y`` from instance features."""

    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: Tuple[int, ...] = (256, 128, 64)) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ScalarL2ANet(nn.Module):
    """Simple scalar MLP baseline for objective approximation."""

    def __init__(self, input_dim: int, hidden_sizes: Tuple[int, ...] = (128, 64, 32)) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _train_mse_model(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
) -> nn.Module:
    x_t = torch.as_tensor(x, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.float32)
    ds = TensorDataset(x_t, y_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    dev = torch.device(device)
    model = model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(dev)
            yb = yb.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()
    return model


def train_dual_net(
    model: DualNet,
    features: np.ndarray,
    y_targets: np.ndarray,
    epochs: int = 200,
    lr: float = 3e-4,
    batch_size: int = 64,
    device: str = "cpu",
    b_targets: np.ndarray | None = None,
    lambda_obj: float = 0.0,
    y_targets_unscaled: np.ndarray | None = None,
    y_mean: np.ndarray | None = None,
    y_std: np.ndarray | None = None,
    sdp_instances: Sequence[Any] | None = None,
    lambda_feas: float = 0.0,
    feas_margin: float = 0.0,
    feas_loss_mode: str = "shift",
    feas_margin_train: float = 0.0,
    label_margin_debug: bool = False,
    interiorize_labels: bool = False,
    interior_delta: float = 1e-3,
    lambda_y: float = 1.0,
    obj_debug: bool = False,
) -> DualNet:
    """Train ``DualNet`` with MSE and optional objective-aware dual loss.

    Objective-aware dual loss is used only for L2CA/L2CA-FI when enabled.
    """
    x_np = np.asarray(features, dtype=float)
    y_np = np.asarray(y_targets, dtype=float)

    x_t = torch.as_tensor(x_np, dtype=torch.float32)
    y_t = torch.as_tensor(y_np, dtype=torch.float32)
    idx_t = torch.arange(x_t.shape[0], dtype=torch.long)

    dev = torch.device(device)
    model = model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    lambda_obj_val = float(lambda_obj)
    obj_enabled = lambda_obj_val > 0.0
    lambda_feas_val = float(lambda_feas)
    feas_enabled = lambda_feas_val > 0.0
    lambda_y_val = float(lambda_y)
    interiorize_labels_enabled = bool(interiorize_labels)
    feas_mode = str(feas_loss_mode).strip().lower()
    if feas_mode not in {"shift", "eig"}:
        if obj_debug:
            print(f"[l2ca-debug] unknown feas_loss_mode='{feas_loss_mode}', falling back to 'shift'.")
        feas_mode = "shift"
    # Backward compatibility: if explicit training margin is not provided,
    # reuse legacy feas_margin value.
    feas_margin_train_val = float(feas_margin_train)
    if float(feas_margin_train_val) == 0.0 and float(feas_margin) != 0.0:
        feas_margin_train_val = float(feas_margin)

    b_t: torch.Tensor | None = None
    y_unscaled_t: torch.Tensor | None = None
    y_mean_t: torch.Tensor | None = None
    y_std_t: torch.Tensor | None = None
    warned_missing_obj = False
    warned_missing_feas = False
    warned_eig_fallback = False

    def _label_slack_margin_stats(y_labels_unscaled_np: np.ndarray) -> tuple[np.ndarray | None, bool]:
        if sdp_instances is None or len(sdp_instances) != y_labels_unscaled_np.shape[0]:
            return None, False
        lam_list: list[float] = []
        for i in range(y_labels_unscaled_np.shape[0]):
            sdp = sdp_instances[i]
            yi = np.asarray(y_labels_unscaled_np[i], dtype=float).reshape(-1)
            S = np.asarray(sdp.C, dtype=float) - np.asarray(sdp.apply_AT(yi), dtype=float)
            S = 0.5 * (S + S.T)
            lam_list.append(float(np.linalg.eigvalsh(S)[0]))
        lam = np.asarray(lam_list, dtype=float)
        frac_1e8 = float(np.mean(lam <= 1e-8))
        frac_1e6 = float(np.mean(lam <= 1e-6))
        boundary = bool(frac_1e8 >= 0.5 or frac_1e6 >= 0.5)
        if bool(label_margin_debug):
            print(
                "[l2ca-debug] label slack lam_min stats: "
                f"mean={float(np.mean(lam)):.3e}, "
                f"median={float(np.median(lam)):.3e}, "
                f"min={float(np.min(lam)):.3e}, "
                f"pct<=1e-8={100.0 * frac_1e8:.1f}%, "
                f"pct<=1e-6={100.0 * frac_1e6:.1f}%"
            )
        return lam, boundary

    def _interiorize_label_via_b(sdp: Any, y_star: np.ndarray, delta: float, t_max: float = 1e3) -> np.ndarray:
        y0 = np.asarray(y_star, dtype=float).reshape(-1).copy()
        S0 = np.asarray(sdp.C, dtype=float) - np.asarray(sdp.apply_AT(y0), dtype=float)
        S0 = 0.5 * (S0 + S0.T)
        lam0 = float(np.linalg.eigvalsh(S0)[0])
        if lam0 >= float(delta):
            return y0

        b = np.asarray(sdp.b, dtype=float).reshape(-1)
        D = np.asarray(sdp.apply_AT(b), dtype=float)
        D = 0.5 * (D + D.T)

        def lam_at_t(t: float) -> float:
            S = S0 + float(t) * D
            S = 0.5 * (S + S.T)
            return float(np.linalg.eigvalsh(S)[0])

        t_lo = 0.0
        t_hi = 1.0
        lam_hi = lam_at_t(t_hi)
        expand_steps = 0
        while lam_hi < float(delta) and t_hi < float(t_max) and expand_steps < 30:
            t_lo = t_hi
            t_hi *= 2.0
            lam_hi = lam_at_t(t_hi)
            expand_steps += 1

        if lam_hi >= float(delta):
            for _ in range(30):
                t_mid = 0.5 * (t_lo + t_hi)
                if lam_at_t(t_mid) >= float(delta):
                    t_hi = t_mid
                else:
                    t_lo = t_mid
            return y0 - float(t_hi) * b

        # Fallback: cheap cut-style subgradient updates.
        y = y0.copy()
        best_y = y.copy()
        best_lam = lam0
        A_list = [np.asarray(Ai, dtype=float) for Ai in sdp.A]
        for _ in range(80):
            S = np.asarray(sdp.C, dtype=float) - np.asarray(sdp.apply_AT(y), dtype=float)
            S = 0.5 * (S + S.T)
            vals, vecs = np.linalg.eigh(S)
            idx = int(np.argmin(vals))
            lam = float(vals[idx])
            if lam > best_lam:
                best_lam = lam
                best_y = y.copy()
            if lam >= float(delta):
                return y
            v = np.asarray(vecs[:, idx], dtype=float).reshape(-1)
            a = np.asarray([float(v.T @ Ai @ v) for Ai in A_list], dtype=float)
            g = -a
            denom = float(np.dot(g, g)) + 1e-12
            step = min(1.0, max(1e-6, (float(delta) - lam) / denom))
            y = y + step * g
        return best_y

    if obj_enabled and b_targets is not None:
        b_np = np.asarray(b_targets, dtype=float)
        if b_np.ndim == 2 and b_np.shape[0] == y_np.shape[0] and b_np.shape[1] == y_np.shape[1]:
            b_t = torch.as_tensor(b_np, dtype=torch.float32)
        elif obj_debug and not warned_missing_obj:
            print("[l2ca-debug] objective-aware loss skipped: b_targets shape mismatch.")
            warned_missing_obj = True
    elif obj_enabled and obj_debug and not warned_missing_obj:
        print("[l2ca-debug] objective-aware loss skipped: b_targets unavailable.")
        warned_missing_obj = True

    if obj_enabled and y_targets_unscaled is not None:
        y_u_np = np.asarray(y_targets_unscaled, dtype=float)
        if y_u_np.ndim == 2 and y_u_np.shape == y_np.shape:
            y_unscaled_t = torch.as_tensor(y_u_np, dtype=torch.float32)
        elif obj_debug and not warned_missing_obj:
            print("[l2ca-debug] objective-aware loss skipped: y_targets_unscaled shape mismatch.")
            warned_missing_obj = True

    if obj_enabled and y_mean is not None and y_std is not None:
        y_mean_np = np.asarray(y_mean, dtype=float).reshape(-1)
        y_std_np = np.asarray(y_std, dtype=float).reshape(-1)
        if y_mean_np.shape[0] == y_np.shape[1] and y_std_np.shape[0] == y_np.shape[1]:
            y_mean_t = torch.as_tensor(y_mean_np, dtype=torch.float32, device=dev)
            y_std_t = torch.as_tensor(y_std_np, dtype=torch.float32, device=dev)
        elif obj_debug and not warned_missing_obj:
            print("[l2ca-debug] objective-aware loss skipped: y_mean/y_std shape mismatch.")
            warned_missing_obj = True

    if feas_enabled and (sdp_instances is None or len(sdp_instances) != y_np.shape[0]):
        if obj_debug and not warned_missing_feas:
            print("[l2ca-debug] feasibility-aware loss skipped: sdp_instances missing or mismatched.")
            warned_missing_feas = True
        feas_enabled = False

    # Optional interior-label training target construction.
    if interiorize_labels_enabled:
        can_build_interiors = (
            y_targets_unscaled is not None
            and y_mean is not None
            and y_std is not None
            and sdp_instances is not None
            and len(sdp_instances) == y_np.shape[0]
        )
        if not can_build_interiors:
            if obj_debug or bool(label_margin_debug):
                print("[l2ca-debug] interior-label feature skipped: missing unscaled labels/sdp_instances.")
            interiorize_labels_enabled = False
        else:
            y_unscaled_np = np.asarray(y_targets_unscaled, dtype=float)
            lam_arr, boundary_labels = _label_slack_margin_stats(y_unscaled_np)
            if lam_arr is None:
                if obj_debug or bool(label_margin_debug):
                    print("[l2ca-debug] interior-label feature skipped: could not compute label margins.")
                interiorize_labels_enabled = False
            elif not boundary_labels:
                print("Labels appear interior; skipping interior-label feature.")
                interiorize_labels_enabled = False
            else:
                delta_val = float(max(interior_delta, 0.0))
                y_cen = np.asarray(y_unscaled_np, dtype=float).copy()
                for i in range(y_cen.shape[0]):
                    sdp_i = sdp_instances[i]
                    y_cen[i] = _interiorize_label_via_b(
                        sdp=sdp_i,
                        y_star=y_cen[i],
                        delta=delta_val,
                        t_max=1e3,
                    )
                y_mean_np = np.asarray(y_mean, dtype=float).reshape(1, -1)
                y_std_np = np.asarray(y_std, dtype=float).reshape(1, -1)
                y_np = (y_cen - y_mean_np) / y_std_np
                y_t = torch.as_tensor(y_np, dtype=torch.float32)
                if obj_debug or bool(label_margin_debug):
                    lam_cen, _ = _label_slack_margin_stats(y_cen)
                    if lam_cen is not None:
                        print(
                            "[l2ca-debug] interiorized label lam_min stats: "
                            f"mean={float(np.mean(lam_cen)):.3e}, "
                            f"median={float(np.median(lam_cen)):.3e}, "
                            f"min={float(np.min(lam_cen)):.3e}"
                        )
    elif bool(label_margin_debug) and y_targets_unscaled is not None:
        _ = _label_slack_margin_stats(np.asarray(y_targets_unscaled, dtype=float))

    def _build_slack_batch(y_pred_phys: torch.Tensor, ib_local: torch.Tensor) -> torch.Tensor | None:
        if sdp_instances is None:
            return None
        idx_list = ib_local.detach().cpu().tolist()
        slacks: list[torch.Tensor] = []
        for local_j, row_idx in enumerate(idx_list):
            sdp = sdp_instances[int(row_idx)]
            A_np = np.asarray(sdp.A, dtype=float)
            C_np = np.asarray(sdp.C, dtype=float)
            A_t = torch.as_tensor(A_np, dtype=torch.float32, device=dev)
            C_t = torch.as_tensor(C_np, dtype=torch.float32, device=dev)
            yj = y_pred_phys[local_j]
            if yj.shape[0] != A_t.shape[0]:
                return None
            # S(y) = C - sum_i y_i A_i
            Sj = C_t - torch.einsum("i,ijk->jk", yj, A_t)
            Sj = 0.5 * (Sj + Sj.transpose(-1, -2))
            slacks.append(Sj)
        if not slacks:
            return None
        return torch.stack(slacks, dim=0)

    def _chol_margin_penalty(S_batch: torch.Tensor) -> torch.Tensor:
        nonlocal warned_eig_fallback
        shifts = (0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1)
        _, n, _ = S_batch.shape
        eye = torch.eye(n, dtype=S_batch.dtype, device=S_batch.device).unsqueeze(0)
        margin_t = torch.as_tensor(float(feas_margin_train_val), dtype=S_batch.dtype, device=S_batch.device)

        if feas_mode == "eig":
            lam = torch.min(torch.linalg.eigvalsh(S_batch), dim=1).values
            return torch.mean(torch.relu(margin_t - lam) ** 2)

        has_chol_ex = hasattr(torch.linalg, "cholesky_ex")
        if not has_chol_ex:
            if obj_debug and not warned_eig_fallback:
                print("[l2ca-debug] torch.linalg.cholesky_ex unavailable; using eigvalsh feasibility loss fallback.")
                warned_eig_fallback = True
            lam = torch.min(torch.linalg.eigvalsh(S_batch), dim=1).values
            return torch.mean(torch.relu(margin_t - lam) ** 2)

        info_list: list[torch.Tensor] = []
        for delta in shifts:
            M = S_batch + float(delta) * eye
            _, infok = torch.linalg.cholesky_ex(M, check_errors=False)
            info_list.append(infok)

        info_stack = torch.stack(info_list, dim=0)  # (K, B)
        success_stack = info_stack.eq(0)
        any_success = torch.any(success_stack, dim=0)  # (B,)
        first_idx = torch.argmax(success_stack.to(torch.int64), dim=0)
        chosen_idx = torch.where(
            any_success,
            first_idx,
            torch.full_like(first_idx, len(shifts) - 1),
        )
        deltas_t = torch.as_tensor(shifts, dtype=S_batch.dtype, device=S_batch.device)
        penalty = deltas_t[chosen_idx]
        penalty = torch.where(
            any_success,
            penalty,
            torch.full_like(penalty, float(shifts[-1]) * 10.0),
        )
        return torch.mean(penalty)

    ds = TensorDataset(x_t, y_t, idx_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    for _ in range(int(epochs)):
        model.train()
        for xb, yb, ib in loader:
            xb = xb.to(dev)
            yb = yb.to(dev)
            ib = ib.to(dev)
            opt.zero_grad()

            yhat = model(xb)
            loss_y = lambda_y_val * loss_fn(yhat, yb)
            loss = loss_y

            if y_mean_t is not None and y_std_t is not None:
                y_pred_phys = yhat * y_std_t + y_mean_t
            else:
                y_pred_phys = yhat

            if obj_enabled and b_t is not None:
                bb = b_t[ib].to(dev)

                # Prefer exact unscaled values when provided; otherwise unscale
                # predictions/targets using y_mean/y_std if available.
                if y_unscaled_t is not None:
                    y_true_obj = y_unscaled_t[ib].to(dev)
                elif y_mean_t is not None and y_std_t is not None:
                    y_true_obj = yb * y_std_t + y_mean_t
                else:
                    y_true_obj = yb

                obj_pred = torch.sum(y_pred_phys * bb, dim=1)
                obj_true = torch.sum(y_true_obj * bb, dim=1)
                loss_obj = loss_fn(obj_pred, obj_true)
                loss = loss + lambda_obj_val * loss_obj
            elif obj_enabled and obj_debug and not warned_missing_obj:
                print("[l2ca-debug] objective-aware loss inactive for this batch (missing b).")
                warned_missing_obj = True

            if feas_enabled:
                S_batch = _build_slack_batch(y_pred_phys, ib)
                if S_batch is not None:
                    loss_feas = _chol_margin_penalty(S_batch)
                    loss = loss + lambda_feas_val * loss_feas
                elif obj_debug and not warned_missing_feas:
                    print("[l2ca-debug] feasibility-aware loss skipped: could not assemble slack batch.")
                    warned_missing_feas = True

            loss.backward()
            opt.step()

    model.eval()
    return model


def train_scalar_l2a(
    model: ScalarL2ANet,
    features: np.ndarray,
    gamma_targets: np.ndarray,
    epochs: int = 200,
    lr: float = 3e-4,
    batch_size: int = 64,
    device: str = "cpu",
) -> ScalarL2ANet:
    """Train scalar L2A network with pure MSE."""
    y = np.asarray(gamma_targets, dtype=float).reshape(-1, 1)
    out = _train_mse_model(
        model=model,
        x=np.asarray(features, dtype=float),
        y=y,
        epochs=int(epochs),
        lr=float(lr),
        batch_size=int(batch_size),
        device=device,
    )
    return out  # type: ignore[return-value]


__all__ = [
    "DualNet",
    "ScalarL2ANet",
    "train_dual_net",
    "train_scalar_l2a",
]
