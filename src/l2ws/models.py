"""Neural models for warm-starting and approximating SDP solutions."""

from __future__ import annotations

from typing import Literal
import warnings

import torch
from torch import nn
from torch.nn import functional as F

from .graph_utils import SystemGraphBuilder, has_torch_geometric


def _activation_layer(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    return nn.ReLU()


def _build_mlp(
    input_dim: int,
    hidden_sizes,
    dropout: float = 0.2,
    use_batchnorm: bool = True,
    activation: str = "relu",
) -> nn.Sequential:
    layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_sizes:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(_activation_layer(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dim
    return nn.Sequential(*layers)


class WarmStartNetLegacy(nn.Module):
    """Legacy diagonal warm-start predictor (x,s scales only)."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes=(128, 64, 32, 16),
        dropout: float = 0.2,
        activation: str = "relu",
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = _build_mlp(
            input_dim,
            hidden_sizes,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
            activation=activation,
        )
        self.head = nn.Linear(hidden_sizes[-1] if hidden_sizes else input_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        features = self.backbone(x) if len(self.backbone) > 0 else x
        return self.head(features)


class GNNBackbone(nn.Module):
    """Graph encoder used by warm-start networks when PyG is available."""

    def __init__(
        self,
        node_feat_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        readout_dim: int = 128,
    ) -> None:
        super().__init__()
        if not has_torch_geometric():  # pragma: no cover - guarded by caller
            raise ModuleNotFoundError("torch_geometric is required for GNNBackbone.")

        from torch_geometric.nn import GCNConv, global_mean_pool

        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_feat_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.pool = global_mean_pool
        self.readout_mlp = nn.Sequential(
            nn.Linear(hidden_dim, readout_dim),
            nn.ReLU(),
        )

    def forward(self, data) -> torch.Tensor:
        x = data.x
        edge_index = data.edge_index
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = self.pool(x, batch)
        return self.readout_mlp(x)


class WarmStartNet(nn.Module):
    """Warm-start predictor with optional MLP/GNN backbone.

    Depending on ``warmstart_type`` this model predicts either:
    - ``cholesky``: flattened lower-triangular factor ``L`` and scalar ``nu``;
      then reconstructs ``X=LL^T`` and ``S=nu * X^{-1}``.
    - ``diagonal``: legacy diagonal scales ``x,s`` and dual ``y``.

    The forward pass always returns ``(X_tilde, y_tilde, S_tilde)``.
    """

    def __init__(
        self,
        n: int,
        m: int,
        input_dim: int | None = None,
        hidden_sizes=(128, 64, 32, 16),
        dropout: float = 0.2,
        activation: str = "relu",
        use_batchnorm: bool = True,
        backbone: Literal["mlp", "gnn"] = "mlp",
        node_feat_dim: int | None = None,
        gnn_hidden_dim: int = 64,
        gnn_num_layers: int = 3,
        gnn_readout_dim: int = 128,
        warmstart_type: Literal["cholesky", "diagonal"] = "cholesky",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.n = int(n)
        self.m = int(m)
        self.warmstart_type = warmstart_type
        self.eps = float(eps)

        if self.n <= 0 or self.m <= 0:
            raise ValueError("n and m must be positive.")
        if warmstart_type not in {"cholesky", "diagonal"}:
            raise ValueError("warmstart_type must be 'cholesky' or 'diagonal'.")

        if warmstart_type == "cholesky":
            self.output_dim = (self.n * (self.n + 1)) // 2 + 1
        else:
            self.output_dim = 2 + self.m

        self.backbone_name = backbone
        encoded_dim: int

        if backbone == "gnn":
            if not has_torch_geometric():
                warnings.warn(
                    "torch_geometric is not installed; falling back to MLP backbone.",
                    RuntimeWarning,
                )
                self.backbone_name = "mlp"
            elif node_feat_dim is None:
                raise ValueError("node_feat_dim is required when backbone='gnn'.")
            else:
                self.gnn_backbone = GNNBackbone(
                    node_feat_dim=node_feat_dim,
                    hidden_dim=gnn_hidden_dim,
                    num_layers=gnn_num_layers,
                    readout_dim=gnn_readout_dim,
                )
                encoded_dim = gnn_readout_dim

        if self.backbone_name == "mlp":
            if input_dim is None:
                raise ValueError("input_dim is required for the MLP backbone.")
            self.mlp_backbone = _build_mlp(
                input_dim,
                hidden_sizes,
                dropout=dropout,
                use_batchnorm=use_batchnorm,
                activation=activation,
            )
            encoded_dim = hidden_sizes[-1] if hidden_sizes else input_dim

        self.head = nn.Linear(encoded_dim, self.output_dim)

        # Optional target de-standardization buffers for raw outputs.
        self.register_buffer("_target_mean", torch.zeros(self.output_dim, dtype=torch.float32))
        self.register_buffer("_target_std", torch.ones(self.output_dim, dtype=torch.float32))
        self._use_output_scaler = False

        tril = torch.tril_indices(self.n, self.n)
        self.register_buffer("_tril_row", tril[0], persistent=False)
        self.register_buffer("_tril_col", tril[1], persistent=False)
        self.register_buffer("_diag_idx", torch.arange(self.n), persistent=False)

    def set_output_scaler(self, mean: torch.Tensor | None, std: torch.Tensor | None) -> None:
        """Set de-standardization stats for raw outputs.

        If provided, raw head outputs are interpreted as standardized values and
        transformed back by ``raw * std + mean`` before reconstruction.
        """
        if mean is None or std is None:
            self._use_output_scaler = False
            return

        mean = mean.reshape(-1).detach().to(dtype=torch.float32, device=self._target_mean.device)
        std = std.reshape(-1).detach().to(dtype=torch.float32, device=self._target_std.device)
        if mean.numel() != self.output_dim or std.numel() != self.output_dim:
            raise ValueError(
                f"Scaler shapes must match output_dim={self.output_dim}, "
                f"got mean={mean.numel()}, std={std.numel()}."
            )
        std = torch.where(std.abs() < 1e-8, torch.ones_like(std), std)
        self._target_mean.copy_(mean)
        self._target_std.copy_(std)
        self._use_output_scaler = True

    def _encode(self, x) -> torch.Tensor:
        if self.backbone_name == "gnn":
            data = x
            if not hasattr(data, "batch"):
                from torch_geometric.data import Batch

                data = Batch.from_data_list([data])
            return self.gnn_backbone(data)

        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.mlp_backbone(x) if len(self.mlp_backbone) > 0 else x

    def _apply_output_scaler(self, raw: torch.Tensor) -> torch.Tensor:
        if self._use_output_scaler:
            return raw * self._target_std.unsqueeze(0) + self._target_mean.unsqueeze(0)
        return raw

    def raw_forward(self, x, apply_scaler: bool = False) -> torch.Tensor:
        """Return raw model head outputs.

        By default this returns standardized network outputs. Set
        ``apply_scaler=True`` to de-standardize using the configured target
        scaler for reconstruction/inference.
        """
        h = self._encode(x)
        raw = self.head(h)
        if apply_scaler:
            raw = self._apply_output_scaler(raw)
        return raw

    def _decode_cholesky_components(self, raw_output: torch.Tensor):
        batch_size = raw_output.shape[0]
        n_tril = self.n * (self.n + 1) // 2

        L_flat = raw_output[:, :n_tril]
        nu_raw = raw_output[:, n_tril : n_tril + 1]

        L = torch.zeros(
            (batch_size, self.n, self.n),
            dtype=raw_output.dtype,
            device=raw_output.device,
        )
        L[:, self._tril_row, self._tril_col] = L_flat
        # Raw outputs are trained to match physical targets directly, so avoid
        # nonlinear reparameterizations here. We only enforce strict positivity.
        L[:, self._diag_idx, self._diag_idx] = torch.clamp_min(
            L[:, self._diag_idx, self._diag_idx],
            1e-6,
        )

        nu = torch.clamp_min(nu_raw, 1e-6)
        return L, nu

    def _reconstruct_cholesky(self, raw_output: torch.Tensor):
        L, nu = self._decode_cholesky_components(raw_output)
        X_tilde = L @ L.transpose(-1, -2)

        eye = torch.eye(self.n, device=X_tilde.device, dtype=X_tilde.dtype).unsqueeze(0)
        eye = eye.expand(X_tilde.size(0), -1, -1)
        X_reg = X_tilde + 1e-3 * eye
        X_inv = torch.linalg.solve(X_reg, eye)
        S_tilde = nu.unsqueeze(-1) * X_inv
        y_tilde = torch.zeros((X_tilde.size(0), self.m), dtype=X_tilde.dtype, device=X_tilde.device)
        return X_tilde, y_tilde, S_tilde

    def _reconstruct_diagonal(self, raw_output: torch.Tensor):
        scales = F.softplus(raw_output[:, :2]) + self.eps
        x_scale = scales[:, 0]
        s_scale = scales[:, 1]
        y_tilde = raw_output[:, 2:]

        eye = torch.eye(self.n, device=raw_output.device, dtype=raw_output.dtype).unsqueeze(0)
        eye = eye.expand(raw_output.size(0), -1, -1)
        X_tilde = x_scale.view(-1, 1, 1) * eye
        S_tilde = s_scale.view(-1, 1, 1) * eye
        return X_tilde, y_tilde, S_tilde

    def _reconstruct_XyS(self, raw_output: torch.Tensor):
        if self.warmstart_type == "cholesky":
            return self._reconstruct_cholesky(raw_output)
        return self._reconstruct_diagonal(raw_output)

    def predict_components(self, x):
        """Return interpreted components prior to full reconstruction.

        Returns:
            - cholesky mode: ``(L, nu)``
            - diagonal mode: ``(x_scale, s_scale, y_tilde)``
        """
        raw_output = self.raw_forward(x, apply_scaler=True)
        if self.warmstart_type == "cholesky":
            return self._decode_cholesky_components(raw_output)

        scales = F.softplus(raw_output[:, :2]) + self.eps
        return scales[:, 0], scales[:, 1], raw_output[:, 2:]

    def forward(self, x):
        raw_output = self.raw_forward(x, apply_scaler=True)
        return self._reconstruct_XyS(raw_output)


class GainApproxNet(nn.Module):
    """Approximates the L2 gain directly from system features.

    Can also be used to predict primal/dual variables by setting ``output_dim > 1``.
    Supports MLP and optional GNN backbones (with graceful fallback to MLP).
    """

    def __init__(
        self,
        input_dim: int | None,
        hidden_sizes=(128, 64, 32, 16),
        output_dim: int = 1,
        dropout: float = 0.0,
        use_batchnorm: bool = False,
        positive_output: bool = True,
        activation: str = "relu",
        backbone: Literal["mlp", "gnn"] = "mlp",
        node_feat_dim: int | None = None,
        gnn_hidden_dim: int = 64,
        gnn_num_layers: int = 3,
        gnn_readout_dim: int = 128,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        encoded_dim: int

        if backbone == "gnn":
            if not has_torch_geometric():
                warnings.warn(
                    "torch_geometric is not installed; falling back to MLP backbone.",
                    RuntimeWarning,
                )
                self.backbone_name = "mlp"
            elif node_feat_dim is None:
                raise ValueError("node_feat_dim is required when backbone='gnn'.")
            else:
                self.gnn_backbone = GNNBackbone(
                    node_feat_dim=node_feat_dim,
                    hidden_dim=gnn_hidden_dim,
                    num_layers=gnn_num_layers,
                    readout_dim=gnn_readout_dim,
                )
                encoded_dim = gnn_readout_dim

        if self.backbone_name == "mlp":
            if input_dim is None:
                raise ValueError("input_dim is required for the MLP backbone.")
            self.mlp_backbone = _build_mlp(
                input_dim,
                hidden_sizes,
                dropout=dropout,
                use_batchnorm=use_batchnorm,
                activation=activation,
            )
            encoded_dim = hidden_sizes[-1] if hidden_sizes else input_dim

        self.head = nn.Linear(encoded_dim, output_dim)
        self.positive_output = positive_output

    def _encode(self, x) -> torch.Tensor:
        if self.backbone_name == "gnn":
            data = x
            if not hasattr(data, "batch"):
                from torch_geometric.data import Batch

                data = Batch.from_data_list([data])
            return self.gnn_backbone(data)

        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.mlp_backbone(x) if len(self.mlp_backbone) > 0 else x

    def forward(self, x) -> torch.Tensor:
        features = self._encode(x)
        out = self.head(features)
        if self.positive_output:
            out = F.softplus(out)
        return out


class CentralPathNet(nn.Module):
    """Predicts an (X, y, S) triple for a fixed-size SDP instance.

    This model is intended for the unsupervised L2A loss (Eq. (23) in the paper),
    where the loss encourages satisfaction of the central-path equations without
    requiring labeled optimal solutions.

    Notes:
    - This is primarily practical for small/medium SDPs because the output is
      quadratic in the SDP dimension n.
    - X and S are projected to the SPD cone via eigenvalue clipping.
    """

    def __init__(
        self,
        input_dim: int,
        n: int,
        m: int,
        hidden_sizes=(256, 256, 128),
        min_eig: float = 1e-4,
    ) -> None:
        super().__init__()
        self.n = int(n)
        self.m = int(m)
        self.min_eig = float(min_eig)
        self.backbone = _build_mlp(
            input_dim, hidden_sizes, dropout=0.0, use_batchnorm=False, activation="relu"
        )
        out_dim = 2 * (self.n * self.n) + self.m
        self.head = nn.Linear(hidden_sizes[-1] if hidden_sizes else input_dim, out_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        h = self.backbone(x) if len(self.backbone) > 0 else x
        out = self.head(h)
        n2 = self.n * self.n
        X_raw = out[:, :n2].reshape(-1, self.n, self.n)
        y = out[:, n2 : n2 + self.m]
        S_raw = out[:, n2 + self.m :].reshape(-1, self.n, self.n)
        X = self._project_spd(X_raw)
        S = self._project_spd(S_raw)
        return X, y, S

    def _project_spd(self, M: torch.Tensor) -> torch.Tensor:
        M = 0.5 * (M + M.transpose(-1, -2))
        eigvals, eigvecs = torch.linalg.eigh(M)
        eigvals = torch.clamp(eigvals, min=self.min_eig)
        return eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1, -2)
