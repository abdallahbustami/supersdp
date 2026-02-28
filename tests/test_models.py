import numpy as np
import pytest
import torch

from l2ws.models import GainApproxNet, WarmStartNet
from l2ws.graph_utils import SystemGraphBuilder, has_torch_geometric


def test_gain_approx_net_output_shape():
    model = GainApproxNet(input_dim=6, hidden_sizes=(8, 4), output_dim=2, positive_output=False)
    x = torch.randn(5, 6)
    y = model(x)
    assert y.shape == (5, 2)


def test_gain_approx_net_fallback_to_mlp(monkeypatch):
    import l2ws.models as models_mod

    monkeypatch.setattr(models_mod, "has_torch_geometric", lambda: False)
    with pytest.warns(RuntimeWarning):
        model = GainApproxNet(input_dim=4, output_dim=1, backbone="gnn", node_feat_dim=3)
    assert model.backbone_name == "mlp"


@pytest.mark.skipif(not has_torch_geometric(), reason="torch_geometric not available")
def test_gain_approx_net_gnn_forward():
    A = np.array([[-1.0, 0.2], [0.1, -2.0]])
    Bw = np.array([[1.0], [0.0]])
    Cz = np.array([[1.0, 2.0]])
    graph = SystemGraphBuilder().build(A, Bw, Cz)

    model = GainApproxNet(
        input_dim=4,
        output_dim=2,
        positive_output=False,
        backbone="gnn",
        node_feat_dim=graph.x.shape[1],
    )
    y = model(graph)
    assert y.shape == (1, 2)


def test_warmstart_net_cholesky_shapes_mlp():
    n = 4
    m = 3
    model = WarmStartNet(n=n, m=m, input_dim=10, hidden_sizes=(16, 8), backbone="mlp")
    x = torch.randn(7, 10)
    X, y, S = model(x)
    assert X.shape == (7, n, n)
    assert y.shape == (7, m)
    assert S.shape == (7, n, n)


def test_warmstart_net_diagonal_mode_shapes():
    n = 3
    m = 2
    model = WarmStartNet(n=n, m=m, input_dim=5, hidden_sizes=(8,), warmstart_type="diagonal")
    x = torch.randn(4, 5)
    X, y, S = model(x)
    assert X.shape == (4, n, n)
    assert y.shape == (4, m)
    assert S.shape == (4, n, n)


def test_warmstart_net_output_scaler_roundtrip():
    n = 2
    m = 2
    model = WarmStartNet(n=n, m=m, input_dim=4, hidden_sizes=(8,), backbone="mlp", use_batchnorm=False)
    mean = torch.ones(model.output_dim)
    std = 2.0 * torch.ones(model.output_dim)
    model.set_output_scaler(mean, std)

    x = torch.zeros(1, 4)
    with torch.no_grad():
        raw = model.raw_forward(x)
    assert raw.shape[-1] == model.output_dim


@pytest.mark.skipif(not has_torch_geometric(), reason="torch_geometric not available")
def test_warmstart_net_gnn_forward():
    n = 2
    m = 3
    A = np.array([[-1.0, 0.2], [0.1, -2.0]])
    Bw = np.array([[1.0], [0.0]])
    Cz = np.array([[1.0, 2.0]])
    graph = SystemGraphBuilder().build(A, Bw, Cz)

    model = WarmStartNet(
        n=n,
        m=m,
        input_dim=4,
        backbone="gnn",
        node_feat_dim=graph.x.shape[1],
        warmstart_type="cholesky",
    )
    X, y, S = model(graph)
    assert X.shape == (1, n, n)
    assert y.shape == (1, m)
    assert S.shape == (1, n, n)
