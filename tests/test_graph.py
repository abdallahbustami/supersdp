import numpy as np
import pytest

from l2ws.data import LTISystem, build_system_graph
from l2ws.graph_utils import SystemGraphBuilder, has_torch_geometric


def _example_system():
    A = np.array([[-1.0, 0.5, 0.0], [0.1, -2.0, 0.3], [0.0, 0.2, -3.0]])
    Bw = np.array([[1.0], [0.0], [0.2]])
    Cz = np.array([[1.0, 0.0, 1.0]])
    return LTISystem(A=A, Bw=Bw, Cz=Cz)


def test_builder_requires_pyg_when_unavailable(monkeypatch):
    import l2ws.graph_utils as gu

    monkeypatch.setattr(gu, "Data", None)
    builder = gu.SystemGraphBuilder()
    sys = _example_system()
    with pytest.raises(ModuleNotFoundError):
        builder.build(sys.A, sys.Bw, sys.Cz)


@pytest.mark.skipif(not has_torch_geometric(), reason="torch_geometric not available")
def test_system_graph_shapes():
    sys = _example_system()
    graph = SystemGraphBuilder().build(sys.A, sys.Bw, sys.Cz)

    assert graph.x.shape[0] == sys.A.shape[0]
    assert graph.x.shape[1] == 1 + sys.Bw.shape[1] + sys.Cz.shape[0]
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_attr.shape[1] == 1


@pytest.mark.skipif(not has_torch_geometric(), reason="torch_geometric not available")
def test_build_system_graph_helper():
    sys = _example_system()
    graph = build_system_graph(sys)
    assert graph.x.shape[0] == sys.A.shape[0]
