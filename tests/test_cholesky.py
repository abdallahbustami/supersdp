import numpy as np

from l2ws.ipm import dF_value, in_neighborhood
from l2ws.toolbox import _warmstart_from_cholesky, _warmstart_targets_cholesky


def _random_cholesky_pair(n: int, seed: int):
    rng = np.random.default_rng(seed)
    L = np.tril(rng.standard_normal((n, n)))
    np.fill_diagonal(L, np.abs(np.diag(L)) + 3.0)
    nu = float(rng.uniform(0.2, 3.0))
    y = rng.standard_normal(n)
    return L, nu, y


def test_random_cholesky_pairs_have_zero_dF():
    sizes = [2, 5, 10, 20]
    seed = 1
    for n in sizes:
        for _ in range(25):
            L, nu, y = _random_cholesky_pair(n, seed)
            seed += 1
            state = _warmstart_from_cholesky(L, nu, y)
            assert dF_value(state.X, state.S) < 1e-3
            assert in_neighborhood(state.X, state.S, gamma=1e-3)


def test_targets_cholesky_contains_expected_layout():
    n = 4
    m = 3
    L, nu, y = _random_cholesky_pair(n, seed=4)
    X = L @ L.T
    S = nu * np.linalg.solve(X, np.eye(n))

    target = _warmstart_targets_cholesky((X, y[:m], S), n=n, m=m)
    n_tril = n * (n + 1) // 2

    assert target.shape[0] == n_tril + 1
    assert target[n_tril] > 0
