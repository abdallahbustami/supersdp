# SuperSDP

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](#)

  _________                          _________________ __________ 
 /   _____/__ ________   ___________/   _____/\______ \\______   \
 \_____  \|  |  \____ \_/ __ \_  __ \_____  \  |    |  \|     ___/
 /        \  |  /  |_> >  ___/|  | \/        \ |    `   \    |    
/_______  /____/|   __/ \___  >__| /_______  //_______  /____|    
        \/      |__|        \/             \/         \/          
        
Learning-accelerated semidefinite programming for control-oriented and general standard-form SDPs, with:
- Infeasible predictor-corrector IPM for standard-form SDPs
- L2WS warm-start models (Cholesky or legacy diagonal)
- Optional GNN backbone (`torch-geometric`) for system-structured inputs
- L2A direct approximation mode
- L2CA dual prediction with certification/repair and optional short refinement

## Installation

### Base install (MLP/backbone-free of PyG)

```bash
pip install -e .
```

### Dev tools

```bash
pip install -e ".[dev]"
```

### Optional GNN support

```bash
pip install -e ".[gnn]"
```

If your environment needs explicit PyG wheels, follow the official PyG install guide and then reinstall `l2ws`.

The top-level package entrypoint (`import l2ws`) intentionally exposes the
stable low-level solver/config API plus a small set of common SDP/system data
helpers. Advanced L2CA utilities and experimental helpers remain available from
their submodules (for example `l2ws.l2ca` and `l2ws.learning`).

## Quick Start

```python
import numpy as np
from l2ws import L2CAConfig, SuperSDP, LTISystem, ProblemConfig, SolverConfig, TrainingConfig, generate_l2_gain_instances

sys = LTISystem(
    A=np.array([[-3.0, -2.0], [1.0, 0.0]], float),
    Bw=np.array([[1.0], [0.0]]),
    Cz=np.array([[2.0, 1.0]]),
)
inst = generate_l2_gain_instances(sys, num_instances=24, seed=42)

solver = SuperSDP(
    ProblemConfig(n=inst[0].sdp.dim, m=inst[0].sdp.num_constraints),
    TrainingConfig(epochs=10, batch_size=8),
    SolverConfig(mode="L2CA"),
    l2ca_config=L2CAConfig(bisect_iters=20, anchor_mode="knn5_best"),
)
solver.fit([x.sdp for x in inst[:16]])
result = solver.solve(inst[16].sdp)
print(result.mode_used, result.dual_feasible, result.dual_obj)
```

## Toolbox CLI Quickstart

List supported toolbox applications:

```bash
PYTHONPATH=src ./l2ws_env/bin/python -m l2ws.cli list-apps
```

Run from a `.mat` file:

```bash
PYTHONPATH=src ./l2ws_env/bin/python -m l2ws.cli run \
  --mat data/systems.mat \
  --mat-key-A A --mat-key-B B --mat-key-Cz Cz \
  --application l2_gain \
  --num-train 80 --num-test 20 \
  --epochs 45 --batch-size 64 --hidden-sizes 128,64,32 \
  --perturb-kind entrywise_uniform --A-scale 0.05 --B-scale 0.02 \
  --algorithms MOSEK IPM L2WS L2A L2CA --seed 118
```

Lyapunov regularized run:

```bash
PYTHONPATH=src ./l2ws_env/bin/python -m l2ws.cli run \
  --mat data/systems.mat \
  --mat-key-A A \
  --application lyapunov_reg \
  --cert-regularize on --cert-eps 1e-6 --cert-Q identity --cert-Q-scale 1.0 \
  --num-train 60 --num-test 20 --algorithms IPM L2CA
```

Expected `.mat` formats:
- Single system: `A` as `(n,n)` plus optional `Bw`/`B` and `Cz`.
- Multiple systems: `A` as `(n,n,k)` or MATLAB cell/array of matrices.
- `Bw`/`Cz` can also be single matrices (broadcast to all systems) or collections with the same count as `A`.
- If `Bw` or `Cz` is missing for `{l2_gain,h2_norm,hinf_norm}`, the CLI defaults them to identity and logs it.

More runnable examples:
- `examples/run_toolbox_minimal.py`
- `examples/run_toolbox_hinf.py`
- `examples/test_l2ca_fi_msd8.py`

## Module Structure

Core algorithm modules are now separated for easier navigation:
- `src/l2ws/l2ws.py`: warm-start solver API entrypoints (`SuperSDP`, configs)
- `src/l2ws/l2a.py`: direct objective-approximation API (`ScalarL2ANet`, trainer)
- `src/l2ws/l2ca.py`: dual certification/correction utilities (feasibility + repair)
- `src/l2ws/learning.py`: shared DNN components/training loops used by L2A/L2CA

Backward-compatible imports are preserved (existing imports from `l2ws.l2ca` and
`l2ws.toolbox` continue to work).

## SuperSDP API

`SuperSDP(problem_config, training_config, solver_config, device="cpu", l2ca_config=...)`

Main methods:
- `fit(instances, solutions=None, graphs=None)`
- `solve(instance, graph=None)`
- `solve_batch(instances, graphs=None)`
- `update(new_instances, new_solutions, graphs=None)` (lifelong mode)
- `save(path)` / `SuperSDP.load(path)`

Core configs:
- `ProblemConfig(n, m, data_format="dense"|"sparse"|"func")`
- `TrainingConfig(..., backbone="mlp"|"gnn")`
- `SolverConfig(mode="L2WS"|"L2A"|"L2CA"|"Auto", warmstart_type="cholesky"|"diagonal")`
- `L2CAConfig(...)` for dual-loss weights and certify/repair settings

Backward compatibility note:
- The older name `L2WSSolver` is still available as an alias, but `SuperSDP` is the primary public class name.

L2CA configuration note:
- With `tier_auto=True` (the default), the shared L2CA inference path uses the cached robust anchor plus structural tier logic, so `anchor_mode` is effectively inactive.
- `anchor_mode` only affects the non-tiered anchor path when `tier_auto=False`.

## Solver Modes

- `L2WS`: predict warm-start state then run IPM.
- `L2A`: direct approximation model (no IPM iterations in inference), with optional `mlp`/`gnn` backbone.
- `L2CA`: predict a dual point, then run the shared fast certification/repair pipeline used by the toolbox runner.
- `Auto`: retained as a backward-compatible legacy alias; it does not add a distinct production solver path beyond the current warm-start/IPM settings.

`SolverConfig.backend` is validated by the low-level API but low-level `SuperSDP.solve()` always uses the internal infeasible IPM. External MOSEK/SCS comparisons are exposed through the runner/CLI layer.

## Run L2-Gain Case Study

```bash
PYTHONPATH=src python -m l2ws.workflows.l2_gain_case_study \
  --instances 50 --episode-size 10 --seed 1 --device cpu --backbone mlp
```

With GNN backbone:

```bash
PYTHONPATH=src python -m l2ws.workflows.l2_gain_case_study \
  --instances 50 --episode-size 10 --seed 1 --device cpu --backbone gnn
```

## Extending to Custom SDP Problems

1. Build `SDPInstance(C, A, b)` objects for your problem family.
2. Decide feature pathway:
- Flat features (`data_format="dense"/"func"`) for MLP
- Graph inputs (`backbone="gnn"`) via `SystemGraphBuilder`/`build_system_graph`
3. Train using `SuperSDP.fit(...)` with labels or auto-labeling.
4. Use `warmstart_type="cholesky"` for guaranteed `d_F=0` warm-start construction.

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```
