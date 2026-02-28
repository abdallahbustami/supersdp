"""User-facing CLI for the L2WS toolbox."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from scipy.io import loadmat

from .applications import list_applications
from .data import LTISystem
from .perturbations import PerturbationSpec


def _loadmat_compat(path: str | Path) -> Dict[str, Any]:
    path_str = str(path)
    try:
        raw = loadmat(path_str, squeeze_me=True, struct_as_record=False, simplify_cells=True)
    except TypeError:
        raw = loadmat(path_str, squeeze_me=True, struct_as_record=False)
    return {k: v for k, v in raw.items() if not str(k).startswith("__")}


def _extract_key(data: Dict[str, Any], key: str) -> Any:
    parts = str(key).split(".")
    cur: Any = data
    walked = []
    for part in parts:
        walked.append(part)
        if isinstance(cur, dict):
            if part not in cur:
                raise KeyError(f"Missing key '{'.'.join(walked)}' in MAT data.")
            cur = cur[part]
            continue
        if hasattr(cur, part):
            cur = getattr(cur, part)
            continue
        if isinstance(cur, np.ndarray) and cur.dtype.names and part in cur.dtype.names:
            cur = cur[part]
            if isinstance(cur, np.ndarray) and cur.shape == ():
                cur = cur.item()
            continue
        raise KeyError(f"Cannot resolve '{part}' in MAT path '{key}'.")
    return cur


def _try_extract_key(data: Dict[str, Any], key: str | None) -> tuple[Any | None, bool]:
    if key is None:
        return None, False
    try:
        return _extract_key(data, key), True
    except KeyError:
        return None, False


def _as_matrix(value: Any, label: str, allow_vector: bool = True) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1 and allow_vector:
        return arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"{label} must be a 2D matrix. Got shape {arr.shape}.")
    return arr


def _as_matrix_list(value: Any, label: str, allow_vector: bool = True) -> List[np.ndarray]:
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            raise ValueError(f"{label} is empty.")
        return [_as_matrix(v, f"{label}[{i}]", allow_vector=allow_vector) for i, v in enumerate(value)]

    arr = np.asarray(value)
    if arr.dtype != object and np.issubdtype(arr.dtype, np.number):
        if arr.ndim == 1 and allow_vector:
            return [arr.reshape(1, -1).astype(float)]
        if arr.ndim == 2:
            return [arr.astype(float)]
        if arr.ndim == 3:
            return [np.asarray(arr[:, :, i], dtype=float) for i in range(arr.shape[2])]
        raise ValueError(f"{label} numeric array must be 2D or 3D. Got shape {arr.shape}.")

    if arr.dtype == object:
        flat = arr.reshape(-1)
        if flat.size == 0:
            raise ValueError(f"{label} object array is empty.")
        return [_as_matrix(v, f"{label}[{i}]", allow_vector=allow_vector) for i, v in enumerate(flat)]

    raise ValueError(f"{label} has unsupported type/shape for matrix extraction.")


def _align_matrix_count(mats: List[np.ndarray] | None, count: int, label: str) -> List[np.ndarray] | None:
    if mats is None:
        return None
    if len(mats) == count:
        return mats
    if len(mats) == 1 and count > 1:
        return [mats[0].copy() for _ in range(count)]
    raise ValueError(f"{label} count mismatch: expected 1 or {count}, got {len(mats)}.")


def _orient_matrix(mat: np.ndarray, n: int, role: str, idx: int) -> np.ndarray:
    """Orient a 2D matrix to satisfy expected SDP dimensions when possible."""
    M = np.asarray(mat, dtype=float)
    if M.ndim != 2:
        raise ValueError(f"{role}[{idx}] must be 2D. Got shape {M.shape}.")
    if role == "Bw":
        if M.shape[0] == n:
            return M
        if M.shape[1] == n:
            return M.T
        raise ValueError(f"Bw[{idx}] must have row count {n} (or be transposed). Got shape {M.shape}.")
    if role == "Cz":
        if M.shape[1] == n:
            return M
        if M.shape[0] == n:
            return M.T
        raise ValueError(f"Cz[{idx}] must have column count {n} (or be transposed). Got shape {M.shape}.")
    raise ValueError(f"Unsupported role '{role}'.")


def load_systems_from_mat(
    mat_path: str | Path,
    mat_key_A: str,
    mat_key_B: str | None,
    mat_key_Bw: str | None,
    mat_key_Cz: str | None,
    application: str,
) -> Dict[str, LTISystem]:
    """Load systems from a MAT file and normalize to ``dict[name -> LTISystem]``."""
    data = _loadmat_compat(mat_path)

    A_raw = _extract_key(data, mat_key_A)
    A_list = _as_matrix_list(A_raw, mat_key_A, allow_vector=False)
    n_sys = len(A_list)

    bw_raw = None
    bw_source = None
    if mat_key_Bw:
        bw_raw, found_bw = _try_extract_key(data, mat_key_Bw)
        if not found_bw:
            raise KeyError(f"--mat-key-Bw '{mat_key_Bw}' was provided but not found in MAT file.")
        bw_source = mat_key_Bw
    elif mat_key_B:
        bw_raw, found_b = _try_extract_key(data, mat_key_B)
        if found_b:
            bw_source = mat_key_B

    cz_raw = None
    if mat_key_Cz:
        cz_raw, found_cz = _try_extract_key(data, mat_key_Cz)
        if not found_cz:
            raise KeyError(f"--mat-key-Cz '{mat_key_Cz}' was provided but not found in MAT file.")

    Bw_list = _as_matrix_list(bw_raw, bw_source or "Bw") if bw_raw is not None else None
    Cz_list = _as_matrix_list(cz_raw, mat_key_Cz or "Cz") if cz_raw is not None else None
    Bw_list = _align_matrix_count(Bw_list, n_sys, "Bw")
    Cz_list = _align_matrix_count(Cz_list, n_sys, "Cz")

    if bw_source is not None and mat_key_Bw is None and mat_key_B is not None:
        print(f"[info] Using '{mat_key_B}' as disturbance matrix Bw.")

    needs_bw_cz = str(application).strip().lower() in {"l2_gain", "h2_norm", "hinf_norm"}
    systems: Dict[str, LTISystem] = {}
    for i, A in enumerate(A_list):
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"A[{i}] must be square. Got shape {A.shape}.")
        n = A.shape[0]

        bw_missing = Bw_list is None
        cz_missing = Cz_list is None
        Bw = np.eye(n, dtype=float) if bw_missing else _orient_matrix(np.asarray(Bw_list[i], dtype=float), n=n, role="Bw", idx=i)
        Cz = np.eye(n, dtype=float) if cz_missing else _orient_matrix(np.asarray(Cz_list[i], dtype=float), n=n, role="Cz", idx=i)

        if needs_bw_cz and (bw_missing or cz_missing):
            missing = []
            if bw_missing:
                missing.append("Bw")
            if cz_missing:
                missing.append("Cz")
            print(f"[info] sys_{i:03d}: missing {', '.join(missing)} in MAT; defaulting to identity.")

        systems[f"sys_{i:03d}"] = LTISystem(A=A, Bw=Bw, Cz=Cz)

    return systems


def _parse_hidden_sizes(value: str) -> tuple[int, ...]:
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("hidden sizes must be a comma-separated list of ints.")
    try:
        out = tuple(int(p) for p in parts)
    except ValueError as exc:  # pragma: no cover
        raise argparse.ArgumentTypeError("hidden sizes must be integers.") from exc
    if any(v <= 0 for v in out):
        raise argparse.ArgumentTypeError("hidden sizes must be positive.")
    return out


def _build_parser() -> argparse.ArgumentParser:
    apps = sorted(list_applications().keys())
    parser = argparse.ArgumentParser(prog="python -m l2ws.cli", description="L2WS toolbox CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list-apps", help="List supported toolbox applications.")
    p_list.set_defaults(func=_cmd_list_apps)

    p_run = sub.add_parser("run", help="Run toolbox experiment(s) from a MAT file.")
    p_run.add_argument("--mat", required=True, type=str, help="Path to MAT file containing system matrices.")
    p_run.add_argument("--mat-key-A", default="A", type=str, help="MAT key/path for state matrix A.")
    p_run.add_argument("--mat-key-B", default=None, type=str, help="Optional MAT key/path used as Bw fallback.")
    p_run.add_argument("--mat-key-Bw", default=None, type=str, help="Optional MAT key/path for disturbance Bw.")
    p_run.add_argument("--mat-key-Cz", default=None, type=str, help="Optional MAT key/path for output Cz.")
    p_run.add_argument("--application", required=True, choices=apps)

    p_run.add_argument("--num-train", type=int, default=50)
    p_run.add_argument("--num-test", type=int, default=20)
    p_run.add_argument("--epochs", type=int, default=50)
    p_run.add_argument("--batch-size", type=int, default=64)
    p_run.add_argument("--hidden-sizes", type=_parse_hidden_sizes, default=(128, 64, 32))
    p_run.add_argument("--seed", type=int, default=0)
    p_run.add_argument("--lr", type=float, default=3e-4)
    p_run.add_argument(
        "--l2ca-lambda-obj",
        type=float,
        default=0.0,
        help="Weight for objective-aware dual loss in L2CA/L2CA-FI dual-net training.",
    )
    p_run.add_argument(
        "--l2ca-tier-auto",
        choices=["on", "off"],
        default="on",
        help="Enable automatic tiered anchor/repair for L2CA.",
    )
    p_run.add_argument(
        "--l2ca-tier0-fallback",
        choices=["off", "fi", "short_ipm"],
        default="off",
        help="Tier-0 guaranteed fallback mode for L2CA.",
    )
    p_run.add_argument(
        "--l2ca-lambda-feas",
        type=float,
        default=0.0,
        help="Weight for feasibility-aware dual loss in L2CA/L2CA-FI dual-net training.",
    )
    p_run.add_argument(
        "--l2ca-feas-loss",
        choices=["shift", "eig"],
        default="shift",
        help="Feasibility-aware loss mode for L2CA/L2CA-FI dual-net training.",
    )
    p_run.add_argument(
        "--l2ca-feas-margin-train",
        type=float,
        default=0.0,
        help="Training-time feasibility margin used when --l2ca-feas-loss eig.",
    )
    p_run.add_argument(
        "--l2ca-label-margin-debug",
        action="store_true",
        default=False,
        help="Print label slack-margin diagnostics for L2CA/L2CA-FI dual training targets.",
    )
    p_run.add_argument(
        "--l2ca-interiorize-labels",
        type=int,
        choices=[0, 1],
        default=1,
        help="Enable interiorized dual labels for certificate L2CA/L2CA-FI training (1=yes, 0=no).",
    )
    p_run.add_argument(
        "--l2ca-interior-delta",
        type=float,
        default=1e-3,
        help="Target slack margin delta for interiorized dual labels.",
    )
    p_run.add_argument(
        "--l2ca-lambda-y",
        type=float,
        default=1.0,
        help="Weight on supervised dual MSE term for L2CA/L2CA-FI training.",
    )

    p_run.add_argument(
        "--perturb-kind",
        choices=["none", "entrywise_uniform", "entrywise_gaussian", "diagonal"],
        default="entrywise_uniform",
    )
    p_run.add_argument("--A-scale", type=float, default=0.05)
    p_run.add_argument("--B-scale", type=float, default=0.02)

    p_run.add_argument(
        "--algorithms",
        nargs="+",
        default=["IPM", "L2WS", "L2A", "L2CA"],
        help="Subset of {MOSEK, SCS, IPM, L2WS, L2A, L2CA}.",
    )

    p_run.add_argument("--cert-regularize", choices=["auto", "on", "off"], default="auto")
    p_run.add_argument("--cert-eps", type=float, default=1e-6)
    p_run.add_argument("--cert-Q", choices=["identity", "diag_from_A", "custom_diag"], default="identity")
    p_run.add_argument("--cert-Q-scale", type=float, default=1.0)

    p_run.add_argument("--prestabilize", choices=["off", "shift"], default="off")
    p_run.add_argument("--prestabilize-margin", type=float, default=1e-3)

    p_run.set_defaults(func=_cmd_run)
    return parser


def _cmd_list_apps(args: argparse.Namespace) -> int:
    _ = args
    apps = list_applications()
    print("Available applications:")
    for key in sorted(apps.keys()):
        spec = apps[key]
        print(f"  - {spec.key}: {spec.description}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from .runner import ExperimentConfig, run_experiment

    systems = load_systems_from_mat(
        mat_path=args.mat,
        mat_key_A=args.mat_key_A,
        mat_key_B=args.mat_key_B,
        mat_key_Bw=args.mat_key_Bw,
        mat_key_Cz=args.mat_key_Cz,
        application=args.application,
    )
    if not systems:
        raise RuntimeError("No systems were loaded from MAT file.")

    perturb = PerturbationSpec(
        kind=str(args.perturb_kind),
        A_scale=float(args.A_scale),
        B_scale=float(args.B_scale),
        seed=int(args.seed),
    )
    cfg = ExperimentConfig(
        application=str(args.application),
        num_train=int(args.num_train),
        num_test=int(args.num_test),
        perturb=perturb,
        seed=int(args.seed),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        hidden_sizes=tuple(args.hidden_sizes),
        lr=float(args.lr),
        l2ca_tier_auto=str(args.l2ca_tier_auto),
        l2ca_tier0_fallback=str(args.l2ca_tier0_fallback),
        l2ca_lambda_obj=float(args.l2ca_lambda_obj),
        l2ca_lambda_feas=float(args.l2ca_lambda_feas),
        l2ca_feas_loss=str(args.l2ca_feas_loss),
        l2ca_feas_margin_train=float(args.l2ca_feas_margin_train),
        l2ca_label_margin_debug=bool(args.l2ca_label_margin_debug),
        l2ca_interiorize_labels=bool(int(args.l2ca_interiorize_labels)),
        l2ca_interior_delta=float(args.l2ca_interior_delta),
        l2ca_lambda_y=float(args.l2ca_lambda_y),
        algorithms=tuple(str(a) for a in args.algorithms),
        cert_regularize=str(args.cert_regularize),
        cert_eps=float(args.cert_eps),
        cert_Q=str(args.cert_Q),
        cert_Q_scale=float(args.cert_Q_scale),
        prestabilize=str(args.prestabilize),
        prestabilize_margin=float(args.prestabilize_margin),
    )

    run_experiment(systems=systems, config=cfg)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
