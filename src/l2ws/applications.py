"""Application registry for the user-facing L2WS toolbox runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict
import warnings

import numpy as np

from .data import (
    LTISystem,
    build_h2_norm_sdp,
    build_hinf_norm_sdp,
    build_l2_gain_sdp,
    build_lyapunov_regularized_sdp,
)
from .problem import SDPInstance


@dataclass(frozen=True)
class ApplicationSpec:
    key: str
    description: str
    needs_control_B: bool
    needs_disturbance_Bw: bool
    needs_output_Cz: bool
    builder: Callable[..., SDPInstance]
    feature_fn: Callable[[LTISystem], np.ndarray]


_DEFAULT_IO_WARNED: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key in _DEFAULT_IO_WARNED:
        return
    _DEFAULT_IO_WARNED.add(key)
    warnings.warn(message, RuntimeWarning, stacklevel=3)


def _to_array_or_none(x: object | None) -> np.ndarray | None:
    if x is None:
        return None
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return None
    return arr


def ensure_system_io(system: LTISystem, app_key: str, warn_token: str | None = None) -> LTISystem:
    """Ensure required system matrices exist for the selected application."""
    A = np.asarray(system.A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("system.A must be square.")
    n = A.shape[0]

    Bw = _to_array_or_none(getattr(system, "Bw", None))
    Cz = _to_array_or_none(getattr(system, "Cz", None))

    needs_io = str(app_key).strip().lower() in {"l2_gain", "h2_norm", "hinf_norm"}
    missing: list[str] = []
    if needs_io:
        if Bw is None:
            Bw = np.eye(n, dtype=float)
            missing.append("Bw")
        if Cz is None:
            Cz = np.eye(n, dtype=float)
            missing.append("Cz")
        if missing:
            token = warn_token or f"{app_key}:{id(system)}"
            _warn_once(
                f"{token}:{','.join(missing)}",
                (
                    f"{app_key}: missing {', '.join(missing)}; "
                    "defaulting to identity matrices."
                ),
            )
    else:
        # Keep LTISystem well-formed even when application does not use Bw/Cz.
        if Bw is None:
            Bw = np.eye(n, dtype=float)
        if Cz is None:
            Cz = np.eye(n, dtype=float)

    return LTISystem(A=A.copy(), Bw=Bw.copy(), Cz=Cz.copy())


def _default_feature_fn(system: LTISystem) -> np.ndarray:
    """Default features: vec(A) plus vec(Bw) when available."""
    A = np.asarray(system.A, dtype=float).reshape(-1)
    Bw = getattr(system, "Bw", None)
    if Bw is None:
        return A.astype(float)
    Bw_arr = np.asarray(Bw, dtype=float)
    if Bw_arr.size == 0:
        return A.astype(float)
    return np.concatenate([A, Bw_arr.reshape(-1)]).astype(float)


def _build_l2_gain(system: LTISystem, **kwargs) -> SDPInstance:
    sys_io = ensure_system_io(system, "l2_gain", warn_token=kwargs.get("_warn_token"))
    return build_l2_gain_sdp(sys_io.A, sys_io.Bw, sys_io.Cz, name=kwargs.get("name", "l2_gain"))


def _build_h2_norm(system: LTISystem, **kwargs) -> SDPInstance:
    sys_io = ensure_system_io(system, "h2_norm", warn_token=kwargs.get("_warn_token"))
    return build_h2_norm_sdp(sys_io.A, sys_io.Bw, sys_io.Cz, name=kwargs.get("name", "h2_norm"))


def _build_lyapunov_reg(system: LTISystem, **kwargs) -> SDPInstance:
    return build_lyapunov_regularized_sdp(
        system.A,
        system.Bw,
        system.Cz,
        name=kwargs.get("name", "lyapunov_reg"),
        Q_mode=str(kwargs.get("cert_Q", "identity")),
        Q_scale=float(kwargs.get("cert_Q_scale", 1.0)),
        eps=float(kwargs.get("cert_eps", 1e-6)),
    )


def _build_hinf_norm(system: LTISystem, **kwargs) -> SDPInstance:
    sys_io = ensure_system_io(system, "hinf_norm", warn_token=kwargs.get("_warn_token"))
    return build_hinf_norm_sdp(sys_io, name=kwargs.get("name", "hinf_norm"))


_REGISTRY: Dict[str, ApplicationSpec] = {
    "l2_gain": ApplicationSpec(
        key="l2_gain",
        description="Continuous-time L2-gain SDP",
        needs_control_B=False,
        needs_disturbance_Bw=True,
        needs_output_Cz=True,
        builder=_build_l2_gain,
        feature_fn=_default_feature_fn,
    ),
    "h2_norm": ApplicationSpec(
        key="h2_norm",
        description="Continuous-time H2 norm SDP",
        needs_control_B=False,
        needs_disturbance_Bw=True,
        needs_output_Cz=True,
        builder=_build_h2_norm,
        feature_fn=_default_feature_fn,
    ),
    "lyapunov_reg": ApplicationSpec(
        key="lyapunov_reg",
        description="Regularized Lyapunov certificate SDP",
        needs_control_B=False,
        needs_disturbance_Bw=False,
        needs_output_Cz=False,
        builder=_build_lyapunov_reg,
        feature_fn=_default_feature_fn,
    ),
    "hinf_norm": ApplicationSpec(
        key="hinf_norm",
        description="Continuous-time H-infinity norm SDP",
        needs_control_B=False,
        needs_disturbance_Bw=True,
        needs_output_Cz=True,
        builder=_build_hinf_norm,
        feature_fn=_default_feature_fn,
    ),
}


def get_application_spec(key: str) -> ApplicationSpec:
    k = str(key).strip().lower()
    if k not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(f"Unknown application '{key}'. Available: {available}")
    return _REGISTRY[k]


def list_applications() -> Dict[str, ApplicationSpec]:
    return dict(_REGISTRY)


__all__ = [
    "ApplicationSpec",
    "ensure_system_io",
    "get_application_spec",
    "list_applications",
]
