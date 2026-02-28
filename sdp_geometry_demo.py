#!/usr/bin/env python3
"""
Conceptual 3D visualization of L2A / L2CA / L2WS on a 2x2 SDP cone.

Cone used:
  S(x,y,z) = [[z + x, y],
              [y, z - x]]
Feasible iff S >= 0  <=>  z >= sqrt(x^2 + y^2), z >= 0
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import subprocess


def slack_matrix(p):
    x, y, z = p
    return np.array([[z + x, y], [y, z - x]], dtype=float)


def is_feasible(p, margin=0.0):
    x, y, z = p
    return bool(z >= np.hypot(x, y) + margin and z >= 0.0)


def project_interior(p, margin=1e-3):
    x, y, z = p
    r = np.hypot(x, y)
    z = max(float(z), float(r + margin), 0.0)
    return np.array([x, y, z], dtype=float)


def bisect_to_feasible(p_pred, p_anchor, iters=70, margin=1e-6):
    if not is_feasible(p_anchor, margin):
        raise ValueError("Anchor must be feasible.")
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        pm = (1.0 - mid) * p_pred + mid * p_anchor
        if is_feasible(pm, margin):
            hi = mid
        else:
            lo = mid
    p_feas = (1.0 - hi) * p_pred + hi * p_anchor
    return p_feas, hi


def line_search_feasible(
    p0,
    direction,
    bisect_iters=70,
    expand_iters=30,
    margin=1e-6,
    t_max=None,
):
    d = np.asarray(direction, dtype=float)
    nd = np.linalg.norm(d)
    if nd < 1e-14:
        return p0.copy(), 0.0
    d = d / nd

    if not is_feasible(p0, margin):
        return p0.copy(), 0.0

    t_lo = 0.0

    # Segment-capped line search (used to lift toward a specific target point).
    if t_max is not None:
        t_hi = max(0.0, float(t_max))
        if t_hi <= 1e-14:
            return p0.copy(), 0.0
        if is_feasible(p0 + t_hi * d, margin):
            return p0 + t_hi * d, t_hi
        for _ in range(bisect_iters):
            tm = 0.5 * (t_lo + t_hi)
            if is_feasible(p0 + tm * d, margin):
                t_lo = tm
            else:
                t_hi = tm
        return p0 + t_lo * d, t_lo

    t_hi = 1.0

    for _ in range(expand_iters):
        if is_feasible(p0 + t_hi * d, margin):
            t_lo = t_hi
            t_hi *= 2.0
        else:
            break

    if is_feasible(p0 + t_hi * d, margin):
        return p0 + t_lo * d, t_lo

    for _ in range(bisect_iters):
        tm = 0.5 * (t_lo + t_hi)
        if is_feasible(p0 + tm * d, margin):
            t_lo = tm
        else:
            t_hi = tm

    return p0 + t_lo * d, t_lo


def simulate_l2ws(start, target, n_steps=30, margin=0.003):
    p = project_interior(start, margin=margin)
    traj = [p.copy()]
    for k in range(n_steps - 1):
        # Predictor: move toward optimum
        alpha = 0.36 + 0.10 * (k / max(1, n_steps - 1))
        p = p + alpha * (target - p)

        # Mild decaying zig-zag in xy-plane (predictor-corrector oscillation proxy).
        vxy = target[:2] - p[:2]
        nvxy = np.linalg.norm(vxy)
        if nvxy > 1e-12:
            txy = np.array([-vxy[1], vxy[0]], dtype=float) / nvxy
            amp = 0.10 * np.exp(-0.33 * k) * (1.0 if (k % 2 == 0) else -1.0)
            p[:2] = p[:2] + amp * txy

        # Corrector: mild interior-centering (smaller than predictor pull).
        center = np.array([0.0, 0.0, p[2]], dtype=float)
        p = p + 0.05 * (center - p)

        # Keep strictly interior
        p = project_interior(p, margin=margin)
        traj.append(p.copy())
    return np.array(traj)


def main():
    plt.style.use("default")
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
            "font.size": 14,
            "axes.labelsize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )

    # Objective direction (for "optimum-like" direction)
    c = np.array([0.45, 0.15, 1.0], dtype=float)
    c = c / np.linalg.norm(c)

    z_cap = 3.2
    c_xy = c[:2]
    c_xy = c_xy / np.linalg.norm(c_xy)
    p_opt = np.array([z_cap * c_xy[0], z_cap * c_xy[1], z_cap], dtype=float)  # boundary point

    # L2A: near-optimal direction but clearly infeasible (for visible repair/lift trajectory).
    t_xy = np.array([-c_xy[1], c_xy[0]], dtype=float)
    r_l2a = 3.6
    off_l2a = 0.1
    p_l2a_xy = r_l2a * c_xy + off_l2a * t_xy
    p_l2a = np.array([p_l2a_xy[0], p_l2a_xy[1], 2.95], dtype=float)  # z < sqrt(x^2+y^2): infeasible

    # L2CA: predict -> bisection-to-feasible toward anchor -> objective lift
    p_pred = p_l2a.copy()
    p_anchor = np.array([0.55, 0.20, 2.00], dtype=float)  # robust feasible anchor
    p_repair, t_rep = bisect_to_feasible(p_pred, p_anchor)
    lift_dir = p_opt - p_repair
    l2ca_lift_fraction = 0.62  # keep L2CA improved but typically short of near-optimal IPM endpoint
    p_lift, t_lift = line_search_feasible(
        p_repair,
        lift_dir,
        t_max=float(l2ca_lift_fraction * np.linalg.norm(lift_dir)),
    )

    # L2WS: warm-started IPM-like interior trajectory (very close to boundary optimum set)
    p_ws_target = project_interior(0.9995 * p_opt, margin=0.0015)
    p_ws0 = np.array([-0.25, 0.18, 0.70], dtype=float)
    ws_path = simulate_l2ws(p_ws0, p_ws_target, n_steps=30, margin=0.003)

    # Cone surface z = sqrt(x^2 + y^2)
    theta = np.linspace(0, 2 * np.pi, 220)
    z = np.linspace(0, z_cap, 120)
    T, Z = np.meshgrid(theta, z)
    R = Z
    X = R * np.cos(T)
    Y = R * np.sin(T)

    fig = plt.figure(figsize=(12, 9), facecolor="white")
    ax = fig.add_subplot(111, projection="3d", facecolor="white")

    # Cone + wireframe
    ax.plot_surface(X, Y, Z, cmap="turbo", alpha=0.22, linewidth=0, antialiased=True)
    ax.plot_wireframe(X[::9, ::12], Y[::9, ::12], Z[::9, ::12], color="#9fb3ff", alpha=0.18, linewidth=0.5)

    # L2A
    ax.plot([0, p_l2a[0]], [0, p_l2a[1]], [0, p_l2a[2]], "--", color="#ff4d6d", linewidth=2.3, alpha=0.95)
    ax.scatter(*p_l2a, color="#ff4d6d", marker="x", s=120)

    # L2CA
    ax.plot(
        [p_pred[0], p_repair[0]],
        [p_pred[1], p_repair[1]],
        [p_pred[2], p_repair[2]],
        color="#06d6a0",
        linewidth=3.0,
        alpha=0.95,
    )
    ax.plot(
        [p_repair[0], p_lift[0]],
        [p_repair[1], p_lift[1]],
        [p_repair[2], p_lift[2]],
        color="#6c757d",
        linewidth=3.8,
        alpha=0.98,
    )
    repair_vec = p_repair - p_pred
    lift_vec = p_lift - p_repair
    ax.quiver(
        p_pred[0],
        p_pred[1],
        p_pred[2],
        repair_vec[0],
        repair_vec[1],
        repair_vec[2],
        color="#06d6a0",
        linewidth=2.0,
        arrow_length_ratio=0.10,
    )
    ax.quiver(
        p_repair[0],
        p_repair[1],
        p_repair[2],
        lift_vec[0],
        lift_vec[1],
        lift_vec[2],
        color="#6c757d",
        linewidth=2.2,
        arrow_length_ratio=0.10,
    )
    ax.scatter(*p_pred, color="#f28e2b", s=55)
    ax.scatter(*p_repair, color="#06d6a0", s=85)
    ax.scatter(*p_lift, color="#adb5bd", s=95)

    # L2WS
    ax.plot(ws_path[:, 0], ws_path[:, 1], ws_path[:, 2], color="#00d4ff", linewidth=3.0, alpha=0.95)
    ax.scatter(ws_path[0, 0], ws_path[0, 1], ws_path[0, 2], color="#00d4ff", s=80)
    ax.scatter(ws_path[-1, 0], ws_path[-1, 1], ws_path[-1, 2], color="#7ae7ff", s=100)

    # Optimum marker
    ax.scatter(*p_opt, color="#ffe66d", marker="*", s=260)

    ax.set_xlabel(r"$x\,(\mathrm{asymmetry\ mode})$")
    ax.set_ylabel(r"$y\,(\mathrm{off\!-\!diagonal\ mode})$")
    ax.set_zlabel(r"$z\,(\mathrm{trace\ mode})$", labelpad=10)

    lim = z_cap + 0.2
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(0.0, z_cap + 0.25)
    ax.view_init(elev=24, azim=-54)

    legend_items = [
        Line2D([0], [0], color="#ff4d6d", lw=2.3, ls="--", label=r"L2A: one-shot prediction"),
        Line2D([0], [0], color="#06d6a0", lw=3.0, label=r"L2CA repair"),
        Line2D([0], [0], color="#6c757d", lw=3.8, label=r"L2CA lift"),
        Line2D([0], [0], color="#00d4ff", lw=3.0, label=r"L2WS: warm-started IPM trajectory"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#ffe66d", lw=0, markersize=14, label=r"optimum direction"),
    ]
    ax.legend(handles=legend_items, loc="upper left", frameon=False)

    # Quick console diagnostics
    def score(p):
        return float(np.dot(c, p))

    print("Feasibility / objective diagnostics")
    print(f"L2A point feasible?   {is_feasible(p_l2a)}   score={score(p_l2a):.4f}")
    print(f"L2CA repaired feasible? {is_feasible(p_repair)} score={score(p_repair):.4f}")
    print(f"L2CA lifted feasible?   {is_feasible(p_lift)}   score={score(p_lift):.4f}")
    print(f"L2WS final feasible?    {is_feasible(ws_path[-1])} score={score(ws_path[-1]):.4f}")
    print(f"L2CA bisection t={t_rep:.6f}, lift t={t_lift:.6f}")

    plt.tight_layout()
    out_path = Path(__file__).resolve().parent / "sdp_geometry_demo.pdf"
    out_path_eps = Path(__file__).resolve().parent / "sdp_geometry_demo.eps"
    out_path_pdf_outlined = Path(__file__).resolve().parent / "sdp_geometry_demo_outlined.pdf"
    out_path_eps_outlined = Path(__file__).resolve().parent / "sdp_geometry_demo_outlined.eps"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    fig.savefig(out_path_eps, format="eps", bbox_inches="tight", facecolor="white")
    print(f"Saved figure to: {out_path}")
    print(f"Saved figure to: {out_path_eps}")

    # Illustrator-safe outlined exports (no external font dependency).
    # Requires Ghostscript (`gs`) available on PATH.
    try:
        subprocess.run(
            [
                "gs",
                "-dNOPAUSE",
                "-dBATCH",
                "-sDEVICE=pdfwrite",
                "-dNoOutputFonts",
                f"-sOutputFile={out_path_pdf_outlined}",
                str(out_path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        subprocess.run(
            [
                "gs",
                "-dNOPAUSE",
                "-dBATCH",
                "-sDEVICE=eps2write",
                "-dNoOutputFonts",
                f"-sOutputFile={out_path_eps_outlined}",
                str(out_path_eps),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print(f"Saved outlined figure to: {out_path_pdf_outlined}")
        print(f"Saved outlined figure to: {out_path_eps_outlined}")
    except FileNotFoundError:
        print("Ghostscript not found; skipped outlined exports.")
    except subprocess.CalledProcessError as exc:
        print("Ghostscript outline conversion failed; keeping standard PDF/EPS.")
        if exc.stderr:
            print(exc.stderr.strip())
    plt.show()


if __name__ == "__main__":
    main()
