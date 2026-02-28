#!/usr/bin/env python3
"""
Cleaner 3D conceptual visualization of L2A / L2CA / L2WS on a 2x2 SDP cone.

Feasible set:
  S(x,y,z) = [[z + x, y],
              [y, z - x]] >= 0
  <=> z >= sqrt(x^2 + y^2), z >= 0
"""

from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def is_feasible(p, margin=0.0):
    x, y, z = p
    return bool(z >= np.hypot(x, y) + margin and z >= 0.0)


def project_interior(p, margin=1e-3):
    x, y, z = p
    r = np.hypot(x, y)
    return np.array([x, y, max(float(z), float(r + margin), 0.0)], dtype=float)


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
    return (1.0 - hi) * p_pred + hi * p_anchor, hi


def line_search_feasible(p0, direction, t_max, bisect_iters=70, margin=1e-6):
    d = np.asarray(direction, dtype=float)
    nd = np.linalg.norm(d)
    if nd < 1e-14:
        return p0.copy(), 0.0
    d = d / nd
    t_lo, t_hi = 0.0, max(0.0, float(t_max))
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


def simulate_l2ws(start, target, n_steps=28, margin=0.003):
    p = project_interior(start, margin=margin)
    traj = [p.copy()]
    for k in range(n_steps - 1):
        alpha = 0.35 + 0.10 * (k / max(1, n_steps - 1))
        p = p + alpha * (target - p)

        vxy = target[:2] - p[:2]
        nvxy = np.linalg.norm(vxy)
        if nvxy > 1e-12:
            txy = np.array([-vxy[1], vxy[0]], dtype=float) / nvxy
            amp = 0.09 * np.exp(-0.30 * k) * (1.0 if (k % 2 == 0) else -1.0)
            p[:2] = p[:2] + amp * txy

        center = np.array([0.0, 0.0, p[2]], dtype=float)
        p = p + 0.045 * (center - p)
        p = project_interior(p, margin=margin)
        traj.append(p.copy())
    return np.array(traj)


def export_outlined(pdf_path: Path, eps_path: Path):
    pdf_out = pdf_path.with_name(pdf_path.stem + "_outlined.pdf")
    eps_out = eps_path.with_name(eps_path.stem + "_outlined.eps")
    try:
        subprocess.run(
            [
                "gs",
                "-dNOPAUSE",
                "-dBATCH",
                "-sDEVICE=pdfwrite",
                "-dNoOutputFonts",
                f"-sOutputFile={pdf_out}",
                str(pdf_path),
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
                f"-sOutputFile={eps_out}",
                str(eps_path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print(f"Saved outlined figure to: {pdf_out}")
        print(f"Saved outlined figure to: {eps_out}")
    except FileNotFoundError:
        print("Ghostscript not found; skipped outlined exports.")
    except subprocess.CalledProcessError as exc:
        print("Ghostscript outline conversion failed; keeping standard PDF/EPS.")
        if exc.stderr:
            print(exc.stderr.strip())


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

    # Geometry + objective direction
    z_cap = 3.2
    c = np.array([0.45, 0.15, 1.0], dtype=float)
    c = c / np.linalg.norm(c)
    c_xy = c[:2] / np.linalg.norm(c[:2])
    t_xy = np.array([-c_xy[1], c_xy[0]], dtype=float)
    p_opt = np.array([z_cap * c_xy[0], z_cap * c_xy[1], z_cap], dtype=float)

    # Trajectory points (easy to tweak)
    p_pred = np.array([2.76, 1.16, 2.55], dtype=float)  # L2A prediction (infeasible)
    p_anchor = np.array([0.55, 0.20, 2.00], dtype=float)  # feasible training anchor
    p_repair, t_rep = bisect_to_feasible(p_pred, p_anchor)
    lift_dir = p_opt - p_repair
    p_lift, t_lift = line_search_feasible(
        p_repair,
        lift_dir,
        t_max=float(0.62 * np.linalg.norm(lift_dir)),
    )

    p_ws0 = np.array([-0.25, 0.18, 0.70], dtype=float)
    p_ws_target = project_interior(0.9995 * p_opt, margin=0.0015)
    ws_path = simulate_l2ws(p_ws0, p_ws_target, n_steps=28, margin=0.003)

    # Cone coordinates
    theta = np.linspace(0, 2 * np.pi, 180)
    zz = np.linspace(0.0, z_cap, 60)
    T, Z = np.meshgrid(theta, zz)
    X = Z * np.cos(T)
    Y = Z * np.sin(T)

    fig = plt.figure(figsize=(11, 8.2), facecolor="white")
    ax = fig.add_subplot(111, projection="3d", facecolor="white")

    # Clean "glass cone": very light fill + sparse rings + generators.
    ax.plot_surface(
        X,
        Y,
        Z,
        color="#90caf9",
        alpha=0.10,
        linewidth=0,
        antialiased=True,
        shade=False,
    )
    ring_levels = np.linspace(0.35, z_cap, 8)
    for z0 in ring_levels:
        xr = z0 * np.cos(theta)
        yr = z0 * np.sin(theta)
        ax.plot(xr, yr, z0 * np.ones_like(theta), color="#78909c", lw=1.1, alpha=0.65)
    for ang in np.linspace(-np.pi, np.pi, 10, endpoint=False):
        ax.plot(
            [0.0, z_cap * np.cos(ang)],
            [0.0, z_cap * np.sin(ang)],
            [0.0, z_cap],
            color="#90a4ae",
            lw=0.9,
            alpha=0.45,
        )

    # Subtle trajectory shadows on z=0 plane (adds depth without clutter).
    ax.plot([0, p_pred[0]], [0, p_pred[1]], [0, 0], "--", color="#f8bbd0", lw=1.3, alpha=0.7)
    ax.plot(
        [p_pred[0], p_repair[0], p_lift[0]],
        [p_pred[1], p_repair[1], p_lift[1]],
        [0, 0, 0],
        color="#cfd8dc",
        lw=1.3,
        alpha=0.55,
    )
    ax.plot(ws_path[:, 0], ws_path[:, 1], 0 * ws_path[:, 0], color="#b3e5fc", lw=1.2, alpha=0.55)

    # L2A
    ax.plot([0, p_pred[0]], [0, p_pred[1]], [0, p_pred[2]], "--", color="#e63946", linewidth=2.5, alpha=0.95)
    ax.scatter(*p_pred, color="#e63946", marker="x", s=120)

    # L2CA: repair + lift
    ax.plot(
        [p_pred[0], p_repair[0]],
        [p_pred[1], p_repair[1]],
        [p_pred[2], p_repair[2]],
        color="#00a896",
        linewidth=3.0,
        alpha=0.95,
    )
    ax.plot(
        [p_repair[0], p_lift[0]],
        [p_repair[1], p_lift[1]],
        [p_repair[2], p_lift[2]],
        color="#6c757d",
        linewidth=3.6,
        alpha=0.98,
    )
    rv = p_repair - p_pred
    lv = p_lift - p_repair
    ax.quiver(p_pred[0], p_pred[1], p_pred[2], rv[0], rv[1], rv[2], color="#00a896", linewidth=2.0, arrow_length_ratio=0.10)
    ax.quiver(
        p_repair[0],
        p_repair[1],
        p_repair[2],
        lv[0],
        lv[1],
        lv[2],
        color="#6c757d",
        linewidth=2.1,
        arrow_length_ratio=0.10,
    )
    ax.scatter(*p_repair, color="#00a896", s=80)
    ax.scatter(*p_lift, color="#adb5bd", s=92)

    # L2WS
    ax.plot(ws_path[:, 0], ws_path[:, 1], ws_path[:, 2], color="#0077b6", linewidth=3.0, alpha=0.97)
    ax.scatter(ws_path[0, 0], ws_path[0, 1], ws_path[0, 2], color="#0077b6", s=72)
    ax.scatter(ws_path[-1, 0], ws_path[-1, 1], ws_path[-1, 2], color="#48cae4", s=98)

    # Optimum marker
    ax.scatter(*p_opt, color="#ffb703", marker="*", s=250, zorder=6)

    # Axes / view
    ax.set_xlabel(r"$x\,(\mathrm{asymmetry\ mode})$")
    ax.set_ylabel(r"$y\,(\mathrm{off\!-\!diagonal\ mode})$")
    ax.set_zlabel(r"$z\,(\mathrm{trace\ mode})$", labelpad=10)
    lim = z_cap + 0.18
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(0.0, z_cap + 0.22)
    ax.view_init(elev=23, azim=-57)
    try:
        ax.set_box_aspect((1, 1, 0.80))
    except Exception:
        pass
    ax.grid(False)

    legend_items = [
        Line2D([0], [0], color="#e63946", lw=2.5, ls="--", label=r"L2A: one-shot (often infeasible)"),
        Line2D([0], [0], color="#00a896", lw=3.0, label=r"L2CA repair"),
        Line2D([0], [0], color="#6c757d", lw=3.6, label=r"L2CA lift"),
        Line2D([0], [0], color="#0077b6", lw=3.0, label=r"L2WS: warm-started IPM trajectory"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#ffb703", lw=0, markersize=14, label=r"optimum direction"),
    ]
    ax.legend(handles=legend_items, loc="upper left", frameon=False)

    # Console diagnostics
    score = lambda p: float(np.dot(c, p))
    print("Feasibility / objective diagnostics")
    print(f"L2A point feasible?     {is_feasible(p_pred)}   score={score(p_pred):.4f}")
    print(f"L2CA repaired feasible? {is_feasible(p_repair)} score={score(p_repair):.4f}")
    print(f"L2CA lifted feasible?   {is_feasible(p_lift)}   score={score(p_lift):.4f}")
    print(f"L2WS final feasible?    {is_feasible(ws_path[-1])} score={score(ws_path[-1]):.4f}")
    print(f"L2CA bisection t={t_rep:.6f}, lift t={t_lift:.6f}")

    plt.tight_layout()
    base = Path(__file__).resolve().parent / "sdp_geometry_demo2"
    out_pdf = base.with_suffix(".pdf")
    out_eps = base.with_suffix(".eps")
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_eps, format="eps", bbox_inches="tight", facecolor="white")
    print(f"Saved figure to: {out_pdf}")
    print(f"Saved figure to: {out_eps}")
    export_outlined(out_pdf, out_eps)
    plt.show()


if __name__ == "__main__":
    main()
