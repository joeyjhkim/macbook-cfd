#!/usr/bin/env python3
"""Generate all analysis figures:
    1. grid_refinement.png    — 2D mesh-independence study (3 grids)
    2. convergence_2d.png     — 2D residual history
    3. convergence_3d.png     — 3D residual history
    4. comparison_2d_3d.png   — side-by-side T fields + metric bar chart
"""

from __future__ import annotations

import copy
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 2x default font sizes for legibility in two-column paper layout.
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 14,
})

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cfd import config as cfg2d, energy as en2d, flow as fl2d, geometry as geom2d, validate as val2d
from cfd3d import config3d as cfg3d, energy3d as en3d, flow3d as fl3d, geometry3d as geom3d, validate3d as val3d

OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)


def _run_2d(nx: int, ny: int, label: str):
    c = cfg2d.load("config/macbook_pro_16.yaml")
    d = replace(c.domain, nx=nx, ny=ny)
    s = replace(c.solver, log_every=999999)
    c = replace(c, domain=d, solver=s)
    g = geom2d.build(c)
    f = fl2d.solve(c, g, log=lambda _: None)
    T = en2d.solve(c, g, f)
    stats = val2d.compute_stats(c, g, T)
    print(f"  {label}: SoC={stats['soc_mean_c']:.1f}  Bat={stats['battery_max_c']:.1f}  "
          f"Palm={stats['palm_mean_c']:.1f}  Exh={stats['exhaust_mean_c']:.1f}")
    return c, g, f, T, stats


def grid_refinement():
    """Run 2D at coarse, baseline, fine grids; plot metric convergence."""
    print("=== Grid refinement study ===")
    grids = [
        (130, 90,  "Coarse 130x90"),
        (260, 180, "Baseline 260x180"),
        (520, 360, "Fine 520x360"),
    ]
    results = []
    for nx, ny, label in grids:
        _, _, _, _, stats = _run_2d(nx, ny, label)
        results.append((nx, ny, stats))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = [
        ("a", "soc_mean_c",     "SoC mean (\u00B0C)",       [80, 100]),
        ("b", "battery_max_c",  "Battery max (\u00B0C)",    [38, 45]),
        ("c", "palm_mean_c",    "Palm rest mean (\u00B0C)", [29, 36]),
        ("d", "exhaust_mean_c", "Exhaust mean (\u00B0C)",   [42, 60]),
    ]
    h_vals = [1.0 / nx for nx, _, _ in results]
    for ax, (panel, key, title, ref_range) in zip(axes.flat, metrics):
        m_vals = [s[key] for _, _, s in results]
        ax.axhspan(ref_range[0], ref_range[1], alpha=0.20, color="#2E7D32",
                   label="Published range", zorder=1)
        ax.plot(h_vals, m_vals, "o-", color="#1565C0", markersize=9, lw=2.2,
                markeredgecolor="white", markeredgewidth=1.2,
                label="Simulation", zorder=3)
        ax.set_xlabel(r"Grid spacing $1/n_x$", fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(title, fontsize=13.2, fontweight="bold")
        ax.tick_params(axis="both", which="major", labelsize=9)
        ax.legend(fontsize=10.8, loc="best")
        ax.grid(True, alpha=0.3)
        ax.text(0.04, 0.93, f"({panel})", transform=ax.transAxes,
                fontsize=15.6, fontweight="bold", family="sans-serif",
                ha="left", va="top",
                bbox=dict(facecolor="white", edgecolor="black",
                          boxstyle="round,pad=0.18", lw=0.6))
    fig.suptitle("2D Grid Refinement \u2014 Mesh Independence",
                 fontsize=16.8, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "grid_refinement.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  -> grid_refinement.png")


def _convergence_panel(flow, title: str, out_path: Path) -> None:
    MOM_COLOR = "#1565C0"
    DIV_COLOR = "#C62828"
    fig, ax1 = plt.subplots(figsize=(10, 4.6))
    iters = np.arange(1, len(flow.mom_history) + 1)
    l1, = ax1.semilogy(iters, flow.mom_history, color=MOM_COLOR, lw=1.4,
                        label=r"Momentum residual  $\max|\Delta\mathbf{u}|$  (m/s)")
    ax1.set_xlabel("Pseudo-time iteration", fontsize=13.2)
    ax1.set_ylabel(r"Momentum residual  (m s$^{-1}$)", color=MOM_COLOR, fontsize=13.2)
    ax1.tick_params(axis="y", labelcolor=MOM_COLOR, labelsize=9)
    ax1.tick_params(axis="x", labelsize=9)
    ax1.grid(True, which="both", alpha=0.25)
    ax2 = ax1.twinx()
    l2, = ax2.semilogy(iters, flow.div_history, color=DIV_COLOR, lw=1.4, alpha=0.85,
                        label=r"Interior divergence  $\|\nabla\!\cdot\!\mathbf{u}\|_{\infty}$  (s$^{-1}$)")
    ax2.set_ylabel(r"Interior divergence  (s$^{-1}$)", color=DIV_COLOR, fontsize=13.2)
    ax2.tick_params(axis="y", labelcolor=DIV_COLOR, labelsize=9)
    ax1.legend([l1, l2], [l1.get_label(), l2.get_label()],
               loc="upper right", fontsize=10.8,
               framealpha=0.92, edgecolor="gray")
    ax1.set_title(title, fontsize=14.4, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def convergence_2d():
    """Run 2D baseline and plot residual history."""
    print("=== 2D convergence ===")
    _, _, flow, _, _ = _run_2d(260, 180, "2D baseline")
    _convergence_panel(flow,
        "2D Solver Convergence \u2014 Pseudo-Transient Chorin Projection",
        OUT / "convergence_2d.png")
    print(f"  -> convergence_2d.png  ({flow.iterations} iters, "
          f"final dU={flow.final_momentum_res:.2e})")
    return flow


def convergence_3d():
    """Run 3D and plot residual history."""
    print("=== 3D convergence ===")
    c = cfg3d.load("config/macbook_pro_16_3d.yaml")
    g = geom3d.build(c)
    flow = fl3d.solve(c, g, log=lambda _: None)
    _convergence_panel(flow,
        "3D Solver Convergence \u2014 Chorin Projection, Direct LU",
        OUT / "convergence_3d.png")
    print(f"  -> convergence_3d.png  ({flow.iterations} iters, "
          f"final dU={flow.final_mom_res:.2e})")
    return c, g, flow


def comparison(c2d, g2d, T2d, stats2d, c3d, g3d, T3d, stats3d):
    """Side-by-side temperature + metric bar chart."""
    print("=== 2D vs 3D comparison ===")
    from matplotlib.patches import Patch

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])

    # -- Top row: temperature fields --
    vmin, vmax = 25.0, max(T2d.T.max(), T3d.T.max())
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.pcolormesh(g2d.x_c * 1000, g2d.y_c * 1000, T2d.T,
                          cmap="inferno", vmin=vmin, vmax=vmax, shading="nearest")
    ax1.set_title(r"$(a)$  2D temperature  $T_{\mathrm{SoC}} = "
                  f"{stats2d['soc_mean_c']:.0f}$" + r"$\,^\circ\mathrm{C}$",
                  fontsize=14.4, fontweight="bold")
    ax1.set_xlabel("x (mm)", fontsize=13.2)
    ax1.set_ylabel("y (mm)", fontsize=13.2)
    ax1.tick_params(labelsize=9)
    ax1.set_aspect("equal")
    fig.colorbar(im1, ax=ax1, label=r"T ($^\circ$C)", pad=0.02)

    ax2 = fig.add_subplot(gs[0, 1])
    soc_k = g3d.nz // 2
    for ci, comp in enumerate(c3d.components):
        if comp.name == "SoC":
            soc_k = int(round(0.5 * (comp.z_range_m[0] + comp.z_range_m[1]) / g3d.dz))
            soc_k = max(0, min(g3d.nz - 1, soc_k))
    im2 = ax2.pcolormesh(g3d.x_c * 1000, g3d.y_c * 1000, T3d.T[soc_k],
                          cmap="inferno", vmin=vmin, vmax=vmax, shading="nearest")
    ax2.set_title(r"$(b)$  3D temperature at $z = "
                  f"{g3d.z_c[soc_k]*1000:.1f}$ mm, "
                  r"$T_{\mathrm{SoC}} = " + f"{stats3d['soc_mean_c']:.0f}"
                  r"\,^\circ\mathrm{C}$",
                  fontsize=14.4, fontweight="bold")
    ax2.set_xlabel("x (mm)", fontsize=13.2)
    ax2.set_ylabel("y (mm)", fontsize=13.2)
    ax2.tick_params(labelsize=9)
    ax2.set_aspect("equal")
    fig.colorbar(im2, ax=ax2, label=r"T ($^\circ$C)", pad=0.02)

    # -- Bottom: validation bar chart --
    ax3 = fig.add_subplot(gs[1, :])
    metric_names = ["SoC mean", "Battery max", "Palm rest", "Exhaust"]
    keys = ["soc_mean_c", "battery_max_c", "palm_mean_c", "exhaust_mean_c"]
    vals_2d = [stats2d[k] for k in keys]
    vals_3d = [stats3d[k] for k in keys]
    ref_lo = [82, 35, 27, 40]
    ref_hi = [98, 41, 37, 50]

    x = np.arange(len(metric_names))
    w = 0.20

    # Published range — drawn first behind the bars
    for i in range(len(metric_names)):
        ax3.fill_between([i - 0.42, i + 0.42], ref_lo[i], ref_hi[i],
                          color="#A5D6A7", alpha=0.55, zorder=1,
                          edgecolor="#2E7D32", linewidth=1.2)

    bars_2d = ax3.bar(x - w, vals_2d, w * 1.7, color="#1565C0",
                       edgecolor="white", linewidth=1.2,
                       label="2D simulation", zorder=3)
    bars_3d = ax3.bar(x + w, vals_3d, w * 1.7, color="#2E7D32",
                       edgecolor="white", linewidth=1.2,
                       label="3D simulation", zorder=3)

    for bar, val in zip(bars_2d, vals_2d):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 1.0,
                 f"{val:.1f}", ha="center", va="bottom",
                 fontsize=10.8, fontweight="bold", color="#0D47A1")
    for bar, val in zip(bars_3d, vals_3d):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 1.0,
                 f"{val:.1f}", ha="center", va="bottom",
                 fontsize=10.8, fontweight="bold", color="#1B5E20")

    # Custom legend with three distinct entries
    handles = [
        Patch(facecolor="#1565C0", edgecolor="white", label="2D simulation"),
        Patch(facecolor="#2E7D32", edgecolor="white", label="3D simulation"),
        Patch(facecolor="#A5D6A7", edgecolor="#2E7D32",
              label="Published range"),
    ]
    ax3.legend(handles=handles, fontsize=13.2, loc="upper right",
               framealpha=0.95, edgecolor="gray")

    ax3.set_xticks(x)
    ax3.set_xticklabels(metric_names, fontsize=13.2)
    ax3.tick_params(axis="y", labelsize=10)
    ax3.set_ylabel(r"Temperature ($^\circ$C)", fontsize=13.2)
    ax3.set_title(r"$(c)$  Validation metrics vs published data",
                  fontsize=14.4, fontweight="bold")
    ax3.grid(True, axis="y", alpha=0.3, zorder=0)
    ax3.set_ylim(20, 110)

    fig.suptitle("MacBook Pro 16\u2033 CFD \u2014 2D vs 3D comparison",
                 fontsize=16.8, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "comparison_2d_3d.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  -> comparison_2d_3d.png")


def main():
    grid_refinement()

    flow_2d = convergence_2d()
    c3d, g3d, flow_3d = convergence_3d()

    # Re-run 2D baseline for comparison data
    c2d, g2d, _, T2d, stats2d = _run_2d(260, 180, "2D (comparison)")
    T3d = en3d.solve(c3d, g3d, flow_3d)
    stats3d = val3d.compute_stats(c3d, g3d, T3d)
    comparison(c2d, g2d, T2d, stats2d, c3d, g3d, T3d, stats3d)

    print("\nAll analysis figures written.")


if __name__ == "__main__":
    main()
