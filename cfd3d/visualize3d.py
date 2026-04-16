"""3D result visualization: mid-height slice + vertical cross-sections.

Three figures:
    layout3d.png        component bounding boxes (mid-z slice + side view)
    temperature3d.png   T on mid-z slice and mid-y slice
    velocity3d.png      speed + streamlines on mid-z and mid-y slices
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 2x default font sizes for legibility in two-column paper layout.
plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "figure.titlesize": 24,
})

from .config3d import Config3D
from .energy3d import TemperatureField3D
from .flow3d import FlowField3D
from .geometry3d import Geometry3D


def _mid_indices(geom: Geometry3D) -> tuple[int, int, int]:
    return geom.nz // 2, geom.ny // 2, geom.nx // 2


def layout(config: Config3D, geom: Geometry3D, out: Path) -> None:
    kz, jy, ix = _mid_indices(geom)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax1, ax2 = axes

    # Top view: mid-z slice
    ax1.set_title(f"Top view (z = {geom.z_c[kz]*1000:.1f} mm)")
    ax1.set_xlabel("x (mm)"); ax1.set_ylabel("y (mm)")
    ax1.set_xlim(0, config.domain.length_x_m * 1000)
    ax1.set_ylim(0, config.domain.length_y_m * 1000)
    ax1.set_aspect("equal")
    for c in config.components:
        x0, x1 = c.x_range_m; y0, y1 = c.y_range_m; z0, z1 = c.z_range_m
        if z0 > geom.z_c[kz] or z1 < geom.z_c[kz]:
            continue
        color = "orange" if c.is_fan else ("lightblue" if c.q_watts == 0 else "salmon")
        ax1.add_patch(plt.Rectangle((x0*1000, y0*1000), (x1-x0)*1000, (y1-y0)*1000,
                                     facecolor=color, edgecolor="k", alpha=0.6))
        ax1.text(0.5*(x0+x1)*1000, 0.5*(y0+y1)*1000, c.name,
                 ha="center", va="center", fontsize=12)
    # Inlet rectangle in faint blue
    ix0, ix1 = config.bc.inlet.x_range_m
    iy0, iy1 = config.bc.inlet.y_range_m
    ax1.add_patch(plt.Rectangle((ix0*1000, iy0*1000), (ix1-ix0)*1000, (iy1-iy0)*1000,
                                 facecolor="none", edgecolor="cyan", linestyle="--", lw=1.5))

    # Side view: mid-y slice (y through centerline)
    ax2.set_title(f"Side view (y = {geom.y_c[jy]*1000:.1f} mm)")
    ax2.set_xlabel("x (mm)"); ax2.set_ylabel("z (mm)")
    ax2.set_xlim(0, config.domain.length_x_m * 1000)
    ax2.set_ylim(0, config.domain.length_z_m * 1000)
    for c in config.components:
        x0, x1 = c.x_range_m; y0, y1 = c.y_range_m; z0, z1 = c.z_range_m
        if y0 > geom.y_c[jy] or y1 < geom.y_c[jy]:
            continue
        color = "orange" if c.is_fan else ("lightblue" if c.q_watts == 0 else "salmon")
        ax2.add_patch(plt.Rectangle((x0*1000, z0*1000), (x1-x0)*1000, (z1-z0)*1000,
                                     facecolor=color, edgecolor="k", alpha=0.6))

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def temperature(config: Config3D, geom: Geometry3D, T: TemperatureField3D,
                out: Path, stats: dict | None = None) -> None:
    kz, jy, _ = _mid_indices(geom)
    # Try to find the SoC z for a more informative slice.
    for ci, c in enumerate(config.components):
        if c.name == "SoC":
            kz = int(round(0.5 * (c.z_range_m[0] + c.z_range_m[1]) / geom.dz))
            kz = max(0, min(geom.nz - 1, kz))
            break

    x_mm = geom.x_c * 1000
    y_mm = geom.y_c * 1000
    z_mm = geom.z_c * 1000

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax1, ax2 = axes
    vmin = float(T.T.min()); vmax = float(T.T.max())
    cmap = "inferno"

    im1 = ax1.pcolormesh(x_mm, y_mm, T.T[kz], cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest")
    ax1.set_title(f"T at z = {z_mm[kz]:.1f} mm (SoC plane)")
    ax1.set_xlabel("x (mm)"); ax1.set_ylabel("y (mm)"); ax1.set_aspect("equal")
    fig.colorbar(im1, ax=ax1, label="T (°C)")

    im2 = ax2.pcolormesh(x_mm, z_mm, T.T[:, jy, :], cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest")
    ax2.set_title(f"T at y = {y_mm[jy]:.1f} mm (mid cross-section)")
    ax2.set_xlabel("x (mm)"); ax2.set_ylabel("z (mm)")
    fig.colorbar(im2, ax=ax2, label="T (°C)")

    if stats is not None:
        fig.suptitle(
            f"T_SoC={stats.get('soc_mean_c', float('nan')):.0f}°C  "
            f"T_bat_max={stats.get('battery_max_c', float('nan')):.0f}°C  "
            f"T_exhaust={stats.get('exhaust_mean_c', float('nan')):.0f}°C",
            fontsize=24,
        )

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def velocity(config: Config3D, geom: Geometry3D, flow: FlowField3D, out: Path) -> None:
    kz, jy, _ = _mid_indices(geom)
    x_mm = geom.x_c * 1000; y_mm = geom.y_c * 1000; z_mm = geom.z_c * 1000

    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    ax1, ax2 = axes

    speed_top = flow.speed_cell[kz]
    vmax = float(flow.speed_cell.max())
    im1 = ax1.pcolormesh(x_mm, y_mm, speed_top, cmap="viridis",
                         vmin=0, vmax=vmax, shading="nearest")
    ax1.streamplot(x_mm, y_mm, flow.u_cell[kz], flow.v_cell[kz],
                   density=1.8, color="white", linewidth=1.2,
                   arrowsize=2.2)
    ax1.set_title(f"Speed at z = {z_mm[kz]:.1f} mm", pad=12)
    ax1.set_xlabel("x (mm)"); ax1.set_ylabel("y (mm)"); ax1.set_aspect("equal")
    cb1 = fig.colorbar(im1, ax=ax1, label="|V| (m/s)")
    cb1.ax.tick_params(labelsize=18)

    speed_side = flow.speed_cell[:, jy, :]
    im2 = ax2.pcolormesh(x_mm, z_mm, speed_side, cmap="viridis",
                         vmin=0, vmax=vmax, shading="nearest")
    ax2.streamplot(x_mm, z_mm, flow.u_cell[:, jy, :], flow.w_cell[:, jy, :],
                   density=1.8, color="white", linewidth=1.2,
                   arrowsize=2.2)
    ax2.set_title(f"Speed at y = {y_mm[jy]:.1f} mm", pad=12)
    ax2.set_xlabel("x (mm)"); ax2.set_ylabel("z (mm)")
    cb2 = fig.colorbar(im2, ax=ax2, label="|V| (m/s)")
    cb2.ax.tick_params(labelsize=18)

    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
