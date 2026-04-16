"""Plot helpers — keeps figure code out of the solver."""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np

# Default font sizes ~2x so figures stay legible when shrunk into
# the two-column paper layout.
plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "figure.titlesize": 24,
})

from .config import Config
from .energy import TemperatureField
from .flow import FlowField
from .geometry import Geometry


def _draw_components(ax, config: Config, *, edgecolor: str = "white", lw: float = 0.7) -> None:
    # Text is drawn in edgecolor with an opposite-tone stroke so labels
    # stay legible on any background (colormap + component fill).
    dark_text = edgecolor.lower() in {"black", "navy", "k", "#000000"}
    stroke_color = "white" if dark_text else "black"
    pe = [patheffects.withStroke(linewidth=2.5, foreground=stroke_color)]
    for comp in config.components:
        x0, x1 = (m * 1e3 for m in comp.x_range_m)
        y0, y1 = (m * 1e3 for m in comp.y_range_m)
        ax.add_patch(patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            lw=lw, edgecolor=edgecolor, facecolor="none",
        ))
        w, h = x1 - x0, y1 - y0
        label = comp.name.replace("_", " ")
        aspect = h / max(w, 1e-6)
        rotation = 90 if (aspect > 2.0 and h > 35) else 0
        base = max(h, w) if rotation else min(w, h)
        fs = min(22, max(13, base / 3.0))
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, label, color=edgecolor,
                fontsize=fs, ha="center", va="center", fontweight="bold",
                rotation=rotation, path_effects=pe)


def _draw_io(ax, config: Config) -> None:
    ix0, ix1 = (m * 1e3 for m in config.bc.inlet_x_range_m)
    iy = config.bc.inlet_y_m * 1e3
    H_mm = config.domain.height_m * 1e3
    L_mm = config.domain.length_m * 1e3
    ax.plot([ix0, ix1], [iy, iy], lw=2.5, color="deepskyblue", zorder=5)
    ax.annotate("INLET (bottom vents)", xy=((ix0 + ix1) / 2, iy - 3),
                fontsize=16, color="deepskyblue", ha="center", va="top",
                fontweight="bold", zorder=6)
    for ox0, ox1 in config.bc.outlet_bands_x_m:
        ax.plot([ox0 * 1e3, ox1 * 1e3], [H_mm, H_mm], lw=3, color="tomato", zorder=5)
    # Place the REAR EXHAUST label INSIDE the plot, below the outlet
    # line, so it can't collide with the figure title.
    ax.annotate("REAR EXHAUST", xy=(L_mm * 0.5, H_mm - 4),
                fontsize=16, color="tomato", ha="center", va="top",
                fontweight="bold", zorder=6)


def layout(config: Config, geom: Geometry, path: Path) -> None:
    L_mm, H_mm = config.domain.length_m * 1e3, config.domain.height_m * 1e3
    cmap = plt.cm.tab20(np.linspace(0, 1, len(config.components)))
    fig, ax = plt.subplots(figsize=(15, 10.5))
    ax.add_patch(patches.Rectangle((0, 0), L_mm, H_mm, lw=2,
                                    edgecolor="black", facecolor="white"))
    for ci, comp in enumerate(config.components):
        x0, x1 = (m * 1e3 for m in comp.x_range_m)
        y0, y1 = (m * 1e3 for m in comp.y_range_m)
        c = cmap[ci]
        ax.add_patch(patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            lw=1, edgecolor=c, facecolor=c, alpha=0.55,
        ))
        w, h = x1 - x0, y1 - y0
        label = comp.name.replace("_", " ")
        aspect = h / max(w, 1e-6)
        rotation = 90 if (aspect > 2.0 and h > 35) else 0
        base = max(h, w) if rotation else min(w, h)
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, label,
                fontsize=min(24, max(14, base / 2.8)),
                rotation=rotation,
                ha="center", va="center", fontweight="bold", color="black")
    _draw_io(ax, config)
    ax.axhline(config.domain.depth.zone_split_y_m * 1e3, color="gray", ls="--", lw=0.5)
    ax.set_xlim(-3, L_mm + 3); ax.set_ylim(-3, H_mm + 3)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_title("MacBook Pro 16\" Component Layout — Rear Exhaust",
                 fontsize=26, pad=18)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def temperature(config: Config, geom: Geometry, T: TemperatureField, path: Path,
                stats: dict | None = None) -> None:
    L_mm, H_mm = config.domain.length_m * 1e3, config.domain.height_m * 1e3
    fig, ax = plt.subplots(figsize=(16, 11))
    levels = np.linspace(T.T.min(), T.T.max(), 40)
    cf = ax.contourf(geom.X_c * 1e3, geom.Y_c * 1e3, T.T, levels=levels, cmap="inferno")
    fig.colorbar(cf, ax=ax, label="Temperature (°C)", pad=0.01, shrink=0.82)
    _draw_components(ax, config, edgecolor="white")
    _draw_io(ax, config)
    title = "Temperature"
    if stats:
        title += (f"  |  T_SoC={stats['soc_mean_c']:.0f}°C  "
                  f"T_bat={stats['battery_max_c']:.0f}°C  "
                  f"T_exhaust={stats['exhaust_mean_c']:.0f}°C")
    ax.set_title(title, fontsize=20, fontweight="bold", pad=18)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_xlim(0, L_mm); ax.set_ylim(0, H_mm)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def velocity(config: Config, geom: Geometry, flow: FlowField, path: Path) -> None:
    L_mm, H_mm = config.domain.length_m * 1e3, config.domain.height_m * 1e3
    fig, ax = plt.subplots(figsize=(16, 11))
    cf = ax.contourf(geom.X_c * 1e3, geom.Y_c * 1e3, flow.speed_cell, 30, cmap="viridis")
    fig.colorbar(cf, ax=ax, label="Speed (m/s)", pad=0.01, shrink=0.82)
    ax.streamplot(
        geom.x_c * 1e3, geom.y_c * 1e3,
        flow.u_cell, flow.v_cell,
        color="white", linewidth=0.8, density=2, arrowsize=0.8,
    )
    for comp in config.components:
        if not comp.is_fan:
            continue
        x0, x1 = (m * 1e3 for m in comp.x_range_m)
        y0, y1 = (m * 1e3 for m in comp.y_range_m)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        ax.annotate("", xy=(cx, y1 - 3), xytext=(cx, y0 + 3),
                    arrowprops=dict(arrowstyle="->", color="yellow", lw=2.5))
        ax.text(cx, cy, "FAN", color="yellow", fontsize=14,
                ha="center", va="center", fontweight="bold")
    _draw_components(ax, config, edgecolor="cyan")
    _draw_io(ax, config)
    ax.set_title("Velocity — Front-to-Back Airflow (Navier–Stokes)",
                 fontsize=22, fontweight="bold", pad=18)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_xlim(0, L_mm); ax.set_ylim(0, H_mm)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def vorticity(config: Config, geom: Geometry, flow: FlowField, path: Path,
              clip: float = 25.0) -> None:
    L_mm, H_mm = config.domain.length_m * 1e3, config.domain.height_m * 1e3
    fig, ax = plt.subplots(figsize=(16, 11))
    levels = np.linspace(-clip, clip, 30)
    cf = ax.contourf(geom.X_c * 1e3, geom.Y_c * 1e3,
                     np.clip(flow.vorticity_cell, -clip, clip),
                     levels=levels, cmap="RdBu_r")
    fig.colorbar(cf, ax=ax, label="Vorticity (1/s)", pad=0.01, shrink=0.82)
    _draw_components(ax, config, edgecolor="navy")
    _draw_io(ax, config)
    ax.set_title("Vorticity Field (now physical — solved from NS, not post-hoc)",
                 fontsize=22, fontweight="bold", pad=18)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_xlim(0, L_mm); ax.set_ylim(0, H_mm)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
