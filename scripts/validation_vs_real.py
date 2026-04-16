#!/usr/bin/env python3
"""Generate validation_vs_real.png — simulation vs published measurements."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)

METRICS = ["SoC / CPU die", "Battery max", "Palm rest surface", "Exhaust air"]
REAL_LO = [82, 35, 27, 40]
REAL_HI = [98, 41, 37, 50]
REAL_SRCS = [
    "Notebookcheck, AppleInsider,\nTom's HW, LaptopMedia",
    "MacRumors, Apple Community\n(coconutBattery)",
    "Notebookcheck (9-zone),\nTom's HW (thermocouple)",
    "Engineering estimate",
]
SIM_2D = [95.7, 39.3, 33.6, 43.4]
SIM_3D = [80.8, 34.5, 34.1, 39.8]


def main():
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(METRICS))
    w = 0.22

    # Published range (drawn first, behind bars)
    for i in range(len(METRICS)):
        ax.fill_between([i - 0.42, i + 0.42], REAL_LO[i], REAL_HI[i],
                         color="#A5D6A7", alpha=0.55, zorder=1,
                         edgecolor="#2E7D32", linewidth=1.2)

    bars_2d = ax.bar(x - w, SIM_2D, w * 1.6, color="#1565C0",
                      edgecolor="white", linewidth=1.2,
                      label="2D simulation", zorder=3)
    bars_3d = ax.bar(x + w, SIM_3D, w * 1.6, color="#2E7D32",
                      edgecolor="white", linewidth=1.2,
                      label="3D simulation", zorder=3)

    for bar, val in zip(bars_2d, SIM_2D):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.2,
                f"{val:.1f}", ha="center", va="bottom",
                fontsize=20, fontweight="bold", color="#0D47A1")
    for bar, val in zip(bars_3d, SIM_3D):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.2,
                f"{val:.1f}", ha="center", va="bottom",
                fontsize=20, fontweight="bold", color="#1B5E20")

    # Source labels on the right edge of each band
    for i, src in enumerate(REAL_SRCS):
        mid = (REAL_LO[i] + REAL_HI[i]) / 2
        ax.text(i + 0.44, mid,
                f"  {REAL_LO[i]}\u2013{REAL_HI[i]}\u00B0C\n  {src}",
                fontsize=15, color="dimgray", va="center", ha="left")

    handles = [
        Patch(facecolor="#1565C0", edgecolor="white", label="2D simulation"),
        Patch(facecolor="#2E7D32", edgecolor="white", label="3D simulation"),
        Patch(facecolor="#A5D6A7", edgecolor="#2E7D32",
              label="Published range"),
    ]
    ax.legend(handles=handles, fontsize=22, loc="upper left",
              framealpha=0.95, edgecolor="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(METRICS, fontsize=24)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_ylabel(r"Temperature ($^\circ$C)", fontsize=24)
    ax.set_ylim(20, 112)
    ax.grid(True, axis="y", alpha=0.25, zorder=0)
    ax.set_title("CFD simulation vs published MacBook Pro 16\u2033 thermal measurements",
                 fontsize=28, fontweight="bold", pad=10)

    fig.text(0.5, 0.01,
             "Green bands: published measurement range from Notebookcheck, Tom's "
             "Hardware, AppleInsider, LaptopMedia, and user-reported data "
             "(sustained CPU load).",
             ha="center", fontsize=16, color="dimgray")

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(OUT / "validation_vs_real.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("-> validation_vs_real.png")


if __name__ == "__main__":
    main()
