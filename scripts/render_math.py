"""Render LaTeX math snippets to tight-cropped PNGs via matplotlib.

Uses matplotlib's `stix` fontset so math baselines align with STIX Two
Text used for body text in the PDF. The equations set (eq1-eq9) serves
the paper body; the nomenclature symbols (sym_*) serve the Nomenclature
table so every Greek letter, subscript, and superscript is rendered in
consistent math mode rather than with HTML entities.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from PIL import Image, ImageChops

# STIX pairs visually with Charter serif body text.
rcParams["mathtext.fontset"] = "stix"

EQ_DIR = Path(__file__).resolve().parent.parent / "figures" / "math"
EQ_DIR.mkdir(parents=True, exist_ok=True)


def _trim_whitespace(path: Path, pad_px: int = 20) -> None:
    """Tight-crop a transparent-background PNG to its drawn content,
    then add a small uniform padding. matplotlib's bbox_inches='tight'
    leaves large axes-padding with axis off; this post-process fixes it.
    """
    with Image.open(path) as im:
        im = im.convert("RGBA")
        alpha = im.getchannel("A")
        bbox = alpha.getbbox()
        if bbox is None:
            return
        cropped = im.crop(bbox)
        w, h = cropped.size
        new = Image.new("RGBA", (w + 2*pad_px, h + 2*pad_px), (255, 255, 255, 0))
        new.paste(cropped, (pad_px, pad_px), cropped)
        new.save(path)


def render(name: str, tex: str, fontsize: int = 28) -> Path:
    """Render a math expression to a tight-cropped PNG at high DPI.

    Large fontsize + high DPI so that the final scaled image preserves
    stroke weight visibly matching 10pt body text in the PDF. Output is
    cropped to the drawn glyphs with a small transparent padding.
    """
    out = EQ_DIR / f"{name}.png"
    fig, ax = plt.subplots(figsize=(8, 1.2))
    ax.axis("off")
    ax.text(0.5, 0.5, f"${tex}$", fontsize=fontsize, color="black",
            ha="center", va="center", transform=ax.transAxes)
    fig.savefig(out, dpi=360, bbox_inches="tight", pad_inches=0.1,
                transparent=True)
    plt.close(fig)
    _trim_whitespace(out, pad_px=16)
    return out


# -- Display equations for the paper body ----------------------------------
EQUATIONS = {
    "eq1":  r"\nabla \cdot \mathbf{u} = 0",
    "eq2":  (r"\frac{\partial \mathbf{u}}{\partial t}"
             r" + (\mathbf{u}\cdot\nabla)\mathbf{u}"
             r" = -\frac{1}{\rho}\nabla p"
             r" + \nabla\!\cdot\!(\nu_{\mathrm{eff}}\,\nabla\mathbf{u})"
             r" + \mathbf{f}_{B}"),
    "eq3":  (r"\rho\,c_{p}\,\mathbf{u}\cdot\nabla T"
             r" = \nabla\!\cdot\!(k\,\nabla T) + \dot{q}'''"),
    "eq4":  r"\mathbf{f}_{B} = -\beta\,(\mathbf{u} - \mathbf{u}_{\mathrm{target}})",
    "eq5":  (r"\nu_{t} = \ell_{m}^{2}\,|\mathbf{S}|,"
             r"\quad |\mathbf{S}| = \sqrt{2\,\mathbf{S}\!:\!\mathbf{S}}"),
    "eq6":  (r"\mathbf{u}^{*} = \frac{\mathbf{u}^{n}"
             r" + \Delta t\,\mathbf{R}(\mathbf{u}^{n},p^{n})"
             r" + \Delta t\,\beta\,\mathbf{u}_{\mathrm{target}}}"
             r"{1 + \Delta t\,\beta}"),
    "eq7":  (r"\mathbf{u}^{*} \leftarrow \mathbf{u}^{n}"
             r" + \alpha_{u}\,(\mathbf{u}^{*} - \mathbf{u}^{n})"),
    "eq8":  r"\nabla^{2}\,\delta p = \frac{\rho}{\Delta t}\,\nabla\!\cdot\!\mathbf{u}^{*}",
    "eq9":  (r"\Delta t = \min\!\left("
             r"\mathrm{CFL}\,\frac{h}{|\mathbf{u}|_{\max}},\;"
             r"\frac{1}{4}\,\frac{h^{2}}{\nu_{\mathrm{eff},\max}}\right)"),
}

# -- Nomenclature symbols (math mode → consistent metrics) ------------------
NOMENCLATURE = [
    ("sym_u",         r"\mathbf{u}"),
    ("sym_uvw",       r"u,\ v,\ w"),
    ("sym_utarget",   r"\mathbf{u}_{\mathrm{target}}"),
    ("sym_p",         r"p,\ \delta p"),
    ("sym_T",         r"T"),
    ("sym_rho",       r"\rho"),
    ("sym_mu",        r"\mu,\ \nu"),
    ("sym_nut",       r"\nu_{t}"),
    ("sym_nueff",     r"\nu_{\mathrm{eff}}"),
    ("sym_beta",      r"\beta"),
    ("sym_k",         r"k"),
    ("sym_cp",        r"c_{p}"),
    ("sym_alpha",     r"\alpha"),
    ("sym_q",         r"\dot{q}'''"),
    ("sym_h",         r"h"),
    ("sym_dt",        r"\Delta t"),
    ("sym_alphas",    r"\alpha_{u},\ \alpha_{p}"),
    ("sym_Re",        r"\mathrm{Re}"),
    ("sym_Pr",        r"\mathrm{Pr}"),
    ("sym_Ra",        r"\mathrm{Ra}"),
    ("sym_Gr",        r"\mathrm{Gr}"),
    ("sym_GCI",       r"\mathrm{GCI}"),
]


def render_all() -> None:
    for k, v in EQUATIONS.items():
        render(k, v)
    # Symbols are shorter → smaller figsize and tight pad for inline use
    for name, tex in NOMENCLATURE:
        render(name, tex, fontsize=24)
    print(f"Rendered {len(EQUATIONS)} equations + {len(NOMENCLATURE)} nomenclature"
          f" symbols to {EQ_DIR}")


if __name__ == "__main__":
    render_all()
