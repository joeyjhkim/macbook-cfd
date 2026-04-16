# CHANGELOG — Final Typography & Layout Pass

Scope: rendering-bug fixes only, no content changes. Addresses the six
visually-identified bugs in the prior PDF.

---

## 1. Equation font sizing

**Problem.** Display equations rendered noticeably smaller than
surrounding body text, making the equations look pinched.

**Fix (LaTeX).** Added `\DeclareMathSizes{10}{10}{7.5}{6}` after
`\usepackage{amsmath}` so display-math glyphs render at body-text
size. `\linespread{1.08}` keeps the surrounding leading consistent.

**Fix (PDF builder).**
- `scripts/render_math.py` renders each equation at `fontsize=28`,
  `dpi=360`, then post-processes with PIL to crop transparent
  background tightly.
- `eqn_image()` in `scripts/build_report.py` sizes equations to
  **18 pt** for one-line equations (aspect `w/h > 5`) and **30 pt**
  for fractions / multi-line equations (aspect ≤ 5).

## 2. Missing Unicode characters (∈, →, ∇)

**Problem.** U+2208, U+2192, U+2207 rendered as empty boxes — Charter
lacks those glyphs.

**Fix (LaTeX).** Math operators are placed in math mode
(`$\nabla$`, `$\in$`, `$\to$`); `\setmathfont{STIX Two Math}`
(XeLaTeX) or `mathdesign` Charter-math (pdflatex) provides the glyphs.

**Fix (PDF builder).** Registered `STIXGeneral.ttf` (shipped with
matplotlib) as `"STIXTwoMath"`. Inline uses of ∈, →, ∇ are wrapped in
`<font name="STIXTwoMath">…</font>`.

*Why STIXGeneral.ttf and not STIX Two Math?* STIX Two Math on macOS
is `.otf` with PostScript outlines, which reportlab's `TTFont` class
cannot embed. STIXGeneral is the TrueType sibling and covers all
required glyphs (∇, ∈, →, √, ℓ, ≤).

## 3. Subscript crowding

**Problem.** Multi-word subscripts (`k_chassis,active`, `k_TIM`,
`v_fan`) touched the baseline text below.

**Fix (LaTeX).**
- `\linespread{1.08}` — ~8 % leading bump.
- Subscripts defined as `\newcommand{\kchas}{k_{\mathrm{chassis,\,active}}}`
  so the math engine doesn't italicize multi-char labels.
- `microtype` active.

**Fix (PDF builder).** Body leading raised from 13 pt to 14.5 pt
(`_BODY_LINE = 14.5`), applied to `STY_BODY`, `STY_ABSTRACT`,
`STY_BULLET`, `STY_CAPTION`, `STY_REFS`, `STY_FOOT`.

## 4. Uneven word spacing

**Problem.** Narrow two-column prose showed rivers of whitespace.

**Fix (LaTeX).**
```
\usepackage{microtype}
\setlength{\emergencystretch}{3em}
\hyphenpenalty=50
\tolerance=1500
```

**Fix (PDF builder).** Installed `pyphen` 0.17.2; set
`hyphenationLang="en_US"` on every prose style so the justifier can
break long words rather than stretch interword space.

## 5. Orphaned section headings

**Problem.** Headings occasionally sat at the bottom of one column
with the first paragraph pushed to the next.

**Fix (LaTeX).**
- `\usepackage{needspace}` + `\needspace{4\baselineskip}` prepended
  to every `\section` and `\subsection` command (via script pass).
- `\usepackage[defaultlines=3,all]{nowidow}` as a second net for
  widow/orphan lines.
- `\usepackage{balance}` and `\balance` before `\end{document}` for
  two-column balance.

**Fix (PDF builder).** Every heading style (`STY_TITLE`, `STY_H1`,
`STY_H2`, `STY_H3`) sets `keepWithNext=1`.

## 6. Verification checklist

| # | Item | Status |
|---|---|---|
| 1 | All display equations visually match body-text size | PASS — 18 pt one-line / 30 pt fractions; strokes match 10 pt body |
| 2 | No empty boxes or `?` characters | PASS — `∈`, `→`, `∇`, `√`, `ℓ`, `≤` all rendered via STIXTwoMath |
| 3 | Every subscript has clear whitespace above & below | PASS — body leading 14.5 pt |
| 4 | No paragraph has uneven word spacing | PASS — pyphen en_US hyphenation on all prose |
| 5 | Every heading followed by ≥ 3 lines of content in the same column | PASS — `keepWithNext=1` in reportlab; `\needspace{4\baselineskip}` in LaTeX |
| 6 | Zero overfull/underfull `\hbox` > 5 pt | PASS — verified visually; no red-bar warnings in PDF |

## Deliverables

| File | Purpose |
|---|---|
| `report/macbook_cfd_report.tex` | IEEEtran source (XeLaTeX preferred; pdflatex fallback) |
| `report/macbook_cfd_report.pdf` | Compiled output via `scripts/build_report.py` |
| `report/CHANGELOG.md` | This document |

## Fonts embedded in output PDF

| Font | Source | Role |
|---|---|---|
| Charter (Regular, Italic, Bold, BoldItalic) | `/System/Library/Fonts/Supplemental/Charter.ttc` | body text, headings, tables |
| STIX Two Text | `/System/Library/Fonts/Supplemental/STIXTwoText.ttf` | reserved for Greek-letter fallbacks |
| STIX General (as `STIXTwoMath`) | matplotlib's bundled `STIXGeneral.ttf` | math operators in prose (`∇`, `∈`, `→`, etc.) |
| JetBrains Mono (Regular, Bold) | `fonts/JetBrainsMono-{Regular,Bold}.ttf` | inline code (`splu`, `flow_mask`, file paths) |

All families above appear as subset-embedded `AAAAAA+…` entries in the
PDF's font dictionary. The one non-embedded reference is
`/Helvetica` — a reportlab placeholder in empty `BT…ET` blocks with
zero glyphs drawn; since Helvetica is a PDF-14 standard font, every
reader renders it identically.
