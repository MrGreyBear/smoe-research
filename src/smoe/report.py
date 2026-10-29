"""Turn result CSVs into booktabs LaTeX tables (no number in the paper is typed by hand)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from smoe import TABLES

TEX = TABLES / "tex"


def fmt(x, nd=3):
    if isinstance(x, (str, bool, np.bool_)):
        return str(x)
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    if isinstance(x, (int, np.integer)):
        return f"{x:,d}"
    if abs(x) >= 1e4:
        return f"{x:,.0f}"
    return f"{x:.{nd}f}"


def stars(p):
    return "^{***}" if p < 0.01 else "^{**}" if p < 0.05 else "^{*}" if p < 0.10 else ""


def _header(c: str) -> str:
    """Escape _ and % outside $...$ only, so math subscripts survive."""
    parts = str(c).split("$")
    for i in range(0, len(parts), 2):
        parts[i] = parts[i].replace("_", "\\_").replace("%", "\\%").replace("\\\\%", "\\%")
    return "$".join(parts)


def to_tex(df: pd.DataFrame, name: str, caption: str, label: str, nd=3, notes: str = "", col_fmt=None,
           size="\\small", bold_min_cols=(), bold_max_cols=()) -> str:
    import re

    caption = re.sub(r"(?<!\\)&", r"\\&", caption)
    notes = re.sub(r"(?<!\\)&", r"\\&", notes)
    cols = list(df.columns)
    col_fmt = col_fmt or ("l" + "r" * (len(cols) - 1))
    lines = [f"\\begin{{table}}[!htbp]\\centering{size}", f"\\caption{{{caption}}}\\label{{{label}}}",
             "\\begin{adjustbox}{max width=\\textwidth}", f"\\begin{{tabular}}{{{col_fmt}}}", "\\toprule",
             " & ".join(_header(c) for c in cols) + "\\\\", "\\midrule"]
    mins = {c: df[c].min() for c in bold_min_cols}
    maxs = {c: df[c].max() for c in bold_max_cols}
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            s = fmt(r[c], nd)
            if isinstance(r[c], str) and "$" not in s and "\\" not in s:
                s = s.replace("_", "\\_")
            if (c in mins and r[c] == mins[c]) or (c in maxs and r[c] == maxs[c]):
                s = f"\\textbf{{{s}}}"
            cells.append(s.replace("&", "\\&").replace("%", "\\%"))
        lines.append(" & ".join(cells) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}"]
    if notes:
        lines += ["\\par\\smallskip\\begin{minipage}{\\textwidth}\\footnotesize", notes, "\\end{minipage}"]
    lines += ["\\end{table}"]
    TEX.mkdir(parents=True, exist_ok=True)
    out = "\n".join(lines)
    (TEX / f"{name}.tex").write_text(out, encoding="utf-8")
    return out
