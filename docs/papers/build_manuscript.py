"""Build the manuscript (.docx) from the graded benchmark and the committed analysis.

Run from the repository root:  python docs/papers/build_manuscript.py [output.docx]
Every number in the text is computed here from data_collection/verified_benchmark.csv
through docs/papers/benchmark_analysis.py, so the manuscript cannot drift from the data.
"""
import html
import json
import math
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from benchmark_analysis import GROUPS, cutoff, load, predict, regression, slab_se, summary  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "EMI_Shielding_paper_revised.docx"
EQ_DIR = HERE / "_equations"
EQ_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------- data
bench = load()
BY = {b["id"]: b for b in bench}
PRIM = [b for b in bench if b["quality"] != "C"]
GRADE_C = [b for b in bench if b["quality"] == "C"]
N, NP = len(bench), len({b["paper"] for b in bench})
NPRIM, NPPRIM = len(PRIM), len({b["paper"] for b in PRIM})


def sg(x, nd=1):  # signed value with a true minus sign
    return f"{x:+.{nd}f}".replace("-", "−")


def ci(s):
    return f"{sg(s['ci_lo'])} to {sg(s['ci_hi'])}"


def dres(i):
    return sg(BY[i]["res"])


S_ALL, S_PRIM = summary(bench), summary(PRIM)
HIGH = [b for b in PRIM if b["sigma"] >= 1e4]
MID = [b for b in PRIM if 1 <= b["sigma"] < 1e4]
LOW = [b for b in PRIM if b["sigma"] < 1]
S_HIGH, S_MID, S_LOW = summary(HIGH), summary(MID), summary(LOW)
S_HIGH_FREE, S_HIGH_WG = summary(HIGH, "res_free"), summary(HIGH, "res_all_wg")
S_HIGH_ALL = summary([b for b in bench if b["sigma"] >= 1e4])
S_HIGH_A = summary([b for b in bench if b["sigma"] >= 1e4 and b["quality"] == "A"])
G = [summary([b for b in PRIM if b["group"] == i]) for i in range(len(GROUPS))]
REG = regression(PRIM)
HIGH_PAPERS = sorted({b["paper"] for b in HIGH})
PM = [float(np.median([b["res"] for b in HIGH if b["paper"] == p])) for p in HIGH_PAPERS]
WG = [b for b in bench if b["fixture"] == "waveguide"]
WG_SHIFT = [b["slab"] - b["slab_free"] for b in WG]
FIX = {k: sum(b["fixture"] == k for b in bench) for k in ("waveguide", "coaxial", "unspecified")}
OUTL = [b for b in PRIM if abs(b["res"]) > 10]
OUT_POS, OUT_NEG = [b for b in OUTL if b["res"] > 0], [b for b in OUTL if b["res"] < 0]
NUMW = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
        11: "eleven", 12: "twelve", 13: "thirteen"}
QUAL = {q: sum(b["quality"] == q for b in bench) for q in "ABC"}
SE90 = sum(b["se"] >= 90 for b in bench)
YEARS = None  # filled after references are formatted


def z_te(f):
    """TE10 wave impedance of the standard waveguide at f, ohm."""
    return 376.730 / math.sqrt(1 - (cutoff(f) / f) ** 2)


def fsig2(v):
    return f"{v:.3g}" if 0.01 <= v < 1e3 else sci(v)


def sci(v, nd=1):
    m, e = f"{v:.{nd}e}".split("e")
    return f"{m} × 10^{{{int(e)}}}".replace("-", "−")


# ---------------------------------------------------------------- references
FOUND = {
    "schelkunoff": "S.A. Schelkunoff, The electromagnetic theory of coaxial transmission lines and cylindrical shields, Bell Syst. Tech. J. 13 (1934) 532–579. https://doi.org/10.1002/j.1538-7305.1934.tb00679.x",
    "schulz": "R.B. Schulz, V.C. Plantz, D.R. Brush, Shielding theory and practice, IEEE Trans. Electromagn. Compat. 30 (1988) 187–201. https://doi.org/10.1109/15.3297",
    "ott": "H.W. Ott, Electromagnetic Compatibility Engineering, Wiley, Hoboken, 2009. https://doi.org/10.1002/9780470508510",
    "celozzi": "S. Celozzi, R. Araneo, G. Lovat, Electromagnetic Shielding, Wiley, Hoboken, 2008. https://doi.org/10.1002/9780470268483",
    "astm": "ASTM D4935-18, Standard test method for measuring the electromagnetic shielding effectiveness of planar materials, ASTM International, West Conshohocken, PA, 2018.",
}
DOI_KEYS = {
    "simon": "10.1080/03602558108067695", "colaneri": "10.1109/19.137363", "joo": "10.1063/1.112717",
    "alsaleh": "10.1016/j.carbon.2009.02.030", "yun": "10.1002/adma.201906769", "iqbalrev": "10.1002/adfm.202000883",
    "chung": "10.1016/j.matchemphys.2020.123587", "peng": "10.1063/5.0075019", "moucka": "10.3390/polym11101603",
    "wilson": "10.1109/15.3302", "ieee2715": "10.1109/MEMC.2023.10201434", "casey": "10.1109/15.3309",
    "bergman": "10.1103/PhysRevLett.39.1222", "clerc": "10.1080/00018739000101501", "mclachlan98": "10.1103/PhysRevB.58.13558",
}
# Crossref "published" is the online date; these volumes carry the following year.
YEAR_FIX = {"10.3390/nano15010006": 2025, "10.1007/s40820-024-01583-2": 2025, "10.1007/s40820-021-00759-4": 2022}
CROSSREF = json.loads((HERE / "references_crossref.json").read_text())


TITLE_FIX = {
    "Ti 3 CNT x (MXene)": "Ti_{3}CNT_{x} (MXene)",
    "Ti 3 C 2 T x MXene-MWCNT@MoS 2 Film": "Ti_{3}C_{2}T_{x} MXene-MWCNT@MoS_{2} Film",
    "Ti3C2T x : MXenes": "Ti_{3}C_{2}T_{x}: MXenes",
    "Ti 3 C 2 T x /(Aramid": "Ti_{3}C_{2}T_{x}/(Aramid",
    "Emi Shielding Through Conductive Plastics": "EMI shielding through conductive plastics",
}


def _clean(text):
    out = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", text or ""))).strip()
    for bad, good in TITLE_FIX.items():
        out = out.replace(bad, good)
    return out


def _author(a):
    initials = "".join(part[0] + "." for part in re.split(r"[\s\-]+", a.get("given", "")) if part)
    return f"{initials} {a.get('family', a.get('name', ''))}".strip()


def format_crossref(doi):
    m = CROSSREF[doi]
    authors = ", ".join(_author(a) for a in m["author"])
    year = YEAR_FIX.get(doi, m["published"]["date-parts"][0][0])
    pages = (m.get("page") or m.get("article-number") or "").replace("-", "–")
    if doi == "10.34133/2022/9780290":
        pages = "9780290"
    vol = m.get("volume")
    if doi == "10.1109/MEMC.2023.10201434":
        vol = "12(2)"
    ref = f"{authors}, {_clean(m['title'][0])}, {_clean(m['container-title'][0])} {vol} ({year}) {pages}. https://doi.org/{doi}"
    if doi == "10.1007/s10853-020-05518-1":
        ref += " (values read from the preprint, arXiv:2005.04875)"
    return ref, year


REFTEXT = dict(FOUND)
for _k, _d in DOI_KEYS.items():
    REFTEXT[_k] = format_crossref(_d)[0]
_years = []
for b in bench:
    text, year = format_crossref(b["paper"])
    REFTEXT.setdefault(b["citation"], text)
    _years.append(year)
YEARS = (min(_years), max(_years))
REFNUM = {}


def num(key):
    key = BY[key]["citation"] if key in BY else key
    if key not in REFNUM:
        REFNUM[key] = len(REFNUM) + 1
    return REFNUM[key]


def c(*keys):
    nums = sorted(dict.fromkeys(num(k) for k in keys))
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(str(nums[i]) if j == i else (f"{nums[i]},{nums[j]}" if j == i + 1 else f"{nums[i]}–{nums[j]}"))
        i = j + 1
    return "[" + ",".join(out) + "]"


# ---------------------------------------------------------------- docx helpers
# Layout follows the NeurIPS proceedings style: US Letter, 5.5 in x 9 in text block,
# 10 pt Times on 11 pt leading, ruled title, centred abstract, booktabs tables.
from docx.enum.section import WD_SECTION  # noqa: E402,F401
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT  # noqa: E402
from docx.oxml import OxmlElement  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402

FONT, MONO, TEXT_W = "Times New Roman", "Courier New", 5.5
doc = Document()
st = doc.styles["Normal"]
st.font.name, st.font.size = FONT, Pt(10)
st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
st.paragraph_format.space_after, st.paragraph_format.space_before = Pt(5.5), Pt(0)
st.paragraph_format.line_spacing = Pt(11)
for sec in doc.sections:
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    sec.left_margin = sec.right_margin = Inches(1.5)
    sec.top_margin, sec.bottom_margin = Inches(1.0), Inches(1.0)
    sec.footer_distance = Inches(0.5)
    fp = sec.footer.paragraphs[0]; fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for kind, text in (("begin", None), (None, "PAGE"), ("end", None)):
        r = fp.add_run(); r.font.size = Pt(10)
        if kind:
            el = OxmlElement("w:fldChar"); el.set(qn("w:fldCharType"), kind)
        else:
            el = OxmlElement("w:instrText"); el.set(qn("xml:space"), "preserve"); el.text = text
        r._r.append(el)
# open in normal (not compatibility) mode in current Word
_compat = doc.settings.element.find(qn("w:compat"))
if _compat is None:
    _compat = OxmlElement("w:compat"); doc.settings.element.append(_compat)
_cs = OxmlElement("w:compatSetting")
for k, v in (("name", "compatibilityMode"), ("uri", "http://schemas.microsoft.com/office/word"), ("val", "15")):
    _cs.set(qn(f"w:{k}"), v)
_compat.append(_cs)

TOKEN = re.compile(r"(_\{[^}]*\}|\^\{[^}]*\}|\*[^*]+\*)")
JUST, CENTER, LEFT = WD_ALIGN_PARAGRAPH.JUSTIFY, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT


def runs(par, text, bold=False, size=None, font=None):
    for part in TOKEN.split(text):
        if not part:
            continue
        if part.startswith("_{"):
            r = par.add_run(part[2:-1]); r.font.subscript = True
        elif part.startswith("^{"):
            r = par.add_run(part[2:-1]); r.font.superscript = True
        elif part.startswith("*"):
            r = par.add_run(part[1:-1]); r.italic = True
        else:
            r = par.add_run(part)
        r.bold = bold or None
        r.font.name = font or FONT
        if size:
            r.font.size = Pt(size)
    return par


def P(text, align=JUST, **kw):
    par = doc.add_paragraph()
    par.alignment = align
    return runs(par, text, **kw)


def H(text, level=1):
    """Section heads: 12 pt bold flush left; subsection heads: 10 pt bold. Number, then an em space."""
    text = re.sub(r"^(\d+)\.\s+", "\\1\u2003", text) if level == 1 else re.sub(r"^(\d+\.\d+)\s+", "\\1\u2003", text)
    par = doc.add_paragraph(); par.alignment = LEFT
    pf = par.paragraph_format
    pf.space_before, pf.space_after, pf.keep_with_next = (Pt(12), Pt(7), True) if level == 1 else (Pt(8), Pt(5), True)
    pf.line_spacing = Pt(14 if level == 1 else 11)
    runs(par, text, bold=True, size=12 if level == 1 else 10)


def _border(par, side, sz):
    pPr = par._p.get_or_add_pPr()
    bdr = pPr.find(qn("w:pBdr"))
    if bdr is None:
        bdr = OxmlElement("w:pBdr"); pPr.append(bdr)
    el = OxmlElement(f"w:{side}")
    for k, v in (("val", "single"), ("sz", str(sz)), ("space", "12"), ("color", "000000")):
        el.set(qn(f"w:{k}"), v)
    bdr.append(el)


def _cell_borders(cell, **sides):
    tcPr = cell._tc.get_or_add_tcPr()
    bdr = tcPr.find(qn("w:tcBorders"))
    if bdr is None:
        bdr = OxmlElement("w:tcBorders"); tcPr.append(bdr)
    for side, sz in sides.items():
        el = OxmlElement(f"w:{side}")
        for k, v in (("val", "single"), ("sz", str(sz)), ("space", "0"), ("color", "000000")):
            el.set(qn(f"w:{k}"), v)
        bdr.append(el)


def _plain_table(ncols, widths):
    t = doc.add_table(rows=1, cols=ncols); t.alignment = WD_TABLE_ALIGNMENT.CENTER; t.autofit = False
    tblPr = t._tbl.tblPr
    lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); tblPr.append(lay)
    return t


def _set_widths(t, widths):
    for row in t.rows:
        for cell, w in zip(row.cells, widths):
            cell.width = Inches(w)


def _repeat_header(row):
    trPr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader"); el.set(qn("w:val"), "true"); trPr.append(el)
    cs = OxmlElement("w:cantSplit"); cs.set(qn("w:val"), "true"); trPr.append(cs)


EQN = [0]


def EQ(tex):
    """Display equation: centred, number flush right, vertically centred on the equation."""
    EQN[0] += 1
    path = EQ_DIR / f"eq{EQN[0]}.png"
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.text(0, 0, "$" + tex.replace(r"\frac", r"\dfrac") + "$", fontsize=10, math_fontfamily="stix")
    fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.02, transparent=False, facecolor="white")
    plt.close(fig)
    from PIL import Image
    w, _ = Image.open(path).size
    widths = [0.45, TEXT_W - 0.9, 0.45]
    t = _plain_table(3, widths); _set_widths(t, widths)
    for cell in t.rows[0].cells:
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        cell.paragraphs[0].paragraph_format.space_after = Pt(0)
        cell.paragraphs[0].paragraph_format.line_spacing = 1.0
    mid, right = t.rows[0].cells[1].paragraphs[0], t.rows[0].cells[2].paragraphs[0]
    mid.alignment = CENTER
    mid.add_run().add_picture(str(path), width=Inches(min(TEXT_W - 1.0, w / 600)))
    right.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    right.add_run(f"({EQN[0]})").font.name = FONT
    gap = doc.add_paragraph(); gap.paragraph_format.space_after = Pt(0); gap.paragraph_format.line_spacing = Pt(5)


def _caption(text, after=5.5, before=0):
    """'Figure 1:' / 'Table 1:' captions, centred when short and justified when long."""
    text = re.sub(r"^((?:Figure|Table) [A-Z]?\d+)\.", r"\1:", text)
    par = doc.add_paragraph()
    par.alignment = CENTER if len(text) < 95 else JUST
    par.paragraph_format.space_after, par.paragraph_format.space_before = Pt(after), Pt(before)
    return runs(par, text)


def FIG(name, caption, width=TEXT_W):
    par = doc.add_paragraph(); par.alignment = CENTER
    par.paragraph_format.keep_with_next = True
    par.paragraph_format.line_spacing = 1.0
    par.paragraph_format.space_before, par.paragraph_format.space_after = Pt(6), Pt(4)
    par.add_run().add_picture(str(HERE / name), width=Inches(min(width, TEXT_W)))
    _caption(caption, after=11)


def TABLE(caption, header, rows, widths, size=9, align=None):
    """Booktabs table: caption above, rules at top, under the header and at the bottom only."""
    cap = _caption(caption, after=4, before=6); cap.paragraph_format.keep_with_next = True
    scale = min(1.0, TEXT_W / sum(widths)); widths = [w * scale for w in widths]
    t = _plain_table(len(header), widths)
    mar = OxmlElement("w:tblCellMar")
    for side in ("left", "right"):
        el = OxmlElement(f"w:{side}"); el.set(qn("w:w"), "45"); el.set(qn("w:type"), "dxa"); mar.append(el)
    t._tbl.tblPr.append(mar)
    align = align or ["l"] + ["c"] * (len(header) - 1)
    for cell, text in zip(t.rows[0].cells, header):
        cell.text = ""; runs(cell.paragraphs[0], text, bold=True, size=size)
    for row in rows:
        for cell, text in zip(t.add_row().cells, row):
            cell.text = ""; runs(cell.paragraphs[0], str(text), size=size)
    _set_widths(t, widths); _repeat_header(t.rows[0])
    last = len(t.rows) - 1
    for i, row in enumerate(t.rows):
        for j, cell in enumerate(row.cells):
            par = cell.paragraphs[0]
            par.alignment = LEFT if align[j] == "l" else CENTER
            par.paragraph_format.space_after = par.paragraph_format.space_before = Pt(1.5)
            par.paragraph_format.line_spacing = Pt(size + 1.5)
            sides = {}
            if i == 0:
                sides.update(top=8, bottom=4)
            if i == last:
                sides["bottom"] = 8
            if sides:
                _cell_borders(cell, **sides)
    gap = doc.add_paragraph(); gap.paragraph_format.space_after = Pt(4)


# ---------------------------------------------------------------- title block
title = doc.add_paragraph(); title.alignment = CENTER
title.paragraph_format.line_spacing = Pt(19)
title.paragraph_format.space_before, title.paragraph_format.space_after = Pt(14), Pt(26)
_border(title, "top", 32); _border(title, "bottom", 8)
for _i, _line in enumerate(("Predict Before You Build:", "A Benchmark for Electromagnetic Shielding")):
    if _i:
        title.add_run().add_break()
    runs(title, _line, bold=True, size=17)

AUTHORS = [
    ("Danush Arun^{\u2217}", ["Electrical and Electronics Engineering", "Manipal Institute of Technology",
                              "Manipal 576104, Karnataka, India"], "danusharun999@gmail.com"),
    ("Prasad S. Godavarti^{\u2217}", ["International Visiting Faculty Member", "Manipal Institute of Technology, MAHE",
                                      "Manipal 576104, Karnataka, India"], "pgodavarti@gmail.com"),
    ("Prabhakar Sastri", ["School of Computer Engineering", "Manipal Institute of Technology",
                          "Manipal 576104, Karnataka, India"], None),
    ("Srikanth Prabhu^{\u2217}", ["School of Computer Engineering", "Manipal Institute of Technology, MAHE",
                                  "Manipal 576104, Karnataka, India"], "srikanth.prabhu@manipal.edu"),
]
grid = _plain_table(2, [TEXT_W / 2] * 2); grid.add_row(); _set_widths(grid, [TEXT_W / 2] * 2)
for cell, (name, affil, mail) in zip([c_ for row in grid.rows for c_ in row.cells], AUTHORS):
    par = cell.paragraphs[0]; par.alignment = CENTER
    par.paragraph_format.space_after, par.paragraph_format.line_spacing = Pt(8), Pt(11)
    runs(par, name, bold=True)
    for line in affil:
        par.add_run().add_break(); runs(par, line)
    if mail:
        par.add_run().add_break(); runs(par, mail, font=MONO, size=9)
note = P("^{\u2217}Corresponding authors.", align=CENTER, size=9)
note.paragraph_format.space_after = Pt(16)

abs_head = doc.add_paragraph(); abs_head.alignment = CENTER
abs_head.paragraph_format.space_after = Pt(8)
runs(abs_head, "Abstract", bold=True, size=12)


def rng(ids, key="res"):
    v = [BY[i][key] for i in ids]
    return f"{sg(min(v))} to {sg(max(v))}"


def ids(prefix, pool=None):
    return [b["id"] for b in (pool or bench) if b["id"].startswith(prefix)]


LOW_EFF = [b["sigma_eff"] for b in LOW]
DERIVED = [b for b in bench if b["sigma_basis"].startswith("derived")]
THIN_ERR = max(abs(slab_se(b["sigma"], b["t"], 0.5 * (b["f_lo"] + b["f_hi"]))
                   - 20 * math.log10(1 + 376.730 * b["sigma"] * b["t"] / 2)) for b in DERIVED)
GRADE_C_REASON = {b["paper"]: b["quality_note"] for b in GRADE_C}
BULK_LOW = [b for b in PRIM if b["group"] in (4, 5) and b["sigma"] < 1e3]
HIGH_F3 = sum(abs(b["log_ratio"]) <= math.log10(3) for b in HIGH)
T_NEED = next(t for t in np.arange(40e-6, 400e-6, 1e-6) if slab_se(BY["iqbal2020_ti3cn_350"]["sigma"], t, 8.2e9) >= 116.0)
BIG_POS_HIGH = max((b for b in OUT_POS if b["sigma"] >= 1e4 and not b["id"].startswith("iqbal2020_ti3cn")), key=lambda b: b["res"])
DENSE = G[2]
A_N = QUAL["A"]

_abs = P(
    f"Whether the shielding effectiveness (SE) of a planned material can be estimated from its conductivity and "
    f"thickness alone is a practical question that is seldom tested on data whose origin can be checked. We compile "
    f"{N} measured SE values from {NP} primary papers on metal films and meshes, MXene and graphene films, "
    f"carbon–polymer composites, foams and aerogels (10 MHz to 40 GHz, 13 nm to 5 mm, 10^{{−3}} to 10^{{7}} S m^{{−1}}). "
    f"Each value is tied to a quoted sentence, table or caption, and each entry is graded by how completely its source "
    f"documents it. Every entry is compared with the exact homogeneous-slab prediction for the stated fixture, "
    f"including the TE_{{10}} wave impedance for waveguide measurements, which changes the prediction by up to "
    f"{max(WG_SHIFT):.1f} dB. For the {NPRIM} entries that meet the documentation criteria, sheets with reported "
    f"conductivity above 10^{{4}} S m^{{−1}} (n = {S_HIGH['n']}, {S_HIGH['papers']} papers) have a median residual of "
    f"{sg(S_HIGH['median'])} dB (95% interval {ci(S_HIGH)} dB, resampling papers) and a root-mean-square residual of "
    f"{S_HIGH['rmse']:.0f} dB; for dense MXene and graphene films it is {DENSE['rmse']:.0f} dB. Below 10^{{4}} S m^{{−1}} "
    f"the reported conductivity does not predict SE: the root-mean-square residual is {S_MID['rmse']:.0f} dB, and "
    f"samples reported below 1 S m^{{−1}} shield as homogeneous sheets of {min(LOW_EFF):.0f} to {max(LOW_EFF):.0f} "
    f"S m^{{−1}} would. Controlled comparisons within single studies do not show a consistent contribution from "
    f"porosity, with one exception. The benchmark, code and tests are openly available.")
_kw = P("*Keywords:* electromagnetic interference shielding; shielding effectiveness; benchmark dataset; waveguide "
        "measurement; MXene; polymer composites; effective conductivity")
for _p in (_abs, _kw):
    _p.paragraph_format.left_indent = _p.paragraph_format.right_indent = Inches(0.5)
_kw.paragraph_format.space_after = Pt(10)

# ---------------------------------------------------------------- 1
H("1. Introduction")
P(f"A shield is usually specified before it exists. The designer chooses a material and a thickness, and the two "
  f"numbers available at that stage are a conductivity and a dimension. For a homogeneous conducting sheet, "
  f"Schelkunoff’s transmission-line analysis {c('schelkunoff')} turns those two numbers into a shielding "
  f"effectiveness (SE) in closed form {c('schulz', 'ott', 'celozzi')}. How far that estimate can be trusted for the "
  f"materials now being developed is the subject of this paper.")
P(f"The comparison between sheet theory and measurement is not new. Simon’s empirical expression "
  f"{c('simon')} has been used for four decades to relate SE to conductivity, thickness and frequency. Colaneri and "
  f"Shacklette {c('colaneri')} and Joo and Epstein {c('joo')} tested thin- and thick-sheet expressions on conducting "
  f"polymers, and Al-Saleh and Sundararaj {c('alsaleh')} examined the mechanisms in carbon nanotube composites. For "
  f"MXenes, Yun et al. {c('yun')} and Han et al. {c('han2020_ti3c2_5um')} compared the measured SE of highly "
  f"conductive films, from assemblies of monolayers to micrometre thickness, with sheet and transfer-matrix "
  f"calculations and found close agreement, while "
  f"Iqbal et al. {c('iqbal2020_ti3cn_350')} reported absorption in annealed Ti_{{3}}CNT_{{x}} far above what its "
  f"conductivity suggests. Reviews tabulate SE with conductivity and thickness across material classes "
  f"{c('iqbalrev', 'chung')}, and common misreadings of SE and of its reflection and absorption parts have been "
  f"documented {c('peng')}.")
P(f"What these sources do not provide is a single, uniformly applied comparison across material classes in which "
  f"every input can be traced to the place where it was printed, and in which the measurement fixture is taken into "
  f"account. Most reported SE values are measured in a rectangular waveguide, where the wave impedance differs from "
  f"that of free space {c('moucka', 'wilson')}; the difference is a few decibels, which is the size of the agreement "
  f"usually claimed. Tabulated values also mix band averages with maxima, and conductivities measured on the specimen "
  f"with values quoted for the material in general.")
P(f"This paper contributes three things. The first is a benchmark of {N} measured SE values from {NP} papers in "
  f"which each value, thickness, frequency and conductivity is linked to a quotation, and each entry carries a "
  f"quality grade that depends only on what its source prints. The second is a fixture-aware comparison with the "
  f"exact slab model, with statistics that treat papers, not individual samples, as independent. The third is an "
  f"answer to the practical question: for sheets above 10^{{4}} S m^{{−1}} the estimate is good to about "
  f"{S_HIGH['rmse']:.0f} dB root mean square, and below that conductivity it is not usable.")

# ---------------------------------------------------------------- 2
H("2. Model")
H("2.1 Homogeneous slab in free space", 2)
P("A slab of thickness *t*, conductivity *σ*, permeability *µ* and permittivity *ε* separates two half-spaces of "
  "free space with impedance *Z*_{0} = 376.73 Ω. At angular frequency *ω* its propagation constant and intrinsic "
  "impedance are")
EQ(r"\gamma = j\omega\sqrt{\mu\,\varepsilon_c},\qquad \eta=\sqrt{\mu/\varepsilon_c},\qquad \varepsilon_c=\varepsilon-j\sigma/\omega")
P("and the shielding effectiveness is the sum of a reflection term, an absorption term and a multiple-reflection "
  "correction,")
EQ(r"SE = R + A + B")
EQ(r"R = 20\log_{10}\frac{|Z_0+\eta|^{2}}{4\,Z_0\,|\eta|},\qquad A = 20\log_{10}\left|e^{\gamma t}\right| = 8.686\,\mathrm{Re}(\gamma)\,t")
EQ(r"B = 20\log_{10}\left|1-q^{2}e^{-2\gamma t}\right|,\qquad q=\frac{Z_0-\eta}{Z_0+\eta}")
P("With *B* included, the sum is algebraically identical to the exact transmission loss of the slab. The "
  "assumptions are a laterally infinite, homogeneous, isotropic and linear sheet, a plane wave at normal incidence, "
  "and no substrate. For a good conductor (|*η*| ≪ *Z*_{0}) the correction *B* is negative and becomes negligible "
  "once *A* exceeds about 10 dB; the skin depth is *δ* = (2/*ωµσ*)^{1/2} and *A* = 8.686 *t*/*δ*.")
P("When the sheet is much thinner than the skin depth the three terms reduce to a result that depends only on the "
  "sheet resistance *R*_{s} = 1/(*σt*):")
EQ(r"SE \approx 20\log_{10}\left(1+\frac{Z_0}{2R_s}\right)")
P(f"For the {len(DERIVED)} entries below whose conductivity is derived from a printed sheet resistance, Eq. (5) "
  f"agrees with Eqs. (2)–(4) to within {THIN_ERR:.1f} dB, so the nominal thickness has almost no influence on those "
  f"predictions.")

H("2.2 Measurement fixture", 2)
P(f"Equations (1)–(5) describe a transverse electromagnetic wave, which is what a coaxial cell {c('astm')} "
  f"launches. In a rectangular waveguide the sample is illuminated by the TE_{{10}} mode. The field is still "
  f"tangential to the sheet, but the wave impedance and propagation constant in each medium become")
EQ(r"Z_{\mathrm{TE}}=\frac{j\omega\mu}{\gamma_{\mathrm{TE}}},\qquad \gamma_{\mathrm{TE}}=\sqrt{k_c^{2}-\omega^{2}\mu\,\varepsilon_c},\qquad k_c=\frac{2\pi f_c}{c}")
P(f"where *f*_{{c}} is the cutoff frequency (6.557 GHz for WR-90, 9.488 GHz for WR-62). Equations (2)–(4) hold with "
  f"*Z*_{{0}} and *η* replaced by the TE_{{10}} impedances of air and of the sheet. In air *Z*_{{TE}} = "
  f"*Z*_{{0}}/[1 − (*f*_{{c}}/*f*)^{{2}}]^{{1/2}}, which is {z_te(8.2e9):.0f} Ω at 8.2 GHz, {z_te(10e9):.0f} Ω at "
  f"10 GHz and {z_te(12.4e9):.0f} Ω at 12.4 GHz. For a reflecting sheet the waveguide therefore reads "
  f"{20 * math.log10(z_te(8.2e9) / 376.73):.1f}, {20 * math.log10(z_te(10e9) / 376.73):.1f} and "
  f"{20 * math.log10(z_te(12.4e9) / 376.73):.1f} dB above the free-space value at those frequencies "
  f"{c('moucka')}. Each benchmark entry is predicted with the model of the fixture its source states: TE_{{10}} "
  f"for waveguide measurements, and Eqs. (1)–(4) for coaxial cells and for sources that do not name the fixture. "
  f"The sensitivity of the results to this assignment is reported in Section 4.1.")

H("2.3 Residual and effective conductivity", 2)
P("For a sample with reported conductivity *σ*, thickness *t* and measurement frequency *f*, the residual is")
EQ(r"\Delta SE = SE_{\mathrm{measured}} - SE_{\mathrm{slab}}(\sigma,\,t,\,f)")
P("with *µ*_{r} = 1 and *ε*_{r} = 1. When a source reports a value for a band, the prediction is the arithmetic "
  "mean of the predicted SE in decibels at 41 frequencies across that band. A residual has a different meaning in "
  "different regimes: when the predicted SE is close to zero the residual is close to the measured SE and cannot be "
  "appreciably negative, and a given residual corresponds to a different change in conductivity for a thin sheet than for one "
  "thicker than the skin depth. We therefore also report the effective conductivity *σ*_{eff}, defined as the "
  "conductivity of the homogeneous sheet that gives the measured SE in the same fixture,")
EQ(r"SE_{\mathrm{slab}}(\sigma_{\mathrm{eff}},\,t,\,f)=SE_{\mathrm{measured}}")
P("and the ratio *σ*_{eff}/*σ*. The residual and the ratio are descriptive. Neither identifies a cause: a ratio "
  "above one can arise from loss that the DC conductivity does not capture, from a conductivity measured on a "
  "different specimen or in a different direction, or from a thickness that is larger than the stated one; a "
  "ratio below one can arise from leakage at the sample edge, cracks, a conductivity that overstates that of the "
  "measured sheet, or a thickness that is smaller.")

H("2.4 Implementation", 2)
P("The analysis code evaluates Eqs. (1)–(6) directly. An independent implementation of the Schelkunoff "
  "decomposition and a transfer-matrix solver in the same repository agree with it to within 10^{−6} dB over "
  "conductivities from 10^{−3} to 6 × 10^{7} S m^{−1}; these identities, and the limit in which the TE_{10} result "
  "reduces to the free-space result, are enforced by automated tests.")

# ---------------------------------------------------------------- 3
H("3. Benchmark")
H("3.1 Inclusion and grading", 2)
P("A measurement was recorded if its primary source prints the measured total SE in decibels, the thickness of the "
  "sample or of its conductive layer, the frequency or band, and the electrical conductivity or sheet resistance. "
  "Values read from plots, values a paper quotes from other work, and calculated SE were not recorded. Conductivity "
  "units were kept as printed and converted once. Where a source gives sheet resistance, the conductivity is "
  "1/(*R*_{s}*t*). Each entry stores the citation, the values as printed, a verbatim quotation that locates them, "
  "the measurement method and any caveat. Every entry was checked against the source text a second time after "
  "extraction; no value was found to differ from its source.")
nC_papers = len({b["paper"] for b in GRADE_C})
P(f"Sources do not document their samples equally, so each entry carries a grade that depends only on what the "
  f"source prints and not on the size or sign of the residual. Grade A ({QUAL['A']} entries): SE, thickness and "
  f"conductivity are stated for the same specimen without qualification. Grade B ({QUAL['B']} entries): one "
  f"documented caveat, such as a conductivity quoted for the film type and not measured on the specimen, a "
  f"thickness measured before a processing step, a two-probe conductivity, an SE printed as approximate, or a "
  f"fixture described only in supplementary material. Grade C ({QUAL['C']} entries from {nC_papers} papers): the "
  f"source is internally inconsistent or incomplete for that sample; Table 3 lists the reason for each. Grades A "
  f"and B form the primary set of {NPRIM} entries from {NPPRIM} papers. Grade C entries are shown in the figures "
  f"with open symbols and are excluded from the statistics unless stated.")

H("3.2 Coverage", 2)
se_all = [b["se"] for b in bench]
P(f"The {N} entries come from {NP} papers published between {YEARS[0]} and {YEARS[1]} (Appendix A). Measured SE "
  f"ranges from {min(se_all):.1f} to {max(se_all):.0f} dB, thickness from 13 nm to 5 mm, and reported conductivity "
  f"from 2.5 × 10^{{−3}} to 1.2 × 10^{{7}} S m^{{−1}}. The fixture is a rectangular waveguide for {FIX['waveguide']} "
  f"entries and a coaxial cell for {FIX['coaxial']}; for {FIX['unspecified']} the source gives only the band or a "
  f"network analyser. Most measurements are in the X band (8.2–12.4 GHz); the others extend from 10 MHz to 40 GHz. "
  f"The entries are a mixture of band averages, single-frequency values and values printed without qualification, "
  f"as the sources give them; the qualifier is recorded for each. {NUMW.get(SE90, SE90).capitalize()} measured "
  f"values are at or above 90 dB, where the dynamic range of the fixture, which most sources do not state, becomes "
  f"a concern.")
P("The sources were restricted to papers whose full text could be read, which favours open-access journals, and "
  "to samples whose values appear in the text, a table or a caption. Published shielding materials are also "
  "selected for shielding well: a composite with low conductivity and negligible SE is rarely reported. This "
  "matters for the interpretation of positive residuals at low conductivity in Section 4.3.")

# ---------------------------------------------------------------- 4
H("4. Results")
H("4.1 Overall comparison", 2)
P(f"Figure 1 compares measured and predicted SE, and Table 1 summarises the residuals. Over the primary set the "
  f"median residual is {sg(S_PRIM['median'])} dB (95% interval {ci(S_PRIM)} dB) and the root-mean-square (RMS) "
  f"residual is {S_PRIM['rmse']:.0f} dB; {S_PRIM['w10']} of {NPRIM} entries lie within ±10 dB and {S_PRIM['w5']} "
  f"within ±5 dB. Intervals are obtained by resampling papers with replacement, because samples from one paper share "
  f"a fixture, a conductivity method and an author. The shaded ±10 dB band in the figures is a visual guide and "
  f"corresponds to a factor of ten in transmitted power.")
P(f"The agreement depends on the reported conductivity (Fig. 1b). Above 10^{{4}} S m^{{−1}} (n = {S_HIGH['n']}, "
  f"{S_HIGH['papers']} papers) the median residual is {sg(S_HIGH['median'])} dB ({ci(S_HIGH)} dB), the RMS residual "
  f"is {S_HIGH['rmse']:.1f} dB, and {S_HIGH['w10']} entries are within ±10 dB, {S_HIGH['w5']} within ±5 dB and "
  f"{S_HIGH['w3']} within ±3 dB. Taking one median per paper, {sum(abs(x) <= 10 for x in PM)} of {len(PM)} papers "
  f"are within ±10 dB. Between 1 and 10^{{4}} S m^{{−1}} (n = {S_MID['n']}) the RMS residual is {S_MID['rmse']:.0f} dB "
  f"and only {S_MID['w10']} entries are within ±10 dB. Below 1 S m^{{−1}} (n = {S_LOW['n']}, {S_LOW['papers']} papers) "
  f"the predicted SE is at most {max(b['slab'] for b in LOW):.0f} dB, so any sample that shields appreciably has a "
  f"positive residual; all {S_LOW['n']} lie between {sg(S_LOW['min'])} and {sg(S_LOW['max'])} dB.")
P(f"Three checks bound the dependence of these numbers on choices made here. If every entry is predicted with the "
  f"free-space model, the median residual above 10^{{4}} S m^{{−1}} becomes {sg(S_HIGH_FREE['median'])} dB (RMS "
  f"{S_HIGH_FREE['rmse']:.1f} dB); if every X- and Ku-band entry without a stated coaxial cell is treated as a "
  f"waveguide measurement, it becomes {sg(S_HIGH_WG['median'])} dB (RMS {S_HIGH_WG['rmse']:.1f} dB). The fixture "
  f"model shifts the centre of the distribution by {abs(S_HIGH_FREE['median'] - S_HIGH_WG['median']):.1f} dB and "
  f"leaves its width unchanged. Restricting to grade A gives n = {S_HIGH_A['n']} from {S_HIGH_A['papers']} papers, "
  f"a median of {sg(S_HIGH_A['median'])} dB and an RMS residual of {S_HIGH_A['rmse']:.1f} dB. Including the grade C "
  f"entries gives n = {S_HIGH_ALL['n']}, a median of {sg(S_HIGH_ALL['median'])} dB and an RMS residual of "
  f"{S_HIGH_ALL['rmse']:.1f} dB.")
FIG("fig1_parity.png",
    f"Figure 1. Measured shielding effectiveness and the homogeneous-slab prediction for the {N} benchmark entries. "
    f"(a) Parity plot. (b) Residual against reported conductivity; the dashed line is the least-squares fit to the "
    f"primary set. Open symbols are grade C entries. The band marks ±10 dB.")
TABLE("Table 1. Residuals of the primary set by material group and by reported conductivity. Median with 95% "
      "interval from resampling papers; IQR: interquartile range; RMS: root mean square.",
      ["Subset", "*n*", "Papers", "Median (dB)", "95% interval", "IQR (dB)", "RMS (dB)", "Within ±10 dB"],
      [[GROUPS[i][0], G[i]["n"], G[i]["papers"], sg(G[i]["median"]),
        ci(G[i]) if G[i]["papers"] > 2 else "–", f"{sg(G[i]['q1'])} to {sg(G[i]['q3'])}", f"{G[i]['rmse']:.1f}",
        f"{G[i]['w10']} of {G[i]['n']}"] for i in range(len(GROUPS))] +
      [[label, s["n"], s["papers"], sg(s["median"]), ci(s) if s["papers"] > 2 else "–",
        f"{sg(s['q1'])} to {sg(s['q3'])}", f"{s['rmse']:.1f}", f"{s['w10']} of {s['n']}"]
       for label, s in (("*σ* ≥ 10^{4} S m^{−1}", S_HIGH), ("1 ≤ *σ* < 10^{4} S m^{−1}", S_MID),
                        ("*σ* < 1 S m^{−1}", S_LOW), ("Primary set", S_PRIM), (f"All {N} entries", S_ALL))],
      widths=[1.72, 0.28, 0.48, 0.55, 0.9, 0.9, 0.42, 0.62], size=7.5)
FIG("fig2_residual_by_group.png",
    "Figure 2. Residuals by material group. Each symbol is one entry (open: grade C). The bar spans the "
    "interquartile range of the primary set and the tick marks its median.")

H("4.2 Sheets above 10^{4} S m^{−1}", 2)
films = ["voronin2025_iai3", "jeon2023_blso", "liu2020_ag_film"]
P(f"Continuous metal and oxide films follow the model within a few decibels. An In_{{2}}O_{{3}}/Ag/In_{{2}}O_{{3}} "
  f"stack with a 13 nm silver layer {c('voronin2025_iai3')}, a 440 nm La-doped BaSnO_{{3}} film {c('jeon2023_blso')} "
  f"and a sintered silver film on paper {c('liu2020_ag_film')} have residuals of {dres('voronin2025_iai3')}, "
  f"{dres('jeon2023_blso')} and {dres('liu2020_ag_film')} dB. Two silver films loaded with nickel particles from the last "
  f"study lie {dres('liu2020_agni560_2to1')} and {dres('liu2020_agni560_1to2')} dB above the prediction; the source does "
  f"not give the permeability of these films, and we do not assign a cause.")
dense_prim = [b for b in PRIM if b["group"] == 2]
han = ids("han2020_")
P(f"Dense MXene and graphene films behave in the same way over three orders of magnitude in thickness "
  f"(n = {DENSE['n']} from {DENSE['papers']} papers; median {sg(DENSE['median'])} dB, RMS {DENSE['rmse']:.1f} dB, "
  f"{DENSE['w10']} within ±10 dB). Sprayed Ti_{{3}}C_{{2}}T_{{x}} films of about 40 and 137 nm and a 5 µm filtered "
  f"film {c('han2020_ti3c2_5um')} have residuals of {dres('han2020_ti3c2_40nm')}, {dres('han2020_ti3c2_137nm')} and "
  f"{dres('han2020_ti3c2_5um')} dB, a 29 nm oxygen-substituted film {c('park2026_mxene_29nm')} {dres('park2026_mxene_29nm')} "
  f"dB, micrometre-thick films {c('wan2022_mxene_lm', 'huang2025_mxene_mx80')} between "
  f"{rng(ids('wan2022_') + ['huang2025_mxene_mx80']).replace(' to ', ' and ')} dB, and 10 to 40 µm films "
  f"{c('iqbal2020_ti3c2_asis', 'hassan2024_al_mxene')} between "
  f"{rng(['iqbal2020_ti3c2_asis', 'iqbal2020_ti3c2_350', 'iqbal2020_ti3cn_asis'] + ids('hassan2024_')).replace(' to ', ' and ')} "
  f"dB. The largest departure in this group is a 12 µm film {c('ahmed2023_ti3c2_12um')} at "
  f"{dres('ahmed2023_ti3c2_12um')} dB. This confirms on data from {DENSE['papers'] - 1} further papers what Han et al. "
  f"{c('han2020_ti3c2_5um')} and Yun et al. {c('yun')} found for their own films: within a scatter of about "
  f"{DENSE['rmse']:.0f} dB, a dense laminate of high conductivity shields as a homogeneous sheet of that "
  f"conductivity and thickness does, and no additional contribution from the laminated structure is detectable at "
  f"that resolution.")
mesh = G[1]
P(f"Metal meshes and coated fabrics are less uniform (n = {mesh['n']} from {mesh['papers']} papers, RMS "
  f"{mesh['rmse']:.1f} dB). Electroplated copper meshes measured at 10 MHz in a coaxial cell "
  f"{c('voronin2022_cumesh_LCM30')} lie between {rng(ids('voronin2022_')).replace(' to ', ' and ')} dB. In waveguide "
  f"measurements at 8–18 GHz, silver meshes embedded in glass {c('li2023_agmesh_W1P25t0.5')} lie between "
  f"{rng(ids('li2023_agmesh')).replace(' to ', ' and ')} dB and silver meshes in PET {c('zarei2024_agmesh_1')} between "
  f"{rng(ids('zarei2024_')).replace(' to ', ' and ')} dB. The mesh pitches are far smaller than the wavelength, so "
  f"the departures are not explained by the mesh geometry in the quasi-static picture {c('casey')}; the substrates, "
  f"which the model omits, and the way the sheet resistance was measured are possible contributors that the sources "
  f"do not allow us to separate.")

H("4.3 Reported conductivity below 10^{4} S m^{−1}", 2)
P(f"Below 10^{{4}} S m^{{−1}} the reported conductivity loses its predictive value. The residuals of the "
  f"{S_MID['n'] + S_LOW['n']} primary entries in this range span {sg(min(b['res'] for b in MID + LOW))} to "
  f"{sg(max(b['res'] for b in MID + LOW))} dB. Over the whole primary set the residual decreases with reported "
  f"conductivity (Spearman *ρ* = {sg(REG['spearman'], 2)}), and a least-squares fit gives {sg(REG['slope'], 1)} dB per "
  f"decade, although the interval obtained by resampling papers, {sg(REG['slope_ci'][0], 1)} to "
  f"{sg(REG['slope_ci'][1], 1)} dB per decade, includes zero.")
epdm, wani = ids("rahaman2023_"), ids("wani2026_")
P(f"Figure 3 shows the same data as effective conductivity. Above 10^{{4}} S m^{{−1}} the effective and reported "
  f"values agree within a factor of three for {HIGH_F3} of the {len(HIGH)} primary entries. At the low end they "
  f"separate completely. "
  f"Carbon-black-filled EPDM sheets 1 mm thick {c('rahaman2023_epdm_cb_20phr')} with DC conductivities of "
  f"2.5 × 10^{{−3}} to 6.3 × 10^{{−2}} S m^{{−1}} are reported to shield {min(BY[i]['se'] for i in epdm):.1f} to "
  f"{max(BY[i]['se'] for i in epdm):.1f} dB at 10 GHz, which corresponds to effective conductivities of "
  f"{min(BY[i]['sigma_eff'] for i in epdm):.0f} to {max(BY[i]['sigma_eff'] for i in epdm):.0f} S m^{{−1}}, four orders "
  f"of magnitude above the DC values; the same source prints 1 kHz conductivities 50 to 200 times the DC values. "
  f"Six graphitic aerogels {c('wani2026_graphitic_aerogel_700_5')} with two-probe conductivities of 0.11 to 2.9 "
  f"S m^{{−1}} correspond to {min(BY[i]['sigma_eff'] for i in wani):.0f} to "
  f"{max(BY[i]['sigma_eff'] for i in wani):.0f} S m^{{−1}}. For the {len(BULK_LOW)} composites, foams and aerogels with "
  f"reported conductivities below 10^{{3}} S m^{{−1}}, a range of more than five decades, the effective conductivity "
  f"stays between {min(b['sigma_eff'] for b in BULK_LOW):.0f} and {max(b['sigma_eff'] for b in BULK_LOW):.0f} S m^{{−1}}.")
P(f"Two readings of this are possible and the benchmark cannot separate them. The first is physical: in a "
  f"composite near its percolation threshold, conducting clusters that are disconnected at DC are coupled "
  f"capacitively at microwave frequencies, so the AC conductivity exceeds the DC value and rises with frequency "
  f"{c('bergman', 'clerc', 'mclachlan98')}. The second is selection and measurement: materials are published "
  f"because they shield, so samples with low conductivity and low SE are under-represented, and DC conductivities "
  f"of poorly conducting, porous or contact-limited samples are easily under-estimated. Either way the practical "
  f"consequence is the same. A DC conductivity below about 10^{{4}} S m^{{−1}} is not an input from which microwave "
  f"SE can be estimated, and the complex permittivity or sheet impedance at the measurement frequency is needed "
  f"instead.")
FIG("fig3_effective_conductivity.png",
    "Figure 3. Effective conductivity, obtained by inverting the measured shielding effectiveness with the slab "
    "model, against reported conductivity. The band marks a factor of ten. Symbols as in Fig. 1.", width=3.3)

H("4.4 Does porosity add shielding?", 2)
porous_prim = [b for b in PRIM if b["porous"]]
P(f"Porous films, foams and aerogels are often described as shielding better than their conductivity implies, "
  f"through internal reflection and absorption in the pore structure. In the primary set the "
  f"{len(porous_prim)} porous entries do lie higher on average, but the difference is not resolved: with reported "
  f"conductivity in the same regression, the porosity term is {sg(REG['porous'], 0)} dB with a 95% interval of "
  f"{sg(REG['porous_ci'][0], 0)} to {sg(REG['porous_ci'][1], 0)} dB. Porous samples also tend to have low reported "
  f"conductivity, so the two variables are confounded across studies.")
P(f"Comparisons within a single study are more informative, because fixture, method and authors are the same "
  f"(Table 2, Fig. 4). A segregated polysulfone/nanotube composite {c('xie2020_psu_cnt_solid')} has a residual of "
  f"{dres('xie2020_psu_cnt_solid')} dB as a solid and {dres('xie2020_psu_cnt_foam')} dB after foaming. A graphene film "
  f"{c('lin2019_graphene_lg4')} has {dres('lin2019_graphene_lg4')} dB when compressed and {dres('lin2019_graphene_plg')} dB "
  f"in its porous state. In both cases foaming or expansion lowers the conductivity and the SE together, as the "
  f"slab model expects, and the residual is unchanged within a few decibels. The exception is annealed "
  f"Ti_{{3}}CNT_{{x}} {c('iqbal2020_ti3cn_350')}: the residual rises from {dres('iqbal2020_ti3cn_asis')} dB as made to "
  f"{dres('iqbal2020_ti3cn_250')} dB after annealing at 250 °C and {dres('iqbal2020_ti3cn_350')} dB at 350 °C, while "
  f"Ti_{{3}}C_{{2}}T_{{x}} films annealed in the same study remain at {dres('iqbal2020_ti3c2_asis')} and "
  f"{dres('iqbal2020_ti3c2_350')} dB. The authors attribute the increase to absorption in a porous layered "
  f"structure that forms on annealing. Our prediction uses the thickness before annealing, since the thickness "
  f"afterwards is not printed; a thicker film would raise the prediction, and the film would have to reach about "
  f"{T_NEED * 1e6:.0f} µm, from 40 µm, to account for the 350 °C result. The measured values of 99 and 116 dB are also the highest "
  f"in the benchmark. We regard this series as the one case in the benchmark where a structural contribution is "
  f"supported by a controlled comparison, and the general statement that porosity adds shielding as not "
  f"established by these data.")
PAIRS = [("Polysulfone/CNT, segregated", "xie2020_psu_cnt_solid", "solid", "xie2020_psu_cnt_foam", "foamed"),
         ("Graphene film", "lin2019_graphene_lg4", "compressed", "lin2019_graphene_plg", "porous"),
         ("Ti_{3}CNT_{x} film", "iqbal2020_ti3cn_asis", "as made", "iqbal2020_ti3cn_250", "250 °C"),
         ("Ti_{3}CNT_{x} film", "iqbal2020_ti3cn_asis", "as made", "iqbal2020_ti3cn_350", "350 °C"),
         ("Ti_{3}C_{2}T_{x} film", "iqbal2020_ti3c2_asis", "as made", "iqbal2020_ti3c2_350", "350 °C"),
         ("Exfoliated graphene film (grade C)", "zhang2023_graphene_egf", "as made", "zhang2023_graphene_mwegf", "expanded")]
TABLE("Table 2. Within-study comparisons of a dense and a porous or annealed state of the same material. "
      "*σ*: reported conductivity; ΔSE: residual.",
      ["Material", "States", "*σ* (S m^{−1})", "SE (dB)", "ΔSE (dB)", "Change (dB)", "Ref."],
      [[name, f"{la} → {lb}", f"{fsig2(BY[a]['sigma'])} → {fsig2(BY[b_]['sigma'])}", f"{BY[a]['se']:.1f} → {BY[b_]['se']:.1f}",
        f"{dres(a)} → {dres(b_)}", sg(BY[b_]["res"] - BY[a]["res"]), f"[{num(a)}]"] for name, a, la, b_, lb in PAIRS],
      widths=[1.25, 0.95, 1.2, 0.75, 0.85, 0.6, 0.4], size=7.5)
FIG("fig4_paired.png",
    "Figure 4. Residuals for the within-study comparisons of Table 2. Dashed line and open symbols: grade C.",
    width=3.6)

H("4.5 Unexplained residuals", 2)
P(f"{len(OUTL)} of the {NPRIM} primary entries have residuals beyond ±10 dB: {len(OUT_POS)} positive and "
  f"{len(OUT_NEG)} negative. We treat both signs in the same way, as cases in which the reported conductivity, "
  f"thickness and SE are not mutually consistent with a homogeneous sheet, and list the candidate causes without "
  f"assigning one. Of the positive residuals, {sum(b['sigma'] < 1e4 for b in OUT_POS)} belong to samples below "
  f"10^{{4}} S m^{{−1}} and are discussed in Section 4.3. The largest positive residual above that value, apart from "
  f"annealed Ti_{{3}}CNT_{{x}}, is {sg(BIG_POS_HIGH['res'])} dB, for a silver mesh in glass "
  f"{c('li2023_agmesh_W1P25t0.5')}. A film of Ti_{{3}}C_{{2}}T_{{x}} "
  f"in an aramid nanofibre matrix {c('zhang2022_mxene_anf_pva')} is reported at 70 dB for 120 µm and 357 S m^{{−1}} "
  f"({dres('zhang2022_mxene_anf_pva')} dB), which would require an effective conductivity "
  f"{BY['zhang2022_mxene_anf_pva']['sigma_eff'] / BY['zhang2022_mxene_anf_pva']['sigma']:.0f} times the stated one.")
P(f"The largest negative residuals are a segregated graphene/polypropylene composite {c('alam2017_gnp_pp_10')} "
  f"({dres('alam2017_gnp_pp_10')} dB), a 4.5 mm carbon nanofibre aerogel {c('chen2025_cnf_pani_aerogel')} "
  f"({dres('chen2025_cnf_pani_aerogel')} dB), whose predicted SE of {BY['chen2025_cnf_pani_aerogel']['slab']:.0f} dB is "
  f"beyond what a waveguide measurement normally resolves, and a stack of two MWCNT/PMMA bars "
  f"{c('pande2009_mwcnt_pmma_scb2')} ({dres('pande2009_mwcnt_pmma_scb2')} dB), in which the bars are joined by an "
  f"insulating adhesive. Candidate causes of a negative residual include leakage around the sample, incomplete "
  f"contact between a free-standing film and the fixture, cracks and pinholes, a measured SE limited by the dynamic "
  f"range, and a conductivity measured in the plane of a network that is poorly connected across the specimen.")
P(f"The grade C entries (Table 3) show why the grading matters. Five cellulose-nanofibre/Ti_{{3}}C_{{2}}T_{{x}} films "
  f"from one study {c('wang2026_tocnf_mxene_M25')} have residuals of {rng(ids('wang2026_'))} dB. The source prints "
  f"the conductivity unit in two different ways, and dividing the tabulated conductivities by 100 brings all five "
  f"predictions to within {max(abs(BY[i]['se'] - predict(BY[i]['sigma'] / 100, BY[i]['t'], BY[i]['f_lo'], BY[i]['f_hi'], False)) for i in ids('wang2026_')):.0f} "
  f"dB of the measurements. We cannot tell from the paper which reading is correct, which is the reason for the "
  f"grade. A comparison of each measured SE with the slab value would expose inconsistencies of this kind at the "
  f"time of writing or review.")
def tidy(note):
    for a_, b_ in ((" um", " µm"), ("ohm", "Ω"), ("S cm-1", "S cm^{−1}"), ("'s m-1'", "“s m-1”"),
                   ("'was calculated according to the Simon formula'", "“was calculated according to the Simon formula”"),
                   ("factor 5-12", "a factor of 5 to 12")):
        note = note.replace(a_, b_)
    return note


c_rows = []
for paper in dict.fromkeys(b["paper"] for b in GRADE_C):
    sub = [b for b in GRADE_C if b["paper"] == paper]
    res = [b["res"] for b in sub]
    c_rows.append([f"[{num(sub[0]['id'])}]", len(sub), tidy(GRADE_C_REASON[paper]),
                   sg(res[0]) if len(sub) == 1 else f"{sg(min(res))} to {sg(max(res))}"])
TABLE("Table 3. Grade C entries: sources that are internally inconsistent or incomplete for the sample. "
      "These are excluded from the primary statistics.",
      ["Ref.", "*n*", "Reason for the grade", "ΔSE (dB)"], c_rows,
      widths=[0.4, 0.3, 3.8, 1.0], size=8, align=["c", "c", "l", "c"])

# ---------------------------------------------------------------- 5
H("5. Discussion")
H("5.1 What can be predicted before a shield is built", 2)
P(f"For a continuous sheet whose conductivity is above 10^{{4}} S m^{{−1}}, the slab model with the target "
  f"conductivity and thickness gives the SE that published samples of that kind were measured to have, with an RMS "
  f"difference of about {S_HIGH['rmse']:.0f} dB across classes and about {DENSE['rmse']:.0f} dB for dense MXene and "
  f"graphene films. That is an estimate with a stated uncertainty, obtained from two numbers, and it is sufficient "
  f"to decide whether a candidate film is in the right range before it is made. It is not a substitute for "
  f"measurement: {S_HIGH['n'] - S_HIGH['w10']} of the {S_HIGH['n']} entries in this range differ from it by more than "
  f"10 dB. For a planar sheet in this regime the estimate depends on the sheet resistance alone, Eq. (5), so the "
  f"quantity to design for is the product of conductivity and thickness.")
P("For composites, foams and aerogels below 10^{4} S m^{−1}, no estimate of comparable quality can be made from "
  "the DC conductivity. The SE of these materials has to be predicted from their response at the frequency of "
  "interest, and a model that supplies only a DC percolation conductivity will under-estimate the shielding of "
  "the samples that are reported.")

H("5.2 Reporting", 2)
P(f"The comparison would be sharper if sources reported a small set of quantities consistently. We suggest the "
  f"following, which add little work to a shielding study: the in-plane conductivity or sheet resistance measured on "
  f"the specimen used for the SE measurement, with the method; the thickness of that specimen as measured; the "
  f"fixture and mode (rectangular waveguide and its designation, or coaxial cell, with the standard followed "
  f"{c('astm', 'ieee2715')}); the frequency of each SE value and whether it is a band average or a maximum; the "
  f"dynamic range of the set-up; and the slab prediction beside the measured value. At normal incidence, and for "
  f"the TE_{{10}} mode, the electric field lies in the plane of the sheet, so it is the in-plane conductivity that "
  f"enters the prediction. Of the {N} entries collected here, {QUAL['A']} meet all of the documentation criteria "
  f"without qualification.")

H("5.3 Limitations", 2)
P(f"The benchmark is small. Several groups rest on a few papers (Table 1), and there are no bulk metal foils, "
  f"ferromagnetic sheets or multilayers with per-layer properties. Intervals from resampling {NPPRIM} papers are "
  f"wide, and the sub-group statistics should be read as descriptive. Sources were limited to full texts that could "
  f"be read and to values printed in text, tables or captions, so samples documented only in supplementary files or "
  f"plots are absent. Band averages, maxima, single-frequency values and unqualified values are pooled as "
  f"reported; sources that average transmitted power and not decibels would differ from our convention where SE "
  f"varies across the band. For {FIX['unspecified']} entries the fixture is not stated, which leaves an uncertainty "
  f"of up to {max(WG_SHIFT):.1f} dB in their predictions. The model takes *µ*_{{r}} = 1 and *ε*_{{r}} = 1 and omits "
  f"substrates, so magnetic loss, dielectric loss and substrate reflections all appear in the residual. No "
  f"uncertainty is propagated from the reported conductivity and thickness, because sources rarely state one.")
P("With a larger and better documented set, the factors that determine shielding could be organised in a "
  "structural model, with composition, filler geometry, processing and thickness as independent variables, "
  "conductivity and porosity as mediating variables, and SE as the outcome. The within-study comparisons of "
  "Section 4.4 indicate the kind of data such a model would need.")

H("6. Conclusions")
P(f"A benchmark of {N} shielding measurements from {NP} papers, each traceable to its source and graded for "
  f"documentation, was compared with the exact homogeneous-slab model for the stated measurement fixture. For "
  f"sheets with reported conductivity above 10^{{4}} S m^{{−1}}, the model reproduces the measurements with a median "
  f"residual of {sg(S_HIGH['median'])} dB and an RMS residual of {S_HIGH['rmse']:.0f} dB ({S_HIGH['w10']} of "
  f"{S_HIGH['n']} within ±10 dB), and with {DENSE['rmse']:.0f} dB for dense MXene and graphene films from 29 nm to "
  f"40 µm. The waveguide wave impedance accounts for up to {max(WG_SHIFT):.1f} dB of the prediction and should be "
  f"included in any such comparison. Below 10^{{4}} S m^{{−1}} the reported DC conductivity does not predict the "
  f"measured SE. The data do not establish that porosity in general adds shielding beyond what conductivity and "
  f"thickness imply; one annealing series supports it. Stating the slab prediction beside each measured SE, with "
  f"the fixture and the in-plane conductivity of the measured specimen, would make such comparisons routine.")

H("Declaration of competing interest")
P("The authors declare that they have no known competing financial interests or personal relationships that could "
  "have appeared to influence the work reported in this paper.")
H("Funding")
P("This research did not receive any specific grant from funding agencies in the public, commercial, or "
  "not-for-profit sectors.")
H("Data availability")
P("The benchmark file with quotations and grades, the analysis code, the automated tests and the scripts that "
  "generate every number, table and figure in this paper are available at "
  "https://github.com/DanushArun/material-emi-shielding (release benchmark-v1).")
H("Declaration of generative AI and AI-assisted technologies in the manuscript preparation process")
P("During the preparation of this work the authors used Claude (Anthropic) to assist with locating and extracting "
  "values from the cited sources, writing and testing the analysis code, and drafting and editing the text. The "
  "authors reviewed and edited the content as needed and take full responsibility for the content of the "
  "published article.")

H("Appendix A. Benchmark entries")


def ft(x):
    return f"{x * 1e9:.3g} nm" if x < 1e-6 else (f"{x * 1e6:.3g} µm" if x < 1e-3 else f"{x * 1e3:.3g} mm")


def ff(b):
    lo, hi = b["f_lo"], b["f_hi"]
    u = (lambda v: f"{v / 1e9:g}") if hi >= 1e9 else (lambda v: f"{v / 1e6:g}")
    unit = "GHz" if hi >= 1e9 else "MHz"
    return f"{u(lo)} {unit}" if lo == hi else f"{u(lo)}–{u(hi)} {unit}"


CHEM = [("Ti3C1.94O0.06Tz", "Ti_{3}C_{1.94}O_{0.06}T_{z}"), ("Ti3C2Tx", "Ti_{3}C_{2}T_{x}"), ("Ti3CNTx", "Ti_{3}CNT_{x}"),
        ("In2O3", "In_{2}O_{3}"), ("BaSnO3", "BaSnO_{3}"), ("MoS2", "MoS_{2}")]


def chem(name):
    for plain, marked in CHEM:
        name = name.replace(plain, marked)
    return name.replace("*", "∗")


def fsig(v):
    if 0.01 <= v < 1e3:
        return f"{v:.3g}"
    m, e = f"{v:.2e}".split("e")
    return f"{m} × 10^{{{int(e)}}}".replace("-", "−")


TABLE("Table A1. All benchmark entries. *t*: thickness; *σ*: reported conductivity in S m^{−1} (for entries "
      "reported as sheet resistance, 1/(*R*_{s}*t*)); Fx.: fixture (W waveguide, C coaxial cell, U not stated); "
      "Gr.: grade; SE: measured; Slab: prediction for the fixture; ΔSE: residual; log *ρ*: "
      "log_{10}(*σ*_{eff}/*σ*).",
      ["Material", "*t*", "Frequency", "*σ*", "Fx.", "Gr.", "SE (dB)", "Slab (dB)", "ΔSE (dB)", "log *ρ*", "Ref."],
      [[chem(b["material"]), ft(b["t"]), ff(b), fsig(b["sigma"]), b["fixture"][0].upper(), b["quality"],
        f"{b['se']:.1f}", f"{b['slab']:.1f}", sg(b["res"]), sg(b["log_ratio"], 2), f"[{num(b['id'])}]"]
       for b in sorted(bench, key=lambda b: (b["group"], b["citation"], b["id"]))],
      widths=[1.3, 0.46, 0.78, 0.64, 0.24, 0.24, 0.36, 0.38, 0.42, 0.38, 0.34], size=7)

H("References")
for key, i in sorted(REFNUM.items(), key=lambda kv: kv[1]):
    par = P(f"[{i}]\t{REFTEXT[key]}", align=LEFT)
    pf = par.paragraph_format
    pf.left_indent, pf.first_line_indent, pf.space_after = Inches(0.3), Inches(-0.3), Pt(4)
    pf.tab_stops.add_tab_stop(Inches(0.3))

doc.save(OUT)
print(f"Saved {OUT}  ({N} entries, {NP} papers, {len(REFNUM)} references, {EQN[0]} equations)")
