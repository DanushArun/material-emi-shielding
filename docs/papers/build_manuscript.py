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
from benchmark_analysis import GROUPS, cutoff, load, predict, slab_se, summary  # noqa: E402

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


def sg(x, nd=1):  # signed value with a true minus sign; zero is unsigned
    text = f"{x:+.{nd}f}"
    if float(text) == 0:
        text = f"{0:.{nd}f}"
    return text.replace("-", "\u2212")


def ci(s):
    return f"{sg(s['ci_lo'])} to {sg(s['ci_hi'])}"


def dres(i):
    return sg(BY[i]["res"])


S_ALL, S_PRIM = summary(bench), summary(PRIM)
# Two populations separated by a gap in reported conductivity (no primary entry lies inside it)
FILM = [b for b in PRIM if b["sigma"] >= 1e4]
BULK = [b for b in PRIM if b["sigma"] < 1e4]
GAP = (max(b["sigma"] for b in BULK), min(b["sigma"] for b in FILM))
assert max(b["t"] for b in FILM) < min(b["t"] for b in BULK)
ANNEALED = ("iqbal2020_ti3cn_250", "iqbal2020_ti3cn_350")
S_FILM, S_BULK = summary(FILM), summary(BULK)
S_FILM_NOTI = summary([b for b in FILM if b["id"] not in ANNEALED])
S_FILM_FREE, S_FILM_STATED = summary(FILM, "res_free"), summary(FILM, "res_stated")
S_FILM_BAND = summary([b for b in FILM if b["band_average"]])
S_FILM_ALL = summary([b for b in bench if b["sigma"] >= 1e4])
S_BULK_ALL = summary([b for b in bench if b["sigma"] < 1e4])
LOW = [b for b in PRIM if b["sigma"] < 1]
G = [summary([b for b in PRIM if b["group"] == i]) for i in range(len(GROUPS))]
G_ALL = [summary([b for b in bench if b["group"] == i]) for i in range(len(GROUPS))]
FILM_PAPERS = sorted({b["paper"] for b in FILM})
PM = [float(np.median([b["res"] for b in FILM if b["paper"] == p])) for p in FILM_PAPERS]
WG = [b for b in bench if "waveguide" in b["fixture"]]
WG_SHIFT = [b["slab"] - b["slab_free"] for b in WG]
FIX = {k: sum(b["fixture"] == k for b in bench) for k in ("waveguide", "coaxial", "assumed-waveguide", "assumed-tem")}
OUTL = [b for b in PRIM if abs(b["res"]) > 10]
OUT_POS, OUT_NEG = [b for b in OUTL if b["res"] > 0], [b for b in OUTL if b["res"] < 0]
NUMW = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
        11: "eleven", 12: "twelve", 13: "thirteen"}
QUAL = {q: sum(b["quality"] == q for b in bench) for q in "ABC"}
SE90 = sum(b["se"] >= 90 for b in bench)
N_BAND = sum(b["band_average"] for b in bench)
YEARS = None  # filled after references are formatted


def z_te(f):
    """TE10 wave impedance of the standard waveguide at f, ohm."""
    return 376.730 / math.sqrt(1 - (cutoff(f, 8.2e9, 12.4e9) / f) ** 2)


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
    "colaneri": "N.F. Colaneri, L.W. Shacklette, EMI shielding measurements of conductive polymer blends, IEEE Trans. Instrum. Meas. 41 (1992) 291–297. https://doi.org/10.1109/19.137363",
    "glover": "R.E. Glover, M. Tinkham, Conductivity of superconducting films for photon energies between 0.3 and 40 *kT*_{c}, Phys. Rev. 108 (1957) 243–256. https://doi.org/10.1103/PhysRev.108.243",
    "astm": "ASTM D4935-18, Standard test method for measuring the electromagnetic shielding effectiveness of planar materials, ASTM International, West Conshohocken, PA, 2018.",
}
DOI_KEYS = {
    "simon": "10.1080/03602558108067695", "dyre": "10.1103/RevModPhys.72.873", "joo": "10.1063/1.112717",
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


def rng(ids_, key="res"):
    v = [BY[i][key] for i in ids_]
    return f"{sg(min(v))} to {sg(max(v))}"


def between(ids_):
    return rng(ids_).replace(" to ", " and ")


def ids(prefix):
    return [b["id"] for b in bench if b["id"].startswith(prefix)]


def rci(s):
    return f"{s['rmse_lo']:.0f} to {s['rmse_hi']:.0f}"


DENSE = G[2]
DERIVED = [b for b in bench if b["sigma_basis"].startswith("derived")]
THIN_ERR = max(abs(slab_se(b["sigma"], b["t"], 0.5 * (b["f_lo"] + b["f_hi"]))
                   - 20 * math.log10(1 + 376.730 * b["sigma"] * b["t"] / 2)) for b in DERIVED)
GRADE_C_REASON = {b["paper"]: b["quality_note"] for b in GRADE_C}
BULK_LOW = [b for b in PRIM if b["group"] in (4, 5) and b["sigma"] < 1e3]
FILM_F3 = sum(abs(b["log_ratio"]) <= math.log10(3) for b in FILM)
TI = BY["iqbal2020_ti3cn_350"]
T_FIXED = next(t for t in np.arange(40e-6, 900e-6, 1e-6) if slab_se(TI["sigma"], t, 8.2e9, 6.557e9) >= TI["se"])
T_RS = next(t for t in np.arange(40e-6, 2000e-6, 1e-6)
            if slab_se(TI["sigma"] * 40e-6 / t, t, 8.2e9, 6.557e9) >= TI["se"])
EPDM = ids("rahaman2023_")
EPS50 = [slab_se(BY[i]["sigma"], BY[i]["t"], 10e9, 6.557e9, eps_r=50.0) for i in EPDM]
W_EPS0 = 2 * math.pi * 10e9 * 8.8541878e-12

_abs = P(
    f"A shield is specified, before it is made, by a conductivity and a thickness. We examine how well the "
    f"closed-form shielding effectiveness (SE) of a homogeneous sheet with those two properties reproduces published "
    f"measurements. We compile {N} measured SE values from {NP} papers on metal films and meshes, MXene and graphene "
    f"films, carbon–polymer composites, foams and aerogels (10 MHz to 40 GHz, 13 nm to 5 mm). Each value is linked to "
    f"a quoted sentence, table or caption, and each entry is graded for documentation by rules that do not use the "
    f"result. Predictions use the wave impedance of the measurement fixture, which for a rectangular waveguide raises "
    f"the predicted SE by 1.4 to 4.8 dB across the X band. The entries fall into two populations. For conductive "
    f"films (13 nm to 40 µm, above 10^{{4}} S m^{{−1}}; n = {S_FILM['n']} from {S_FILM['papers']} papers after "
    f"grading) the median residual is {sg(S_FILM['median'])} dB (95% interval {ci(S_FILM)} dB, resampling papers), "
    f"the median absolute residual is {S_FILM['medabs']:.0f} dB and the root-mean-square residual is "
    f"{S_FILM['rmse']:.0f} dB, or {S_FILM_NOTI['rmse']:.0f} dB without one annealing series. For bulk composites, "
    f"foams and aerogels (87 µm to 5 mm, below 10^{{4}} S m^{{−1}}; n = {S_BULK['n']}) the root-mean-square residual "
    f"is {S_BULK['rmse']:.0f} dB and the reported DC conductivity is not a usable input. Only {QUAL['A']} of the {N} "
    f"entries are documented without any caveat. The data cannot test whether porosity adds shielding. The data, "
    f"code and tests are openly available.")
_kw = P("*Keywords:* electromagnetic interference shielding; shielding effectiveness; benchmark dataset; waveguide "
        "measurement; MXene; polymer composites; sheet resistance")
for _p in (_abs, _kw):
    _p.paragraph_format.left_indent = _p.paragraph_format.right_indent = Inches(0.5)
_kw.paragraph_format.space_after = Pt(10)

# ---------------------------------------------------------------- 1
H("1. Introduction")
P(f"A shield is usually specified before it exists. The designer chooses a material and a thickness, and the two "
  f"numbers available at that stage are a conductivity and a dimension. For a homogeneous conducting sheet, "
  f"Schelkunoff’s transmission-line analysis {c('schelkunoff')} turns those two numbers into a shielding "
  f"effectiveness (SE) in closed form {c('schulz', 'ott', 'celozzi')}. This paper asks how closely that estimate "
  f"matches what has been measured on the materials now being developed, and where it fails.")
P(f"The comparison between sheet theory and measurement has a long history. The thin-film transmission formula "
  f"goes back to Glover and Tinkham {c('glover')}, and Simon’s empirical expression {c('simon')} has been used for "
  f"four decades to relate SE to conductivity, thickness and frequency. Colaneri and Shacklette {c('colaneri')} and "
  f"Joo and Epstein {c('joo')} tested thin- and thick-sheet expressions on conducting polymers, and Al-Saleh and "
  f"Sundararaj {c('alsaleh')} examined the mechanisms in carbon nanotube composites. For MXenes, Yun et al. "
  f"{c('yun')} and Han et al. {c('han2020_ti3c2_5um')} compared the measured SE of highly conductive films, from "
  f"assemblies of monolayers to micrometre thickness, with sheet and transfer-matrix calculations and found close "
  f"agreement, while Iqbal et al. {c('iqbal2020_ti3cn_350')} reported absorption in annealed Ti_{{3}}CNT_{{x}} far "
  f"above what a transfer-matrix calculation gives. Reviews tabulate SE with conductivity and thickness across "
  f"material classes {c('iqbalrev', 'chung')}, and common misreadings of SE and of its reflection and absorption "
  f"parts have been documented {c('peng')}. That a thin conducting film shields according to its sheet resistance "
  f"is therefore established.")
P(f"What is less clear is how such comparisons hold up when one model is applied uniformly to data from many "
  f"groups, with every input traced to where it was printed. Two practical matters are usually left aside. Most SE "
  f"values are measured in a rectangular waveguide, where the wave impedance differs from that of free space "
  f"{c('moucka', 'wilson')} by an amount comparable to the agreement being claimed. And tabulated values mix band "
  f"averages with maxima, and conductivities measured on the specimen with values quoted for the material in "
  f"general.")
P(f"We make three contributions. The first is a curated set of {N} measured SE values from {NP} papers in which "
  f"each value, thickness, frequency and conductivity is linked to a quotation, with a documentation grade "
  f"assigned by stated rules. The second is a comparison with the slab model that accounts for the fixture and "
  f"that treats papers, not samples, as independent. The third is a quantified statement of what follows and "
  f"what does not: the size and uncertainty of the agreement for conductive films, the failure of the reported DC "
  f"conductivity as an input for bulk composites, and the extent to which the published record is documented well "
  f"enough to support either conclusion. The set is a sample of the literature and not a systematic review, and we "
  f"report no new measurements; Section 5.3 states what this limits.")

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
P("With *B* included, the sum is algebraically identical to the transmission loss of the idealised slab. The "
  "idealisation is a laterally infinite, homogeneous, isotropic, linear sheet with local response, a plane wave at "
  "normal incidence, and no substrate. For a good conductor (|*η*| ≪ *Z*_{0}) the correction *B* is negative and "
  "becomes negligible once *A* exceeds about 10 dB; the skin depth is *δ* = (2/*ωµσ*)^{1/2} and *A* = 8.686 *t*/*δ*.")
P(f"When the sheet is much thinner than the skin depth the three terms reduce to a result that depends only on the "
  f"sheet resistance *R*_{{s}} = 1/(*σt*) {c('glover')}:")
EQ(r"SE \approx 20\log_{10}\left(1+\frac{Z_0}{2R_s}\right)")
P(f"For the {len(DERIVED)} entries below whose conductivity is derived from a printed sheet resistance, Eq. (5) "
  f"agrees with Eqs. (2)–(4) to within {THIN_ERR:.1f} dB, so the nominal thickness has almost no influence on those "
  f"predictions. In a waveguide the same holds with *Z*_{{0}} replaced by the wave impedance of Section 2.2.")

H("2.2 Measurement fixture", 2)
P(f"Equations (1)–(5) describe a transverse electromagnetic wave, which is what a coaxial cell {c('astm')} "
  f"launches. In a rectangular waveguide the sample is illuminated by the TE_{{10}} mode. The field is still "
  f"tangential to the sheet, but the wave impedance and propagation constant in each medium become")
EQ(r"Z_{\mathrm{TE}}=\frac{j\omega\mu}{\gamma_{\mathrm{TE}}},\qquad \gamma_{\mathrm{TE}}=\sqrt{k_c^{2}-\omega^{2}\mu\,\varepsilon_c},\qquad k_c=\frac{2\pi f_c}{c}")
P(f"where *f*_{{c}} is the cutoff frequency (6.557 GHz for WR-90, 9.488 GHz for WR-62). Equations (2)–(4) hold with "
  f"*Z*_{{0}} and *η* replaced by the TE_{{10}} impedances of air and of the sheet, provided the sample fills the "
  f"cross-section without gaps and higher-order modes are not excited. In air *Z*_{{TE}} = "
  f"*Z*_{{0}}/[1 − (*f*_{{c}}/*f*)^{{2}}]^{{1/2}}, which in WR-90 is {z_te(8.0e9):.0f} Ω at 8.0 GHz, {z_te(10e9):.0f} Ω "
  f"at 10 GHz and {z_te(12.4e9):.0f} Ω at 12.4 GHz. For a reflecting sheet the waveguide therefore reads "
  f"{20 * math.log10(z_te(8.0e9) / 376.73):.1f}, {20 * math.log10(z_te(10e9) / 376.73):.1f} and "
  f"{20 * math.log10(z_te(12.4e9) / 376.73):.1f} dB above the free-space value at those frequencies {c('moucka')}.")
P(f"Each entry is predicted with a fixture model chosen as follows. Where the source states a rectangular "
  f"waveguide ({FIX['waveguide']} entries) the TE_{{10}} model is used, and where it states a coaxial cell "
  f"({FIX['coaxial']}) the TEM model. Where the text we read states neither, an entry whose frequencies all lie "
  f"between 8 and 18 GHz is assumed to be a waveguide measurement ({FIX['assumed-waveguide']} entries), because those "
  f"bands are waveguide designations, and a broadband entry is treated as TEM ({FIX['assumed-tem']}). An entry "
  f"confined to 8–13 GHz is taken as a single WR-90 run; entries that span the X and Ku bands switch to WR-62 above "
  f"12.4 GHz. Supplementary files, where the fixture is often described, were not consulted. Section 4.1 reports "
  f"the results under the alternative assumptions.")

H("2.3 Residual and effective conductivity", 2)
P("For a sample with reported conductivity *σ*, thickness *t* and measurement frequency *f*, the residual is")
EQ(r"\Delta SE = SE_{\mathrm{measured}} - SE_{\mathrm{slab}}(\sigma,\,t,\,f)")
P(f"with *µ*_{{r}} = 1 and *ε*_{{r}} = 1. When a source reports a value for a band, the prediction is the arithmetic "
  f"mean of the predicted SE in decibels at 41 equally spaced frequencies across that band. Setting *ε*_{{r}} = 1 is "
  f"adequate when *σ* ≫ *ωε*_{{0}}, which is {W_EPS0:.2f} S m^{{−1}} at 10 GHz. It is not adequate for the least "
  f"conducting samples, which are lossy dielectrics; for these the prediction is close to zero, the residual is "
  f"close to the measured SE, and it carries little information.")
P("A given residual corresponds to a different change in conductivity for a thin sheet than for one thicker than "
  "the skin depth. We therefore also report the effective conductivity *σ*_{eff}, the conductivity of the "
  "homogeneous sheet that gives the measured SE in the same fixture model,")
EQ(r"SE_{\mathrm{slab}}(\sigma_{\mathrm{eff}},\,t,\,f)=SE_{\mathrm{measured}}")
P("and the ratio *r* = *σ*_{eff}/*σ*. The predicted SE increases monotonically with conductivity for every entry "
  "over 10^{−6} to 10^{9} S m^{−1}, so the inversion is unique here. The residual and the ratio are descriptive and "
  "do not identify a cause. A ratio above one can arise from loss that the DC conductivity does not capture, from "
  "a conductivity measured on a different specimen, from a thickness larger than the stated one, or from an SE "
  "reported as a maximum; a ratio below one can arise from leakage at the sample edge, cracks, incomplete contact "
  "with the fixture, a conductivity that overstates that of the measured sheet, or a thickness that is smaller.")

H("2.4 Implementation", 2)
P("The analysis code evaluates Eqs. (1)–(6) directly. An independent implementation of the Schelkunoff "
  "decomposition and a transfer-matrix solver in the same repository agree with it to within 10^{−6} dB over "
  "conductivities from 10^{−3} to 6 × 10^{7} S m^{−1}; these identities, the limit in which the TE_{10} result "
  "reduces to the free-space result, and the inversion for *σ*_{eff} are enforced by automated tests.")

# ---------------------------------------------------------------- 3
H("3. Data")
H("3.1 Collection", 2)
P("A measurement was recorded if its primary source prints the measured total SE in decibels, the thickness of the "
  "sample or of its conductive layer, the frequency or band, and the electrical conductivity or sheet resistance. "
  "Values read from plots, values a paper quotes from other work, and calculated SE were not recorded. Where a "
  "source prints a band average it is used in preference to a maximum or a single-frequency value. Conductivity "
  "units were kept as printed and converted once; where a source gives sheet resistance, the conductivity is "
  "1/(*R*_{s}*t*). Each entry stores the citation, the values as printed, a verbatim quotation that locates them, "
  "the measurement method and any caveat.")
P("The set is a curated sample and not a systematic review. Sources were found by keyword search and by "
  "following citations, and were kept when an open full text was available and the criteria above were met; no "
  "count of the papers screened was kept. Extraction was assisted by a large language model. Every entry was "
  "afterwards compared with the source text in a separate pass, which found no recorded value that differed from "
  "its source. Supplementary files were not consulted, and no second human extraction was made.")

H("3.2 Documentation grade", 2)
nC_papers = len({b["paper"] for b in GRADE_C})
P(f"Sources do not document their samples equally. Each entry is graded by rules that refer only to what the "
  f"source prints; the rules and their application to each entry are in the repository. Grade C ({QUAL['C']} "
  f"entries from {nC_papers} papers) is given when two values printed for the same quantity of the sample differ "
  f"by more than 10%, when the conductivity unit is printed inconsistently, when the stated conductivity "
  f"contradicts the printed sheet resistance and thickness, when conductivity and thickness refer to different "
  f"objects, or when we could read only the abstract (Table 3). Grade B ({QUAL['B']} entries) is given for one or "
  f"more of the following: a conductivity quoted for the film type or as a range and not measured on the specimen; "
  f"a thickness that is nominal, approximate or measured before a processing step; an SE or conductivity printed "
  f"as approximate; a sample on a substrate or made of more than one layer; a two-probe or low-frequency AC "
  f"conductivity; an SE whose frequency or averaging is not explicit; a fixture not stated in the text. Grade A "
  f"({QUAL['A']} entries) has none of these. Grades A and B form the primary set of {NPRIM} entries from {NPPRIM} "
  f"papers. Results are given for the primary set and for all {N} entries; grade C entries are drawn with open "
  f"symbols. The grading was done by the authors and was not repeated by an independent rater.")

H("3.3 Coverage", 2)
se_all = [b["se"] for b in bench]
P(f"The {N} entries come from {NP} papers published between {YEARS[0]} and {YEARS[1]} (Appendix A). Measured SE "
  f"ranges from {min(se_all):.1f} to {max(se_all):.0f} dB, thickness from 13 nm to 5 mm, and reported conductivity "
  f"from 2.5 × 10^{{−3}} to 1.2 × 10^{{7}} S m^{{−1}}. Most measurements are in the X band (8.2–12.4 GHz); the others "
  f"extend from 10 MHz to 40 GHz. Of the {N} values, {N_BAND} are stated to be band averages; the others are "
  f"single-frequency values, maxima, or values printed without qualification. {NUMW.get(SE90, SE90).capitalize()} measured values "
  f"are at or above 90 dB, where the dynamic range of the fixture, which most sources do not state, becomes a "
  f"concern.")
P(f"The entries form two populations with no overlap. Every primary entry with a reported conductivity above "
  f"10^{{4}} S m^{{−1}} is a film, coating or mesh between 13 nm and 40 µm thick, usually characterised by four-probe "
  f"sheet resistance; every primary entry below it is a bulk composite, foam, aerogel or coated fabric between 87 µm "
  f"and 5 mm thick. No primary entry has a reported conductivity between {sci(GAP[0])} and {sci(GAP[1])} "
  f"S m^{{−1}}. Conductivity is therefore confounded with thickness, material class and the way conductivity was "
  f"measured, and the value 10^{{4}} S m^{{−1}} used below labels the gap between the populations; it is not a "
  f"threshold identified by the data. Published shielding materials are also selected for shielding well: a "
  f"composite with low conductivity and negligible SE is rarely reported.")

# ---------------------------------------------------------------- 4
H("4. Results")
H("4.1 Overall comparison", 2)
P(f"Figure 1 compares measured and predicted SE, and Table 1 summarises the residuals. Intervals are obtained by "
  f"resampling papers with replacement, because samples from one paper share a fixture, a conductivity method and "
  f"an author. The shaded ±10 dB band in the figures is a visual guide and corresponds to a factor of ten in "
  f"transmitted power.")
P(f"For the conductive films of the primary set (n = {S_FILM['n']} from {S_FILM['papers']} papers) the median "
  f"residual is {sg(S_FILM['median'])} dB (95% interval {ci(S_FILM)} dB) and the median absolute residual is "
  f"{S_FILM['medabs']:.1f} dB; {S_FILM['w10']} entries are within ±10 dB, {S_FILM['w5']} within ±5 dB and "
  f"{S_FILM['w3']} within ±3 dB. The root-mean-square (RMS) residual is {S_FILM['rmse']:.1f} dB (95% interval "
  f"{rci(S_FILM)} dB). It is dominated by the two annealed Ti_{{3}}CNT_{{x}} films of Section 4.4: without them it is "
  f"{S_FILM_NOTI['rmse']:.1f} dB ({rci(S_FILM_NOTI)} dB). The RMS of the {len(PM)} per-paper medians is "
  f"{S_FILM['paper_rms']:.1f} dB, and {sum(abs(x) <= 10 for x in PM)} of the {len(PM)} are within ±10 dB. Six "
  f"papers supply {sum(sorted([sum(b['paper'] == p for b in FILM) for p in FILM_PAPERS])[-6:])} of the "
  f"{S_FILM['n']} entries. Including the grade C entries (n = {S_FILM_ALL['n']} from {S_FILM_ALL['papers']} papers) "
  f"the median is {sg(S_FILM_ALL['median'])} dB, the median absolute residual {S_FILM_ALL['medabs']:.1f} dB and the "
  f"RMS residual {S_FILM_ALL['rmse']:.1f} dB.")
P(f"For the bulk population (n = {S_BULK['n']} from {S_BULK['papers']} papers) the RMS residual is "
  f"{S_BULK['rmse']:.0f} dB ({rci(S_BULK)} dB), the residuals span {sg(S_BULK['min'])} to {sg(S_BULK['max'])} dB, and "
  f"{S_BULK['w10']} entries are within ±10 dB.")
P(f"The dependence on the choices of Section 2.2 is as follows. If the entries with an assumed fixture are "
  f"predicted with the TEM model, the median residual of the films becomes {sg(S_FILM_STATED['median'])} dB (RMS "
  f"{S_FILM_STATED['rmse']:.1f} dB); if every entry is predicted with the free-space model, it becomes "
  f"{sg(S_FILM_FREE['median'])} dB (RMS {S_FILM_FREE['rmse']:.1f} dB). The fixture model moves the centre of the "
  f"distribution by {abs(S_FILM_FREE['median'] - S_FILM['median']):.1f} dB and leaves its width almost unchanged. "
  f"Restricting the films to values stated as band averages (n = {S_FILM_BAND['n']} from {S_FILM_BAND['papers']} "
  f"papers) gives a median of {sg(S_FILM_BAND['median'])} dB and an RMS residual of {S_FILM_BAND['rmse']:.1f} dB. "
  f"Moving the dividing conductivity anywhere between 3 × 10^{{3}} and 3 × 10^{{4}} S m^{{−1}} changes the film "
  f"population by at most one entry.")
FIG("fig1_parity.png",
    f"Figure 1. Measured shielding effectiveness and the homogeneous-slab prediction for all {N} entries. "
    f"(a) Parity plot. (b) Residual against reported conductivity. Open symbols are grade C entries. The band marks "
    f"±10 dB.")


def trow(label, s, interval=True):
    return [label, s["n"], s["papers"], sg(s["median"]), ci(s) if (interval and s["papers"] >= 10) else "–",
            f"{s['medabs']:.1f}", f"{s['rmse']:.1f}", f"{s['w10']} of {s['n']}"]


TABLE("Table 1. Residuals ΔSE by population and material group. The 95% interval of the median is from resampling "
      "papers and is given where at least ten papers contribute. |ΔSE|: absolute residual; RMS: root mean square.",
      ["Subset", "*n*", "Papers", "Median (dB)", "95% interval (dB)", "Median |ΔSE| (dB)", "RMS (dB)", "Within ±10 dB"],
      [trow("Conductive films, primary set", S_FILM),
       trow("  without annealed Ti_{3}CNT_{x}", S_FILM_NOTI),
       trow("  all grades", S_FILM_ALL),
       trow("Bulk composites, foams, aerogels, primary set", S_BULK),
       trow("  all grades", S_BULK_ALL)] +
      [trow(f"{GROUPS[i][0]}, primary set", G[i], interval=False) for i in range(len(GROUPS))] +
      [trow("Primary set", S_PRIM), trow(f"All {N} entries", S_ALL)],
      widths=[2.0, 0.28, 0.45, 0.52, 0.8, 0.62, 0.42, 0.6], size=7.5)
FIG("fig2_residual_by_group.png",
    "Figure 2. Residuals by material group. Each symbol is one entry (open: grade C). The bar spans the "
    "interquartile range of the primary set and the tick marks its median.")

H("4.2 Conductive films", 2)
P(f"Continuous metal and oxide films follow the model within a few decibels. An In_{{2}}O_{{3}}/Ag/In_{{2}}O_{{3}} "
  f"stack with a 13 nm silver layer {c('voronin2025_iai3')}, a 440 nm La-doped BaSnO_{{3}} film on a 0.5 mm "
  f"MgAl_{{2}}O_{{4}} substrate {c('jeon2023_blso')} and a sintered silver film on paper {c('liu2020_ag_film')} have "
  f"residuals of {dres('voronin2025_iai3')}, {dres('jeon2023_blso')} and {dres('liu2020_ag_film')} dB. Two silver "
  f"films loaded with nickel particles from the last study have {dres('liu2020_agni560_2to1')} and "
  f"{dres('liu2020_agni560_1to2')} dB.")
P(f"Dense MXene and graphene films behave in the same way over three orders of magnitude in thickness "
  f"(n = {DENSE['n']} from {DENSE['papers']} papers; median {sg(DENSE['median'])} dB, median absolute residual "
  f"{DENSE['medabs']:.1f} dB, RMS {DENSE['rmse']:.1f} dB, {DENSE['w10']} within ±10 dB). Sprayed "
  f"Ti_{{3}}C_{{2}}T_{{x}} films of about 40 and 137 nm and a 5 µm filtered film {c('han2020_ti3c2_5um')} have "
  f"residuals of {dres('han2020_ti3c2_40nm')}, {dres('han2020_ti3c2_137nm')} and {dres('han2020_ti3c2_5um')} dB, a "
  f"29 nm oxygen-substituted film {c('park2026_mxene_29nm')} {dres('park2026_mxene_29nm')} dB, micrometre-thick films "
  f"{c('wan2022_mxene_lm', 'huang2025_mxene_mx80')} between {between(ids('wan2022_') + ['huang2025_mxene_mx80'])} dB, "
  f"and 10 to 40 µm films {c('iqbal2020_ti3c2_asis', 'hassan2024_al_mxene')} between "
  f"{between(['iqbal2020_ti3c2_asis', 'iqbal2020_ti3c2_350', 'iqbal2020_ti3cn_asis'] + ids('hassan2024_'))} dB. The "
  f"largest departure in the group is a 12 µm film {c('ahmed2023_ti3c2_12um')} at {dres('ahmed2023_ti3c2_12um')} dB. "
  f"With all grades included the group has n = {G_ALL[2]['n']} and an RMS residual of {G_ALL[2]['rmse']:.1f} dB; the "
  f"difference is five films from one study (Section 4.5).")
P(f"These results agree with what Han et al. {c('han2020_ti3c2_5um')} and Yun et al. {c('yun')} found for their own "
  f"films: at a resolution of about {DENSE['rmse']:.0f} dB, a dense laminate of high conductivity shields as a "
  f"homogeneous sheet of that conductivity and thickness does. The evidence is not fully independent. For the "
  f"thinnest films the thickness was obtained optically and the conductivity is a value quoted for the film type, "
  f"and several of the studies come from related groups.")
mesh = G[1]
P(f"Metal meshes and coated fabrics are less uniform (n = {mesh['n']} from {mesh['papers']} papers, RMS "
  f"{mesh['rmse']:.1f} dB). Electroplated copper meshes measured at 10 MHz in a coaxial cell "
  f"{c('voronin2022_cumesh_LCM30')} lie between {between(ids('voronin2022_'))} dB. In waveguide measurements at "
  f"8–18 GHz, silver meshes in PET {c('zarei2024_agmesh_1')} lie between {between(ids('zarei2024_'))} dB and silver "
  f"meshes embedded in glass {c('li2023_agmesh_W1P25t0.5')} between {between(ids('li2023_agmesh'))} dB. A mesh whose "
  f"pitch is small compared with the wavelength behaves as a sheet with impedance *R*_{{s}} + *jωL*_{{s}}, where "
  f"the inductance *L*_{{s}} is set by the geometry {c('casey')}. The inductive term is negligible at 10 MHz and "
  f"lowers the SE at microwave frequencies, which is the direction of the residuals for the meshes in PET and is "
  f"consistent with the agreement at 10 MHz. It does not account for the positive residuals of the meshes in "
  f"glass. We have not included the inductive term, because the mesh dimensions are not available for every "
  f"sample, so the homogeneous-sheet comparison is of limited value for meshes at these frequencies.")

H("4.3 Bulk composites, foams and aerogels", 2)
wani = ids("wani2026_")
P(f"For the bulk population the reported conductivity does not predict the measured SE (Fig. 1b). Figure 3 shows "
  f"the same data as effective conductivity. For the films, the effective and reported values agree within a "
  f"factor of three for {FILM_F3} of the {len(FILM)} primary entries, which is the same agreement expressed in "
  f"different units. For the {len(BULK_LOW)} composites, foams and aerogels with reported conductivities below "
  f"10^{{3}} S m^{{−1}}, the reported values cover more than five decades while the effective conductivity stays "
  f"between {min(b['sigma_eff'] for b in BULK_LOW):.0f} and {max(b['sigma_eff'] for b in BULK_LOW):.0f} S m^{{−1}}.")
P(f"This narrow range is partly a consequence of how the data arise. A sample a few millimetres thick that shields "
  f"10 to 50 dB in the X band must have an effective conductivity of this order, and samples that shield less are "
  f"seldom published. For the {len(LOW)} primary entries reported below 1 S m^{{−1}}, which come from two papers, "
  f"the reported conductivity is comparable to or below *ωε*_{{0}}, the slab prediction with *ε*_{{r}} = 1 is at most "
  f"{max(b['slab'] for b in LOW):.0f} dB, and a positive residual follows by construction. The most extreme case "
  f"should be read with caution: carbon-black-filled EPDM sheets 1 mm thick {c('rahaman2023_epdm_cb_20phr')} with "
  f"DC conductivities of 2.5 × 10^{{−3}} to 6.3 × 10^{{−2}} S m^{{−1}}, measured with a DC resistance meter, are "
  f"reported to shield {min(BY[i]['se'] for i in EPDM):.1f} to {max(BY[i]['se'] for i in EPDM):.1f} dB at 10 GHz with "
  f"a scalar analyser. Reproducing those values requires a loss equivalent to "
  f"{min(BY[i]['sigma_eff'] for i in EPDM):.0f} to {max(BY[i]['sigma_eff'] for i in EPDM):.0f} S m^{{−1}}; a relative "
  f"permittivity of 50 with the DC conductivity gives about {max(EPS50):.0f} dB, by reflection. The same source "
  f"prints 1 kHz conductivities 50 to 200 times the DC values.")
P(f"One physical reading, which we state as a hypothesis and do not test, is that the AC conductivity of a "
  f"disordered conductor–insulator composite exceeds its DC value and rises with frequency "
  f"{c('bergman', 'clerc', 'mclachlan98', 'dyre')}. It cannot be the whole account, because several bulk entries "
  f"well above their percolation threshold have negative residuals (Section 4.5). The conclusion we draw is "
  f"limited to the practical one: for this population a DC conductivity, however obtained, is not an input from "
  f"which microwave SE can be estimated, and the complex permittivity or sheet impedance at the measurement "
  f"frequency is needed instead.")
FIG("fig3_effective_conductivity.png",
    "Figure 3. Effective conductivity, obtained by inverting the measured shielding effectiveness with the slab "
    "model, against reported conductivity. The band marks a factor of ten. Symbols as in Fig. 1.", width=3.3)

H("4.4 Porous and annealed samples", 2)
P(f"Porous films, foams and aerogels are often described as shielding better than their conductivity implies, "
  f"through internal reflection and absorption in the pore structure. The data here cannot test that statement. "
  f"Across studies, porosity is confounded with low reported conductivity and with thickness. Within studies, the "
  f"set contains only three pairs in which a dense and a porous state of one material are both documented "
  f"(Table 2), and they do not agree.")
P(f"A segregated polysulfone/nanotube composite {c('xie2020_psu_cnt_solid')} has a residual of "
  f"{dres('xie2020_psu_cnt_solid')} dB as a solid and {dres('xie2020_psu_cnt_foam')} dB after foaming at the same "
  f"2 mm thickness. A graphene film {c('lin2019_graphene_lg4')} has {dres('lin2019_graphene_lg4')} dB when compressed "
  f"to 14 µm and {dres('lin2019_graphene_plg')} dB in its porous state at 35.6 µm. In both, conductivity and SE fall "
  f"together roughly as the slab model expects. Annealed Ti_{{3}}CNT_{{x}} {c('iqbal2020_ti3cn_350')} is different: "
  f"the residual rises from {dres('iqbal2020_ti3cn_asis')} dB as made to {dres('iqbal2020_ti3cn_250')} dB after "
  f"annealing at 250 °C and {dres('iqbal2020_ti3cn_350')} dB at 350 °C, while Ti_{{3}}C_{{2}}T_{{x}} films in the same "
  f"study change from {dres('iqbal2020_ti3c2_asis')} to {dres('iqbal2020_ti3c2_350')} dB.")
P(f"Three cautions apply to that series. Our prediction uses the 40 µm thickness before annealing; the measured "
  f"SE would require about {T_FIXED * 1e6:.0f} µm at the stated conductivity, or about {T_RS * 1e6:.0f} µm if the "
  f"product of conductivity and thickness is conserved, and the porosity that the source reports for the annealed "
  f"films is far too small for either. The values are taken at 8.2 GHz, the lower edge of the band, and the 99 and "
  f"116 dB reported are the highest in the set and at or beyond the usual dynamic range of a waveguide "
  f"measurement. And the source itself compared its measurements with transfer-matrix calculations that include "
  f"the porosity and found that they do not account for the result, so our residual restates the authors’ finding "
  f"and does not add to it. The series remains the only controlled comparison in the set that points to a "
  f"contribution beyond conductivity and thickness.")
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

H("4.5 Residuals beyond ±10 dB", 2)
BIG_POS_FILM = max((b for b in OUT_POS if b["sigma"] >= 1e4 and b["id"] not in ANNEALED), key=lambda b: b["res"])
P(f"Of the {NPRIM} primary entries, {len(OUTL)} have residuals beyond ±10 dB, {len(OUT_POS)} positive and "
  f"{len(OUT_NEG)} negative. We treat both signs in the same way: the reported conductivity, thickness and SE are "
  f"not mutually consistent with a homogeneous sheet, and we list possible causes without assigning one. "
  f"Of the positive residuals, {sum(b['sigma'] < 1e4 for b in OUT_POS)} belong to the bulk population. Among them, a "
  f"film of Ti_{{3}}C_{{2}}T_{{x}} in an aramid nanofibre matrix {c('zhang2022_mxene_anf_pva')} is reported at 70 dB "
  f"for 120 µm and 357 S m^{{−1}} ({dres('zhang2022_mxene_anf_pva')} dB), which corresponds to an effective "
  f"conductivity {BY['zhang2022_mxene_anf_pva']['sigma_eff'] / BY['zhang2022_mxene_anf_pva']['sigma']:.0f} times the "
  f"stated one. Among the films, the largest positive residual apart from annealed Ti_{{3}}CNT_{{x}} is "
  f"{sg(BIG_POS_FILM['res'])} dB, for a silver mesh in glass {c('li2023_agmesh_W1P25t0.5')}.")
P(f"The largest negative residuals are a segregated graphene/polypropylene composite {c('alam2017_gnp_pp_10')} "
  f"({dres('alam2017_gnp_pp_10')} dB), a 4.5 mm carbon nanofibre aerogel {c('chen2025_cnf_pani_aerogel')} "
  f"({dres('chen2025_cnf_pani_aerogel')} dB), whose predicted SE of {BY['chen2025_cnf_pani_aerogel']['slab']:.0f} dB "
  f"is beyond what a waveguide measurement normally resolves, and a stack of two MWCNT/PMMA bars "
  f"{c('pande2009_mwcnt_pmma_scb2')} ({dres('pande2009_mwcnt_pmma_scb2')} dB) joined by an insulating adhesive. "
  f"Possible causes of a negative residual include leakage around the sample, incomplete contact between a "
  f"free-standing film and the fixture, cracks and pinholes, a measured SE limited by the dynamic range, and a "
  f"conductivity that overstates that of the measured specimen.")
P(f"The grade C entries are listed in Table 3. Five cellulose-nanofibre/Ti_{{3}}C_{{2}}T_{{x}} films from one study "
  f"{c('wang2026_tocnf_mxene_M25')} were graded C because the source prints the conductivity unit as S cm^{{−1}} in "
  f"its equation and as “s m-1” in its table. Their residuals are {rng(ids('wang2026_'))} dB as tabulated, and "
  f"within {max(abs(BY[i]['se'] - predict(BY[i]['sigma'] / 100, BY[i]['t'], BY[i]['f_lo'], BY[i]['f_hi'], False)) for i in ids('wang2026_')):.0f} "
  f"dB if the tabulated conductivities are divided by 100. We cannot tell from the paper which reading is correct. "
  f"The case shows both the use of the comparison as a check and the weight that single studies carry in a set of "
  f"this size: these five entries alone change the RMS residual of the dense films from {G[2]['rmse']:.1f} to "
  f"{G_ALL[2]['rmse']:.1f} dB.")


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
TABLE("Table 3. Grade C entries and the reason for the grade.",
      ["Ref.", "*n*", "Reason for the grade", "ΔSE (dB)"], c_rows,
      widths=[0.4, 0.3, 3.8, 1.0], size=8, align=["c", "c", "l", "c"])

# ---------------------------------------------------------------- 5
H("5. Discussion")
H("5.1 What can be estimated before a shield is built", 2)
P(f"For a conductive film, the slab model with the target sheet resistance gives the SE that published films of "
  f"that sheet resistance were measured to have, typically within {S_FILM['medabs']:.0f} dB and with an RMS "
  f"difference of {S_FILM_NOTI['rmse']:.0f} to {S_FILM['rmse']:.0f} dB depending on whether one anomalous series is "
  f"counted. An uncertainty of this size corresponds to a factor of two to three in sheet resistance. It is "
  f"enough to decide whether a candidate film is in the right range before it is made, and not enough to replace "
  f"a measurement or to rank films that differ by a few decibels. The result restates, with an uncertainty "
  f"obtained from many groups and with the fixture accounted for, what is known for individual material families; "
  f"it is not a new design rule.")
P("For bulk composites, foams and aerogels, no estimate of comparable quality can be made from the DC "
  "conductivity, and a model that supplies only a DC percolation conductivity will not predict the shielding of "
  "the samples that are reported. Because the two populations do not overlap in thickness or conductivity, the "
  "data do not show where between them the sheet estimate stops being useful.")

H("5.2 Reporting", 2)
n_fix_unstated = FIX["assumed-waveguide"] + FIX["assumed-tem"]
P(f"The most direct finding concerns the record itself. Of {N} entries, {QUAL['A']} are documented without a "
  f"caveat, {n_fix_unstated} do not state the fixture in the text, and only {N_BAND} are stated to be band "
  f"averages. The comparison would be sharper if sources reported a small set of quantities consistently: "
  f"the in-plane conductivity or sheet resistance measured on the specimen used for the SE measurement, with the "
  f"method; the thickness of that specimen as measured; the fixture and mode (rectangular waveguide and its "
  f"designation, or coaxial cell, with the standard followed {c('astm', 'ieee2715')}); the frequency of each SE "
  f"value and whether it is a band average or a maximum; the dynamic range of the set-up; and the slab prediction "
  f"beside the measured value. At normal incidence, and for the TE_{{10}} mode, the electric field lies in the "
  f"plane of the sheet, so it is the in-plane conductivity that enters the prediction.")

H("5.3 Limitations", 2)
P(f"The set is small and is a convenience sample: {NP} papers with open full text, found without a recorded "
  f"search protocol, out of a literature of thousands. Extraction was assisted by a language model and verified "
  f"against the source text, but not duplicated by a second person, and supplementary files, where fixtures and "
  f"methods are often given, were not read. One early and widely cited study could be read only as an abstract "
  f"and is graded C for that reason. The grading rules were written by the authors, who knew the residuals, and "
  f"were not applied by an independent rater; results for all entries are given beside those for the primary set "
  f"so that the effect of the grading can be judged.")
P(f"The two populations are confounded in conductivity, thickness, material class and conductivity method, so "
  f"no statement about a conductivity threshold can be made. Several groups rest on a few papers (Table 1), six "
  f"papers supply most of the film entries, and there are no bulk metal foils, ferromagnetic sheets or "
  f"multilayers with per-layer properties. Band averages, maxima, single-frequency values and unqualified values "
  f"are pooled, and a maximum compared with a band-mean prediction biases the residual upward. For "
  f"{n_fix_unstated} entries the fixture is assumed. The model takes *µ*_{{r}} = 1 and *ε*_{{r}} = 1 and omits "
  f"substrates and mesh inductance, so magnetic loss, dielectric loss, substrate reflections and mesh geometry "
  f"all appear in the residual. No uncertainty is propagated from the reported conductivity and thickness, "
  f"because sources rarely state one. We report no measurements of our own; reference sheets of known sheet "
  f"resistance measured in both a waveguide and a coaxial cell would test the fixture correction directly.")
P("A larger and better documented set would allow the factors that determine shielding to be organised in a "
  "structural model, with composition, processing and thickness as independent variables, conductivity and "
  "porosity as mediating variables, and SE as the outcome.")

H("6. Conclusions")
P(f"A curated set of {N} shielding measurements from {NP} papers, each traceable to its source and graded for "
  f"documentation, was compared with the closed-form homogeneous-slab model for the measurement fixture. For "
  f"conductive films between 13 nm and 40 µm the model reproduces the measurements with a median residual of "
  f"{sg(S_FILM['median'])} dB ({ci(S_FILM)} dB), a median absolute residual of {S_FILM['medabs']:.0f} dB and an RMS "
  f"residual of {S_FILM_NOTI['rmse']:.0f} to {S_FILM['rmse']:.0f} dB. The waveguide wave impedance contributes 1.4 "
  f"to 4.8 dB to the prediction across the X band and belongs in any such comparison. For bulk composites, foams "
  f"and aerogels the reported DC conductivity does not predict the measured SE. The set is too small, and its two "
  f"populations too different, to locate a boundary between these regimes or to test whether porosity adds "
  f"shielding. Only {QUAL['A']} of {N} entries are documented without a caveat; stating the fixture, the in-plane "
  f"conductivity of the measured specimen, the averaging of the SE value and the slab prediction would make such "
  f"comparisons routine.")

H("Declaration of competing interest")
P("The authors declare that they have no known competing financial interests or personal relationships that could "
  "have appeared to influence the work reported in this paper.")
H("Funding")
P("This research did not receive any specific grant from funding agencies in the public, commercial, or "
  "not-for-profit sectors.")
H("Data availability")
P("The data file with quotations and grades, the grading rules, the analysis code, the automated tests and the "
  "scripts that generate every number, table and figure in this paper are available at "
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


FX = {"waveguide": "W", "coaxial": "C", "assumed-waveguide": "W\u2217", "assumed-tem": "T\u2217"}
TABLE("Table A1. All benchmark entries. *t*: thickness; *σ*: reported conductivity in S m^{−1} (for entries "
      "reported as sheet resistance, 1/(*R*_{s}*t*)); Fx.: fixture (W waveguide, C coaxial cell, T transverse electromagnetic; an asterisk "
      "marks an assumed fixture); Gr.: grade; SE: measured; Slab: prediction for the fixture; ΔSE: residual; "
      "log *r*: log_{10}(*σ*_{eff}/*σ*).",
      ["Material", "*t*", "Frequency", "*σ*", "Fx.", "Gr.", "SE (dB)", "Slab (dB)", "ΔSE (dB)", "log *r*", "Ref."],
      [[chem(b["material"]), ft(b["t"]), ff(b), fsig(b["sigma"]), FX[b["fixture"]], b["quality"],
        f"{b['se']:.1f}", f"{b['slab']:.1f}", sg(b["res"]), sg(b["log_ratio"], 2), f"[{num(b['id'])}]"]
       for b in sorted(bench, key=lambda b: (b["group"], b["citation"], b["id"]))],
      widths=[1.28, 0.46, 0.78, 0.64, 0.27, 0.23, 0.36, 0.38, 0.42, 0.38, 0.34], size=7)

H("References")
for key, i in sorted(REFNUM.items(), key=lambda kv: kv[1]):
    par = P(f"[{i}]\t{REFTEXT[key]}", align=LEFT)
    pf = par.paragraph_format
    pf.left_indent, pf.first_line_indent, pf.space_after = Inches(0.3), Inches(-0.3), Pt(4)
    pf.tab_stops.add_tab_stop(Inches(0.3))

doc.save(OUT)
print(f"Saved {OUT}  ({N} entries, {NP} papers, {len(REFNUM)} references, {EQN[0]} equations)")

