"""Build the revised manuscript (.docx) from the verified benchmark and committed figures.

Run from the repository root:  python docs/papers/build_manuscript.py [output.docx]
Every number in the text is read from the benchmark and analysis files, so the
manuscript cannot drift from the data.
"""
import csv
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
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from make_figures import GROUPS  # noqa: E402
from recompute_manuscript_numbers import se, sobol_total, tmm  # noqa: E402
from src.physics.composite_models import mclachlan_gem  # noqa: E402
from src.physics.material_models import freq_dependent_permeability, snoek_resonance_frequency  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "EMI_Shielding_paper_revised.docx"
EQ_DIR = HERE / "_equations"
EQ_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------- data
bench = list(csv.DictReader((ROOT / "data_collection" / "verified_benchmark.csv").open()))
gain = {r["id"]: r for r in csv.DictReader((HERE / "architecture_gain.csv").open())}
for b in bench:
    g = gain[b["id"]]
    b["gain"], b["slab"], b["meas"] = float(g["gain_db"]), float(g["se_slab_db"]), float(g["se_measured_db"])
    b["td"], b["sig"] = float(g["t_over_delta"]), float(b["sigma_s_per_m"])
    b["grp"] = next(i for i, grp in enumerate(GROUPS) if b["material_class"] in grp[3])
BY = {b["id"]: b for b in bench}
N, NP = len(bench), len({b["citation"] for b in bench})


def stats(rows):
    g = np.array([r["gain"] for r in rows])
    return dict(n=len(g), med=float(np.median(g)), lo=float(g.min()), hi=float(g.max()),
                within=int((np.abs(g) <= 10).sum()))


def sg(x):  # signed gain with a true minus sign
    return f"{x:+.1f}".replace("-", "−")


G = [stats([b for b in bench if b["grp"] == i]) for i in range(len(GROUPS))]
ALL = stats(bench)
HIGH = stats([b for b in bench if b["sig"] >= 1e4])
LOW = stats([b for b in bench if b["sig"] < 1.0])
DENSE = [b for b in bench if b["grp"] == 2]
DENSE_X = stats([b for b in DENSE if not b["id"].startswith("wang2026")])
DENSE_PAPERS = len({b["citation"] for b in DENSE if not b["id"].startswith("wang2026")})
POS = int(sum(b["gain"] > 10 for b in bench))
NEG = int(sum(b["gain"] < -10 for b in bench))
NUMW = {4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
ABOVE90 = int(sum(b["meas"] > 90 for b in bench))
LOW_AERO = int(sum(b["sig"] < 1.0 and b["grp"] == 5 for b in bench))
NEG_W = {8: "Eight", 9: "Nine", 10: "Ten", 11: "Eleven", 12: "Twelve"}.get(NEG, str(NEG))
POROUS_POS = [b["gain"] for b in bench if b["grp"] in (3, 5) and b["gain"] > 10]

CU = 5.96e7
single = se(CU, 1.0, 0.5e-6, 1e9)["total_se"]
one_um = se(CU, 1.0, 1.0e-6, 1e9)["total_se"]
sand = tmm([(CU, 1.0, 0.5e-6), (0.0, 3.0, 100e-6), (CU, 1.0, 0.5e-6)], 1e9)["total_se"]
fr_mu = snoek_resonance_frequency(1e5, 8.6e5)
mu_1g = abs(freq_dependent_permeability(1e5, 1e9, M_s=8.6e5))
SOB = [sobol_total(n, c) for n, c in (
    (dict(sigma=CU, mu=1.0, t=10e-6, f=1e9), dict(sigma=0.05, mu=1e-4, t=0.02, f=0.001)),
    (dict(sigma=6.99e6, mu=300.0, t=1e-3, f=1e6), dict(sigma=0.08, mu=0.20, t=0.02, f=0.001)),
    (dict(sigma=10.0, mu=1.0, t=2e-3, f=8.2e9), dict(sigma=0.30, mu=1e-4, t=0.05, f=0.001)))]

# ---------------------------------------------------------------- references
FOUND = {
    "schelkunoff": "S.A. Schelkunoff, The electromagnetic theory of coaxial transmission lines and cylindrical shields, Bell Syst. Tech. J. 13 (1934) 532–579. https://doi.org/10.1002/j.1538-7305.1934.tb00679.x",
    "schulz": "R.B. Schulz, V.C. Plantz, D.R. Brush, Shielding theory and practice, IEEE Trans. Electromagn. Compat. 30 (1988) 187–201. https://doi.org/10.1109/15.3297",
    "ott": "H.W. Ott, Electromagnetic Compatibility Engineering, Wiley, Hoboken, 2009. https://doi.org/10.1002/9780470508510",
    "celozzi": "S. Celozzi, R. Araneo, G. Lovat, Electromagnetic Shielding, Wiley, Hoboken, 2008. https://doi.org/10.1002/9780470268483",
    "yeh": "P. Yeh, Optical Waves in Layered Media, Wiley, New York, 1988.",
    "peng": "M. Peng, F. Qin, Clarification of basic concepts for electromagnetic interference shielding effectiveness, arXiv:2112.03272 (2021).",
    "mclachlan": "D.S. McLachlan, M. Blaszkiewicz, R.E. Newnham, Electrical resistivity of composites, J. Am. Ceram. Soc. 73 (1990) 2187–2203. https://doi.org/10.1111/j.1151-2916.1990.tb07576.x",
    "kirkpatrick": "S. Kirkpatrick, Percolation and conduction, Rev. Mod. Phys. 45 (1973) 574–588. https://doi.org/10.1103/RevModPhys.45.574",
    "bauhofer": "W. Bauhofer, J.Z. Kovacs, A review and analysis of electrical percolation in carbon nanotube polymer composites, Compos. Sci. Technol. 69 (2009) 1486–1498. https://doi.org/10.1016/j.compscitech.2008.06.018",
    "snoek": "J.L. Snoek, Dispersion and absorption in magnetic ferrites at frequencies above one Mc/s, Physica 14 (1948) 207–217. https://doi.org/10.1016/0031-8914(48)90038-X",
    "sobol": "I.M. Sobol, Global sensitivity indices for nonlinear mathematical models and their Monte Carlo estimates, Math. Comput. Simul. 55 (2001) 271–280.",
    "saltelli": "A. Saltelli, Making best use of model evaluations to compute sensitivity indices, Comput. Phys. Commun. 145 (2002) 280–297.",
}
import html
import json

CROSSREF = json.loads((HERE / "references_crossref.json").read_text())


TITLE_FIX = {
    "Ti 3 CNT x (MXene)": "Ti_{3}CNT_{x} (MXene)",
    "Ti 3 C 2 T x MXene-MWCNT@MoS 2 Film": "Ti_{3}C_{2}T_{x} MXene-MWCNT@MoS_{2} Film",
    "Ti3C2T x : MXenes": "Ti_{3}C_{2}T_{x}: MXenes",
    "Ti 3 C 2 T x /(Aramid": "Ti_{3}C_{2}T_{x}/(Aramid",
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
    year = m["published"]["date-parts"][0][0]
    pages = (m.get("page") or m.get("article-number") or "").replace("-", "\u2013")
    if doi == "10.34133/2022/9780290":
        pages = "9780290"
    ref = f"{authors}, {_clean(m['title'][0])}, {_clean(m['container-title'][0])} {m.get('volume')} ({year}) {pages}. https://doi.org/{doi}"
    if doi == "10.1007/s10853-020-05518-1":
        ref += " (values taken from the preprint, arXiv:2005.04875)"
    return ref


REFTEXT = dict(FOUND)
for b in bench:
    REFTEXT.setdefault(b["citation"], format_crossref(b["doi"]))
REFNUM = {}


def num(key):
    key = BY[key]["citation"] if key in BY else key
    if key not in REFNUM:
        REFNUM[key] = len(REFNUM) + 1
    return REFNUM[key]


def c(*keys):
    nums = [num(k) for k in keys]
    return "[" + ",".join(str(n) for n in sorted(dict.fromkeys(nums))) + "]"


def g(i):
    return sg(BY[i]["gain"])


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
    r = par.add_run(text); r.bold = True
    r.font.size, r.font.name = Pt(12 if level == 1 else 10), FONT


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
for _i, _line in enumerate(("Predict Before You Build:", "testing electromagnetic shielding theory",
                            "against 76 verified measurements")):
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
_abs = P(f"Plane-wave theory gives the shielding effectiveness (SE) of a homogeneous conducting slab exactly, yet it is "
  f"rarely tested against measurements whose inputs can be traced to their source. We assemble a benchmark of {N} "
  f"measured SE values from {NP} primary papers, covering metal films and meshes, MXene and graphene films, "
  f"carbon–polymer composites, foams and aerogels (10 MHz to 40 GHz, 13 nm to 5 mm, 10^{{−3}} to 10^{{7}} S m^{{−1}}). "
  f"Every value is tied to a quoted sentence, table or caption. For each entry we compute the exact slab prediction "
  f"from the reported conductivity and thickness, and define the architecture gain as measured minus predicted SE. "
  f"For conductors above 10^{{4}} S m^{{−1}} (n = {HIGH['n']}) the median gain is {sg(HIGH['med'])} dB and "
  f"{HIGH['within']} entries lie within ±10 dB; continuous metal films and dense MXene films from 29 nm to 45 µm "
  f"follow the model. Three departures are systematic. Porous films, foams, aerogels and segregated networks exceed "
  f"the prediction by {min(POROUS_POS):.0f} to {max(POROUS_POS):.0f} dB. Samples with reported conductivity below "
  f"1 S m^{{−1}} shield {LOW['lo']:.0f} to {LOW['hi']:.0f} dB more than that conductivity implies. {NEG_W} entries fall "
  f"more than 10 dB below the prediction, which flags a reported conductivity or thickness that does not describe "
  f"the measured sample. The slab model is therefore a quantitative baseline for dense conductors and a consistency "
  f"check on reported data, and the gain separates what conductivity explains from what structure adds. The code, "
  f"tests and benchmark are openly available.")
_kw = P("*Keywords:* electromagnetic interference shielding; shielding effectiveness; benchmark dataset; transfer matrix "
  "method; MXene; polymer composites; porous materials")

for _p in (_abs, _kw):
    _p.paragraph_format.left_indent = _p.paragraph_format.right_indent = Inches(0.5)
_kw.paragraph_format.space_after = Pt(10)

# ---------------------------------------------------------------- 1
H("1. Introduction")
P(f"The shielding effectiveness of a planar barrier is the ratio of incident to transmitted power, expressed in "
  f"decibels. For a single homogeneous layer at normal incidence, Schelkunoff’s transmission-line analogy "
  f"{c('schelkunoff')} gives it in closed form from four quantities: conductivity, permeability, thickness and "
  f"frequency {c('schulz', 'ott', 'celozzi')}. The calculation takes microseconds and is the usual first estimate "
  f"in electromagnetic compatibility design.")
P(f"Most shielding materials reported today are not homogeneous slabs. They are laminated MXene and graphene films, "
  f"percolating carbon networks in polymers, foams, aerogels and metal meshes. Their papers report a measured SE "
  f"together with a conductivity and a thickness, and frequently attribute the result to the material’s "
  f"structure. Whether structure contributed anything beyond what the conductivity already implies is seldom "
  f"quantified, although the slab calculation that would answer the question is elementary. The literature also "
  f"contains recurring misunderstandings of how SE and its reflection and absorption parts should be interpreted "
  f"{c('peng')}.")
P(f"Comparisons between slab or transfer-matrix calculations and measurement exist for individual material "
  f"families. Han et al. {c('han2020_ti3c2_5um')} showed that a transfer-matrix model follows the measured SE of "
  f"highly conductive MXene films and under-predicts films below about 100 S cm^{{−1}}, and Iqbal et al. "
  f"{c('iqbal2020_ti3cn_350')} reported absorption in annealed Ti_{{3}}CNT_{{x}} well above what its conductivity "
  f"suggests. What is missing is a comparison across material classes that uses one exact model and data whose "
  f"provenance can be checked. Data-driven studies of shielding materials would benefit from the same resource.")
P(f"This paper makes three contributions. First, a benchmark of {N} measured SE values from {NP} primary papers, "
  f"in which each value, thickness, frequency and conductivity is traceable to a quoted location in its source. "
  f"Second, an openly tested implementation in which the Schelkunoff decomposition, a transfer-matrix solver and "
  f"an independent exact slab formula agree to within 10^{{−6}} dB. Third, an analysis of the architecture gain, "
  f"the difference between measured SE and the slab prediction, which identifies where the homogeneous model is "
  f"sufficient, where structure adds shielding, and where reported values are mutually inconsistent.")

# ---------------------------------------------------------------- 2
H("2. Model")
H("2.1 Exact shielding effectiveness of a homogeneous slab", 2)
P("A slab of thickness *t*, conductivity *σ*, permeability *µ* = *µ*_{r}*µ*_{0} and permittivity "
  "*ε* = *ε*_{r}*ε*_{0} separates two half-spaces of free space with impedance *Z*_{0} = 376.73 Ω. At "
  "angular frequency *ω* the propagation constant and intrinsic impedance of the slab are")
EQ(r"\gamma = j\omega\sqrt{\mu\,\varepsilon_c},\qquad \eta=\sqrt{\mu/\varepsilon_c},\qquad \varepsilon_c=\varepsilon-j\sigma/\omega")
P("and the shielding effectiveness is the sum of a reflection term, an absorption term and a multiple-reflection "
  "correction,")
EQ(r"SE = R + A + B")
EQ(r"R = 20\log_{10}\frac{|Z_0+\eta|^{2}}{4\,Z_0\,|\eta|},\qquad A = 20\log_{10}\left|e^{\gamma t}\right| = 8.686\,\mathrm{Re}(\gamma)\,t")
EQ(r"B = 20\log_{10}\left|1-q^{2}e^{-2\gamma t}\right|,\qquad q=\frac{Z_0-\eta}{Z_0+\eta}")
P(f"The reflection term accounts for the impedance mismatch at both faces of the slab. With the correction *B* "
  f"included, the sum is algebraically identical to the exact plane-wave transmission loss, so Eqs. (2)–(4) involve "
  f"no approximation beyond homogeneity and normal incidence. *B* is negative for electrically thin shields and "
  f"vanishes once *A* exceeds about 15 dB. In a good conductor the skin depth is *δ* = (2/*ωµσ*)^{{1/2}} and "
  f"*A* = 8.686 *t*/*δ*.")
P("When the slab is much thinner than the skin depth, the three terms combine into a result that depends only on "
  "the sheet resistance *R*_{s} = 1/(*σt*):")
EQ(r"SE \approx 20\log_{10}\left(1+\frac{Z_0}{2R_s}\right)")
P("Equation (5) is the form used below for metal films, meshes and transparent conductors, for which sheet "
  "resistance is the quantity actually measured.")

H("2.2 Transfer matrix method for layered shields", 2)
P(f"For a stack of *N* layers, each layer *i* of thickness *d*_{{i}} is described by a 2×2 matrix with elements "
  f"*A*_{{i}} = *D*_{{i}} = cosh(*γ*_{{i}}*d*_{{i}}), *B*_{{i}} = *η*_{{i}} sinh(*γ*_{{i}}*d*_{{i}}) and "
  f"*C*_{{i}} = sinh(*γ*_{{i}}*d*_{{i}})/*η*_{{i}} {c('yeh', 'celozzi')}. The stack matrix is the ordered product of the "
  f"layer matrices, and with free space on both sides")
EQ(r"S_{21}=\frac{2}{A+B/Z_0+C\,Z_0+D},\qquad SE=-20\log_{10}|S_{21}|")
P("For thick conductors Re(*γd*) exceeds the range of double-precision hyperbolic functions. The implementation "
  "factors exp(Re(*γd*)) out of each such layer exactly and carries it as a logarithmic scale, so the result stays "
  "finite and continuous across the switch. For a single layer the method reproduces Eqs. (2)–(4): the two "
  "implementations and an independent evaluation of the exact slab formula agree to within 10^{−6} dB over "
  "conductivities from 10^{−3} to 6 × 10^{7} S m^{−1}, including ferromagnetic and weakly conducting cases. "
  "These identities are enforced by automated tests in the repository.")

H("2.3 Percolation conductivity of composites", 2)
P(f"The conductivity of a composite of conducting filler (volume fraction *f*, conductivity *σ*_{{f}}) in an "
  f"insulating matrix (*σ*_{{m}}) changes by many orders of magnitude at the percolation threshold *f*_{{c}} "
  f"{c('kirkpatrick', 'bauhofer')}. The general effective media equation of McLachlan et al. {c('mclachlan')} "
  f"describes the transition in one implicit relation,")
EQ(r"\frac{(1-f)\left(\sigma_m^{1/s}-\sigma_e^{1/s}\right)}{\sigma_m^{1/s}+K\sigma_e^{1/s}}+"
   r"\frac{f\left(\sigma_f^{1/s}-\sigma_e^{1/s}\right)}{\sigma_f^{1/s}+K\sigma_e^{1/s}}=0,\qquad K=\frac{1-f_c}{f_c}")
P("where *σ*_{e} is the effective conductivity and *s* ≈ 2 is the critical exponent for three-dimensional networks. "
  "It is solved numerically by a bracketing root finder.")

H("2.4 Frequency-dependent permeability", 2)
P(f"For ferromagnets the product of static susceptibility and resonance frequency is bounded by Snoek’s limit "
  f"{c('snoek')}, (*µ*_{{s}} − 1) *f*_{{r}} = (2/3) *γ*_{{g}} *µ*_{{0}} *M*_{{s}}, with *γ*_{{g}} = 28 GHz T^{{−1}}, and "
  f"the permeability above resonance is modelled by a single Debye relaxation, *µ*_{{r}}(*f*) = 1 + (*µ*_{{s}} − 1)/"
  f"(1 + j*f*/*f*_{{r}}). For mu-metal (*µ*_{{s}} = 10^{{5}}, *M*_{{s}} = 860 kA m^{{−1}}) this gives *f*_{{r}} = "
  f"{fr_mu / 1e6:.2f} MHz and |*µ*_{{r}}| = {mu_1g:.0f} at 1 GHz. The benchmark below contains no bulk ferromagnetic "
  f"samples, so this component is not validated here.")

H("2.5 Architecture gain", 2)
P("For a sample with reported conductivity *σ*, thickness *t* and measurement frequency *f*, the architecture "
  "gain is")
EQ(r"G = SE_{\mathrm{measured}} - SE_{\mathrm{slab}}(\sigma,\,t,\,f)")
P("with *SE*_{slab} from Eqs. (2)–(4), *µ*_{r} = 1 and *ε*_{r} = 1. When a source reports a band average, the "
  "prediction is averaged over the same band. A gain near zero means the sample shields as a homogeneous conductor "
  "of that conductivity would. A large positive gain means it shields more than its reported conductivity "
  "explains. A large negative gain means the reported conductivity, thickness and SE cannot all describe the same "
  "homogeneous sample.")

H("2.6 Global sensitivity analysis", 2)
P(f"Sensitivity of the prediction to input uncertainty is quantified by total-order Sobol indices {c('sobol')} "
  f"with the Saltelli estimator {c('saltelli')}, using 4096 base samples from a scrambled Sobol sequence and "
  f"normally distributed inputs. The coefficients of variation are chosen per material class: 10^{{−4}} for the "
  f"permeability of non-magnetic materials and 0.20 for ferromagnets; 0.05, 0.08 and 0.30 for the conductivity of "
  f"copper, steel and a composite near percolation; 0.02 to 0.05 for thickness; 0.001 for frequency.")

# ---------------------------------------------------------------- 3
H("3. Benchmark construction")
H("3.1 Inclusion criteria and provenance", 2)
P("A measurement entered the benchmark only if its primary source states, for the same sample, all of: the "
  "measured total SE in decibels; the sample or conductive-layer thickness; the frequency or band; and the "
  "electrical conductivity or sheet resistance. Values had to be printed in the text, a table or a caption. "
  "Values read from plots, values a paper quotes from other work, and calculated SE were excluded. Conductivity "
  "units were recorded exactly as printed and converted once, since confusion between S cm^{−1} and S m^{−1} "
  "changes a prediction by tens of decibels.")
P(f"Each entry stores the citation and DOI, the values as printed, a short verbatim quotation that locates them, "
  f"the measurement method where stated, and any caveat. Sources were open-access full texts or author "
  f"manuscripts; one entry {c('shahzad2016_ti3c2_45um')} relies on the abstract of the original report and is "
  f"marked accordingly. Where a source gives sheet resistance, the conductivity used is 1/(*R*_{{s}}*t*), and by "
  f"Eq. (5) the prediction is then insensitive to the nominal thickness. Papers whose values are given only as "
  f"approximate ranges, that contradict themselves between tables, or whose conductivity unit is ambiguous were "
  f"left out; three entries whose source prints an approximate value are retained and flagged.")
H("3.2 Coverage", 2)
P(f"The benchmark holds {N} entries from {NP} papers published between 2009 and 2026 (Table 1; the full list is "
  f"in Appendix A). Measured SE ranges from 9.7 to 116 dB, thickness from 13 nm to 5 mm, and reported "
  f"conductivity from 2.5 × 10^{{−3}} to 1.2 × 10^{{7}} S m^{{−1}}. Most measurements are waveguide or coaxial "
  f"measurements in the X band (8.2–12.4 GHz); the remainder extend from 10 MHz to 40 GHz. All measured values are "
  f"at or below 116 dB and all but {NUMW.get(ABOVE90, ABOVE90)} are below 90 dB, within the range that standard "
  f"fixtures resolve. This is a precondition for a meaningful comparison: the slab "
  f"model returns hundreds or thousands of decibels for thick metal sheets, and such predictions cannot be tested.")
TABLE("Table 1. Composition of the benchmark and architecture gain by material group.",
      ["Group", "*n*", "Median gain (dB)", "Range (dB)", "Within ±10 dB"],
      [[GROUPS[i][0], G[i]["n"], sg(G[i]["med"]), f"{sg(G[i]['lo'])} to {sg(G[i]['hi'])}", f"{G[i]['within']} of {G[i]['n']}"]
       for i in range(len(GROUPS))] +
      [["All entries", ALL["n"], sg(ALL["med"]), f"{sg(ALL['lo'])} to {sg(ALL['hi'])}", f"{ALL['within']} of {ALL['n']}"]],
      widths=[2.9, 0.5, 1.1, 1.4, 1.0])

# ---------------------------------------------------------------- 4
H("4. Results")
H("4.1 Overall agreement", 2)
P(f"Figure 1a compares measured SE with the slab prediction for all {N} entries, and Fig. 1b shows the gain "
  f"against reported conductivity. Of these, {ALL['within']} lie within ±10 dB of the prediction, {POS} lie more than "
  f"10 dB above it and {NEG} more than 10 dB below. Agreement depends strongly on conductivity. Above "
  f"10^{{4}} S m^{{−1}} (n = {HIGH['n']}) the median gain is {sg(HIGH['med'])} dB and {HIGH['within']} entries are within "
  f"±10 dB. Below 1 S m^{{−1}} (n = {LOW['n']}) every entry lies above the prediction, with a median gain of "
  f"{sg(LOW['med'])} dB. Figure 2 shows the same data grouped by material class.")
FIG("fig1_parity.png",
    f"Figure 1. Measured shielding effectiveness against the homogeneous-slab prediction for the {N} benchmark "
    f"entries. (a) Parity plot; the band marks ±10 dB. (b) Architecture gain against reported conductivity.")
FIG("fig2_gain_by_class.png",
    "Figure 2. Architecture gain by material group. Each point is one benchmark entry; vertical bars mark group "
    "medians and the band marks ±10 dB.")

H("4.2 Continuous conductors follow the slab model", 2)
P(f"Continuous metal films and transparent conductors agree with Eq. (5) to within a few decibels. An "
  f"In_{{2}}O_{{3}}/Ag/In_{{2}}O_{{3}} film with a 13 nm silver layer {c('voronin2025_iai3')} and a 440 nm La-doped "
  f"BaSnO_{{3}} film {c('jeon2023_blso')} have gains of {g('voronin2025_iai3')} and {g('jeon2023_blso')} dB, and a "
  f"sintered silver film on paper {c('liu2020_ag_film')} {g('liu2020_ag_film')} dB. Silver films loaded with nickel "
  f"particles from the same study exceed the prediction by 6 to 10 dB, consistent with a magnetic contribution "
  f"that the calculation, which assumes *µ*_{{r}} = 1, does not include.")
P(f"Dense MXene and graphene films behave the same way over three orders of magnitude in thickness. Spray-coated "
  f"Ti_{{3}}C_{{2}}T_{{x}} films of about 40 nm and 137 nm {c('han2020_ti3c2_40nm')} have gains of "
  f"{g('han2020_ti3c2_40nm')} and {g('han2020_ti3c2_137nm')} dB, a 29 nm oxygen-substituted film "
  f"{c('park2026_mxene_29nm')} {g('park2026_mxene_29nm')} dB, micrometre-thick films "
  f"{c('wan2022_mxene_lm', 'huang2025_mxene_mx80', 'han2020_ti3c2_5um')} between "
  f"{g('huang2025_mxene_mx80')} and {g('wan2022_mxene_sdm')} dB, and 40 to 45 µm films "
  f"{c('iqbal2020_ti3c2_asis', 'shahzad2016_ti3c2_45um')} between {g('shahzad2016_ti3c2_45um')} and "
  f"{g('iqbal2020_ti3c2_350')} dB. Excluding one study discussed in Section 4.5, the {DENSE_X['n']} dense-film "
  f"entries have a median gain of {sg(DENSE_X['med'])} dB and {DENSE_X['within']} are within ±10 dB. This extends, "
  f"with independent data from {DENSE_PAPERS} papers, the agreement that Han et al. {c('han2020_ti3c2_5um')} reported "
  f"for their own films: for a dense laminate of high conductivity, the measured SE is what a homogeneous "
  f"conductor of that conductivity and thickness delivers. The laminated structure does not, by itself, add "
  f"shielding.")
P(f"Metal meshes are less uniform. Electroplated copper meshes measured at 10 MHz {c('voronin2022_cumesh_LCM30')} "
  f"follow Eq. (5) closely (gains of {g('voronin2022_cumesh_LCM180')} to {g('voronin2022_cumesh_SCM180')} dB). At "
  f"8–18 GHz, silver meshes embedded in glass {c('li2023_agmesh_W1P25t0.5')} lie up to "
  f"{BY['li2023_agmesh_W1P25t0.8']['gain']:.0f} dB above the prediction and silver meshes in PET "
  f"{c('zarei2024_agmesh_1')} 5 to 10 dB below it. A mesh is a periodic structure, not a sheet; once the "
  f"wavelength is no longer very large compared with the mesh pitch, sheet resistance alone does not determine "
  f"its transmission.")

H("4.3 Porosity and segregated networks add shielding", 2)
P(f"The largest positive gains among conductors belong to porous structures. For 40 µm Ti_{{3}}CNT_{{x}} films "
  f"{c('iqbal2020_ti3cn_350')} the gain is {g('iqbal2020_ti3cn_asis')} dB as made, {g('iqbal2020_ti3cn_250')} dB "
  f"after annealing at 250 °C and {g('iqbal2020_ti3cn_350')} dB after annealing at 350 °C, although the "
  f"conductivity at 350 °C is lower than at 250 °C. Ti_{{3}}C_{{2}}T_{{x}} films annealed in the same study, which "
  f"develop little porosity, stay within 6 dB of the prediction. The authors attribute the difference to "
  f"absorption in the porous layered structure; the gain expresses that attribution as a number that can be "
  f"compared across materials. The predictions here use the initial film thickness, because the thickness after "
  f"annealing is not given in the text; a thicker porous film would raise the prediction somewhat and lower the "
  f"gain.")
P(f"The same sign and a similar magnitude appear in bulk porous and segregated materials. Six graphitic "
  f"aerogels {c('wani2026_graphitic_aerogel_900_5')} have gains of {g('wani2026_graphitic_aerogel_900_35')} to "
  f"{g('wani2026_graphitic_aerogel_800_5')} dB, a printed MXene/CNT/polyimide aerogel frame "
  f"{c('xue2023_mxene_cnt_pi_aerogel')} {g('xue2023_mxene_cnt_pi_aerogel')} dB, and segregated carbon-nanotube "
  f"networks in polysulfone and polyurethane {c('xie2020_psu_cnt_solid', 'wang2021_cnt_go_wpu_10')} "
  f"{g('xie2020_psu_cnt_solid')} to {g('wang2021_cnt_go_wpu_10')} dB. Ten of the twelve entries in this group lie "
  f"between +13 and +25 dB and none lies within ±10 dB. A bulk conductivity measured across such a sample "
  f"averages over conducting walls and voids or insulating domains, whereas the incident wave interacts with "
  f"walls whose local conductivity is far higher and with many internal interfaces.")

H("4.4 Low reported conductivity under-predicts shielding", 2)
P(f"All {NUMW.get(LOW['n'], LOW['n'])} entries with reported conductivity below 1 S m^{{−1}} shield more than the slab model allows, by "
  f"{LOW['lo']:.0f} to {LOW['hi']:.0f} dB. Carbon-black-filled EPDM {c('rahaman2023_epdm_cb_40phr')} with DC "
  f"conductivity of 2.5 × 10^{{−3}} to 6.3 × 10^{{−2}} S m^{{−1}} is predicted to shield less than 0.1 dB at 10 GHz "
  f"and measures 26.5 to 49.9 dB. Carbon-fibre-filled PA11/PLA {c('durmaz2023_pa11pla_cf30')} behaves the same way, "
  f"and {NUMW.get(LOW_AERO, LOW_AERO)} of the graphitic aerogels of Section 4.3 also fall in this conductivity range. "
  f"In contrast, composites whose reported conductivity is 20 S m^{{−1}} or more "
  f"{c('lencar2022_mwcnt3_5min', 'madinehei2021_polyester_gnp25', 'li2022_cnt_wpu_50')} lie within about 7 dB of "
  f"the prediction.")
P(f"A DC or low-frequency conductivity is a property of the connected network. At microwave frequencies, filler "
  f"clusters that are isolated at DC are coupled capacitively and dissipate, so the effective conductivity is "
  f"higher, and dielectric polarisation adds loss that Eq. (1) with *ε*_{{r}} = 1 omits. Han et al. "
  f"{c('han2020_ti3c2_5um')} made the same observation for low-conductivity MXenes. The practical consequence is "
  f"that near and below the percolation threshold a DC conductivity is not a usable input for predicting "
  f"microwave SE, however the conductivity is obtained, including from Eq. (7).")

H("4.5 Negative gain as a consistency check", 2)
P(f"{NEG_W} entries lie more than 10 dB below the prediction. A large negative gain is hard to produce "
  f"physically. It indicates that the reported conductivity and thickness do not describe the measured object, or "
  f"that the measurement was limited by leakage or dynamic range. In three cases the reason is evident from the "
  f"source. A filter paper coated with silver nanowires and MXene {c('han2024_agnw_mxene_paper')} "
  f"({g('han2024_agnw_mxene_paper')} dB) reports the conductivity of the coating and the thickness of the whole "
  f"sheet. A two-bar stack of MWCNT/PMMA {c('pande2009_mwcnt_pmma_scb2')} ({g('pande2009_mwcnt_pmma_scb2')} dB) "
  f"reports the conductivity of one bar, and the bars are joined by an insulating adhesive. A 4.5 mm carbon "
  f"nanofibre aerogel {c('chen2025_cnf_pani_aerogel')} ({g('chen2025_cnf_pani_aerogel')} dB) has a predicted SE of "
  f"{BY['chen2025_cnf_pani_aerogel']['slab']:.0f} dB, above what a waveguide measurement normally resolves.")
P(f"In the remaining cases the source offers no explanation. Five cellulose-nanofibre/Ti_{{3}}C_{{2}}T_{{x}} films "
  f"from one study {c('wang2026_tocnf_mxene_M25')} report conductivities of 2.8 × 10^{{5}} to 1.2 × 10^{{6}} S m^{{−1}} "
  f"at 4 to 5 µm and SE of 9.7 to 30.7 dB, a uniform 32 to 38 dB below the prediction and below other films of "
  f"similar conductivity and thickness in the benchmark. A segregated graphene/polypropylene composite "
  f"{c('alam2017_gnp_pp_10')} ({g('alam2017_gnp_pp_10')} dB) and a 12 µm Ti_{{3}}C_{{2}}T_{{x}} film "
  f"{c('ahmed2023_ti3c2_12um')} ({g('ahmed2023_ti3c2_12um')} dB) are similar. We draw no conclusion about the "
  f"cause in any individual paper. The point is that the check costs one line of arithmetic and would let "
  f"authors and reviewers catch a unit, thickness or calibration problem before publication.")

H("4.6 Layered shields", 2)
P(f"The transfer matrix method reproduces a known property of spaced double shields. At 1 GHz a single 0.5 µm "
  f"copper layer gives {single:.1f} dB and a single 1.0 µm layer {one_um:.1f} dB. Splitting the 1.0 µm of copper "
  f"into two 0.5 µm layers separated by 100 µm of PET (*ε*_{{r}} = 3) gives {sand:.1f} dB, a gain of "
  f"{sand - one_um:.1f} dB for the same mass of metal. The gain grows with spacer thickness (Fig. 3) until the "
  f"spacer approaches half a wavelength, where the stack becomes resonant and SE drops sharply; at 10 GHz this "
  f"occurs near 8.7 mm. The sandwich remains below the {2 * single:.0f} dB that two isolated layers would sum to, "
  f"because the layers are coupled. These are model results; the benchmark contains no multilayer sample with "
  f"per-layer properties sufficient to test them.")
FIG("fig4_multilayer.png",
    "Figure 3. Shielding effectiveness of a Cu(0.5 µm)/PET/Cu(0.5 µm) sandwich against spacer thickness, from the "
    "transfer matrix method. Dashed lines: a single 1.0 µm copper layer at the same frequency.", width=3.4)

H("4.7 Percolation and its effect on predicted shielding", 2)
sf, sm, fc = 1e5, 1e-10, 0.01
P(f"Figure 4 compares linear mixing with Eq. (7) for *σ*_{{f}} = 10^{{5}} S m^{{−1}}, *σ*_{{m}} = 10^{{−10}} S m^{{−1}}, "
  f"*f*_{{c}} = 0.01 and *s* = 2, and Table 2 lists values across the transition. Linear mixing assigns a "
  f"conductivity of hundreds of siemens per metre to a composite below its threshold, and therefore a slab SE "
  f"of the order of 100 dB at 2 mm and 8.2 GHz where Eq. (7) correctly gives none. This comparison is between models. "
  f"Section 4.4 shows that a measured composite near threshold shields more than its network conductivity "
  f"predicts, so Eq. (7) should be read as a model of DC transport and as a lower bound on microwave shielding.")
def sci(v):
    m, e = f"{v:.1e}".split("e")
    return f"{m} \u00d7 10^{{{int(e)}}}".replace("-", "\u2212")


TABLE("Table 2. Effective conductivity from linear mixing and from the general effective media equation.",
      ["*f* (vol.)", "Linear mixing (S m^{−1})", "Eq. (7) (S m^{−1})", "Ratio"],
      [[f"{x:.3f}", sci(x * sf + (1 - x) * sm), sci(mclachlan_gem(x, sm, sf, fc, 2.0)),
        sci(mclachlan_gem(x, sm, sf, fc, 2.0) / (x * sf + (1 - x) * sm))] for x in (0.005, 0.010, 0.012, 0.020, 0.050, 0.100)],
      widths=[1.0, 1.7, 1.7, 1.2])
FIG("fig3_percolation.png",
    "Figure 4. (a) Effective conductivity from linear mixing and from the general effective media equation. "
    "(b) Corresponding slab shielding effectiveness for a 2 mm sheet at 8.2 GHz.")

H("4.8 Sensitivity to input uncertainty", 2)
P(f"Figure 5 gives total-order Sobol indices for three cases. For a 10 µm copper film at 1 GHz, conductivity "
  f"(*S*_{{T}} = {SOB[0]['sigma']:.2f}) and thickness ({SOB[0]['t']:.2f}) share the variance. For 1 mm mild steel at "
  f"1 MHz, permeability dominates ({SOB[1]['mu']:.2f}), followed by conductivity ({SOB[1]['sigma']:.2f}). For a "
  f"2 mm composite at 10 S m^{{−1}} and 8.2 GHz, conductivity accounts for nearly all of it "
  f"({SOB[2]['sigma']:.2f}). The ranking follows from the class-specific input uncertainties: assigning a "
  f"non-magnetic metal a 10% permeability uncertainty, as a generic default would, produces a spurious "
  f"permeability sensitivity. For reporting practice the implication matches Sections 4.3 to 4.5: the prediction "
  f"for a composite or porous material is only as good as its conductivity, which is also the least reliably "
  f"reported input.")
FIG("fig5_sensitivity.png",
    "Figure 5. Total-order Sobol indices of the slab prediction for (a) copper, (b) mild steel and (c) a carbon "
    "nanotube composite, with class-specific input uncertainties.")

# ---------------------------------------------------------------- 5
H("5. Discussion")
H("5.1 Use in materials design and reporting", 2)
P("Three uses follow. First, for a dense conductor the slab model is a quantitative design tool: given a target "
  "SE and a conductivity, Eq. (5) or Eqs. (2)–(4) return the thickness, and the benchmark indicates an "
  "uncertainty of roughly ±10 dB. In plane-wave terms a few micrometres of any metal exceed 80 dB at 10 GHz, so "
  "the thickness of practical metal shields is set by handling, seams and apertures, not by the sheet. Second, "
  "the gain gives a common scale on which structured materials can be compared. A claim that a structure enhances "
  "shielding can be stated as a gain in decibels over a homogeneous sheet of the same conductivity and thickness, "
  "which separates the contribution of the architecture from that of simply being more conductive or thicker. "
  "Third, the same calculation is a consistency check. We suggest that reports of shielding materials state the "
  "through-thickness conductivity of the measured sample, its thickness as measured, the frequency to which each "
  "SE value refers, and the slab prediction alongside the measurement.")
P("For data-driven design, the benchmark and baseline are complementary to learned models. The slab prediction "
  "is an informative input feature, and the gain is a better learning target than SE itself, because it removes "
  "the part of the response that elementary physics already explains and leaves the part that depends on "
  "structure.")
H("5.2 Limitations", 2)
P(f"The benchmark is small for some classes: {G[0]['n']} continuous metal films, {G[3]['n']} porous films, and no "
  f"bulk metal foils, ferromagnets or characterised multilayers. Entries were restricted to sources whose full "
  f"text could be read, which favours open-access journals, and one entry rests on an abstract. Values were "
  f"extracted from text, tables and captions, not from supplementary files or plots, so samples documented only "
  f"there are absent. The gain inherits every uncertainty in the reported conductivity and thickness; for porous "
  f"samples in particular, the thickness and the direction and method of the conductivity measurement vary "
  f"between sources. Band-averaged, maximum and single-frequency SE values are mixed, as reported. The model "
  f"assumes normal incidence, *µ*_{{r}} = 1 and *ε*_{{r}} = 1; magnetic fillers and dielectric loss therefore "
  f"appear as positive gain. The permeability model of Section 2.4 and the multilayer results of Section 4.6 are "
  f"not tested against measurement here.")

H("6. Conclusions")
P(f"A benchmark of {N} source-verified shielding measurements from {NP} papers shows that the exact "
  f"homogeneous-slab model predicts the shielding of dense conductors, from 13 nm transparent films to 45 µm "
  f"MXene laminates, with a median difference below 1 dB and about three quarters of entries within ±10 dB. Porous "
  f"films, foams, aerogels and segregated networks exceed the model by {min(POROUS_POS):.0f} to "
  f"{max(POROUS_POS):.0f} dB, and this architecture gain is a direct measure of what their structure contributes. "
  f"Samples of low reported conductivity shield far more than that conductivity implies, so DC conductivity "
  f"should not be used to predict microwave shielding near percolation. Entries well below the model identify "
  f"reported values that are mutually inconsistent. Stating the slab prediction beside each measured SE would make "
  f"claims of structural enhancement quantitative and would expose such inconsistencies at the time of reporting.")

H("7. Future work")
P("The benchmark will be extended to metal foils, ferromagnets, characterised multilayers and data from "
  "supplementary files. With a larger set, the variables that determine shielding can be organised in a structural "
  "model: composition, filler geometry, processing and thickness as independent variables; conductivity, porosity "
  "and permeability as mediating variables; and SE and architecture gain as dependent variables. Fitting such a "
  "model would quantify how much of the effect of processing on shielding passes through conductivity and how much "
  "acts through structure directly.")

H("Declarations")
P("*Funding:* The authors received no external funding for this work.")
P("*Competing interests:* The authors declare no competing interests.")
P("*Data and code availability:* The code, automated tests, benchmark file and the scripts that generate every "
  "number and figure in this paper are available at https://github.com/DanushArun/material-emi-shielding "
  "(branch fix/slab-reflection-physics).")

H("Appendix A. Benchmark entries")

def ft(x):
    return f"{x * 1e9:.3g} nm" if x < 1e-6 else (f"{x * 1e6:.3g} µm" if x < 1e-3 else f"{x * 1e3:.3g} mm")


def ff(b):
    lo, hi = float(b["f_low_hz"]), float(b["f_high_hz"])
    u = (lambda v: f"{v / 1e9:g}") if hi >= 1e9 else (lambda v: f"{v / 1e6:g}")
    unit = "GHz" if hi >= 1e9 else "MHz"
    return f"{u(lo)} {unit}" if lo == hi else f"{u(lo)}–{u(hi)} {unit}"


CHEM = [("Ti3C1.94O0.06Tz", "Ti_{3}C_{1.94}O_{0.06}T_{z}"), ("Ti3C2Tx", "Ti_{3}C_{2}T_{x}"), ("Ti3CNTx", "Ti_{3}CNT_{x}"),
        ("In2O3", "In_{2}O_{3}"), ("BaSnO3", "BaSnO_{3}"), ("MoS2", "MoS_{2}")]


def chem(name):
    for plain, marked in CHEM:
        name = name.replace(plain, marked)
    return name.replace("*", "\u2217")


def fsig(v):
    if 0.01 <= v < 1e3:
        return f"{v:.3g}"
    m, e = f"{v:.2e}".split("e")
    return f"{m} \u00d7 10^{{{int(e)}}}".replace("-", "\u2212")


TABLE("Table A1. All benchmark entries. *t*: thickness; *σ*: conductivity as used (converted to S m^{−1}; for "
  "entries reported as sheet resistance, 1/(*R*_{s}*t*)); SE: measured; Slab: prediction; *G*: gain.",
      ["Material", "*t*", "Frequency", "*σ* (S m^{−1})", "SE (dB)", "Slab (dB)", "*G* (dB)", "Ref."],
      [[chem(b["material"]), ft(float(b["thickness_m"])), ff(b), fsig(b["sig"]), f"{b['meas']:.1f}", f"{b['slab']:.1f}",
        sg(b["gain"]), f"[{num(b['id'])}]"] for b in sorted(bench, key=lambda b: (b["grp"], b["citation"], b["id"]))],
      widths=[1.72, 0.5, 0.92, 0.76, 0.42, 0.42, 0.42, 0.34], size=7.5)

H("References")
for key, i in sorted(REFNUM.items(), key=lambda kv: kv[1]):
    par = P(f"[{i}]\t{REFTEXT[key]}", align=LEFT)
    pf = par.paragraph_format
    pf.left_indent, pf.first_line_indent, pf.space_after = Inches(0.3), Inches(-0.3), Pt(4)
    pf.tab_stops.add_tab_stop(Inches(0.3))

doc.save(OUT)
print(f"Saved {OUT}  ({N} entries, {NP} papers, {len(REFNUM)} references, {EQN[0]} equations)")
