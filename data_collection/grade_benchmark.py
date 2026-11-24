"""Assign a measurement fixture and a quality grade to every benchmark entry.

Run from the repository root:  python data_collection/grade_benchmark.py
Rewrites data_collection/verified_benchmark.csv with three columns:

  fixture       waveguide | coaxial | unspecified   (from the method the source states)
  quality       A  SE, thickness and conductivity are stated for the same specimen without caveat
                B  one documented caveat (listed in quality_note); retained in the primary analysis
                C  the source is internally inconsistent or incomplete for this entry, so it fails
                   the inclusion criteria; listed and analysed separately, excluded from primary statistics
  quality_note  the reason for a B or C grade

Grades depend only on what the source prints, never on the size or sign of the residual.
"""
import csv
from pathlib import Path

PATH = Path(__file__).with_name("verified_benchmark.csv")

C = {
    "shahzad2016_ti3c2_45um": "Only the abstract was read; band, method and total-SE status unverified",
    "wang2026_tocnf_mxene_": "Conductivity unit printed inconsistently (S cm-1 in the equation, 's m-1' in the table header); "
                             "Methods also state SE 'was calculated according to the Simon formula'",
    "zhang2023_graphene_": "Stated conductivity contradicts the sheet resistance and thickness printed for the same film (factor 5-12)",
    "liu2020_agni60_6to1": "Two sheet resistances printed for this film (0.78 and about 1.86 ohm)",
    "ahmed2023_mxene_mwcnt_mos2_15": "Two thicknesses printed for this film (16 and 9.2 um)",
    "durmaz2023_pa11pla_": "Printed DC volume resistivity contradicts the 0.01 Hz conductivity used",
    "han2024_agnw_mxene_paper": "Conductivity is that of the coating, thickness that of the whole paper sheet; "
                                "141 um is attributed to two different samples",
}
B = {
    "iqbal2020_ti3c2_350": "Conductivity assigned from a stated range; thickness is the value before annealing",
    "iqbal2020_ti3cn_250": "Thickness is the value before annealing",
    "iqbal2020_ti3cn_350": "Thickness is the value before annealing",
    "han2020_ti3c2_": "Conductivity stated for the film type, not measured on this specimen",
    "park2026_mxene_29nm": "Conductivity stated for the composition, not measured on this specimen",
    "zarei2024_agmesh_": "Thickness is the trench depth, not the silver thickness",
    "voronin2022_cumesh_": "Thickness printed for the mesh before embedding",
    "voronin2025_iai3": "SE printed as approximate over 0.01-40 GHz",
    "jeon2023_blso": "SE printed as approximate; film measured on a 0.5 mm MgAl2O4 substrate",
    "palanisamy2021_cuni_fabric": "SE printed as approximate",
    "wan2022_mxene_": "Measurement method given only in supplementary information; averaging over the band not stated",
    "zhang2022_mxene_anf_pva": "Measurement method not stated in the main text",
    "wani2026_graphitic_aerogel_": "Two-probe conductivity",
    "rahaman2023_epdm_cb_": "DC conductivity used; the source also prints 1 kHz values 50-200 times higher",
    "ma2022_pvdf_mwcnt_gnp_film": "Conductivity measured at 100 Hz",
    "pande2009_mwcnt_pmma_scb2": "Two stacked bars; conductivity measured on a single bar",
    "ahmed2023_ti3c2_12um": "Frequency within the band not stated",
    "weng2025_mwcnt_mxene_50": "Frequency inferred from the surrounding text (8.2 GHz)",
    "xue2023_mxene_cnt_pi_aerogel": "Printed as 'highest' in the text and 'average' in the abstract",
    "chen2025_cnf_pani_aerogel": "Average or maximum not stated; introduction prints 84.5 dB",
    "liu2020_ag": "Values read from the arXiv preprint; thickness printed as approximately 2 um for all films",
    "lin2019_graphene_lg4": "A second conductivity (6960 S cm-1) is printed once for this film",
    "madinehei2021_polyester_gnp25": "Thickness is the mould thickness",
    "li2022_cnt_wpu_50": "",
}
EDITS = {
    "weng2025_mwcnt_mxene_50": {"f_low_hz": "8.2e9", "f_high_hz": "8.2e9", "se_qualifier": "stated (at 8.2 GHz, from context)"},
    "xue2023_mxene_cnt_pi_aerogel": {"se_qualifier": "stated (average per abstract; 'highest' in text)"},
}


def fixture(method: str) -> str:
    m = method.lower()
    return "coaxial" if "coaxial" in m else ("waveguide" if "waveguide" in m else "unspecified")


def lookup(table: dict, key: str):
    for prefix, note in table.items():
        if key.startswith(prefix):
            return note
    return None


def main() -> None:
    rows = list(csv.DictReader(PATH.open()))
    fields = [f for f in rows[0].keys() if f not in ("fixture", "quality", "quality_note")]
    fields = fields[:fields.index("notes")] + ["fixture", "quality", "quality_note", "notes"]
    for r in rows:
        r.update(EDITS.get(r["id"], {}))
        r["fixture"] = fixture(r["method"])
        c, b = lookup(C, r["id"]), lookup(B, r["id"])
        r["quality"], r["quality_note"] = ("C", c) if c else (("B", b) if b else ("A", ""))
    with PATH.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    for q in "ABC":
        print(q, sum(r["quality"] == q for r in rows))


if __name__ == "__main__":
    main()
