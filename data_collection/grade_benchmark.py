"""Assign a measurement fixture and a documentation grade to every benchmark entry.

Run from the repository root:  python data_collection/grade_benchmark.py
Rewrites the fixture, quality and quality_note columns of verified_benchmark.csv.

Fixture
    waveguide          the source states a rectangular waveguide
    coaxial            the source states a coaxial cell or coaxial line
    assumed-waveguide  not stated in the text we read; every frequency lies in 8-18 GHz
    assumed-tem        not stated in the text we read; broadband or outside 8-18 GHz

Grade (rules are applied to what the source prints; the residual is not consulted)
    C  any of: two printed values for the same quantity of this sample that differ by more
       than 10 %; a conductivity unit printed inconsistently; a stated conductivity that
       contradicts the printed sheet resistance and thickness; conductivity and thickness
       that refer to different objects; full text not read
    B  none of the above and at least one of the caveats listed in CAVEATS
    A  no caveat
"""
import csv
from pathlib import Path

PATH = Path(__file__).with_name("verified_benchmark.csv")

C = {
    "shahzad2016_ti3c2_45um": "Only the abstract was available to us; band, method and total-SE status not verified",
    "wang2026_tocnf_mxene_": "Conductivity unit printed inconsistently (S cm-1 in the equation, 's m-1' in the table header); "
                             "Methods also state SE 'was calculated according to the Simon formula'",
    "zhang2023_graphene_": "Stated conductivity contradicts the sheet resistance and thickness printed for the same film (factor 5-12)",
    "liu2020_agni60_6to1": "Two sheet resistances printed for this film (0.78 and about 1.86 ohm)",
    "ahmed2023_mxene_mwcnt_mos2_15": "Two thicknesses printed for this film (16 and 9.2 um)",
    "durmaz2023_pa11pla_": "Printed DC volume resistivity contradicts the 0.01 Hz conductivity used",
    "han2024_agnw_mxene_paper": "Conductivity is that of the coating, thickness that of the whole paper sheet; "
                                "141 um is attributed to two different samples",
}

# caveat -> (text, id prefixes)
CAVEATS = {
    "sigma_not_specimen": ("conductivity stated for the film type or as a range, not measured on this specimen",
                           ["han2020_ti3c2_", "park2026_mxene_29nm", "iqbal2020_ti3c2_asis", "iqbal2020_ti3c2_350"]),
    "thickness_nominal": ("thickness is nominal, approximate or measured before a processing step",
                          ["iqbal2020_ti3c2_350", "iqbal2020_ti3cn_250", "iqbal2020_ti3cn_350", "zarei2024_agmesh_",
                           "voronin2022_cumesh_", "madinehei2021_polyester_gnp25", "li2022_cnt_wpu_50", "liu2020_ag",
                           "alam2017_gnp_pp_10", "han2020_ti3c2_40nm", "han2020_ti3c2_137nm", "park2026_mxene_29nm"]),
    "approximate_value": ("SE or conductivity printed as approximate",
                          ["voronin2025_iai3", "jeon2023_blso", "palanisamy2021_cuni_fabric", "xia2022_mxene_go"]),
    "substrate_or_layered": ("measured on a substrate, embedded in one, or not a single homogeneous layer",
                             ["jeon2023_blso", "huang2025_mxene_mx80", "li2023_agmesh_", "zarei2024_agmesh_",
                              "voronin2022_cumesh_", "voronin2025_iai3", "hassan2024_mxene_cnt_janus",
                              "pande2009_mwcnt_pmma_scb2", "liu2020_ag", "palanisamy2021_cuni_fabric"]),
    "sigma_method": ("two-probe, DC-meter or low-frequency AC conductivity",
                     ["wani2026_graphitic_aerogel_", "rahaman2023_epdm_cb_", "ma2022_pvdf_mwcnt_gnp_film"]),
    "se_qualifier_unclear": ("frequency or averaging of the SE value not stated explicitly",
                             ["ahmed2023_ti3c2_12um", "weng2025_mwcnt_mxene_50", "xue2023_mxene_cnt_pi_aerogel",
                              "chen2025_cnf_pani_aerogel", "wan2022_mxene_"]),
    "second_value": ("a second value within 10 % is printed for the same quantity",
                     ["lin2019_graphene_lg4", "chen2025_cnf_pani_aerogel"]),
    "preprint": ("values read from the preprint of the cited article", ["liu2020_ag"]),
}
FIXTURE_CAVEAT = "fixture not stated in the text read"


def fixture(method: str, f_lo: float, f_hi: float) -> str:
    m = method.lower()
    if "coaxial" in m:
        return "coaxial"
    if "waveguide" in m:
        return "waveguide"
    return "assumed-waveguide" if (f_lo >= 8.0e9 and f_hi <= 18.0e9) else "assumed-tem"


def main() -> None:
    rows = list(csv.DictReader(PATH.open()))
    fields = [f for f in rows[0].keys() if f not in ("fixture", "quality", "quality_note")]
    fields = fields[:fields.index("notes")] + ["fixture", "quality", "quality_note", "notes"]
    for r in rows:
        r["fixture"] = fixture(r["method"], float(r["f_low_hz"]), float(r["f_high_hz"]))
        c = next((note for prefix, note in C.items() if r["id"].startswith(prefix)), None)
        caveats = [text for text, prefixes in CAVEATS.values() if any(r["id"].startswith(p) for p in prefixes)]
        if r["fixture"].startswith("assumed"):
            caveats.append(FIXTURE_CAVEAT)
        if c:
            r["quality"], r["quality_note"] = "C", c
        elif caveats:
            r["quality"], r["quality_note"] = "B", "; ".join(caveats)
        else:
            r["quality"], r["quality_note"] = "A", ""
    with PATH.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    for q in "ABC":
        print(q, sum(r["quality"] == q for r in rows))
    for f in ("waveguide", "coaxial", "assumed-waveguide", "assumed-tem"):
        print(f, sum(r["fixture"] == f for r in rows))


if __name__ == "__main__":
    main()
