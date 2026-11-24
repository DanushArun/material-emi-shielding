"""Benchmark analysis: measured SE against the homogeneous-slab prediction.

For every entry of data_collection/verified_benchmark.csv this computes

  * the slab prediction in free space (TEM, plane wave at normal incidence),
  * the slab prediction for the fixture the source states (TE10 mode of the
    rectangular waveguide covering the measurement frequency, or TEM for
    coaxial cells and unspecified fixtures),
  * the residual  dSE = SE_measured - SE_slab  for both,
  * the effective conductivity sigma_eff that reproduces the measured SE in
    the same fixture model, and  log10(sigma_eff / sigma_reported).

It then writes summary statistics that treat papers, not entries, as the
independent unit (paper-clustered bootstrap).

Run from the repository root:  python docs/papers/benchmark_analysis.py
Writes benchmark_residuals.csv and benchmark_statistics.txt next to this script.
The slab formula here is written out independently of src/physics so that the
two can be compared (tests/test_benchmark_analysis.py).
"""
import cmath
import csv
import math
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
DATA = ROOT / "data_collection" / "verified_benchmark.csv"
OUT_CSV = HERE / "benchmark_residuals.csv"
OUT_TXT = HERE / "benchmark_statistics.txt"

MU0 = 4e-7 * math.pi
C0 = 299792458.0
EPS0 = 1.0 / (MU0 * C0 ** 2)
Z0 = MU0 * C0
BAND_POINTS = 41
N_BOOT = 10000
SEED = 20261124

# Standard rectangular waveguides: (lowest frequency, highest frequency, TE10 cutoff), Hz
WAVEGUIDES = [
    (8.0e9, 12.4e9, 6.557e9),    # WR-90, X band (8.0 GHz accepted: several sources quote 8-12 GHz)
    (12.4e9, 18.0e9, 9.488e9),   # WR-62, Ku band
    (18.0e9, 26.5e9, 14.051e9),  # WR-42, K band
]

# Analysis groups (label, member material classes)
GROUPS = [
    ("Continuous metal and oxide films", {"Metal film", "Transparent conductor", "Metal film (magnetic filler)"}),
    ("Metal meshes, coated fabric and paper", {"Metal mesh", "Metal-coated fabric", "Coated paper"}),
    ("Dense MXene and graphene films", {"MXene film", "Graphene film", "MXene hybrid film"}),
    ("Porous MXene and graphene films", {"MXene film (porous after annealing)", "Graphene film (porous)",
                                         "MXene hybrid film (porous)"}),
    ("Solid carbon-polymer composites", {"Carbon polymer composite", "CNT polymer composite"}),
    ("Segregated composites, foams, aerogels", {"Segregated composite", "Foam", "Aerogel"}),
]
POROUS_CLASSES = GROUPS[3][1] | {"Foam", "Aerogel"}


def cutoff(f: float):
    """TE10 cutoff of the standard waveguide whose band contains f, or None."""
    for lo, hi, fc in WAVEGUIDES:
        if lo <= f <= hi:
            return fc
    return None


def slab_se(sigma: float, t: float, f: float, fc: float | None = None) -> float:
    """Exact transmission loss (dB) of a homogeneous slab, mu_r = eps_r = 1.

    fc = None: plane wave at normal incidence in free space (also the TEM mode
    of a coaxial cell). fc given: TE10 mode of an air-filled rectangular
    waveguide with that cutoff, the slab filling the cross-section.
    """
    w = 2 * math.pi * f
    kc2 = (2 * math.pi * fc / C0) ** 2 if fc else 0.0
    eps_c = EPS0 - 1j * sigma / w
    g_air = cmath.sqrt(kc2 - w * w * MU0 * EPS0)      # j*beta above cutoff
    g_m = cmath.sqrt(kc2 - w * w * MU0 * eps_c)
    if g_m.real < 0:
        g_m = -g_m
    z_air, z_m = 1j * w * MU0 / g_air, 1j * w * MU0 / g_m   # TE wave impedances
    x = g_m * t
    back = cmath.exp(-2 * x) if x.real < 300 else 0.0
    denom = (z_air + z_m) ** 2 - (z_air - z_m) ** 2 * back
    return 20 * math.log10(abs(denom / (4 * z_air * z_m))) + (20 / math.log(10)) * x.real


def predict(sigma: float, t: float, f_lo: float, f_hi: float, waveguide: bool) -> float:
    """Band-averaged prediction in dB (arithmetic mean of dB values over the band)."""
    freqs = [f_lo] if f_lo == f_hi else np.linspace(f_lo, f_hi, BAND_POINTS)
    return float(np.mean([slab_se(sigma, t, f, cutoff(f) if waveguide else None) for f in freqs]))


def effective_sigma(se: float, t: float, f_lo: float, f_hi: float, waveguide: bool) -> float:
    """Conductivity of the homogeneous slab that gives the measured SE."""
    fn = lambda ls: predict(10.0 ** ls, t, f_lo, f_hi, waveguide) - se  # noqa: E731
    lo, hi = -9.0, 10.0
    if fn(lo) > 0:
        return float("nan")
    return 10.0 ** brentq(fn, lo, hi, xtol=1e-10)


def fixture_of(method: str) -> str:
    m = method.lower()
    if "coaxial" in m:
        return "coaxial"
    if "waveguide" in m:
        return "waveguide"
    return "unspecified"


def load() -> list[dict]:
    rows = []
    for r in csv.DictReader(DATA.open()):
        sigma, t = float(r["sigma_s_per_m"]), float(r["thickness_m"])
        f_lo, f_hi, se = float(r["f_low_hz"]), float(r["f_high_hz"]), float(r["se_db"])
        fixture = r.get("fixture") or fixture_of(r["method"])
        wg = fixture == "waveguide"
        f_mid = 0.5 * (f_lo + f_hi)
        free = predict(sigma, t, f_lo, f_hi, False)
        fix = predict(sigma, t, f_lo, f_hi, wg)
        as_wg = predict(sigma, t, f_lo, f_hi, fixture != "coaxial" and cutoff(f_lo) is not None)
        s_eff = effective_sigma(se, t, f_lo, f_hi, wg)
        rows.append({
            "id": r["id"], "material": r["material"], "material_class": r["material_class"],
            "group": next(i for i, (_, members) in enumerate(GROUPS) if r["material_class"] in members),
            "porous": r["material_class"] in POROUS_CLASSES,
            "paper": r["doi"], "citation": r["citation"], "fixture": fixture,
            "quality": r.get("quality", ""), "quality_note": r.get("quality_note", ""),
            "se_qualifier": r["se_qualifier"], "sigma_basis": r["sigma_basis"],
            "sigma": sigma, "t": t, "f_lo": f_lo, "f_hi": f_hi,
            "t_over_delta": t / math.sqrt(2 / (2 * math.pi * f_mid * MU0 * sigma)),
            "se": se, "slab_free": free, "slab": fix, "slab_all_wg": as_wg,
            "res_free": se - free, "res": se - fix, "res_all_wg": se - as_wg,
            "sigma_eff": s_eff, "log_ratio": math.log10(s_eff / sigma) if s_eff == s_eff else float("nan"),
        })
    return rows


def cluster_boot(rows: list[dict], stat, n_boot: int = N_BOOT, seed: int = SEED):
    """95% interval of stat(rows) resampling papers with replacement."""
    rng = np.random.default_rng(seed)
    papers = sorted({r["paper"] for r in rows})
    by = {p: [r for r in rows if r["paper"] == p] for p in papers}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(len(papers), len(papers))
        sample = [r for i in pick for r in by[papers[i]]]
        vals.append(stat(sample))
    return tuple(np.nanpercentile(vals, [2.5, 97.5]))


def summary(rows: list[dict], key: str = "res") -> dict:
    v = np.array([r[key] for r in rows])
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    lo, hi = cluster_boot(rows, lambda s: float(np.median([r[key] for r in s]))) if len({r["paper"] for r in rows}) > 2 \
        else (float("nan"), float("nan"))
    return {"n": len(v), "papers": len({r["paper"] for r in rows}), "median": float(med), "q1": float(q1),
            "q3": float(q3), "ci_lo": float(lo), "ci_hi": float(hi), "rmse": float(np.sqrt(np.mean(v ** 2))),
            "mae": float(np.mean(np.abs(v))), "min": float(v.min()), "max": float(v.max()),
            "w3": int(np.sum(np.abs(v) <= 3)), "w5": int(np.sum(np.abs(v) <= 5)), "w10": int(np.sum(np.abs(v) <= 10))}


def fmt(s: dict) -> str:
    return (f"n={s['n']:>2} papers={s['papers']:>2}  median={s['median']:+6.1f} [{s['ci_lo']:+.1f}, {s['ci_hi']:+.1f}]  "
            f"IQR={s['q1']:+.1f}..{s['q3']:+.1f}  RMSE={s['rmse']:5.1f}  MAE={s['mae']:5.1f}  "
            f"range={s['min']:+.1f}..{s['max']:+.1f}  within 3/5/10 dB: {s['w3']}/{s['w5']}/{s['w10']}")


def regression(rows: list[dict]) -> dict:
    """OLS of residual on log10(sigma) and a porosity indicator, paper-clustered bootstrap intervals."""
    def fit(sample):
        y = np.array([r["res"] for r in sample])
        x = np.column_stack([np.ones(len(sample)), [math.log10(r["sigma"]) for r in sample],
                             [1.0 if r["porous"] else 0.0 for r in sample]])
        return np.linalg.lstsq(x, y, rcond=None)[0]
    beta = fit(rows)
    rng = np.random.default_rng(SEED)
    papers = sorted({r["paper"] for r in rows})
    by = {p: [r for r in rows if r["paper"] == p] for p in papers}
    boots = np.array([fit([r for i in rng.choice(len(papers), len(papers)) for r in by[papers[i]]])
                      for _ in range(4000)])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    rho, p = spearmanr([math.log10(r["sigma"]) for r in rows], [r["res"] for r in rows])
    return {"intercept": beta[0], "slope": beta[1], "slope_ci": (lo[1], hi[1]),
            "porous": beta[2], "porous_ci": (lo[2], hi[2]), "spearman": float(rho), "spearman_p": float(p)}


def main() -> None:
    rows = load()
    fields = ["id", "material", "material_class", "fixture", "quality", "se_qualifier", "sigma", "t", "f_lo", "f_hi",
              "t_over_delta", "se", "slab_free", "slab", "res_free", "res", "res_all_wg", "sigma_eff", "log_ratio",
              "citation"]
    with OUT_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in r.items() if k in fields})

    out = []
    say = out.append
    say(f"Entries: {len(rows)}; papers: {len({r['paper'] for r in rows})}")
    say("Fixtures: " + ", ".join(f"{k} {sum(r['fixture'] == k for r in rows)}" for k in ("waveguide", "coaxial", "unspecified")))
    if any(r["quality"] for r in rows):
        say("Quality: " + ", ".join(f"{q} {sum(r['quality'] == q for r in rows)}" for q in sorted({r['quality'] for r in rows})))
    wg = [r for r in rows if r["fixture"] == "waveguide"]
    shift = np.array([r["slab"] - r["slab_free"] for r in wg])
    say(f"Waveguide correction (TE10 minus free space) over {len(wg)} entries: "
        f"median {np.median(shift):+.2f} dB, range {shift.min():+.2f} to {shift.max():+.2f} dB")
    say("")
    primary = [r for r in rows if r["quality"] != "C"]
    for label, sub in (("All entries", rows), ("Primary set (quality A and B)", primary),
                       ("Quality A only", [r for r in rows if r["quality"] == "A"])):
        if not sub or (label != "All entries" and not any(r["quality"] for r in rows)):
            continue
        say(f"== {label} ==")
        say("  fixture-specific : " + fmt(summary(sub)))
        say("  free space       : " + fmt(summary(sub, "res_free")))
        say("  all X/Ku as TE10 : " + fmt(summary(sub, "res_all_wg")))
        for name, cond in (("sigma >= 1e4 S/m", lambda r: r["sigma"] >= 1e4), ("1 <= sigma < 1e4", lambda r: 1 <= r["sigma"] < 1e4),
                           ("sigma < 1 S/m", lambda r: r["sigma"] < 1)):
            s2 = [r for r in sub if cond(r)]
            if s2:
                say(f"  {name:<17}: " + fmt(summary(s2)))
        for i, (g, _) in enumerate(GROUPS):
            s2 = [r for r in sub if r["group"] == i]
            if s2:
                say(f"  {g[:38]:<38}: " + fmt(summary(s2)))
        hi = [r for r in sub if r["sigma"] >= 1e4]
        pm = [float(np.median([r["res"] for r in hi if r["paper"] == p])) for p in sorted({r["paper"] for r in hi})]
        say(f"  per-paper medians, sigma >= 1e4: {len(pm)} papers, {sum(abs(x) <= 10 for x in pm)} within 10 dB, "
            f"{sum(abs(x) <= 5 for x in pm)} within 5 dB")
        reg = regression(sub)
        say(f"  residual = {reg['intercept']:+.1f} {reg['slope']:+.2f}*log10(sigma) {reg['porous']:+.1f}*porous;  "
            f"slope CI [{reg['slope_ci'][0]:+.2f}, {reg['slope_ci'][1]:+.2f}], porous CI [{reg['porous_ci'][0]:+.1f}, "
            f"{reg['porous_ci'][1]:+.1f}];  Spearman rho = {reg['spearman']:+.2f} (p = {reg['spearman_p']:.1e})")
        lr = np.array([r["log_ratio"] for r in sub if r["log_ratio"] == r["log_ratio"]])
        say(f"  log10(sigma_eff/sigma_reported): median {np.median(lr):+.2f}, IQR {np.percentile(lr, 25):+.2f}..{np.percentile(lr, 75):+.2f}")
        say("")
    say("== Entries with |residual| > 10 dB (fixture-specific) ==")
    for r in sorted(rows, key=lambda r: r["res"]):
        if abs(r["res"]) > 10:
            say(f"  {r['res']:+6.1f} dB  log ratio {r['log_ratio']:+5.2f}  sigma {r['sigma']:.3g}  t/delta {r['t_over_delta']:.3g}  "
                f"{r['fixture']:<11} {r['quality'] or '-'}  {r['material'][:50]}")
    OUT_TXT.write_text("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\nWrote {OUT_CSV.name} and {OUT_TXT.name}")


if __name__ == "__main__":
    main()
