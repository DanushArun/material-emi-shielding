# An integrated physics framework for EMI shielding design: percolation-resolved composite prediction and multilayer optimization

> **Note:** This is a Markdown mirror of the LaTeX manuscript intended for *Materials & Design*. The authoritative file is `materials-and-design-manuscript.tex`. All numbers in this document are reproducible from `validation_metrics_raw.txt` against the open Python implementation and the curated benchmark dataset in `data_collection/experimental_benchmark_data.py`.

---

## Highlights

- An open multi-physics framework couples Schelkunoff plane-wave theory, the transfer matrix method, the McLachlan general effective media equation, Snoek--Debye permeability, the Mayadas--Shatzkes grain-boundary model, and Monte Carlo uncertainty propagation in sub-second computation per evaluation.
- Validation against 106 published experimental shielding effectiveness measurements is reported in three explicit measurement regimes; in the validatable regime (*t*/*δ* ≤ 3 and *SE*<sub>exp</sub> ≤ 80 dB) the model agrees with non-magnetic metal benchmarks at 6.6 dB mean absolute error and with the full subset at 13.9 dB.
- The McLachlan equation correctly resolves the percolation conductivity transition that linear mixing under-estimates by twelve orders of magnitude immediately below the threshold and three orders of magnitude immediately above.
- Layered MXene composites display a systematic 45--61 dB excess of measured over homogeneous-effective-medium-predicted shielding effectiveness, identifying lamellar internal reflections as the dominant uncaptured mechanism in current isotropic models.
- Sobol total-order sensitivity indices computed with material-class-appropriate coefficient-of-variation values, rather than uniform defaults, yield actionable manufacturing-tolerance priorities that depend on the underlying physics regime.

---

## Abstract

The classical Schelkunoff theory of electromagnetic interference (EMI) shielding is closed-form, fast, and physically transparent, yet it routinely produces predictions that exceed the dynamic range of standard measurement apparatus by orders of magnitude. Modern composite, multilayer, and ferromagnetic shielding materials further violate the homogeneous, frequency-independent, single-layer assumptions on which the original theory rests. Here we present an integrated analytical framework that extends Schelkunoff theory with five coupled physics modules: a numerically stable transfer matrix method for *N*-layer shields, the McLachlan general effective media equation with percolation threshold estimation for composite conductivity, the Snoek--Debye model for frequency-dependent ferromagnetic permeability, the Mayadas--Shatzkes model for grain-boundary scattering, and Monte Carlo propagation with Sobol global sensitivity analysis. We benchmark the framework against 106 published shielding effectiveness measurements drawn from peer-reviewed literature spanning 1 MHz to 77 GHz, and we explicitly stratify the dataset into three measurement regimes determined by the dimensionless thickness *t*/*δ* and the experimental dynamic range. In the validatable regime, the model reproduces non-magnetic metal benchmarks at a mean absolute error of 6.6 dB (*n* = 6, *R*² = 0.68) and the full validatable subset at 13.9 dB (*n* = 20); in the extrapolation regime, plane-wave predictions diverge from reported values by amounts that closely track the censoring imposed by a typical 120 dB measurement ceiling. Coupling the McLachlan equation to the analytical shielding calculation correctly resolves the percolation conductivity transition that linear mixing rules cannot capture, and identifies the lamellar internal reflection in layered MXene films, rather than bulk effective conductivity, as the dominant outstanding modelling gap for high-performance two-dimensional shielding materials. A material-class-specific Sobol analysis with physically realistic coefficient-of-variation values shifts the dominant uncertainty contributor from conductivity (non-magnetic metals) to permeability (ferromagnets) to filler conductivity (composites near percolation), providing direct manufacturing-tolerance guidance. The complete simulation engine is implemented as an open Python library with a representational state transfer (REST) interface and is suitable for integration with optimization, machine-learning, and inverse-design workflows.

**Keywords:** electromagnetic interference shielding; shielding effectiveness; composite materials; percolation theory; transfer matrix method; Monte Carlo uncertainty quantification; MXene; design framework

---

## 1. Introduction

Electromagnetic interference (EMI) shielding is a fundamental requirement in essentially every modern electronic system. The accelerating densification of high-speed digital and analogue circuitry, the migration of fifth-generation wireless networks into the millimetre-wave bands at 24--77 GHz, the deployment of automotive radar, and the proliferation of medical and aerospace electronics together impose increasingly stringent shielding requirements while simultaneously demanding lighter, thinner, and mechanically compliant shield architectures (Wanasinghe et al., 2020; Hong et al., 2017; Liu et al., 2025; Wang et al., 2025). Two decades of research on conductive polymer composites (Al-Saleh & Sundararaj, 2009; Bauhofer & Kovacs, 2009; Arjmand et al., 2011; Thomassin et al., 2013), two-dimensional transition-metal carbides and nitrides (MXenes) (Shahzad et al., 2016; Iqbal et al., 2020; Han et al., 2020; Hu et al., 2026; Liu et al., 2024), graphene-based aerogels (Shen et al., 2014; Yan et al., 2015; Zhang et al., 2016; He et al., 2024), carbon nanotube networks (Li et al., 2015), and hybrid magneto-conductive systems (Song et al., 2014) has expanded the EMI shielding design space far beyond the metallic enclosures envisaged in the foundational work of Schelkunoff (1934) and the standard-defining experiments of Schulz et al. (1988).

The shielding effectiveness *SE* of a planar barrier is conventionally defined as the logarithmic ratio of incident to transmitted electromagnetic power and is decomposed into reflection loss *SE*<sub>R</sub> at the air--material interfaces, absorption loss *SE*<sub>A</sub> from ohmic and magnetic dissipation within the bulk, and a multiple-reflection correction *SE*<sub>M</sub>. The framework is the workhorse of electromagnetic compatibility engineering (Ott, 2009; Paul, 2006; Pozar, 2011; Celozzi et al., 2008) but is increasingly inadequate for four classes of contemporary shielding problem: percolation-dominated nanocomposites, ferromagnetic materials operating above the low megahertz range, multilayer shield stacks, and reliability-based design under manufacturing variability.

This paper presents a framework that extends Schelkunoff theory to address each of these limitations while preserving the speed and physical transparency that make the analytical approach valuable in the first place. Our contributions are five: (i) we construct an integrated computational engine that couples Schelkunoff plane-wave theory, a numerically stable TMM for thick conductors, the McLachlan general effective media equation with percolation-threshold estimation, Snoek--Debye permeability, Mayadas--Shatzkes grain-boundary scattering, and Monte Carlo propagation with Sobol global sensitivity analysis; (ii) we benchmark against 106 published experimental measurements with explicit reporting in three measurement regimes; (iii) we demonstrate that the McLachlan equation resolves the percolation transition that linear mixing cannot capture; (iv) we identify the lamellar internal reflection in layered MXene films as the dominant outstanding modelling gap for two-dimensional high-performance shielding materials; and (v) we show that Sobol total-order sensitivity indices computed with physically realistic, material-class-specific coefficient-of-variation values produce qualitatively different and quantitatively actionable manufacturing-tolerance guidance.

---

## 2. Computational framework

The framework is structured as five physics modules that share a common Python application programming interface and that can be combined to compute the shielding effectiveness of single-layer, multilayer, and composite shields under user-specified frequency, thickness, temperature, and microstructural conditions.

### 2.1 Schelkunoff plane-wave shielding theory

The foundational module implements the transmission-line analogy of Schelkunoff (1934). The total shielding effectiveness decomposes as

  *SE*<sub>total</sub> = *SE*<sub>R</sub> + *SE*<sub>A</sub> + *SE*<sub>M</sub>   (dB),

with

  γ = jω √(μ ε<sub>complex</sub>),   η = √(jωμ / (σ + jωε<sub>r</sub>ε<sub>0</sub>)),

  δ = √(2 / ωμσ),

and *SE*<sub>A</sub> ≈ 8.686 *t*/δ (dB) in the good-conductor limit. The reflection loss is computed from the power reflection coefficient as *SE*<sub>R</sub> = −10 log<sub>10</sub>(1 − |Γ|²). The multiple-reflection correction is significant only when *SE*<sub>A</sub> < 15 dB.

### 2.2 Numerically stable transfer matrix method

For an *N*-layer stack, each layer *i* is represented by a 2×2 transfer matrix:

  **T**<sub>i</sub> = [ [cosh(γ<sub>i</sub>d<sub>i</sub>), η<sub>i</sub> sinh(γ<sub>i</sub>d<sub>i</sub>)], [sinh(γ<sub>i</sub>d<sub>i</sub>)/η<sub>i</sub>, cosh(γ<sub>i</sub>d<sub>i</sub>)] ].

The total stack matrix **T**<sub>total</sub> = **T**<sub>1</sub>**T**<sub>2</sub>...**T**<sub>N</sub> with elements *A*, *B*, *C*, *D* yields the transmission coefficient *S*<sub>21</sub> = 2 / (*A* + *B*/*Z*<sub>0</sub> + *CZ*<sub>0</sub> + *D*) and *SE* = −20 log<sub>10</sub>|*S*<sub>21</sub>|. For thick conductors at high frequency where Re(γ<sub>i</sub>d<sub>i</sub>) > 200, the implementation factors the dominant exponential out analytically and accumulates an explicit log-domain scale factor α<sub>d</sub>, preserving O(*N*) complexity without overflow.

We verified the TMM against the Schelkunoff result for a single 0.5 µm copper layer at 1 GHz: both methods return *SE* = 33.4 dB, agreeing to better than 0.01 dB. For a Cu(0.5 µm)/PET(100 µm)/Cu(0.5 µm) sandwich at 1 GHz, the TMM returns *SE* = 102.5 dB (reflection 34.5 dB, absorption 68.0 dB). Naive summation of the two single-layer copper losses yields only *SE* = 66.9 dB. The 35.6 dB difference arises entirely from the additional reflection at the two extra impedance boundaries introduced by the dielectric core.

### 2.3 Composite material conductivity

For a composite with conductive filler (volume fraction *f*, conductivity σ<sub>f</sub>) in an insulating matrix (conductivity σ<sub>m</sub>), the conductivity follows a percolation power law above the threshold *f*<sub>c</sub> with exponent *t* ≈ 2 for three-dimensional random networks (Kirkpatrick, 1973). Excluded-volume theory gives *f*<sub>c</sub> ≈ 0.7/(*L*/*D*) for rods and *f*<sub>c</sub> ≈ 0.5/(*R*/*t*<sub>f</sub>) for disks (Balberg et al., 1984; Celzard et al., 1996).

The McLachlan general effective media (GEM) equation (McLachlan et al., 1990) unifies percolation behaviour with classical effective-medium theory:

  (1 − *f*)(σ<sub>m</sub><sup>1/t</sup> − σ<sub>eff</sub><sup>1/t</sup>) / (σ<sub>m</sub><sup>1/t</sup> + Aσ<sub>eff</sub><sup>1/t</sup>) + *f*(σ<sub>f</sub><sup>1/t</sup> − σ<sub>eff</sub><sup>1/t</sup>) / (σ<sub>f</sub><sup>1/t</sup> + Aσ<sub>eff</sub><sup>1/t</sup>) = 0,

with *A* = (1 − *f*<sub>c</sub>)/*f*<sub>c</sub>. We solve this with Brent's method to a relative tolerance of 10⁻¹². Table 1 reports the linear-mixing and McLachlan predictions for a representative system (σ<sub>f</sub> = 10⁵ S/m, σ<sub>m</sub> = 10⁻¹⁰ S/m, *f*<sub>c</sub> = 0.01, *t* = 2). Just below the threshold (*f* = 0.005), linear mixing returns σ<sub>eff</sub> = 500 S/m while McLachlan returns 4.0×10⁻¹⁰ S/m, a discrepancy of twelve orders of magnitude. Just above the threshold (*f* = 0.012), linear mixing returns 1200 S/m while McLachlan returns 0.41 S/m, a discrepancy of three orders of magnitude.

**Table 1.** Linear-mixing versus McLachlan GEM prediction of composite conductivity. Filler 10⁵ S/m, matrix 10⁻¹⁰ S/m, *f*<sub>c</sub> = 0.01, *t* = 2.

| *f* (vol.) | σ<sub>linear</sub> (S/m) | σ<sub>GEM</sub> (S/m) | ratio |
|---|---|---|---|
| 0.005 | 5.0×10² | 4.0×10⁻¹⁰ | 8.0×10⁻¹³ |
| 0.010 | 1.0×10³ | 3.2×10⁻⁵ | 3.2×10⁻⁸ |
| 0.012 | 1.2×10³ | 4.1×10⁻¹ | 3.4×10⁻⁴ |
| 0.020 | 2.0×10³ | 1.0×10¹ | 5.1×10⁻³ |
| 0.050 | 5.0×10³ | 1.6×10² | 3.3×10⁻² |
| 0.100 | 1.0×10⁴ | 8.3×10² | 8.3×10⁻² |

Maxwell--Garnett, Bruggeman, and Hashin--Shtrikman forms are also implemented for cross-checking; a high-level dispatcher selects the appropriate model on the basis of declared composite type.

### 2.4 Frequency- and temperature-dependent properties

For ferromagnetic materials, the Snoek limit (Snoek, 1948) gives

  (μ<sub>s</sub> − 1) *f*<sub>r</sub> = (2/3) γ<sub>gyro</sub> μ<sub>0</sub> *M*<sub>s</sub>,

with γ<sub>gyro</sub> = 2.8×10¹⁰ Hz/T. For mu-metal (μ<sub>r</sub> = 100,000, *M*<sub>s</sub> = 860 kA/m), *f*<sub>r</sub> ≈ 0.2 MHz. Above the resonance, the Debye relaxation μ(*f*) = 1 + (μ<sub>s</sub> − 1) / (1 + j*f*/*f*<sub>r</sub>) yields |μ<sub>r</sub>| ≈ 20 at 1 GHz, a five-thousand-fold reduction from the static value. Without this correction, the Schelkunoff calculation predicts *SE* > 140 dB for a 1 mm mu-metal sheet at 1 GHz, a value no 1 mm metallic barrier can deliver.

The temperature dependence of the conductivity above the Debye temperature follows σ(*T*) = σ<sub>ref</sub> / [1 + α(*T* − *T*<sub>ref</sub>)] with values from Matula (1979): copper α = 0.00393 K⁻¹, aluminium 0.00390, nickel 0.00690, iron 0.00651, steel 1018 0.00600, stainless 304 0.00094, mu-metal 0.00200. For copper at 423 K, this predicts σ = 4.0×10⁷ S/m (33% reduction).

### 2.5 Mayadas--Shatzkes grain-boundary scattering

The closed-form correction (Mayadas & Shatzkes, 1970) is

  σ<sub>eff</sub>/σ<sub>bulk</sub> = 1 − (3/2)α + 3α² − 3α³ ln(1 + 1/α),

with α = (ℓ/*d*<sub>g</sub>) · *R*/(1 − *R*), ℓ = 40 nm for copper at room temperature, and *R* ≈ 0.25. For nanocrystalline copper with *d*<sub>g</sub> = 30 nm, this predicts σ<sub>eff</sub>/σ<sub>bulk</sub> = 0.50, in agreement with the experimental conductivity reduction of Mayadas and Shatzkes (1970). The benchmark dataset records a 23 dB difference in shielding effectiveness between nanocrystalline (*d*<sub>g</sub> = 30 nm, *SE* = 95 dB) and conventionally annealed (*d*<sub>g</sub> = 50 µm, *SE* = 118 dB) copper foils at 0.1 mm thickness and 1 GHz.

### 2.6 Monte Carlo and Sobol with material-class-specific CVs

Manufacturing variability is propagated via Monte Carlo sampling. Each input parameter is drawn from an appropriate probability distribution (normal for σ, *t*, μ, *f*; log-normal for grain size). For *N* = 10,000 samples, the complete forward model is evaluated at each sample point. Latin hypercube sampling (McKay et al., 1979) is provided as an alternative.

For global sensitivity analysis, we implement the Sobol decomposition (Sobol, 2001) with the Saltelli estimator (Saltelli, 2002). The implementation uses scrambled quasi-random Sobol sequences from `scipy.stats.qmc.Sobol` and requires *N*(2*D* + 2) model evaluations.

**Material-class-specific coefficient of variation.** The CV assigned to each input parameter is itself an input to the Sobol calculation, and uniform default CVs across material classes can produce qualitatively misleading sensitivity rankings. The diamagnetic relative permeability of pure copper is approximately 0.999994 and varies metallurgically by perhaps one part in 10⁶; assigning the conventional default CV<sub>μ</sub> = 0.10 to such a material is unphysical and inflates the apparent permeability sensitivity. We therefore use class-specific CV values: CV<sub>μ</sub> = 10⁻⁴ for non-magnetic metals and composites, CV<sub>μ</sub> = 0.20 for ferromagnetic materials, CV<sub>σ</sub> = 0.30 for composites near percolation. Other CVs: CV<sub>σ</sub> = 0.05 for metals, CV<sub>t</sub> = 0.02--0.05, CV<sub>f</sub> = 0.001.

---

## 3. Validation methodology and benchmark dataset

### 3.1 Curated dataset of *n* = 106 experimental measurements

We curated 106 published shielding effectiveness measurements with full citations and measurement methods, organized into six categories:

**Table 2.** Composition of the 106-entry benchmark dataset.

| Category | *N* | Materials | Frequency range |
|---|---|---|---|
| Pure metals | 35 | Cu, Al, Steel, SS304, Ni, Ag | 1 MHz -- 10 GHz |
| Composites | 24 | CFRP, MWCNT, MXene, graphene, paint | 1 -- 12 GHz |
| Multilayer | 7 | Metal/dielectric sandwiches | 1 MHz -- 10 GHz |
| Temperature effects | 11 | Cu, Al at −269 to 200 °C | 1 GHz |
| Microstructure | 15 | Steel and Cu, varied grain size | 1 GHz |
| Frequency bands | 14 | 5G, Wi-Fi, automotive radar | 2.4 -- 77 GHz |

The dataset spans *SE* from 2 dB to 260 dB, thickness from 200 nm to 11 mm, and conductivity from 10⁻⁴ to 5×10⁹ S/m.

### 3.2 Three-regime classification

Standard ASTM D4935 coaxial fixtures and rectangular waveguide setups exhibit dynamic ranges of approximately 80--120 dB. Many "experimental" values reported above 120 dB are obtained by extrapolation through the Schelkunoff equations themselves rather than by direct transmission measurement; comparing the same calculation against such values is circular. We therefore stratify the 106 entries into three regimes:

- **Regime A (validatable):** *t*/δ ≤ 3 AND *SE*<sub>exp</sub> ≤ 80 dB. The shield is electrically thin in absorption, the experimental value lies within measurement dynamic range. *n* = 21.
- **Regime B (near ceiling):** 3 < *t*/δ ≤ 10 AND 80 dB < *SE*<sub>exp</sub> ≤ 120 dB. The experimental value approaches the apparatus ceiling. *n* = 12.
- **Regime C (extrapolation):** *t*/δ > 10 OR *SE*<sub>exp</sub> > 120 dB. The reported value is essentially calculated, not measured. *n* = 65.

Headline accuracy claims concern Regime A only; Regime B and C behaviour is reported for completeness.

### 3.3 Accuracy metrics

We report the mean absolute error (MAE), root mean square error (RMSE), median absolute error, systematic bias, and coefficient of determination *R*². No model parameter was fitted to the benchmark dataset. Percolation thresholds are those reported by original experimental authors; TCR coefficients from Matula (1979); GEM critical exponent at theoretical *t* = 2.

---

## 4. Results

### 4.1 Pure metal validation in three regimes

Restricting to non-magnetic metals (copper, aluminium, silver, stainless 304) within Regime A, the model achieves MAE = **6.6 dB** (*n* = 6, RMSE = 7.3 dB, *R*² = 0.68). The model is unbiased in this subset to within a fraction of a decibel.

For the full Regime A subset (*n* = 20, including non-magnetic metals, composites, and frequency-band entries), the model achieves MAE = **13.9 dB**, RMSE = 23.5 dB, median |err| = 9.5 dB, and bias = **−13 dB**. The model systematically *under*-predicts within the validatable subset; per-material analysis shows composites with strongly anisotropic or layered microstructure are the dominant contributors.

In Regime B (*n* = 12), residuals grow systematically as *SE*<sub>exp</sub> approaches the apparatus ceiling. In Regime C (*n* = 65), predicted *SE* exceeds reported values by hundreds to thousands of dB; these residuals have no physical interpretation.

### 4.2 Per-material accuracy in Regime A

Three bands of behaviour:

1. **MAE < 5 dB**: CFRP at moderate carbon-fibre loading (1.3 dB), conductive paint (1.0 dB).
2. **MAE 5--15 dB**: Cu, Al, Ag, SS304, MWCNT/PMMA across four loadings (8.1 dB), MWCNT/PVDF (6.6 dB), short-carbon-fibre/epoxy (9.7 dB across 3 loadings).
3. **MAE > 15 dB**: layered MXene composites: Ti₃C₂T<sub>x</sub>/cellulose-nanofibre (60.7 dB, exp 72 dB), nacre-like Ti₃C₂T<sub>x</sub>/sodium-alginate film (59.9 dB, exp 57 dB), Ti₃C₂T<sub>x</sub> MXene films at 45 µm (44.1 dB, exp 50 dB).

The systematic 45--61 dB excess of measured over bulk-effective-medium-predicted *SE* for layered MXene films is attributed to lamellar inter-flake reflections that the homogeneous-slab assumption cannot capture. This is the central outstanding modelling gap for two-dimensional shielding materials.

### 4.3 McLachlan integration on composites

For four MWCNT/PMMA loadings spanning the percolation transition (0.1, 0.5, 2, 5 wt%, *SE* 2--30 dB) reported by Al-Saleh & Sundararaj (2009) and Arjmand et al. (2011), linear mixing produces MAE = 18.5 dB compared to 8.1 dB for the McLachlan equation, a 56% reduction driven entirely by the correct treatment of the percolation transition. The same improvement pattern is observed for graphene, MXene, and metal-particle-filled composites.

### 4.4 Frequency-dependent permeability for ferromagnetic materials

Without the Snoek--Debye correction, the Schelkunoff calculation predicts *SE* > 433 dB for mild steel 1018 at 1 MHz (0.5 mm, μ<sub>r</sub> = 300) and > 10000 dB for nickel at 1 GHz (μ<sub>r</sub> = 50); experimental values are 80 dB and 155 dB respectively. With the Debye relaxation, predicted values fall to 74 dB and ~1500 dB. The remaining nickel residual at GHz frequencies indicates that even the corrected model retains substantial systematic error for ferromagnetic shields.

### 4.5 Multilayer transfer matrix method

For the Cu(0.5 µm)/PET(100 µm)/Cu(0.5 µm) sandwich, TMM returns *SE* = 102.5 dB at 1 GHz; naive summation yields 66.9 dB. The 35.6 dB difference is the signature of constructive inter-layer interference at the four impedance boundaries.

### 4.6 Material-class-specific Sobol sensitivity

Total-order Sobol indices computed with material-class-appropriate CVs:

| Material class | *S*<sub>T</sub>(σ) | *S*<sub>T</sub>(μ) | *S*<sub>T</sub>(*t*) | *S*<sub>T</sub>(*f*) |
|---|---|---|---|---|
| Non-magnetic Cu (10 µm, 1 GHz) | 0.65 | <10⁻³ | 0.34 | <10⁻³ |
| Ferromagnetic mild steel (1 mm, 1 MHz) | 0.13 | 0.84 | 0.03 | <10⁻⁴ |
| MWCNT/polymer (2 mm, 8.2 GHz) | 0.95 | <10⁻³ | 0.04 | <10⁻⁴ |

Manufacturing implications: tighten thickness for non-magnetic metallic shields; control permeability via annealing for ferromagnetic shields; control filler dispersion uniformly for percolation-dominated composites.

---

## 5. Discussion

The validation results support three positive conclusions. First, within Regime A the framework reproduces non-magnetic metals at the tens-of-decibels level (MAE 6.6 dB, *n* = 6). Second, the McLachlan equation reduces composite MAE from 18.5 dB to 8.1 dB on MWCNT/PMMA across the percolation transition. Third, the TMM captures the 35.6 dB multilayer reflection enhancement that single-layer summation misses.

The negative conclusion is equally important. For layered MXene composites, the bulk-effective-medium description systematically under-predicts *SE* by 45--61 dB. This is not a failure of the bulk physics but of the homogeneous-slab assumption to capture the lamellar microstructure that gives these materials their exceptional thickness efficiency. Bridging this gap analytically will likely require either an anisotropic effective-medium framework or explicit TMM enumeration of representative flake stacks.

**Implications for analytical-vs-measured comparison.** Single-aggregate-statistic validation across the full 106-entry dataset is misleading by construction because the absorption term grows exponentially in *t*/δ. We recommend that future analytical-vs-measured benchmarking in EMI shielding adopt explicit regime stratification.

### 5.1 Limitations

- Plane-wave normal-incidence assumption neglects oblique-incidence and edge-diffraction effects.
- Isotropic effective-medium assumption unsuitable for aligned-fibre composites and layered MXene architectures.
- Linear TCR valid only above Debye temperature; cryogenic accuracy requires Bloch--Grüneisen.
- Single-exponent McLachlan implementation is a simplification of the two-exponent form.
- Benchmark dataset not split into formal training/holdout subsets (no model parameter was fitted, so this is not a fitting concern, but a follow-up study with new measurements made specifically for blind validation would strengthen the empirical case).
- Sobol analyses assume independent input distributions; correlated uncertainties would warrant copula-based propagation.

---

## 6. Design applications

**Sub-second screening of 5G shielding candidates.** Six candidate materials evaluated across 1 MHz -- 40 GHz: copper 0.1 mm (>100 dB), aluminium 6061 1.5 mm (>150 dB), CFRP 2 mm (~50--55 dB), MXene films 45 µm (~85--92 dB at X-band, exceptional thickness efficiency), MWCNT/PMMA 1 mm (~28--32 dB), ITO 200 nm (~18--22 dB at 28 GHz). Each evaluation completes in under 50 ms.

**Multilayer weight optimization.** A Cu/PET/Cu sandwich scaled uniformly for *SE* ≥ 80 dB at 10 GHz returns 12 µm Cu / 500 µm PET / 12 µm Cu achieving 82 dB at 0.72 kg/m², approximately ten percent of the areal density of an equivalent solid copper sheet.

**Reliability-based composite design.** A CNT/epoxy shield required to achieve *SE* ≥ 40 dB at 8.2 GHz with 95% reliability requires deterministic threshold 4.2 wt% CNT but reliability-based threshold 6.8 wt% (a 62% increase), once the steep conductivity--filler-fraction relationship near percolation is propagated through the deterministic forward model.

---

## 7. Conclusions

In the validatable regime (*t*/δ ≤ 3, *SE*<sub>exp</sub> ≤ 80 dB), the framework reproduces non-magnetic metal benchmarks at 6.6 dB MAE and the full validatable subset at 13.9 dB MAE with a systematic under-prediction bias of −13 dB traceable to the homogeneous-slab approximation breaking down on layered composites. The McLachlan equation reduces composite MAE from 18.5 dB to 8.1 dB by capturing the order-of-magnitude conductivity discontinuity at percolation. The TMM recovers the 35.6 dB multilayer reflection enhancement that single-layer summation misses. Outside the validatable regime, plane-wave predictions diverge from reported values by amounts that closely track measurement censoring, and we recommend that future analytical-vs-measured benchmarking in EMI shielding adopt the same regime stratification.

The most significant outstanding modelling gap is the systematic 45--61 dB under-prediction of *SE* for layered Ti₃C₂T<sub>x</sub> MXene films, a failure of the homogeneous-slab assumption rather than the bulk physics. The Sobol sensitivity analysis with material-class-specific CV values yields direct manufacturing-tolerance guidance: thickness control dominates for non-magnetic metallic shields; permeability control dominates for ferromagnetic shields; and filler-loading uniformity dominates for percolation-dominated composites.

The complete framework is implemented as an open Python library with a REST interface and is suitable for integration with optimization, machine-learning, and inverse-design workflows.

---

## Data availability

The complete 106-entry benchmark dataset and the simulation source code are available in the project repository. The master validation metrics file (`validation_metrics_raw.txt`) records the full output of the validation pipeline at the resolution of this paper. All numbers cited above are reproducible from the open implementation against the published dataset.

---

## References

(Full bibliography in the .tex file. Selected key citations: Schelkunoff 1934; Schulz et al. 1988; McLachlan et al. 1990; Mayadas & Shatzkes 1970; Snoek 1948; Matula 1979; Sobol 2001; Saltelli 2002; Celozzi et al. 2008; Ott 2009; Bauhofer & Kovacs 2009; Al-Saleh & Sundararaj 2009; Arjmand et al. 2011; Shahzad et al. 2016; Iqbal et al. 2020; Han et al. 2020; Wanasinghe et al. 2020; Liu et al. 2024 *Carbon* 221, 118948; He et al. 2024 *Chem. Eng. J.* 501, 157507; Liu et al. 2025 *Adv. Funct. Mater.*; Wang et al. 2025 *Nanoscale Adv.*; Hu et al. 2026 *Small*.)
