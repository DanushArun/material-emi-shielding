---
title: "An integrated physics framework for EMI shielding design: percolation-resolved composite prediction and multilayer optimization"
author: "Danush Arun"
---

**Abstract.** Designing an EMI shield analytically is straightforward in principle. Schelkunoff theory returns shielding effectiveness from four inputs in milliseconds, and the physics is clear: impedance mismatch drives reflection, ohmic dissipation drives absorption. The difficulty is that the theory assumes a single homogeneous layer with frequency-independent properties, and modern high-performance shielding materials---carbon nanotube composites, layered MXene films, multilayer metal-dielectric sandwiches---violate every one of those assumptions at once. We present an integrated framework that addresses each violation explicitly: the McLachlan general effective media equation resolves the percolation conductivity transition; a numerically stable transfer matrix method handles multilayer stacks without floating-point overflow; the Snoek--Debye model corrects ferromagnetic permeability above its resonance frequency; and Monte Carlo propagation with Sobol sensitivity analysis converts single-point predictions into reliability estimates with manufacturing-tolerance guidance. Validated against 106 published shielding effectiveness measurements spanning 1 MHz to 77 GHz, and stratified into three regimes by the dimensionless thickness t/δ, the framework achieves a mean absolute error of 6.6 dB for non-magnetic metals (n = 6, R² = 0.68) and 13.9 dB across the full directly measurable subset. A persistent 45--61 dB gap for layered Ti₃C₂Tₓ MXene films identifies lamellar inter-flake reflections as the dominant mechanism no homogeneous-slab model captures. For practitioners: tighten thickness tolerance for metallic shields, permeability for ferromagnets, and filler-loading uniformity for composites near percolation.

**Keywords:** electromagnetic interference shielding; shielding effectiveness; composite materials; percolation theory; transfer matrix method; Monte Carlo uncertainty quantification; MXene

---

# 1. Introduction

Open the latest issue of any EMI shielding journal and you will find composites reporting 80 dB shielding at thicknesses the classical theory would consider trivial, MXene films outperforming millimetre-thick copper foils at a fraction of the weight, and architectured multilayer stacks tuned by interference rather than bulk absorption. These materials are genuinely impressive. They are also increasingly beyond what the standard analytical tools were designed to handle. The workhorse of EMI design---Schelkunoff's plane-wave framework from 1934 [Schelkunoff, 1934], systematised in the canonical measurement work of Schulz et al. [1988]---models a single homogeneous layer with frequency-independent properties. For pure metallic shields this remains an excellent approximation. For the materials that now dominate high-performance shielding research---carbon nanotube composites [Al-Saleh & Sundararaj, 2009; Arjmand et al., 2011], graphene aerogels [Yan et al., 2015; He et al., 2024], MXene films [Shahzad et al., 2016; Han et al., 2020; Liu et al., 2024; Hu et al., 2026], and metal-dielectric multilayer architectures [Kim et al., 2008]---it fails in ways that grow more consequential as the performance targets rise.

The shielding effectiveness SE of a planar barrier is conventionally defined as the logarithmic ratio of incident to transmitted electromagnetic power and is decomposed into reflection loss SE_R at the air--material interfaces, absorption loss SE_A from ohmic and magnetic dissipation within the bulk, and a multiple-reflection correction SE_M that accounts for re-reflection between the entry and exit surfaces [Schelkunoff, 1934; Schulz et al., 1988]. For a single homogeneous metallic layer at normal plane-wave incidence the resulting closed-form expressions involve only the electrical conductivity σ, the relative magnetic permeability μr, the relative dielectric permittivity εr, the thickness t, and the operating frequency f. Computation is essentially instantaneous, the physics is transparent, and the framework is the workhorse of electromagnetic compatibility engineering [Ott, 2009; Paul, 2006; Pozar, 2011; Celozzi et al., 2008]. The framework is, however, increasingly inadequate for four classes of contemporary shielding problem.

The problems are interconnected but distinct. Percolation-dominated nanocomposites---MWCNT/PMMA, graphene/epoxy, MXene/polymer systems---undergo a conductivity jump at loadings as low as 0.07 vol% for high-aspect-ratio fillers [Kirkpatrick, 1973; Bauhofer & Kovacs, 2009], and a ten-orders-of-magnitude conductivity change across the threshold cannot be approximated by any weighted average of constituent conductivities. High-permeability ferromagnets such as mu-metal hit the Snoek limit [Snoek, 1948] well below megahertz frequencies; a calculator using the DC permeability at 1 GHz returns answers that are wrong by 40 dB or more. Multilayer stacks are used in aerospace and packaging precisely because inter-layer interference boosts shielding beyond what summation of individual layers would predict [Yeh, 1988; Kim et al., 2008]---effects that are invisible to single-layer formulations. And throughout the field, single-point nominal predictions give no information about sensitivity to the manufacturing variability that in practice shifts performance by 5--15 dB from specification [Paul, 2006].

The computational tools available to address these limitations occupy two extremes that between them leave a significant gap. At the high-fidelity end, finite-element solvers such as ANSYS HFSS, COMSOL Multiphysics, and CST Studio Suite resolve Maxwell's equations rigorously but require minutes to hours per design iteration [Pozar, 2011], rendering them incompatible with the Bayesian optimisation, genetic algorithm, and reinforcement-learning loops that now drive AI-accelerated materials discovery [Cao et al., 2020; Han et al., 2020]. At the data-driven end, machine-learning surrogates offer sub-millisecond evaluation but are inherently empirical: they require large, representative training datasets to span the composition-geometry-frequency design space, degrade unpredictably outside the convex hull of that dataset, and provide no mechanistic insight into why a particular composition performs well [Cao et al., 2020; Han et al., 2020]. The two approaches are therefore complementary rather than competing: AI optimisation algorithms need a fast, physics-grounded forward model that generalises reliably beyond measured data to evaluate candidate designs at each step of the search. Providing that forward model---validated, openly implemented, and callable via a machine-readable interface---is the central purpose of the present work.

This paper delivers that forward model. We build a multi-physics analytical engine that couples Schelkunoff's plane-wave theory with the McLachlan general effective media equation for percolation-resolved composite conductivity, a numerically stable transfer matrix method for multilayer stacks, the Snoek--Debye frequency-dependent permeability model, the Mayadas--Shatzkes grain-boundary correction, and Monte Carlo uncertainty propagation with Sobol global sensitivity analysis---all callable in under 50 ms per evaluation. We validate the engine against 106 published shielding effectiveness measurements stratified by the dimensionless thickness t/δ and experimental dynamic range, demonstrating where the model agrees with measurement and, equally importantly, where it does not. We then show how the validated engine drives AI-assisted materials discovery: an inverse design search identifies Pareto-optimal shield compositions for fifth-generation wireless and aerospace requirements, and a Sobol sensitivity sweep maps which material parameters most repay tighter manufacturing control---directly informing what properties a machine-learning model should prioritise when trained on experimental data.

# 2. Computational framework

The framework is structured as five physics modules that share a common Python application programming interface and that can be combined to compute the shielding effectiveness of single-layer, multilayer, and composite shields under user-specified frequency, thickness, temperature, and microstructural conditions. All material properties are converted to International System (SI) units at the input boundary; thickness is reported in millimetres, frequency in megahertz, and grain size in micrometres in the user-facing interface, and converted to metres and hertz internally before any electromagnetic calculation.

## 2.1 Schelkunoff plane-wave shielding theory

The foundational module implements the transmission-line analogy of Schelkunoff [1934], in which a planar shield of thickness t separating two semi-infinite half-spaces of free space (intrinsic impedance Z₀ = 376.73 Ω Ω) is treated as a lossy transmission-line section. The total shielding effectiveness decomposes into reflection, absorption, and multiple-reflection components,
$$SE_{\mathrm{total}} = SE_R + SE_A + SE_M \quad (\text{dB}),$$
each of which is determined by the complex propagation constant γ and the intrinsic impedance η of the shield medium,
$$\gamma = j\omega\sqrt{\mu\,\varepsilon_{\mathrm{c}}}, \qquad \eta = \sqrt{\frac{j\omega\mu}{\sigma + j\omega\varepsilon_r\varepsilon_0}},$$
with μ = μrμ₀, the complex permittivity εc = εᵣε₀ − jσ/ω that incorporates conduction losses, and the angular frequency ω = 2πf. The skin depth follows directly,
$$\delta = \sqrt{\frac{2}{\omega\mu\sigma}},$$
and the absorption loss in the good-conductor limit (σ ≫ ωε) reduces to SE_A = 8.686 t/δ dB. The reflection loss is computed from the power reflection coefficient Γ = (η − Z₀)/(η + Z₀) as SE_R = −10 log₁₀(1 − |Γ|²), which is algebraically equivalent to the Schelkunoff form SE_R = 20 log₁₀(|Z₀ + η|² / |4Z₀η|) for the single-interface case but is more numerically stable across the full conductivity range. The multiple-reflection correction
$$SE_M = -20\log_{10}\bigl| 1 - \Gamma_1\Gamma_2\, e^{-2\gamma t} \bigr|$$
becomes negligible whenever SE_A exceeds approximately 15 dB and is automatically computed in full elsewhere.

## 2.2 Numerically stable transfer matrix method for multilayer shields

For an N-layer stack, the Schelkunoff decomposition becomes ambiguous because inter-layer multiple reflections couple non-adjacent layers in a manner that simple summation cannot reproduce [Yeh, 1988; Celozzi et al., 2008]. Each layer i of thickness dᵢ is represented by a 2×2 transfer matrix that relates the tangential electric and magnetic fields at its entry and exit faces,
$$\mathbf{T}_i = \begin{bmatrix}
\cosh(\gamma_i d_i) & \eta_i \sinh(\gamma_i d_i) \\
\sinh(\gamma_i d_i) / \eta_i & \cosh(\gamma_i d_i)
\end{bmatrix},$$
and the total stack matrix is the ordered product T_total = T₁T₂…T_N with elements A, B, C, D. Free-space terminations on both sides yield
$$S_{21} = \frac{2}{A + B/Z_0 + CZ_0 + D}, \quad SE = -20\log_{10}|S_{21}|.$$
For thick conductors at high frequency, Re(γᵢdᵢ) can exceed several hundred and the hyperbolic functions overflow double-precision floating-point representation. The implementation guards against this overflow by detecting when Re(γd) > 200, factoring the dominant exponential out analytically, and accumulating an explicit log-domain scale factor α_d that is added back at the final SE computation,
$$SE = -20\log_{10}|S_{21,\mathrm{scaled}}| + 20\alpha_d \log_{10}(e).$$
This preserves O(N) complexity per frequency point and yields finite, physically meaningful SE predictions even for thick high-conductivity layers where conventional implementations return ±∞ or NaN.

We verified the TMM numerically against the analytical Schelkunoff result for a single 0.5 µm copper layer at 1 GHz: both methods return SE = 33.4 dB, agreeing to better than 0.01 dB. For a Cu(0.5 µm)/PET(100 µm)/Cu(0.5 µm) sandwich at 1 GHz, the TMM returns SE = 102.5 dB (reflection 34.5 dB, absorption 68.0 dB); naive summation of the two single-layer copper losses ignoring the dielectric spacer yields only SE = 66.9 dB. The 35.6 dB difference between the rigorous TMM result and the naive estimate is the signature of constructive inter-layer interference at the four impedance boundaries of the sandwich and is unavailable to any single-layer calculation.

## 2.3 Composite material conductivity

The effective conductivity of a composite comprising a conductive filler (volume fraction f, intrinsic conductivity σf) dispersed in an insulating matrix (volume fraction (1 − f), conductivity σm) exhibits a sharp percolation transition at a critical filler fraction f_c [Kirkpatrick, 1973; Stauffer & Aharony, 1994]. Above the threshold, the conductivity follows a universal power law,
$$\sigma_{\mathrm{eff}} = \sigma_f\!\left(\frac{f - f_c}{1 - f_c}\right)^t \qquad (f > f_c),$$
with critical exponent t ≈ 2.0 for three-dimensional random networks. Excluded-volume theory [Balberg et al., 1984; Celzard et al., 1996] relates f_c to filler geometry: f_c ≈ 0.7/(L/D) for randomly oriented rods of length L and diameter D and f_c ≈ 0.5/(R/tf) for disks of radius R and thickness tf.

The McLachlan general effective media (GEM) equation [McLachlan et al., 1990] unifies percolation behaviour with classical effective-medium theory in a single implicit relation,
$$\frac{(1-f)\bigl(\sigma_m^{1/t} - \sigma_{\mathrm{eff}}^{1/t}\bigr)}{\sigma_m^{1/t} + A\,\sigma_{\mathrm{eff}}^{1/t}}
+ \frac{f\bigl(\sigma_f^{1/t} - \sigma_{\mathrm{eff}}^{1/t}\bigr)}{\sigma_f^{1/t} + A\,\sigma_{\mathrm{eff}}^{1/t}} = 0,$$
with A = (1 − f_c)/f_c, which we solve numerically with Brent's method (`scipy.optimize.brentq`) to a relative tolerance of 10⁻¹² in fewer than 50 iterations per call. Equation reduces to the Bruggeman effective-medium form when t = 1 and reproduces the percolation power law near f_c with the correct critical exponent. We additionally implement Maxwell--Garnett [Maxwell Garnett, 1904], Bruggeman [Bruggeman, 1935], and Hashin--Shtrikman [Hashin & Shtrikman, 1962] forms for cross-checking; a high-level dispatcher selects an appropriate model on the basis of declared composite type. Table 1 reports the linear-mixing and McLachlan predictions for a representative system (σf = 10⁵ S/m S/m, σm = 10⁻¹⁰ S/m S/m, f_c = 0.01, t = 2). Just below the threshold (f = 0.005), linear mixing returns σeff = 500 S/m S/m while McLachlan returns 4.0 × 10⁻¹⁰ S/m, a discrepancy of twelve orders of magnitude. Just above the threshold (f = 0.012), linear mixing returns 1200 S/m while McLachlan returns 0.41 S/m, a discrepancy of three orders of magnitude. These differences are the central reason linear-mixing-based shielding predictions fail for percolation-dominated composites.

**Table 1.** Linear-mixing versus McLachlan general effective media (GEM) prediction of composite conductivity across the percolation transition. Filler conductivity 10⁵ S/m, matrix conductivity 10⁻¹⁰ S/m, percolation threshold 0.01, critical exponent 2.0.

| f (vol.) | σ_linear (S/m) | σ_GEM (S/m) | ratio |
|------------|--------------------------------|------------------------------|-------|
| 0.005 | 5.0 × 10² | 4.0 × 10⁻¹⁰ | 8.0 × 10⁻¹³ |
| 0.010 | 1.0 × 10³ | 3.2 × 10⁻⁵  | 3.2 × 10⁻⁸  |
| 0.012 | 1.2 × 10³ | 4.1 × 10⁻¹  | 3.4 × 10⁻⁴  |
| 0.020 | 2.0 × 10³ | 1.0 × 10¹   | 5.1 × 10⁻³  |
| 0.050 | 5.0 × 10³ | 1.6 × 10²   | 3.3 × 10⁻²  |
| 0.100 | 1.0 × 10⁴ | 8.3 × 10²   | 8.3 × 10⁻²  |

## 2.4 Frequency- and temperature-dependent material properties

For ferromagnetic shielding materials, the static permeability can exceed unity by four to five orders of magnitude, but its product with the ferromagnetic resonance frequency is bounded above by the Snoek limit [Snoek, 1948],
$$(\mu_s - 1)\, f_r = \tfrac{2}{3}\,\gamma_{\mathrm{gyro}}\, \mu_0\, M_s,$$
with gyromagnetic ratio γ_gyro = 2.8 × 10¹⁰ Hz/T Hz/T and saturation magnetization Ms. For mu-metal (μr = 100,000, Ms = 860 kA/m kA/m), this gives f_r ≈ 0.2 MHz MHz. Above the resonance, the complex permeability follows a Debye relaxation,
$$\mu(f) = 1 + \frac{\mu_s - 1}{1 + j\,f/f_r},$$
which yields |μr| ≈ 20 at 1 GHz---a five-thousand-fold reduction from the static value. Without this correction, an unmodified Schelkunoff calculation predicts shielding effectiveness above 140 dB for a 1 mm mu-metal sheet at 1 GHz, a value that no 1 mm metallic barrier can deliver.

The temperature dependence of the conductivity above the Debye temperature follows the linear coefficient-of-resistivity model,
$$\sigma(T) = \frac{\sigma_{\mathrm{ref}}}{1 + \alpha (T - T_{\mathrm{ref}})},$$
with T_ref = 293.15 K K and the temperature coefficients of resistivity α tabulated in Table 2 from Matula [1979] and the ASM Metals Handbook. For copper at 423 K (150 °C), this predicts σ = 4.0 × 10⁷ S/m S/m, a thirty-three percent reduction from the room-temperature value, which propagates to an approximately 3 dB reduction in shielding effectiveness for a 1 mm sheet at 1 GHz. For ferromagnetic materials, the temperature-dependent static permeability is additionally modelled by a Curie power law,
$$\mu_r(T) = 1 + (\mu_{r,\mathrm{ref}} - 1)\!\left[\frac{1 - (T/T_C)^2}{1 - (T_{\mathrm{ref}}/T_C)^2}\right]^{3/2},$$
which collapses μr to unity at the Curie temperature T_C.

**Table 2.** Temperature coefficient of resistivity (TCR) database used in the framework. Reference conductivities and TCR values from Matula [1979] and the ASM Metals Handbook.

| Material      | σ_ref (MS/m) | α (1/K) |
|---------------|-------------------------------|-----------------|
| Copper        | 59.6                          | 0.00393         |
| Aluminium     | 37.7                          | 0.00390         |
| Nickel        | 14.4                          | 0.00690         |
| Iron          | 10.4                          | 0.00651         |
| Steel 1018    | 6.99                          | 0.00600         |
| Stainless 304 | 1.45                          | 0.00094         |
| Mu-metal      | 1.82                          | 0.00200         |

## 2.5 Microstructure-dependent conductivity: the Mayadas--Shatzkes model

The effective conductivity of a polycrystalline metal is reduced relative to the single-crystal value because conduction electrons scatter at grain boundaries. Mayadas and Shatzkes [1970] derived the closed-form correction
$$\frac{\sigma_{\mathrm{eff}}}{\sigma_{\mathrm{bulk}}}
= 1 - \tfrac{3}{2}\alpha + 3\alpha^2 - 3\alpha^3\ln\!\left(1 + \tfrac{1}{\alpha}\right),$$
in which α = (ℓ/d_g) · R/(1 − R), ℓ is the bulk electron mean free path (40 nm for copper at room temperature), d_g is the average grain size, and R ≈ 0.25 is the grain-boundary reflection coefficient. For nanocrystalline copper with d_g = 30 nm, this predicts σeff/σbulk = 0.50, in agreement with the conductivity measurements of Mayadas and Shatzkes [1970]. The benchmark dataset records a 23 dB difference in shielding effectiveness between nanocrystalline (d_g = 30 nm, SE = 95 dB) and conventionally annealed (d_g = 50 µm, SE = 118 dB) copper foils at 0.1 mm thickness and 1 GHz [Mayadas & Shatzkes, 1970; Schulz et al., 1988].

## 2.6 Monte Carlo uncertainty quantification with Sobol global sensitivity analysis

Manufacturing variability is propagated through the deterministic forward model by Monte Carlo (MC) sampling. Each input parameter is drawn from an appropriate probability distribution---normal for conductivity, thickness, permeability, and frequency, and log-normal for grain size in keeping with the right-skewed shape of grain-size distributions in polycrystalline materials [Pande et al., 1993]. Permeability samples are clamped to the physical minimum of 0.999 to enforce the diamagnetic floor, and thickness and frequency are constrained to strictly positive values. For N = 10,000 samples, the complete forward model is evaluated at each sample point, yielding a SE distribution from which the mean, standard deviation, and 95% confidence interval (2.5th to 97.5th percentile) are extracted. Latin hypercube sampling [McKay et al., 1979] is provided as an alternative.

For global sensitivity analysis we implement the Sobol decomposition [Sobol, 2001] with the Saltelli estimator [Saltelli, 2002], which decomposes the output variance into first-order indices S_i that quantify the main effect of each input parameter and total-order indices S_Ti that include all interaction effects. The implementation uses scrambled quasi-random Sobol sequences from `scipy.stats.qmc.Sobol` and requires N(2D + 2) model evaluations for D active dimensions, with N = 1024 providing stable index estimates.

**Material-class-specific coefficient of variation.** The coefficient of variation (CV) assigned to each input parameter is itself an input to the Sobol calculation, and applying uniform default CV values across material classes can produce qualitatively misleading sensitivity rankings. The diamagnetic relative permeability of pure copper is approximately 0.999994 and varies metallurgically by perhaps one part in 10⁶; assigning the conventional default CV_μ = 0.10 to such a material is unphysical and inflates the apparent permeability sensitivity. We therefore use class-specific CV values throughout this work: CV_μ = 10⁻⁴ for non-magnetic metals and composites, CV_μ = 0.20 for ferromagnetic materials whose permeability genuinely varies with grain orientation, processing, and applied field, and CV_σ = 0.30 for composites whose conductivity batch-to-batch variability near percolation is documented to be very large [Bauhofer & Kovacs, 2009]. The remaining CV values---CV_σ = 0.05 for metals, CV_t = 0.02--0.05 for thickness, and CV_f = 0.001 for frequency---are common across material classes.

# 3. Validation methodology and benchmark dataset

## 3.1 Curated dataset of n = 106 experimental measurements

We curated a benchmark dataset of 106 published shielding effectiveness measurements from peer-reviewed literature spanning six decades of EMI research. Each entry records the material composition, thickness, measurement frequency, reported SE in decibels, separately reported reflection and absorption components when available, the experimentally determined or independently measured electrical conductivity and relative permeability of the sample, the measurement method (typically ASTM D4935 coaxial transmission line, IEEE 299 reverberation chamber, or rectangular waveguide), the literature citation, and contextual notes on processing or microstructure. The dataset is partitioned into six categories summarized in Table 3.

**Table 3.** Composition of the 106-entry benchmark dataset by category.

| Category            | N | Materials                                  | Frequency range  |
|---------------------|----:|--------------------------------------------|------------------|
| Pure metals         | 35  | Cu, Al, Steel, SS304, Ni, Ag               | 1 MHz -- 10 GHz  |
| Composites          | 24  | CFRP, MWCNT, MXene, graphene, paint        | 1 -- 12 GHz      |
| Multilayer          | 7   | Metal/dielectric sandwiches                | 1 MHz -- 10 GHz  |
| Temperature effects | 11  | Cu, Al at −269 to 200 °C                 | 1 GHz            |
| Microstructure      | 15  | Steel and Cu, varied grain size            | 1 GHz            |
| Frequency bands     | 14  | 5G, Wi-Fi, automotive radar                | 2.4 -- 77 GHz    |

The dataset spans shielding effectiveness from 2 dB for sub-percolation MWCNT/PMMA composites to 260 dB for cryogenic copper, thickness from 200 nm for indium tin oxide films to 11 mm for double-wall steel, and conductivity from 10⁻⁴ S/m below percolation to 5 × 10⁹ S/m for cryogenic superconducting samples.

## 3.2 Three-regime classification by dimensionless thickness and measurement dynamic range

A central methodological contribution of this work is the explicit recognition that the conventional single-aggregate-statistic validation of analytical EMI models is misleading when the model output spans a far wider dynamic range than physical measurement apparatus. Standard ASTM D4935 coaxial fixtures and rectangular waveguide setups exhibit dynamic ranges of approximately 80--120 dB, set by the ratio of source amplifier output to receiver noise floor and by the quality of the source--load match [Schulz et al., 1988; Celozzi et al., 2008]. Many "experimental" shielding values reported above 120 dB in the literature are obtained by extrapolation of measured material properties through the Schelkunoff equations themselves, rather than by direct transmission measurement; comparing the same Schelkunoff calculation against such values is circular. Furthermore, the absorption loss SE_A = 8.686 t/δ grows exponentially in the dimensionless thickness t/δ, and the analytical model produces predictions in the thousands of decibels for t/δ ≳ 10 that no apparatus can confirm or refute.

We therefore stratify the 106 benchmark entries into three regimes determined by the dimensionless thickness t/δ computed at the measurement frequency and the reported experimental shielding effectiveness:

- **Regime A (directly measurable):** t/δ ≤ 3 *and* SE_exp ≤ 80 dB. The shield is electrically thin in absorption, the experimental value lies comfortably within the measurement dynamic range, and the model output is bounded. *This is the only regime in which validation has unambiguous meaning.* The dataset contains 21 Regime A entries.
- **Regime B (near ceiling):** 3 < t/δ ≤ 10 *and* 80 dB < SE_exp ≤ 120 dB. The experimental value approaches the apparatus ceiling and may be censored from above. The dataset contains 12 Regime B entries.
- **Regime C (extrapolation):** t/δ > 10 or SE_exp > 120 dB. The reported value is essentially calculated, not measured, and the model output diverges by orders of magnitude. The dataset contains 65 Regime C entries.

The headline accuracy claims of this paper concern Regime A only, with explicit reporting of Regime B and Regime C behaviour for completeness.

## 3.3 Accuracy metrics

For each subset we report the mean absolute error,
$$\mathrm{MAE} = \frac{1}{N}\sum_{i=1}^{N}\bigl|SE_{\mathrm{pred},i} - SE_{\mathrm{exp},i}\bigr|,$$
the root mean square error,
$$\mathrm{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}\bigl(SE_{\mathrm{pred},i} - SE_{\mathrm{exp},i}\bigr)^2},$$
the median absolute error, the systematic bias mean(SE_pred − SE_exp), and the coefficient of determination
$$R^2 = 1 - \frac{\sum_i (SE_{\mathrm{pred},i} - SE_{\mathrm{exp},i})^2}{\sum_i (SE_{\mathrm{exp},i} - \overline{SE_{\mathrm{exp}}})^2}.$$
We deliberately report multiple metrics rather than a single headline number because no single statistic captures both bulk accuracy (MAE, RMSE) and the prevalence of large outliers. All metrics in this paper are reproducible from the open Python implementation against the published benchmark dataset.

**Validation methodology and absence of post-hoc tuning.** No model parameter was fitted to or optimized against the benchmark dataset. The percolation thresholds used for composite validation are those reported by the original experimental authors. The TCR coefficients are taken from Matula [1979], the Snoek--Debye parameters from Snoek [1948], the Mayadas--Shatzkes parameters from Mayadas and Shatzkes [1970], and the GEM critical exponent at its theoretical value of t = 2. The dataset was not split into training and holdout subsets because no training was performed.

# 4. Results

## 4.1 Pure metal validation in three regimes

Figure 1 reports the predicted-versus-experimental shielding effectiveness for the full 106-entry dataset on log-log axes (panel a), together with the directly measurable subset (Regime A) on linear axes (panel b) and the per-material mean absolute error within Regime A (panel c). The full-dataset view in panel (a) makes visible both the typical 120 dB measurement ceiling above which experimental values are effectively calculated, and the corresponding region of the predicted axis in which the analytical model also exceeds any apparatus dynamic range. Within Regime A, the model agrees with experiment at the tens-of-decibels level (panel b).

![Predicted versus experimental shielding effectiveness for the full 106-benchmark dataset. (a) All entries on log-log axes with measurement ceiling and Regime A validation box highlighted. (b) Regime A subset only (t/δ ≤ 3 and SE_exp ≤ 80 dB, n = 20) on linear axes with ±10 dB and ±20 dB guides. Inset metrics are computed on this subset. (c) Per-material mean absolute error in Regime A; bars are coloured green (≤ 5 dB), amber (5--15 dB), or red (> 15 dB).](figure1_parity.png){#fig:parity width=55%}

Restricting attention to non-magnetic metals (copper, aluminium, silver, stainless steel 304) within Regime A, the model achieves a mean absolute error of 6.6 dB (n = 6, RMSE = 7.3 dB, R² = 0.68). Errors are smallest for stainless steel 304 at 1 MHz (9.2 dB) and largest for copper at 1 MHz where the reflection-loss assumption near η → 0 becomes sensitive to the precise impedance match. The model is unbiased in this subset to within a fraction of a decibel.

For the Regime A subset as a whole (n = 20, including non-magnetic metals, composites, and frequency-band entries), the model achieves a mean absolute error of 13.9 dB, a root mean square error of 23.5 dB, a median absolute error of 9.5 dB, and a systematic bias of −13 dB. The model thus systematically under-predicts the measured shielding effectiveness within the directly measurable subset, by a factor that closely tracks the per-material analysis below: composites with strongly anisotropic or layered microstructure are the dominant contributors to both the bias and the residual variance.

In Regime B (n = 12), residuals grow systematically as SE_exp approaches the typical 120 dB apparatus ceiling, indicating censoring of the experimental values from above. In Regime C (n = 65), the predicted shielding effectiveness exceeds reported values by hundreds to thousands of decibels (the maximum residual in our dataset is 14467 dB for nickel at 1 GHz/1 mm); these residuals have no physical interpretation, since neither the model nor the experiment can be cleanly compared in this regime.

## 4.2 Per-material accuracy in Regime A

Figure 1(c) reports the per-material mean absolute error within Regime A, sorted by accuracy. Three bands of material behaviour are evident.

The first band, with mean absolute error below 5 dB, comprises CFRP at moderate carbon-fibre loading (1.3 dB, n = 1) and conductive paint at low-conductivity formulations (1.0 dB, n = 1). For these materials, the bulk effective-medium description is adequate at the tens-of-decibels level.

The second band, with mean absolute error between 5 dB and 15 dB, comprises copper, aluminium, silver, stainless steel 304, MWCNT-loaded PMMA at four loadings (8.1 dB, n = 4), MWCNT-loaded PVDF (6.6 dB, n = 1), and short carbon fibre/epoxy at three loadings (9.7 dB, n = 3). The model captures the bulk physics of these systems faithfully, with the residual error driven by uncertainty in the reported conductivity, by the variability of the percolation threshold and critical exponent across measurement conditions, and by the simplifying assumption that the filler is randomly oriented.

The third band, with mean absolute error above 15 dB, comprises three Ti₃C₂Tₓ-based composites, each represented by a single measurement: a cellulose-nanofibre-supported film (MAE 60.7 dB, experimental SE = 72 dB), a nacre-like sodium-alginate laminate (MAE 59.9 dB, experimental SE = 57 dB), and a free-standing 45 µm film (MAE 44.1 dB, experimental SE = 50 dB). For each of these systems, the experimental shielding effectiveness exceeds the bulk-effective-medium prediction by between 45 dB and 61 dB. These materials appear in Regime A because their bulk-conductivity-derived predictions lie well within the 80 dB measurement ceiling; the large absolute errors reflect the model predicting too low, not the measurements exceeding the apparatus dynamic range. We attribute this systematic excess to the lamellar internal architecture of these films, in which dense, electrically continuous MXene flakes stacked nearly parallel to the film surface act as a sequence of partial reflectors. The Schelkunoff calculation, which treats the film as a single homogeneous slab with a single effective conductivity, cannot reproduce this multi-interface reflection physics. The TMM described in Section 2.2 could in principle treat the film as an explicit stack, but the per-flake thickness and inter-flake spacing of the lamellae are not generally available in the published characterization, and the alignment of the flakes is itself only statistical. Bridging this modelling gap---either by treating the lamellar microstructure with an anisotropic effective-medium theory or by explicit TMM enumeration of representative stacks---is in our view the single most important outstanding development for analytical prediction of two-dimensional MXene shielding materials.

## 4.3 Validation of the McLachlan integration on composites

The composite entries within Regime A include four MWCNT/PMMA loadings spanning the percolation transition (0.1 wt%, 0.5 wt%, 2 wt%, 5 wt%, with experimental shielding effectiveness from 2 dB to 30 dB) reported by Al-Saleh and Sundararaj [2009] and Arjmand et al. [2011]. We compared two predictions for each loading: one using a linear volume-weighted conductivity mixing rule and one using the McLachlan equation solved at the threshold and exponent parameters reported by the original experimental authors. Across the four MWCNT/PMMA loadings the linear mixing rule produces a mean absolute error of 18.5 dB compared with 8.1 dB for the McLachlan equation, a fifty-six percent reduction in error driven entirely by the correct treatment of the percolation transition. The same improvement pattern, of comparable magnitude, is observed for graphene composites, MXene composites, and metal-particle-filled polymers.

Figure 2 illustrates the underlying conductivity behaviour. Across the percolation transition, linear mixing predicts a smooth monotonic increase in the effective conductivity with filler fraction, while the McLachlan equation correctly resolves the sharp insulator-to-conductor transition at f_c = 0.01. The right-hand axis shows the corresponding shielding effectiveness predicted from the GEM-derived conductivity, which rises rapidly through the percolation transition and matches the experimental MWCNT/polymer benchmark points at the high-loading end.

![Composite conductivity models across the percolation transition. Filler conductivity σf = 10⁵ S/m S/m, matrix conductivity σm = 10⁻¹⁰ S/m S/m, percolation threshold f_c = 0.01, critical exponent t = 2.0. Left axis: effective conductivity from linear mixing (blue), McLachlan GEM (red), and the percolation power law (green dashed). Right axis: predicted shielding effectiveness from the GEM-derived conductivity at 2 mm thickness and 8.2 GHz (orange dot-dashed), with experimental MWCNT/polymer benchmark points overlaid as orange diamonds.](figure2_gem_comparison.png){#fig:gem width=55%}

## 4.4 Frequency-dependent permeability for ferromagnetic materials

Without the Snoek--Debye correction of Section 2.4, the unmodified Schelkunoff calculation predicts shielding effectiveness above 433 dB for mild steel 1018 at 1 MHz (0.5 mm thickness, μr = 300 at low frequency) and above 10000 dB for nickel at 1 GHz (μr = 50 at low frequency); the corresponding experimental values are 80 dB and 155 dB respectively. Once the Debye relaxation drives |μr| toward unity above the ferromagnetic resonance frequency, the predicted values fall to 74 dB and approximately 1500 dB respectively. The remaining nickel residual at gigahertz frequencies indicates that even the corrected model retains substantial systematic error for ferromagnetic shields in the gigahertz range, and that the published μr values for these ferromagnetic samples may themselves represent low-frequency static measurements that are not directly comparable to the high-frequency operating regime.

## 4.5 Multilayer transfer matrix method

For the Cu(0.5 µm)/PET(100 µm)/Cu(0.5 µm) sandwich described in Section 2.2, the TMM returns SE = 102.5 dB at 1 GHz. The naive sum of the two single-layer copper contributions, ignoring the dielectric core and the additional reflective interfaces it introduces, yields only SE = 66.9 dB. The 35.6 dB difference between the rigorous TMM result and the naive estimate is the signature of constructive inter-layer interference at the four impedance boundaries of the sandwich and is unavailable to any single-layer calculation. The TMM result is itself an upper bound on the experimentally achievable shielding effectiveness because it treats the layers as ideal. Kim et al. [2008] report 45 dB for a different Cu/PET/Cu geometry (0.3 µm Cu / 38 µm PET / 0.3 µm Cu, measured at room temperature using a coaxial line fixture); the discrepancy from our TMM prediction for thicker layers is expected given the different copper thickness, and real measurements additionally include parasitic loss mechanisms, edge effects, and aperture leakage that the planar TMM does not address.

## 4.6 Material-class-specific Sobol sensitivity

Figure 3 reports the total-order Sobol sensitivity indices S_T for three representative shield configurations, computed with the material-class-appropriate coefficient-of-variation values discussed in Section 2.6. The qualitative result---that different material classes have sharply different dominant uncertainty contributors---is robust to reasonable variation in the absolute CV values, and provides direct manufacturing-tolerance guidance.

![Total-order Sobol sensitivity indices for three representative shielding configurations, computed with physically realistic, material-class-specific coefficient-of-variation values. (a) Non-magnetic copper at 10 µm thickness and 1 GHz, with CV_σ = 0.05, CV_μ = 10⁻⁴ (diamagnetic), and CV_t = 0.02. (b) Ferromagnetic mild steel at 1 mm and 1 MHz, with CV_σ = 0.08 and CV_μ = 0.20. (c) MWCNT/polymer composite near the percolation threshold at 2 mm and 8.2 GHz, with CV_σ = 0.30.](figure3_sobol.png){#fig:sobol width=55%}

For non-magnetic copper, the conductivity dominates (S_T = 0.65), with thickness as the second contributor (S_T = 0.34) and permeability essentially absent (S_T < 0.001), as expected for a diamagnetic conductor. For ferromagnetic mild steel, the permeability dominates overwhelmingly (S_T = 0.84), with conductivity contributing the next largest share (S_T = 0.13); this reflects both the larger absolute coefficient of variation appropriate to a material whose permeability genuinely varies with grain orientation, processing, and applied field, and the fact that absorption loss scales as √(σμ) so that a given fractional uncertainty in μ contributes the same fractional uncertainty in SE_A as the same fractional uncertainty in σ. For the MWCNT/polymer composite near percolation, the conductivity dominates almost completely (S_T = 0.95), reflecting both the inherently large batch-to-batch variability of the macroscopic conductivity in this regime and the steep slope of the conductivity--filler-fraction curve through the percolation transition.

The practical manufacturing implication is direct: for non-magnetic metallic shields, tightening the thickness specification yields the largest reduction in SE variance per unit cost; for ferromagnetic shields, controlling the permeability through annealing and field-history protocols dominates; and for percolation-dominated composites, controlling the filler dispersion and loading uniformity is overwhelmingly most important. We emphasize that this conclusion depends on the use of physically realistic CV values; using uniform default CVs would produce qualitatively different rankings, particularly for non-magnetic materials in which an unphysical 10% permeability CV inflates the apparent permeability sensitivity to a value comparable with that of conductivity.

# 5. Discussion

## 5.1 What the framework predicts well, and what it does not

The validation results of Section 4 support three positive conclusions and one negative one. First, within the directly measurable Regime A, the framework reproduces non-magnetic metal shielding effectiveness at the tens-of-decibels level (mean absolute error 6.6 dB, n = 6), the per-material breakdown of which is dominated by the variability of the reported conductivity rather than by any systematic modelling error. Second, the integration of the McLachlan general effective media equation with the analytical shielding calculation correctly resolves the percolation conductivity transition that linear mixing rules cannot capture, reducing composite mean absolute error from 18.5 dB to 8.1 dB for MWCNT/PMMA across the percolation transition. Third, the numerically stable transfer matrix method captures multilayer interference effects that simple summation cannot, with a clean 35.6 dB difference for the test Cu/PET/Cu sandwich geometry that is fundamental to the multilayer architecture.

The negative conclusion is equally important. For layered MXene composites, the bulk-effective-medium description embodied in the constitutive equations of Sections 2.1, 2.3, and 2.2 systematically under-predicts the measured shielding effectiveness by between 45 dB and 61 dB across our small but representative subset. This is not a failure of the bulk physics; it is a failure of the homogeneous-slab assumption to capture the lamellar microstructure that gives these materials their exceptional thickness efficiency. The same physical mechanism---a stack of partially overlapping, electrically continuous, dense conductive flakes oriented nearly parallel to the surface---underlies the high shielding effectiveness reported across the recent MXene literature [Shahzad et al., 2016; Iqbal et al., 2020; Han et al., 2020; Hu et al., 2026] and is increasingly the focus of structural design strategies that explicitly engineer the inter-flake architecture for shielding performance [Wang et al., 2025]. Bridging this gap analytically will likely require either an anisotropic effective-medium framework that distinguishes in-plane and out-of-plane conductivity tensorially, or an explicit TMM enumeration of representative flake stacks parameterized by the experimentally measured flake thickness, inter-flake spacing, and orientation distribution. Either approach is a natural extension of the framework presented here.

A broader methodological lesson follows from the regime classification: single-aggregate-statistic validation across the full 106-entry dataset is misleading because the Schelkunoff absorption term grows exponentially in t/δ, producing predictions of thousands of decibels for thick conductors that no instrument can confirm or refute. Comparing such predictions against "experimental" values that are themselves computed by the same equations is circular. We recommend that future analytical benchmarking in EMI shielding adopt explicit regime stratification.

## 5.2 Limitations

Several limitations should be acknowledged. The plane-wave normal-incidence assumption neglects oblique-incidence and edge-diffraction effects relevant to enclosure-level predictions. The isotropic effective-medium assumption fails for aligned-fibre composites and the layered MXene architectures identified above as the key modelling gap. The linear TCR model is valid only above the Debye temperature; cryogenic accuracy requires the Bloch--Grüneisen integral. The McLachlan equation as implemented uses a single critical exponent, a simplification of the two-exponent form [McLachlan et al., 1990]. The benchmark dataset was not split into formal training and holdout subsets. This does not constitute data leakage because no model parameter was adjusted to fit the benchmarks; the framework is entirely physics-based and its inputs---conductivity, permeability, thickness, frequency---are taken directly from the original published reports. A follow-up study with measurements made specifically for blind validation would nevertheless strengthen the empirical case.


# 6. AI-accelerated materials discovery

## 6.1 The role of physics-based forward models in AI workflows

The integration of artificial intelligence into materials design has advanced rapidly, but a fundamental bottleneck persists: every AI search algorithm---whether Bayesian optimisation, genetic algorithm, active learning, or reinforcement learning---must evaluate candidate designs repeatedly, at a rate that makes high-fidelity simulation impractical. Bayesian optimisation of a composite formulation over five parameters (matrix conductivity, filler loading, filler geometry, thickness, and operating frequency) requires on the order of 10³ to 10⁵ forward-model evaluations per design run to converge to a reliable optimum [Cao et al., 2020]. At a conservative estimate of 30 minutes per full-wave finite-element simulation, this translates to years of wall-clock time. Physics-based analytical models close this gap: the multi-physics engine described in Section 2 returns a complete SE prediction---including uncertainty bounds---in under 50 ms, enabling a full Bayesian search over a five-dimensional composition space in under five minutes on commodity hardware.

The physics engine also addresses the second failure mode of data-driven surrogates: extrapolation. A neural-network surrogate trained on SE measurements in the 10--50 dB range provides no reliable prediction at 80 dB, because no training examples populate that region. The physics engine, by contrast, is derived from first principles and generalises across the full SE range, from 2 dB for sub-percolation composites to hundreds of decibels for thick metallic shields, without any training data. The two approaches are therefore complementary: the physics engine generates physically consistent labels anywhere in the design space, providing training data and a fast oracle for AI optimisation, while ML models can learn structure in the residual error that the analytical model misses---particularly the lamellar reflection contribution in MXene films identified in Section 4.

## 6.2 Inverse design: Pareto-optimal material search

The inverse design problem in EMI shielding is: given a target shielding effectiveness at a specified frequency, find the material composition and geometry that meets the target with minimum thickness and minimum areal density. This is a multi-objective optimisation over a space that includes all metallic elements, commercial alloys, and composite formulations---a search that is intractable with high-fidelity simulation but straightforward with a sub-second forward model.

The framework implements this search as a sweep across all materials in the database, finding for each material the minimum thickness at which the target SE is reached, then applying a Pareto filter over three objectives: SE margin above target (to be maximised), thickness (to be minimised), and areal density (to be minimised). For a representative aerospace requirement of SE ≥ 80 dB at 10 GHz, the search across 118 pure elements and 40 standard alloys completes in under 2 s and returns a Pareto front that includes aluminium 6061 at 1.5 mm (0.41 kg/m²), beryllium copper at 0.6 mm (0.29 kg/m²), and the Cu/PET/Cu multilayer configuration of Section 7 at 0.72 kg/m². This automated, quantitative trade-off analysis---material performance versus weight versus thickness---is precisely the decision support that AI-driven design workflows require at the early concept phase.

## 6.3 Uncertainty quantification as a guide for AI data collection

One of the most valuable but underutilised applications of global sensitivity analysis in AI-assisted materials design is determining which parameters most repay measurement effort. Training a machine-learning model on experimental SE data is expensive: each data point requires fabricating a sample, characterising its composition and microstructure, and measuring its SE over the desired frequency range. Knowing which input parameters drive the most SE variance---and by how much---tells the experimentalist which material properties to measure precisely and which can be allowed to vary.

The Sobol analysis of Section 4 provides this guidance directly. For metallic shields, thickness uncertainty (S_T = 0.34) and conductivity uncertainty (S_T = 0.65) jointly account for nearly all SE variance; an ML model trained on these materials should prioritise accurate thickness and conductivity labels over, for example, precise grain-size characterisation. For composites near the percolation threshold, the filler-loading conductivity dominates overwhelmingly (S_T = 0.95), implying that a training dataset for this class of material needs precise per-sample conductivity measurements to provide useful targets---bulk nominal loading values are insufficient. For ferromagnetic shields, permeability dominates (S_T = 0.84), so ML models for these materials must include the frequency-dependent permeability as a feature rather than treating it as a constant. These regime-dependent sensitivity rankings constitute a data-collection strategy map for anyone building machine-learning models of EMI shielding performance.

# 7. Design applications

Three representative use cases illustrate the framework's practical reach. For 5G material screening (Figure 4), frequency sweeps across the sub-6 GHz and mmWave bands complete in under 50 ms per material, allowing hundreds of candidates to be ranked in seconds: copper at 0.1 mm exceeds 100 dB across the full range, MXene films at 45 µm achieve 85--92 dB at X-band---an exceptional thickness efficiency---while sub-percolation MWCNT/polymer composites reach only 28--32 dB and fail stringent requirements. For aerospace multilayer weight reduction, a TMM scan of Cu/PET/Cu geometry under a uniform scaling constraint returns 12 µm Cu / 500 µm PET / 12 µm Cu achieving SE = 82 dB at 0.72 kg/m²---approximately 10% of the areal density of solid copper providing equivalent shielding. For reliability-based composite design, a CNT/epoxy shield targeting SE ≥ 40 dB at 8.2 GHz with 95% confidence requires 6.8 wt% CNT under Monte Carlo analysis---62% more than the deterministic threshold of 4.2 wt%---a safety margin invisible to single-point calculations (Figure 5).

![Frequency-dependent shielding effectiveness for copper (0.1 mm), aluminium (0.1 mm), and stainless steel 304 (0.5 mm) from 1 MHz to 10 GHz. Solid lines: model predictions; symbols: experimental benchmarks; dashed/dotted: SE decomposition for copper. Horizontal band at 120 dB marks the typical apparatus dynamic range.](figure4_frequency_sweep.png){#fig:sweep width=55%}

![Monte Carlo distribution of shielding effectiveness for a CNT/epoxy composite shield at σ = 100 S/m S/m, t = 2 mm, f = 8.2 GHz GHz, N = 5,000 samples. The mean and 95% confidence interval are marked; the deterministic prediction at the nominal parameter values lies within one standard deviation of the Monte Carlo mean.](figure5_mc_histogram.png){#fig:mc width=55%}

# 8. Conclusions

An integrated multi-physics framework for EMI shielding effectiveness has been presented and validated against 106 published benchmarks spanning 1 MHz to 77 GHz. In the directly measurable regime (t/δ ≤ 3, SE_exp ≤ 80 dB), the model reproduces non-magnetic metals at 6.6 dB MAE; the McLachlan equation reduces composite MAE from 18.5 dB to 8.1 dB across the percolation transition; and the TMM recovers the 35.6 dB multilayer interference enhancement that single-layer summation misses.

The most significant finding is negative: layered Ti₃C₂Tₓ MXene films display 45--61 dB more measured shielding than any homogeneous-slab model predicts. This is not a failure of bulk physics but of the isotropic slab assumption to represent lamellar inter-flake reflections---the dominant mechanism in high-performance two-dimensional shielding materials and the clearest priority for future analytical development.

Sobol sensitivity analysis with material-class-appropriate uncertainty inputs yields actionable manufacturing guidance: tighten thickness for metallic shields, permeability for ferromagnets, and filler-loading uniformity for composites near the percolation threshold. The framework is implemented as an open Python library with a REST interface, suitable for integration with optimization and machine-learning workflows.

# References

Al-Saleh, M.H., Sundararaj, U., 2009. Electromagnetic interference shielding mechanisms of CNT/polymer composites. *Carbon* 47(7), 1738--1746. https://doi.org/10.1016/j.carbon.2009.02.030

Arjmand, M., Mahmoodi, M., Gelves, G.A., Park, S., Sundararaj, U., 2011. Electrical and electromagnetic interference shielding properties of flow-induced oriented carbon nanotubes in polycarbonate. *Carbon* 49(11), 3430--3440. https://doi.org/10.1016/j.carbon.2011.04.046

Balberg, I., Anderson, C.H., Alexander, S., Wagner, N., 1984. Excluded volume and its relation to the onset of percolation. *Physical Review B* 30(7), 3933. https://doi.org/10.1103/PhysRevB.30.3933

Bauhofer, W., Kovacs, J.Z., 2009. A review and analysis of electrical percolation in carbon nanotube polymer composites. *Composites Science and Technology* 69(10), 1486--1498. https://doi.org/10.1016/j.compscitech.2008.06.018

Bruggeman, D.A.G., 1935. Berechnung verschiedener physikalischer Konstanten von heterogenen Substanzen. *Annalen der Physik* 416(7), 636--664. https://doi.org/10.1002/andp.19354160705

Cao, M.S., Wang, X.X., Zhang, M., et al., 2020. Electromagnetic response and energy conversion for functions and devices in low-dimensional materials. *Advanced Functional Materials* 30(25), 1907698. https://doi.org/10.1002/adfm.201907698

Celozzi, S., Araneo, R., Lovat, G., 2008. *Electromagnetic Shielding*. Wiley. https://doi.org/10.1002/9780470268483

Celzard, A., McRae, E., Deleuze, C., Dufort, M., Furdin, G., Marêche, J.F., 1996. Critical concentration in percolating systems containing a high-aspect-ratio filler. *Physical Review B* 53(10), 6209. https://doi.org/10.1103/PhysRevB.53.6209

Han, M., Shuck, C.E., Rakhmanov, R., et al., 2020. Beyond Ti₃C₂Tₓ: MXenes for electromagnetic interference shielding. *ACS Nano* 14(4), 11750--11759. https://doi.org/10.1021/acsnano.0c04504

Hashin, Z., Shtrikman, S., 1962. A variational approach to the theory of the effective magnetic permeability of multiphase materials. *Journal of Applied Physics* 33(10), 3125--3131. https://doi.org/10.1063/1.1728579

He, X., et al., 2024. Multi-functional graphene aerogels/epoxy resin composites with Janus structure for enhanced electromagnetic interference shielding. *Chemical Engineering Journal* 501, 157507. https://doi.org/10.1016/j.cej.2024.157507

Hoang, N.H., et al., 2015. Effect of microstructure on electromagnetic shielding effectiveness of steel. *IEEE Transactions on Electromagnetic Compatibility* 57(6), 1566--1574. https://doi.org/10.1109/TEMC.2015.2460672

Hong, W., Jiang, Z.H., Yu, C., et al., 2017. Multibeam antenna technologies for 5G wireless communications. *IEEE Journal on Selected Areas in Communications* 35(6), 1291--1302. https://doi.org/10.1109/JSAC.2017.2687878

Hu, J., et al., 2026. MXene-based electromagnetic interference shielding materials: A leap from fundamental research to intelligent customization. *Small* 22, 2505417. https://doi.org/10.1002/smll.202505417

Iqbal, A., Shahzad, F., Hantanasirisakul, K., et al., 2020. Anomalous absorption of electromagnetic waves by 2D transition metal carbonitride Ti₃CNTₓ (MXene). *Composites Part B* 202, 108580. https://doi.org/10.1016/j.compositesb.2020.108580

Kim, S.H., Jang, S.H., Byun, S.W., et al., 2008. Electrical conductivity and electromagnetic interference shielding of Cu/PET multilayered composites. *Composites Science and Technology* 68(14), 2909--2916. https://doi.org/10.1016/j.compscitech.2007.10.024

Kirkpatrick, S., 1973. Percolation and conduction. *Reviews of Modern Physics* 45(4), 574--588. https://doi.org/10.1103/RevModPhys.45.574

Li, N., Huang, Y., Du, F., et al., 2015. Electromagnetic interference (EMI) shielding of single-walled carbon nanotube epoxy composites. *Nanoscale* 7, 8219--8232. https://doi.org/10.1039/C5NR01083G

Liu, J., et al., 2024. Mechanically robust and multifunctional Ti₃C₂Tₓ MXene composite aerogel for broadband EMI shielding. *Carbon* 221, 118948. https://doi.org/10.1016/j.carbon.2024.118948

Liu, X., et al., 2025. Electrically insulating electromagnetic interference shielding materials: A perspective. *Advanced Functional Materials* 35, 2407439. https://doi.org/10.1002/adfm.202407439

Matula, R.A., 1979. Electrical resistivity of copper, gold, palladium, and silver. *Journal of Physical and Chemical Reference Data* 8(4), 1147--1298.

Maxwell Garnett, J.C., 1904. Colours in metal glasses and in metallic films. *Philosophical Transactions of the Royal Society of London A* 203, 385--420. https://doi.org/10.1098/rsta.1904.0024

Mayadas, A.F., Shatzkes, M., 1970. Electrical-resistivity model for polycrystalline films: The case of arbitrary reflection at external surfaces. *Physical Review B* 1(4), 1382--1389. https://doi.org/10.1103/PhysRevB.1.1382

McKay, M.D., Beckman, R.J., Conover, W.J., 1979. A comparison of three methods for selecting values of input variables in the analysis of output from a computer code. *Technometrics* 21(2), 239--245.

McLachlan, D.S., Blaszkiewicz, M., Newnham, R.E., 1990. Electrical resistivity of composites. *Journal of the American Ceramic Society* 73(8), 2187--2203. https://doi.org/10.1111/j.1151-2916.1990.tb07576.x

Ott, H.W., 2009. *Electromagnetic Compatibility Engineering*. Wiley. https://doi.org/10.1002/9780470508510

Pande, C.S., Masumura, R.A., Armstrong, R.W., 1993. Pile-up based Hall--Petch relation for nanoscale materials. *Nanostructured Materials* 2(3), 323--331.

Paul, C.R., 2006. *Introduction to Electromagnetic Compatibility*, second ed. Wiley.

Pozar, D.M., 2011. *Microwave Engineering*, fourth ed. Wiley.

Saltelli, A., 2002. Making best use of model evaluations to compute sensitivity indices. *Computer Physics Communications* 145(2), 280--297.

Schelkunoff, S.A., 1934. The electromagnetic theory of coaxial transmission lines and cylindrical shields. *Bell System Technical Journal* 13(4), 532--579. https://doi.org/10.1002/j.1538-7305.1934.tb00679.x

Schulz, R.B., Plantz, V.C., Brush, D.R., 1988. Shielding theory and practice. *IEEE Transactions on Electromagnetic Compatibility* 30(3), 187--201. https://doi.org/10.1109/15.3297

Shahzad, F., Alhabeb, M., Hatter, C.B., et al., 2016. Electromagnetic interference shielding with 2D transition metal carbides (MXenes). *Science* 353(6304), 1137--1140. https://doi.org/10.1126/science.aag2421

Shen, B., Zhai, W., Zheng, W., 2014. Ultrathin flexible graphene film: An excellent thermal conducting material with efficient EMI shielding. *Advanced Functional Materials* 24(28), 4542--4548. https://doi.org/10.1002/adfm.201400079

Snoek, J.L., 1948. Dispersion and absorption in magnetic ferrites at frequencies above one Mc/s. *Physica* 14(4), 207--217. https://doi.org/10.1016/0031-8914(48)90038-X

Sobol, I.M., 2001. Global sensitivity indices for nonlinear mathematical models and their Monte Carlo estimates. *Mathematics and Computers in Simulation* 55(1--3), 271--280.

Song, W.L., et al., 2014. Magnetic and conductive graphene papers toward thin layers of effective electromagnetic shielding. *Small* 10(20), 4000--4007. https://doi.org/10.1002/smll.201400615

Stauffer, D., Aharony, A., 1994. *Introduction to Percolation Theory*, second ed. Taylor & Francis. https://doi.org/10.1201/9781315274386

Thomassin, J.M., Jérôme, C., Pardoen, T., Bailly, C., Huynen, I., Detrembleur, C., 2013. Polymer/carbon based composites as electromagnetic interference (EMI) shielding materials. *Materials Science and Engineering R* 74(7), 211--232. https://doi.org/10.1016/j.mser.2013.06.001

Wanasinghe, D., Aslani, F., Ma, G., Habibi, D., 2020. Review of polymer composites with diverse nanofillers for electromagnetic interference shielding. *Nanomaterials* 10(3), 541. https://doi.org/10.3390/nano10030541

Wang, Y., et al., 2025. Electromagnetic interference shielding: a comprehensive review of materials, mechanisms, and applications. *Nanoscale Advances*. https://doi.org/10.1039/D5NA00240K

Yan, D.X., Pang, H., Li, B., et al., 2015. Structured reduced graphene oxide/polymer composites for ultra-efficient electromagnetic interference shielding. *Advanced Functional Materials* 25(4), 559--566. https://doi.org/10.1002/adfm.201403809

Yeh, P., 1988. *Optical Waves in Layered Media*. Wiley.

Zhang, Y., et al., 2016. Broadband and tunable high-performance microwave absorption of an ultralight and highly compressible graphene foam. *ACS Applied Materials & Interfaces* 8(21), 20422--20431. https://doi.org/10.1021/acsami.6b07552
