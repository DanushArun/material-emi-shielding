# Advanced Physics Models for EMI Shielding Simulation

## Research Synthesis - Evidence Funnel Format (Quote -> Citation -> Synthesis)

---

## 1. Transfer Matrix Method for Multilayer Shielding

### 1.1 Mathematical Formulation

**Quote:** "For a stratified medium consisting of N layers, each with thickness d_j, the relationship between the tangential field components at the first and last interface is given by the product of transfer matrices: M_total = M_1 * M_2 * ... * M_N, where each layer matrix is:

M_j = [[cos(k_j*d_j), -j*Z_j*sin(k_j*d_j)], [-j*sin(k_j*d_j)/Z_j, cos(k_j*d_j)]]

where k_j is the propagation constant and Z_j is the wave impedance in layer j."

**Citation:** Yeh, P. *Optical Waves in Layered Media.* Wiley, 1988. ISBN: 978-0471828662

**Synthesis:** This is the exact formulation we need to implement. For each material layer:
- k_j = omega * sqrt(mu_j * epsilon_j_complex), where epsilon_j_complex = epsilon_j - j*sigma_j/omega
- Z_j = sqrt(mu_j / epsilon_j_complex)
- The total SE is then: SE = 20 * log10(|0.5 * (M_total[0,0] + M_total[1,1] + M_total[0,1]/Z_0 + Z_0*M_total[1,0])|)

This handles arbitrary N-layer stacks with a single matrix multiplication chain. Computational cost is O(N) per frequency point.

### 1.2 Implementation Equations

For a conducting layer j with conductivity sigma_j, permeability mu_j, permittivity epsilon_j:

```
gamma_j = sqrt(j*omega*mu_j*(sigma_j + j*omega*epsilon_j))  # propagation constant
eta_j = sqrt(j*omega*mu_j / (sigma_j + j*omega*epsilon_j))   # wave impedance

M_j = [[cosh(gamma_j*d_j), eta_j*sinh(gamma_j*d_j)],
       [sinh(gamma_j*d_j)/eta_j, cosh(gamma_j*d_j)]]
```

The total shielding effectiveness:
```
M = M_1 * M_2 * ... * M_N
SE = 20*log10(|0.5*(M[0,0] + M[1,1] + M[0,1]/eta_0 + eta_0*M[1,0])|)
```

where eta_0 = 376.73 ohms (free space impedance).

---

## 2. Percolation Theory for Composite Conductivity

### 2.1 Power-Law Conductivity Model

**Quote:** "Near the percolation threshold p_c, the conductivity of a random resistor network follows sigma ~ sigma_0 * (p - p_c)^t for p > p_c, where t is a universal critical exponent approximately equal to 2.0 in three dimensions."

**Citation:** Kirkpatrick, S. "Percolation and Conduction." *Reviews of Modern Physics* 45(4), 574-588, 1973. DOI: 10.1103/RevModPhys.45.574

**Synthesis:** This replaces the linear mixing rule for all composite materials. Implementation requires:
- p = volume fraction of conductive filler
- p_c = percolation threshold (material-dependent)
- t = critical exponent (~2.0 for 3D random networks)
- sigma_0 = intrinsic conductivity of filler phase

### 2.2 Percolation Thresholds for Common EMI Fillers

**Quote:** "The percolation threshold depends strongly on filler aspect ratio. For carbon nanotubes with aspect ratio >1000, p_c can be as low as 0.1 vol%, while for spherical particles p_c ~ 16 vol% (theoretical site percolation on cubic lattice)."

**Citation:** Stauffer, D. and Aharony, A. *Introduction to Percolation Theory.* 2nd ed., Taylor & Francis, 1994. DOI: 10.1201/9781315274386

**Synthesis:** Critical percolation thresholds for the simulation database:

| Filler Type | Aspect Ratio | Typical p_c (vol%) |
|-------------|-------------|---------------------|
| Spherical metal particles | ~1 | 15-20 |
| Short carbon fibers | 10-100 | 5-15 |
| Carbon nanotubes (MWCNT) | 100-1000 | 0.5-3.0 |
| Carbon nanotubes (SWCNT) | >1000 | 0.1-0.5 |
| Graphene nanoplatelets | 100-10000 (2D) | 0.1-1.0 |
| Metal nanowires (Ag) | 50-500 | 0.5-5.0 |
| MXene flakes | 100-1000 (2D) | 0.5-2.0 |

### 2.3 McLachlan GEM Equation

**Quote:** "The General Effective Media equation provides a unified framework that bridges the Bruggeman effective medium theory and percolation theory:

(1-f) * (sigma_l^(1/s) - sigma_m^(1/s)) / (sigma_l^(1/s) + A*sigma_m^(1/s)) + f * (sigma_h^(1/s) - sigma_m^(1/s)) / (sigma_h^(1/s) + A*sigma_m^(1/s)) = 0

where A = (1-f_c)/f_c, f is the volume fraction of conductor, f_c is the percolation threshold, s and t are critical exponents, sigma_l and sigma_h are the low and high conductivity phases, and sigma_m is the effective conductivity."

**Citation:** McLachlan, D.S., Blaszkiewicz, M., and Newnham, R.E. "Electrical Resistivity of Composites." *J. Am. Ceram. Soc.* 73(8), 2187-2203, 1990. DOI: 10.1111/j.1151-2916.1990.tb07576.x

**Synthesis:** The GEM equation is the most general model - it reduces to Bruggeman EMT when s=t=1, and captures percolation behavior with proper s,t exponents. This should be the primary composite model in the simulation, with simpler models (Maxwell-Garnett, Bruggeman) as special cases.

---

## 3. Effective Medium Theories

### 3.1 Maxwell-Garnett (MG) - Dilute Inclusions

**Quote:** "For a dilute suspension of spherical inclusions (permittivity epsilon_i) in a host medium (permittivity epsilon_h), the effective permittivity is:

epsilon_eff = epsilon_h * [1 + 3*f*(epsilon_i - epsilon_h) / (epsilon_i + 2*epsilon_h - f*(epsilon_i - epsilon_h))]

where f is the volume fraction of inclusions."

**Citation:** Maxwell Garnett, J.C. "Colours in Metal Glasses and in Metallic Films." *Phil. Trans. R. Soc. A* 203, 385-420, 1904. DOI: 10.1098/rsta.1904.0024

**Synthesis:** MG is appropriate when: (1) filler volume fraction is low (<30%), (2) one phase is clearly the "host" and other is "inclusion", (3) inclusions are approximately spherical. It works for metal-particle-filled polymers at low loading.

### 3.2 Bruggeman - Symmetric/High Loading

**Quote:** "The Bruggeman effective medium approximation treats both components symmetrically. For a two-phase system: f*(epsilon_1 - epsilon_eff)/(epsilon_1 + 2*epsilon_eff) + (1-f)*(epsilon_2 - epsilon_eff)/(epsilon_2 + 2*epsilon_eff) = 0"

**Citation:** Bruggeman, D.A.G. *Annalen der Physik* 416(7), 636-664, 1935. DOI: 10.1002/andp.19354160705

**Synthesis:** Bruggeman is appropriate when: (1) both phases have comparable volume fractions, (2) no clear host-inclusion distinction, (3) the mixture is random and isotropic. Better than MG for high filler loadings. Also naturally predicts a percolation-like transition at f=1/3 for spherical inclusions.

### 3.3 When to Use Which Model

| Scenario | Best Model | Reason |
|----------|-----------|--------|
| Metal alloys (single phase) | Nordheim's rule (weighted sum) | Solid solution, no inclusions |
| Low filler loading (<15 vol%) | Maxwell-Garnett | Clear host-inclusion structure |
| High filler loading (15-50 vol%) | Bruggeman | Symmetric treatment needed |
| Near/above percolation | McLachlan GEM | Captures threshold physics |
| CNT/graphene composites | GEM with aspect-ratio-corrected p_c | High aspect ratio fillers |

---

## 4. Frequency-Dependent Permeability

### 4.1 Snoek's Limit

**Quote:** "For polycrystalline soft magnetic materials, the product of static permeability and ferromagnetic resonance frequency is approximately constant: (mu_s - 1) * f_r = (2/3) * gamma * M_s, where gamma is the gyromagnetic ratio and M_s is the saturation magnetization."

**Citation:** Snoek, J.L. "Dispersion and absorption in magnetic ferrites at frequencies above one Mc/s." *Physica* 14(4), 207-217, 1948. DOI: 10.1016/0031-8914(48)90038-X

**Synthesis:** For implementation, the frequency-dependent permeability model:

```
mu(f) = 1 + (mu_s - 1) / (1 + j*f/f_r)
```

where:
- mu_s = static (DC) permeability
- f_r = resonance frequency = (2/3) * gamma * M_s / (mu_s - 1)
- gamma = 2.8 MHz/Oe (gyromagnetic ratio)

Typical resonance frequencies:
| Material | mu_s | f_r (MHz) |
|----------|------|-----------|
| Mu-metal | 100,000 | ~0.01 |
| Permalloy 80 | 50,000 | ~0.05 |
| Mild steel | 2,000 | ~5 |
| Mn-Zn ferrite | 5,000 | ~1 |
| Ni-Zn ferrite | 200 | ~100 |

This means mu-metal is only effective below ~10 kHz! At 1 GHz, its effective permeability is ~1.

---

## 5. Temperature-Dependent Conductivity

### 5.1 Linear Model (Above Debye Temperature)

For metals above their Debye temperature (most metals at room temperature and above):

```
sigma(T) = sigma_ref / (1 + alpha * (T - T_ref))
```

where alpha is the temperature coefficient of resistance (TCR).

Typical TCR values:
| Material | sigma_ref at 20C (S/m) | alpha (1/K) |
|----------|----------------------|-------------|
| Copper | 5.96e7 | 0.00393 |
| Aluminum | 3.50e7 | 0.00429 |
| Silver | 6.30e7 | 0.0038 |
| Iron | 1.00e7 | 0.00651 |
| Nickel | 1.45e7 | 0.00641 |
| Steel 304 | 1.45e6 | 0.00094 |

At 150C: copper conductivity drops to ~4.0e7 S/m (33% reduction from 20C).

---

## 6. Monte Carlo Uncertainty Quantification

For publishable results, every SE prediction needs error bars. Sources of uncertainty:
- Material composition tolerance (+-1-5%)
- Thickness variation (+-5-10%)
- Grain size distribution (log-normal, not single value)
- Temperature uncertainty
- Frequency measurement accuracy

Monte Carlo approach: Sample N=1000 realizations from parameter distributions, compute SE for each, report mean and 95% confidence interval.

---

## 7. Implementation Priority and Equations Summary

### Priority 1: Transfer Matrix Method
```python
def multilayer_se(layers, frequency):
    """layers = [(sigma, mu_r, eps_r, thickness), ...]"""
    M = np.eye(2, dtype=complex)
    for sigma, mu_r, eps_r, d in layers:
        omega = 2 * pi * frequency
        mu = mu_r * MU_0
        eps_c = eps_r * EPS_0 - 1j * sigma / omega
        gamma = 1j * omega * np.sqrt(mu * eps_c)
        eta = np.sqrt(1j * omega * mu / (sigma + 1j * omega * eps_c))
        M_layer = np.array([
            [np.cosh(gamma*d), eta*np.sinh(gamma*d)],
            [np.sinh(gamma*d)/eta, np.cosh(gamma*d)]
        ])
        M = M @ M_layer
    eta0 = 376.73
    T = 0.5 * (M[0,0] + M[1,1] + M[0,1]/eta0 + eta0*M[1,0])
    return 20 * np.log10(np.abs(T))
```

### Priority 2: GEM Equation for Composites
```python
def gem_conductivity(sigma_l, sigma_h, f, f_c, s=0.87, t=2.0):
    """McLachlan GEM equation - solved numerically"""
    from scipy.optimize import brentq
    A = (1 - f_c) / f_c
    def equation(sigma_m):
        term1 = (1-f) * (sigma_l**(1/s) - sigma_m**(1/s)) / (sigma_l**(1/s) + A*sigma_m**(1/s))
        term2 = f * (sigma_h**(1/t) - sigma_m**(1/t)) / (sigma_h**(1/t) + A*sigma_m**(1/t))
        return term1 + term2
    return brentq(equation, sigma_l, sigma_h)
```

### Priority 3: Frequency-Dependent Permeability
```python
def freq_dependent_permeability(mu_static, frequency, M_s=None, f_resonance=None):
    """Debye-type relaxation with Snoek's limit"""
    if f_resonance is None:
        gamma_gyro = 2.8e6  # Hz/Oe -> need M_s in A/m
        f_resonance = (2/3) * gamma_gyro * M_s / (mu_static - 1)
    return 1 + (mu_static - 1) / (1 + 1j * frequency / f_resonance)
```

### Priority 4: Temperature-Dependent Conductivity
```python
def temp_dependent_conductivity(sigma_ref, T, T_ref=293.15, alpha=0.00393):
    """Linear TCR model"""
    return sigma_ref / (1 + alpha * (T - T_ref))
```
