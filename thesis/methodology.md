# EMI Shielding Calculation Methodology

## 1. Overview

This document provides detailed technical documentation of the calculation methods used in the EMI Shield Designer application. All formulations are based on classical electromagnetic theory with specific implementations for composite materials.

## 2. Core Electromagnetic Calculations

### 2.1 Fundamental Constants
```python
MU_0 = 4 * π * 1e-7          # Permeability of free space (H/m)
EPSILON_0 = 8.854187817e-12  # Permittivity of free space (F/m)
Z_0 = 376.73031346177        # Impedance of free space (Ω)
C = 299792458                # Speed of light (m/s)
```

### 2.2 Skin Depth Calculation

The skin depth represents the distance at which the electromagnetic field amplitude decreases to 1/e (≈37%) of its surface value.

```python
def calculate_skin_depth(σ, μ, f):
    """
    δ = √(2/(ωμσ))
    
    where:
    - σ: conductivity (S/m)
    - μ: absolute permeability (H/m)
    - f: frequency (Hz)
    - ω = 2πf: angular frequency
    """
    ω = 2 * π * f
    δ = sqrt(2 / (ω * μ * σ))
    return δ
```

### 2.3 Intrinsic Impedance

The intrinsic impedance determines the ratio of electric to magnetic field in the material.

```python
def calculate_intrinsic_impedance(σ, μ_r, ε_r, f):
    """
    η = √(μ/ε*)
    
    where ε* = ε - j(σ/ω) is the complex permittivity
    """
    ω = 2 * π * f
    μ = μ_r * MU_0
    ε = ε_r * EPSILON_0
    
    # Complex permittivity accounting for conduction
    ε_complex = ε - 1j * (σ / ω)
    
    # Intrinsic impedance
    η = sqrt(μ / ε_complex)
    return η
```

### 2.4 Propagation Constant

The propagation constant describes how electromagnetic waves attenuate and phase-shift through the material.

```python
def calculate_propagation_constant(σ, μ_r, ε_r, f):
    """
    γ = jω√(με*) = α + jβ
    
    where:
    - α: attenuation constant (Np/m)
    - β: phase constant (rad/m)
    """
    ω = 2 * π * f
    μ = μ_r * MU_0
    ε = ε_r * EPSILON_0
    ε_complex = ε - 1j * (σ / ω)
    
    γ = 1j * ω * sqrt(μ * ε_complex)
    return γ
```

## 3. Shielding Effectiveness Components

### 3.1 Reflection Loss (R)

Reflection loss occurs at material boundaries due to impedance mismatch.

```python
def calculate_reflection_loss(η):
    """
    R = -10 * log₁₀(1 - |Γ|²)
    
    where Γ = (η - Z₀)/(η + Z₀) is the reflection coefficient
    """
    Γ = (η - Z_0) / (η + Z_0)
    R_coeff = abs(Γ) ** 2
    
    if R_coeff < 1:
        R = -10 * log10(1 - R_coeff)
    else:
        R = 0
    
    return R
```

### 3.2 Absorption Loss (A)

Absorption loss occurs as waves propagate through the material.

```python
def calculate_absorption_loss(thickness, γ):
    """
    A = 8.686 * α * t
    
    where:
    - α = Re(γ): attenuation constant
    - t: material thickness (m)
    - 8.686 = 20/ln(10): conversion factor from Np to dB
    """
    α = γ.real
    A = 8.686 * α * thickness
    return A
```

### 3.3 Multiple Reflection Loss (B)

Multiple reflections occur between material boundaries, significant for thin shields.

```python
def calculate_multiple_reflection_loss(η, thickness, γ):
    """
    B = -20 * log₁₀|K|
    
    where K = (1 - Γ₁Γ₂e^(-2γt))/(1 + Γ₁Γ₂e^(-2γt))
    """
    Γ₁ = (η - Z_0) / (η + Z_0)
    Γ₂ = (Z_0 - η) / (Z_0 + η)
    
    exp_term = exp(-2 * γ * thickness)
    K = (1 - Γ₁ * Γ₂ * exp_term) / (1 + Γ₁ * Γ₂ * exp_term)
    
    B = -20 * log10(abs(K))
    
    # Multiple reflections negligible for high absorption
    if A > 15:  # dB
        B = 0
    
    return B
```

### 3.4 Total Shielding Effectiveness

```python
def calculate_total_se(σ, μ_r, ε_r, thickness, frequency):
    """
    SE_total = R + A + B
    """
    η = calculate_intrinsic_impedance(σ, μ_r, ε_r, frequency)
    γ = calculate_propagation_constant(σ, μ_r, ε_r, frequency)
    
    R = calculate_reflection_loss(η)
    A = calculate_absorption_loss(thickness, γ)
    B = calculate_multiple_reflection_loss(η, thickness, γ)
    
    SE_total = R + A + B
    return SE_total, R, A, B
```

## 4. Composite Material Properties

### 4.1 Current Implementation (Simple Mixing Rules)

#### 4.1.1 Conductivity - Arithmetic Mean
```python
σ_eff = Σ(σᵢ × wᵢ)
```
- Assumes parallel conduction paths
- Valid for well-connected phases
- Overestimates for dispersed particles

#### 4.1.2 Permeability - Geometric Mean
```python
μ_r_eff = Π(μᵢ^wᵢ)
```
- Assumes uniform field distribution
- Reasonable for non-magnetic/weakly magnetic composites
- May underestimate for magnetic composites

#### 4.1.3 Permittivity - Geometric Mean
```python
ε_r_eff = Π(εᵢ^wᵢ)
```
- Similar assumptions to permeability
- Works well for low-loss dielectrics

### 4.2 Limitations of Current Approach

1. **No Microstructure Consideration**
   - Particle size effects ignored
   - Distribution uniformity assumed
   - No percolation threshold

2. **Linear Mixing Assumption**
   - No interaction between phases
   - No interfacial effects
   - No frequency dispersion

3. **Isotropic Assumption**
   - No directional properties
   - Uniform material properties
   - No texture effects

## 5. Advanced Mixing Models (Proposed)

### 5.1 Maxwell-Garnett Model

For spherical inclusions in a host matrix:

```python
def maxwell_garnett_permittivity(ε_host, ε_inclusion, volume_fraction):
    """
    ε_eff = ε_h * (1 + 3f(ε_i - ε_h)/(ε_i + 2ε_h - f(ε_i - ε_h)))
    
    where:
    - ε_h: host permittivity
    - ε_i: inclusion permittivity
    - f: volume fraction of inclusions
    """
    numerator = ε_inclusion - ε_host
    denominator = ε_inclusion + 2*ε_host - volume_fraction*(ε_inclusion - ε_host)
    
    ε_eff = ε_host * (1 + 3*volume_fraction*numerator/denominator)
    return ε_eff
```

### 5.2 Bruggeman Effective Medium Theory

Self-consistent approach for arbitrary concentrations:

```python
def bruggeman_emt(ε₁, ε₂, f₁):
    """
    f₁(ε₁ - ε_eff)/(ε₁ + 2ε_eff) + f₂(ε₂ - ε_eff)/(ε₂ + 2ε_eff) = 0
    
    Solved iteratively for ε_eff
    """
    # Requires numerical solution
    pass
```

### 5.3 Percolation Theory

For conductive fillers:

```python
def percolation_conductivity(σ_filler, volume_fraction, f_critical=0.16):
    """
    σ_eff = σ_filler * (f - f_c)^t  for f > f_c
    σ_eff ≈ 0                       for f < f_c
    
    where:
    - f_c: percolation threshold (~0.16 for 3D random)
    - t: critical exponent (~2 for 3D)
    """
    if volume_fraction > f_critical:
        σ_eff = σ_filler * (volume_fraction - f_critical)**2
    else:
        σ_eff = 1e-10  # Essentially insulating
    
    return σ_eff
```

## 6. Frequency-Dependent Effects

### 6.1 Relaxation Phenomena

```python
def debye_relaxation(ε_static, ε_infinite, relaxation_time, frequency):
    """
    ε*(ω) = ε_∞ + (ε_s - ε_∞)/(1 + jωτ)
    """
    ω = 2 * π * frequency
    ε_complex = ε_infinite + (ε_static - ε_infinite)/(1 + 1j*ω*relaxation_time)
    return ε_complex
```

### 6.2 Skin Effect Correction

For high frequencies where skin depth << thickness:

```python
def high_frequency_correction(SE_dc, skin_depth, thickness):
    """
    SE_hf = SE_dc * (1 - exp(-thickness/skin_depth))
    """
    correction = 1 - exp(-thickness/skin_depth)
    SE_corrected = SE_dc * correction
    return SE_corrected
```

## 7. Validation Approach

### 7.1 Test Cases

1. **Pure Metals** (Known SE values)
   - Copper at 1 GHz, 1mm: ~120 dB
   - Aluminum at 1 GHz, 1mm: ~100 dB
   - Steel at 1 GHz, 1mm: ~130 dB

2. **Standard Composites**
   - 30% Carbon in polymer: ~30-40 dB at 1 GHz
   - Nickel-coated carbon fiber: ~60-80 dB

### 7.2 Error Metrics

```python
def calculate_error_metrics(calculated, measured):
    """
    Absolute Error: |SE_calc - SE_meas|
    Relative Error: |SE_calc - SE_meas|/SE_meas * 100%
    RMS Error: √(Σ(SE_calc - SE_meas)²/n)
    """
    abs_error = abs(calculated - measured)
    rel_error = abs_error / measured * 100
    return abs_error, rel_error
```

## 8. Computational Considerations

### 8.1 Numerical Stability

- Check for division by zero (σ = 0)
- Handle complex logarithms properly
- Validate physical bounds (μ_r ≥ 0.999, ε_r ≥ 1)

### 8.2 Performance Optimization

- Cache material properties
- Vectorize frequency sweeps
- Pre-compute constants

### 8.3 Precision Requirements

- Use complex128 for impedance calculations
- Maintain at least 6 significant figures
- Round final results to 0.1 dB

## 9. Implementation Checklist

- [x] Basic SE calculations (R, A, B)
- [x] Simple mixing rules
- [x] Frequency sweep capability
- [ ] Maxwell-Garnett model
- [ ] Bruggeman EMT
- [ ] Percolation effects
- [ ] Frequency-dependent properties
- [ ] Anisotropic materials
- [ ] Temperature effects
- [ ] Uncertainty quantification

---

*Last Updated: 2025-07-03*