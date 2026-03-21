"""Monte Carlo uncertainty quantification for EMI shielding predictions.

Propagates manufacturing variability (composition, thickness, grain size)
through the physics engine to produce SE predictions with confidence intervals.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from src.physics.emi_calculations import EMICalculator


@dataclass
class UncertaintySpec:
    """Specification for parameter uncertainty."""
    conductivity_cv: float = 0.05     # coefficient of variation (5%)
    thickness_cv: float = 0.02        # 2%
    grain_size_cv: float = 0.30       # 30% (log-normal)
    permeability_cv: float = 0.10     # 10%
    frequency_cv: float = 0.001       # 0.1% (measurement precision)


# Module-level calculator instance reused across all functions to avoid
# repeated initialisation overhead.
_calculator = EMICalculator()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _se_from_params(
    conductivity: float,
    permeability: float,
    permittivity: float,
    thickness: float,
    frequency: float,
    grain_size: Optional[float],
) -> float:
    """Evaluate total SE (dB) for a single parameter set.

    ``include_confidence=False`` skips the internal confidence heuristic so
    that only the physics output contributes to the MC distribution.

    Physical bounds enforced here mirror the validators in
    ``src.utils.constants``:
    - conductivity >= 0
    - relative_permeability >= 0.999  (diamagnetic floor)
    - relative_permittivity >= 1.0
    - thickness > 0
    - frequency > 0
    """
    conductivity  = max(conductivity,  0.0)
    permeability  = max(permeability,  0.999)
    permittivity  = max(permittivity,  1.0)
    thickness     = max(thickness,     1e-12)
    frequency     = max(frequency,     1e-6)

    result = _calculator.calculate_shielding_effectiveness(
        conductivity=conductivity,
        relative_permeability=permeability,
        relative_permittivity=permittivity,
        thickness=thickness,
        frequency=frequency,
        grain_size=grain_size,
        include_confidence=False,
    )
    return float(result["total_se"])


def _sample_normal(
    nominal: float, cv: float, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw *n* samples from N(nominal, (cv * nominal)^2), clipped to > 0."""
    if cv == 0.0:
        return np.full(n, nominal)
    std = cv * abs(nominal)
    samples = rng.normal(nominal, std, size=n)
    # Physical parameters must be strictly positive.
    return np.clip(samples, 1e-30, None)


def _sample_lognormal(
    nominal: float, cv: float, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw *n* log-normal samples whose median equals *nominal*.

    The log-normal sigma parameter is derived from the desired CV so that
    the distribution mean and variance reproduce the expected CV.
    """
    if cv == 0.0:
        return np.full(n, nominal)
    # Relationship: CV^2 = exp(sigma^2) - 1  =>  sigma = sqrt(ln(1 + CV^2))
    sigma_ln = np.sqrt(np.log(1.0 + cv ** 2))
    mu_ln = np.log(nominal) - 0.5 * sigma_ln ** 2
    return rng.lognormal(mean=mu_ln, sigma=sigma_ln, size=n)


def _build_param_samples(
    conductivity: float,
    permeability: float,
    permittivity: float,
    thickness: float,
    frequency: float,
    grain_size: Optional[float],
    uncertainty: UncertaintySpec,
    n: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Return perturbed arrays for each physical parameter."""
    sigma_arr = _sample_normal(conductivity, uncertainty.conductivity_cv, n, rng)
    mu_arr = _sample_normal(permeability, uncertainty.permeability_cv, n, rng)
    mu_arr = np.clip(mu_arr, 0.999, None)  # validator requires >= 0.999
    eps_arr = _sample_normal(permittivity, 0.0, n, rng)   # permittivity treated as exact
    eps_arr = np.clip(eps_arr, 1.0, None)  # validator requires >= 1.0
    t_arr = _sample_normal(thickness, uncertainty.thickness_cv, n, rng)
    t_arr = np.clip(t_arr, 1e-9, None)  # must be positive
    f_arr = _sample_normal(frequency, uncertainty.frequency_cv, n, rng)
    f_arr = np.clip(f_arr, 1.0, None)  # must be positive

    gs_arr: Optional[np.ndarray] = None
    if grain_size is not None and grain_size > 0:
        gs_arr = _sample_lognormal(grain_size, uncertainty.grain_size_cv, n, rng)

    return sigma_arr, mu_arr, eps_arr, t_arr, f_arr, gs_arr


def _run_samples(
    sigma_arr: np.ndarray,
    mu_arr: np.ndarray,
    eps_arr: np.ndarray,
    t_arr: np.ndarray,
    f_arr: np.ndarray,
    gs_arr: Optional[np.ndarray],
) -> np.ndarray:
    """Evaluate SE for every row of the pre-built sample arrays."""
    n = len(sigma_arr)
    se_vals = np.empty(n)
    for i in range(n):
        gs_i = float(gs_arr[i]) if gs_arr is not None else None
        se_vals[i] = _se_from_params(
            float(sigma_arr[i]),
            float(mu_arr[i]),
            float(eps_arr[i]),
            float(t_arr[i]),
            float(f_arr[i]),
            gs_i,
        )
    return se_vals


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def monte_carlo_se(
    conductivity: float,
    permeability: float,
    permittivity: float,
    thickness: float,
    frequency: float,
    grain_size: Optional[float] = None,
    uncertainty: Optional[UncertaintySpec] = None,
    n_samples: int = 1000,
    seed: Optional[int] = None,
) -> Dict:
    """Run Monte Carlo uncertainty propagation for SE predictions.

    Each input parameter is sampled independently:
    - conductivity, thickness, permeability, frequency: normal distribution
    - grain_size: log-normal distribution (right-skewed manufacturing spread)

    Parameters
    ----------
    conductivity : float
        Nominal electrical conductivity (S/m).
    permeability : float
        Nominal relative permeability.
    permittivity : float
        Nominal relative permittivity.
    thickness : float
        Nominal shield thickness (m).
    frequency : float
        Operating frequency (Hz).
    grain_size : float, optional
        Nominal grain size (m). If None, grain boundary effects are ignored.
    uncertainty : UncertaintySpec, optional
        Coefficient-of-variation settings for each parameter. Defaults to the
        class defaults when not provided.
    n_samples : int
        Number of Monte Carlo draws (default 1000).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        se_mean          - mean SE across all samples (dB)
        se_std           - standard deviation (dB)
        se_ci_lower      - 2.5th percentile (95% CI lower bound)
        se_ci_upper      - 97.5th percentile (95% CI upper bound)
        se_distribution  - numpy array of all SE values (length n_samples)
        n_samples        - number of valid samples used
        deterministic_se - SE at the nominal parameter values
    """
    if uncertainty is None:
        uncertainty = UncertaintySpec()

    rng = np.random.default_rng(seed)

    sigma_arr, mu_arr, eps_arr, t_arr, f_arr, gs_arr = _build_param_samples(
        conductivity, permeability, permittivity, thickness, frequency,
        grain_size, uncertainty, n_samples, rng,
    )

    se_dist = _run_samples(sigma_arr, mu_arr, eps_arr, t_arr, f_arr, gs_arr)

    deterministic = _se_from_params(
        conductivity, permeability, permittivity, thickness, frequency, grain_size
    )

    return {
        "se_mean": float(np.mean(se_dist)),
        "se_std": float(np.std(se_dist, ddof=1)),
        "se_ci_lower": float(np.percentile(se_dist, 2.5)),
        "se_ci_upper": float(np.percentile(se_dist, 97.5)),
        "se_distribution": se_dist,
        "n_samples": n_samples,
        "deterministic_se": deterministic,
    }


def latin_hypercube_se(
    conductivity: float,
    permeability: float,
    permittivity: float,
    thickness: float,
    frequency: float,
    grain_size: Optional[float] = None,
    uncertainty: Optional[UncertaintySpec] = None,
    n_samples: int = 100,
    seed: Optional[int] = None,
) -> Dict:
    """Latin Hypercube Sampling (LHS) variant of the MC uncertainty estimate.

    LHS stratifies the unit hypercube so that each one-dimensional marginal is
    sampled uniformly. This typically achieves the same accuracy as pure MC
    with roughly an order of magnitude fewer samples.

    Parameters mirror :func:`monte_carlo_se`; *n_samples* defaults to 100
    because LHS is more efficient.

    Returns
    -------
    dict with the same keys as :func:`monte_carlo_se`.
    """
    from scipy.stats import norm, lognorm
    from scipy.stats.qmc import LatinHypercube

    if uncertainty is None:
        uncertainty = UncertaintySpec()

    use_grain = grain_size is not None and grain_size > 0
    n_dims = 5 + int(use_grain)

    sampler = LatinHypercube(d=n_dims, seed=seed)
    # shape: (n_samples, n_dims), values in [0, 1]
    unit_samples = sampler.random(n=n_samples)

    def _to_normal(col: np.ndarray, nominal: float, cv: float) -> np.ndarray:
        if cv == 0.0:
            return np.full(len(col), nominal)
        std = cv * abs(nominal)
        vals = norm.ppf(col, loc=nominal, scale=std)
        return np.clip(vals, 1e-30, None)

    def _to_lognormal(col: np.ndarray, nominal: float, cv: float) -> np.ndarray:
        if cv == 0.0:
            return np.full(len(col), nominal)
        sigma_ln = np.sqrt(np.log(1.0 + cv ** 2))
        mu_ln = np.log(nominal) - 0.5 * sigma_ln ** 2
        # lognorm in scipy uses s=sigma, scale=exp(mu)
        vals = lognorm.ppf(col, s=sigma_ln, scale=np.exp(mu_ln))
        return np.clip(vals, 1e-30, None)

    # Map each unit column to its physical distribution.
    sigma_arr = _to_normal(unit_samples[:, 0], conductivity, uncertainty.conductivity_cv)
    mu_arr    = _to_normal(unit_samples[:, 1], permeability,  uncertainty.permeability_cv)
    eps_arr   = np.full(n_samples, permittivity)           # permittivity exact
    t_arr     = _to_normal(unit_samples[:, 2], thickness,    uncertainty.thickness_cv)
    f_arr     = _to_normal(unit_samples[:, 3], frequency,    uncertainty.frequency_cv)

    gs_arr: Optional[np.ndarray] = None
    if use_grain:
        gs_arr = _to_lognormal(unit_samples[:, 4], grain_size, uncertainty.grain_size_cv)

    se_dist = _run_samples(sigma_arr, mu_arr, eps_arr, t_arr, f_arr, gs_arr)

    deterministic = _se_from_params(
        conductivity, permeability, permittivity, thickness, frequency, grain_size
    )

    return {
        "se_mean": float(np.mean(se_dist)),
        "se_std": float(np.std(se_dist, ddof=1)),
        "se_ci_lower": float(np.percentile(se_dist, 2.5)),
        "se_ci_upper": float(np.percentile(se_dist, 97.5)),
        "se_distribution": se_dist,
        "n_samples": n_samples,
        "deterministic_se": deterministic,
    }


def sobol_sensitivity(
    conductivity: float,
    permeability: float,
    permittivity: float,
    thickness: float,
    frequency: float,
    grain_size: Optional[float] = None,
    n_samples: int = 1024,
    seed: Optional[int] = None,
) -> Dict:
    """Compute first-order and total-order Sobol sensitivity indices.

    Uses the Saltelli (2002) estimator with quasi-random Sobol sequences for
    sample generation.  The sensitivity index of a parameter quantifies what
    fraction of the total output variance is attributable to that parameter.

    Parameters
    ----------
    conductivity, permeability, permittivity, thickness, frequency :
        Nominal parameter values (used to define the +-3 sigma sampling range).
    grain_size : float, optional
        Include grain size as a sensitivity dimension when provided.
    n_samples : int
        Base sample count *N*; total evaluations are N*(2D+2) where D is the
        number of dimensions.  Must be a power of two for Sobol sequences.
    seed : int, optional
        Random seed passed to the Sobol sampler.

    Returns
    -------
    dict with keys:
        parameters  - list of parameter names in index order
        S1          - dict mapping parameter name -> first-order index
        ST          - dict mapping parameter name -> total-order index
        S1_array    - raw numpy array of first-order indices
        ST_array    - raw numpy array of total-order indices
    """
    from scipy.stats.qmc import Sobol

    # Fixed uncertainty spec used for defining the sampling bounds.
    unc = UncertaintySpec()

    use_grain = grain_size is not None and grain_size > 0
    param_names: List[str] = ["conductivity", "permeability", "thickness", "frequency"]
    nominals    = [conductivity, permeability, thickness, frequency]
    cvs         = [unc.conductivity_cv, unc.permeability_cv, unc.thickness_cv, unc.frequency_cv]

    if use_grain:
        param_names.append("grain_size")
        nominals.append(grain_size)         # type: ignore[arg-type]
        cvs.append(unc.grain_size_cv)

    D = len(param_names)

    # Saltelli scheme requires 2*D columns in the quasi-random base matrix.
    # scipy.stats.qmc.Sobol generates in [0, 1]^(2D).
    sobol_engine = Sobol(d=2 * D, scramble=True, seed=seed)
    # n_samples must be a power of two for Sobol; use as-is and let scipy handle it.
    raw = sobol_engine.random(n=n_samples)   # shape (N, 2D)

    A = raw[:, :D]       # "A" sample matrix
    B = raw[:, D:]       # "B" sample matrix

    def _quantile_to_physical(col: np.ndarray, nominal: float, cv: float, lognorm_flag: bool) -> np.ndarray:
        """Convert uniform [0,1] quantiles to physical parameter values."""
        if cv == 0.0:
            return np.full(len(col), nominal)
        if lognorm_flag:
            from scipy.stats import lognorm
            sigma_ln = np.sqrt(np.log(1.0 + cv ** 2))
            mu_ln = np.log(nominal) - 0.5 * sigma_ln ** 2
            vals = lognorm.ppf(np.clip(col, 1e-10, 1 - 1e-10), s=sigma_ln, scale=np.exp(mu_ln))
        else:
            from scipy.stats import norm
            std = cv * abs(nominal)
            vals = norm.ppf(np.clip(col, 1e-10, 1 - 1e-10), loc=nominal, scale=std)
        return np.clip(vals, 1e-30, None)

    lognorm_flags = [False, False, False, False]
    if use_grain:
        lognorm_flags.append(True)

    def _matrix_to_physical(mat: np.ndarray) -> List[np.ndarray]:
        """Convert an (N, D) unit matrix to a list of D physical parameter arrays."""
        return [
            _quantile_to_physical(mat[:, j], nominals[j], cvs[j], lognorm_flags[j])
            for j in range(D)
        ]

    A_phys = _matrix_to_physical(A)
    B_phys = _matrix_to_physical(B)

    def _eval_matrix(phys: List[np.ndarray]) -> np.ndarray:
        """Evaluate SE for every row of a physical parameter list."""
        n = len(phys[0])
        out = np.empty(n)
        for i in range(n):
            gs_i: Optional[float] = None
            if use_grain:
                gs_i = float(phys[4][i])
            out[i] = _se_from_params(
                float(phys[0][i]),   # conductivity
                float(phys[1][i]),   # permeability
                permittivity,        # kept exact
                float(phys[2][i]),   # thickness
                float(phys[3][i]),   # frequency
                gs_i,
            )
        return out

    Y_A = _eval_matrix(A_phys)
    Y_B = _eval_matrix(B_phys)

    # Build AB_j matrices: A with column j replaced by the B column.
    # Saltelli (2010) estimators:
    #   S1_j  = (1/N) * sum(Y_B * (Y_AB_j - Y_A)) / Var(Y)
    #   ST_j  = (1/N) * sum((Y_A - Y_AB_j)^2 / 2)  / Var(Y)

    Y_all = np.concatenate([Y_A, Y_B])
    total_var = float(np.var(Y_all, ddof=1))

    # Guard against degenerate (zero variance) output.
    if total_var < 1e-20:
        zero_s1 = {p: 0.0 for p in param_names}
        zero_st = {p: 0.0 for p in param_names}
        return {
            "parameters": param_names,
            "S1": zero_s1,
            "ST": zero_st,
            "S1_array": np.zeros(D),
            "ST_array": np.zeros(D),
        }

    S1_arr = np.empty(D)
    ST_arr = np.empty(D)

    for j in range(D):
        # Build AB_j: copy of A with column j taken from B.
        AB_j_phys = [arr.copy() for arr in A_phys]
        AB_j_phys[j] = B_phys[j]

        Y_AB_j = _eval_matrix(AB_j_phys)

        # Saltelli (2010) estimators.
        S1_arr[j] = float(np.mean(Y_B * (Y_AB_j - Y_A))) / total_var
        ST_arr[j] = float(np.mean((Y_A - Y_AB_j) ** 2) / 2.0) / total_var

    # Clip to [0, 1] to remove small numerical artefacts.
    S1_arr = np.clip(S1_arr, 0.0, 1.0)
    ST_arr = np.clip(ST_arr, 0.0, 1.0)

    S1_dict = {name: float(S1_arr[i]) for i, name in enumerate(param_names)}
    ST_dict = {name: float(ST_arr[i]) for i, name in enumerate(param_names)}

    return {
        "parameters": param_names,
        "S1": S1_dict,
        "ST": ST_dict,
        "S1_array": S1_arr,
        "ST_array": ST_arr,
    }


def reliability_se(
    conductivity: float,
    permeability: float,
    permittivity: float,
    thickness: float,
    frequency: float,
    grain_size: Optional[float] = None,
    target_se: float = 40.0,
    uncertainty: Optional[UncertaintySpec] = None,
    n_samples: int = 1000,
    seed: Optional[int] = None,
) -> Dict:
    """Calculate the probability that SE exceeds a design target.

    Runs a Monte Carlo simulation and reports the empirical exceedance
    probability and the conservative (5th-percentile) reliable SE value.

    Parameters
    ----------
    conductivity, permeability, permittivity, thickness, frequency :
        Nominal parameter values.
    grain_size : float, optional
        Nominal grain size (m).
    target_se : float
        Required SE threshold (dB). Default 40 dB.
    uncertainty : UncertaintySpec, optional
        Parameter uncertainty specification.
    n_samples : int
        Monte Carlo sample count.
    seed : int, optional
        Random seed.

    Returns
    -------
    dict with keys:
        probability_exceeds_target - fraction of samples above target_se
        se_95_reliable             - 5th percentile SE (dB); the level
                                     exceeded in 95% of manufacturing runs
        se_mean                    - mean SE (dB)
        se_std                     - standard deviation (dB)
        target_se                  - the requested target (echoed back)
        n_samples                  - number of samples used
    """
    mc = monte_carlo_se(
        conductivity=conductivity,
        permeability=permeability,
        permittivity=permittivity,
        thickness=thickness,
        frequency=frequency,
        grain_size=grain_size,
        uncertainty=uncertainty,
        n_samples=n_samples,
        seed=seed,
    )

    se_dist = mc["se_distribution"]
    prob_exceeds = float(np.mean(se_dist >= target_se))
    se_5th = float(np.percentile(se_dist, 5.0))

    return {
        "probability_exceeds_target": prob_exceeds,
        "se_95_reliable": se_5th,
        "se_mean": mc["se_mean"],
        "se_std": mc["se_std"],
        "target_se": target_se,
        "n_samples": n_samples,
    }
