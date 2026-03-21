"""Transfer Matrix Method for multilayer EMI shielding.

Computes shielding effectiveness for arbitrary N-layer material stacks
using the standard 2x2 transfer matrix cascade approach.

References:
    Celozzi, Araneo, Lovat (2008). Electromagnetic Shielding. Wiley.
    Yeh (1988). Optical Waves in Layered Media. Wiley.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from src.utils.constants import MU_0, EPSILON_0, Z_0


@dataclass
class ShieldLayer:
    """A single layer in a multilayer shield stack."""
    conductivity: float          # S/m
    relative_permeability: float
    relative_permittivity: float
    thickness: float             # meters
    name: str = ""


class MultilayerShield:
    """Computes SE for an N-layer shield using the Transfer Matrix Method.

    The cascade product of individual layer transfer matrices gives the
    overall ABCD matrix that relates the fields at the input face to the
    fields at the output face.  The S21 transmission coefficient is then
    extracted assuming free-space terminations on both sides (impedance Z_0).

    Usage::

        shield = MultilayerShield()
        shield.add_layer(ShieldLayer(conductivity=5.96e7, relative_permeability=1.0,
                                     relative_permittivity=1.0, thickness=0.001,
                                     name="copper"))
        result = shield.calculate_se(frequency=1e9)
        print(result['total_se'])  # dB
    """

    def __init__(self):
        self.layers: List[ShieldLayer] = []

    def add_layer(self, layer: ShieldLayer) -> None:
        """Append a layer to the end of the stack (closest to the source last)."""
        self.layers.append(layer)

    def clear_layers(self) -> None:
        """Remove all layers from the stack."""
        self.layers.clear()

    # ------------------------------------------------------------------
    # Core TMM machinery
    # ------------------------------------------------------------------

    def _layer_transfer_matrix(self, layer: ShieldLayer, frequency: float) -> np.ndarray:
        """Compute the 2x2 ABCD transfer matrix for a single homogeneous layer.

        The matrix relates (E, H) fields at the entry face to (E, H) fields at
        the exit face under the assumption of a plane wave at normal incidence:

            [E_in ]   [cosh(gd)       eta*sinh(gd)] [E_out]
            [H_in ] = [sinh(gd)/eta   cosh(gd)    ] [H_out]

        For good conductors the real part of ``gd`` can exceed ~700, causing
        cosh/sinh to overflow float64.  In that regime the exponential terms
        ``exp(+gd)`` dominate and the matrix entries simplify to:

            cosh(gd) ~ sinh(gd) ~ exp(gd) / 2

        so the matrix is replaced by its asymptotic form, which is exact in the
        limit ``real(gd) >> 1`` (the wave decays almost entirely inside the
        layer).

        Args:
            layer: Material layer parameters.
            frequency: Frequency in Hz.

        Returns:
            2x2 complex numpy array (the ABCD transfer matrix).
        """
        omega = 2.0 * np.pi * frequency

        mu = layer.relative_permeability * MU_0

        # Complex permittivity accounts for conduction losses
        eps_complex = (layer.relative_permittivity * EPSILON_0
                       - 1j * layer.conductivity / omega)

        # Propagation constant  gamma = j*omega*sqrt(mu * eps_c)
        gamma = 1j * omega * np.sqrt(mu * eps_complex)

        # Wave impedance inside the layer
        eta = np.sqrt(1j * omega * mu
                      / (layer.conductivity + 1j * omega * layer.relative_permittivity * EPSILON_0))

        gd = gamma * layer.thickness

        # Float64 cosh/sinh overflow when real(gd) > ~709.
        # This occurs for good conductors at high frequencies or large thickness
        # (e.g. copper at 10 GHz with 1 mm -- ~1500 skin depths).
        #
        # Asymptotic treatment: when real(gd) >> 1,
        #   cosh(gd) ~ exp(gd) / 2  and  sinh(gd) ~ exp(gd) / 2.
        # We work in a numerically scaled frame.  Define E = exp(-real(gd))
        # as the attenuation factor and write:
        #   cosh(gd) = cosh_scaled / E   where cosh_scaled = cosh(gd) * E
        #
        # Substituting into the matrix and tracking the overall scale factor
        # lets us compute S21 without ever materialising the huge values.
        #
        # Concretely:
        #   M_scaled = diag(E, E) @ M @ diag(1, 1)
        # which equals:
        #   [[E*cosh(gd),    E*eta*sinh(gd)],
        #    [E*sinh(gd)/eta, E*cosh(gd)   ]]
        # All entries remain O(1) because E * cosh(gd) ~ 0.5 in the asymptotic.
        # We propagate this scale through the cascade by accumulating log_scale.

        # Threshold chosen so individual matrix entries stay below ~e^200 ~ 10^87,
        # keeping cascades of many layers within float64 (max ~10^308).
        _OVERFLOW_THRESHOLD = 200.0
        alpha_d = gd.real  # real part drives the overflow

        if alpha_d > _OVERFLOW_THRESHOLD:
            # Work with E*cosh(gd) and E*sinh(gd) where E = exp(-alpha_d).
            # exp(gd) = exp(alpha_d + j*imag_d).
            # E * exp(gd) = exp(j*imag_d)  (magnitude 1).
            # E * exp(-gd) = exp(-j*imag_d) (magnitude 1).
            # E * cosh(gd) = (exp(j*beta_d) + exp(-j*beta_d)) / 2 = cos(beta_d)
            # E * sinh(gd) = (exp(j*beta_d) - exp(-j*beta_d)) / 2 = j*sin(beta_d)
            beta_d = gd.imag
            scaled_cosh = np.cos(beta_d)          # E * cosh(gd)
            scaled_sinh = 1j * np.sin(beta_d)     # E * sinh(gd)

            # The true matrix is (1/E) * M_scaled.  We record the log of the
            # scale factor on the instance so _cascade_matrices can apply it.
            # For a product of N matrices each contributing scale 1/E_k:
            #   M_total = (prod 1/E_k) * M_scaled_total
            # so S21 gains a factor of prod(E_k) relative to the scaled product.
            self._log_scale += alpha_d   # accumulate log(1/E) = alpha_d

            M = np.array([
                [scaled_cosh,            eta * scaled_sinh],
                [scaled_sinh / eta,      scaled_cosh      ],
            ], dtype=complex)
        else:
            M = np.array([
                [np.cosh(gd),        eta * np.sinh(gd)],
                [np.sinh(gd) / eta,  np.cosh(gd)      ],
            ], dtype=complex)

        return M

    def _cascade_matrices(self) -> Tuple[np.ndarray, float]:
        """Return the cascade product of all layer matrices plus the accumulated log-scale.

        When any layer triggers the overflow-safe path in ``_layer_transfer_matrix``,
        its transfer matrix is stored in a scaled form where each entry is
        multiplied by ``E = exp(-alpha_d)``.  The corresponding ``log(1/E) = alpha_d``
        is accumulated in ``self._log_scale``.

        Returns:
            Tuple of (M_scaled_total, log_scale) where the physical total matrix
            equals ``M_scaled_total * exp(log_scale)``.
        """
        self._log_scale = 0.0
        M_total = np.eye(2, dtype=complex)
        for layer in self.layers:
            M_total = M_total @ self._layer_transfer_matrix(layer, self._current_frequency)
        return M_total, self._log_scale

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def calculate_se(self, frequency: float) -> Dict[str, float]:
        """Calculate total shielding effectiveness at a single frequency.

        Assumes a plane wave in free space incident on the first layer and
        free-space termination at the exit of the last layer.

        Args:
            frequency: Frequency in Hz (must be positive).

        Returns:
            Dictionary with keys:
                ``total_se``               -- total SE in dB (positive = attenuated)
                ``reflection_loss``        -- reflection contribution in dB
                ``absorption_loss``        -- absorption contribution in dB
                ``transmission_coefficient`` -- |S21| (linear)
                ``reflection_coefficient`` -- |S11| (linear)
        """
        if frequency <= 0:
            raise ValueError("Frequency must be positive")

        if not self.layers:
            return {
                'total_se': 0.0,
                'reflection_loss': 0.0,
                'absorption_loss': 0.0,
                'transmission_coefficient': 1.0,
                'reflection_coefficient': 0.0,
            }

        # Store frequency so _cascade_matrices can access it without changing
        # the public API of _layer_transfer_matrix.
        self._current_frequency = frequency

        M_scaled, log_scale = self._cascade_matrices()
        A, B, C, D = M_scaled[0, 0], M_scaled[0, 1], M_scaled[1, 0], M_scaled[1, 1]

        # S-parameter extraction (free-space ports, Z_ref = Z_0):
        #   S21 = 2 / (A + B/Z_0 + C*Z_0 + D)
        #   S11 = (A + B/Z_0 - C*Z_0 - D) / (A + B/Z_0 + C*Z_0 + D)
        #
        # When log_scale > 0 the stored matrix M_scaled represents the physical
        # matrix divided by exp(log_scale).  Since S21 is 2 / (sum of ABCD
        # products), and all ABCD entries are scaled equally by 1/exp(log_scale),
        # the denominator is also divided by exp(log_scale), giving:
        #   S21_physical = S21_scaled / exp(log_scale)
        # i.e.  |S21_physical| = |S21_scaled| * exp(-log_scale).
        # In log space: log10(|S21|) = log10(|S21_scaled|) - log_scale/ln(10).
        denom = A + B / Z_0 + C * Z_0 + D
        S21_scaled = 2.0 / denom
        S11 = (A + B / Z_0 - C * Z_0 - D) / denom

        abs_S21_scaled = float(np.abs(S21_scaled))
        abs_S11 = float(np.abs(S11))

        # SE in dB, computed in log domain to avoid underflow/overflow.
        # |S21_physical| = |S21_scaled| * exp(-log_scale)
        # SE = -20 * log10(|S21_physical|)
        #    = -20 * log10(|S21_scaled|) + 20 * log_scale * log10(e)
        _log10e = np.log10(np.e)  # ~0.4343
        if abs_S21_scaled > 0.0:
            total_se = float(
                -20.0 * np.log10(abs_S21_scaled) + 20.0 * log_scale * _log10e
            )
        else:
            total_se = float('inf')

        # Physical |S21| for reporting; guard against underflow
        if log_scale > 700.0:
            abs_S21 = 0.0
        else:
            abs_S21 = float(abs_S21_scaled * np.exp(-log_scale))

        # Reflection loss from power not entering the shield
        s11_sq = abs_S11 ** 2
        if s11_sq < 1.0:
            reflection_loss = float(-10.0 * np.log10(1.0 - s11_sq))
        else:
            reflection_loss = 0.0

        absorption_loss = max(0.0, total_se - reflection_loss)

        return {
            'total_se': total_se,
            'reflection_loss': reflection_loss,
            'absorption_loss': absorption_loss,
            'transmission_coefficient': abs_S21,
            'reflection_coefficient': float(abs_S11),
        }

    def frequency_sweep(self, freq_start: float = 1e6, freq_end: float = 10e9,
                        num_points: int = 100) -> Dict[str, np.ndarray]:
        """Compute SE across a logarithmically spaced frequency range.

        Args:
            freq_start: Start frequency in Hz (default 1 MHz).
            freq_end: End frequency in Hz (default 10 GHz).
            num_points: Number of frequency points (default 100).

        Returns:
            Dictionary with numpy arrays:
                ``frequencies``      -- Hz
                ``total_ses``        -- dB
                ``reflection_losses``-- dB
                ``absorption_losses``-- dB
        """
        if freq_start <= 0 or freq_end <= freq_start:
            raise ValueError("freq_start must be positive and less than freq_end")
        if num_points < 1:
            raise ValueError("num_points must be >= 1")

        frequencies = np.logspace(np.log10(freq_start), np.log10(freq_end), num_points)
        total_ses = np.zeros(num_points)
        reflection_losses = np.zeros(num_points)
        absorption_losses = np.zeros(num_points)

        for i, freq in enumerate(frequencies):
            result = self.calculate_se(float(freq))
            total_ses[i] = result['total_se']
            reflection_losses[i] = result['reflection_loss']
            absorption_losses[i] = result['absorption_loss']

        return {
            'frequencies': frequencies,
            'total_ses': total_ses,
            'reflection_losses': reflection_losses,
            'absorption_losses': absorption_losses,
        }

    def optimize_layer_thicknesses(self, frequency: float, target_se: float,
                                   max_total_thickness: float = 0.01) -> Dict:
        """Optimize individual layer thicknesses to reach a target SE.

        Uses a simple proportional scaling of all thicknesses together.
        For independent per-layer optimization use scipy.optimize.minimize
        directly with the ``calculate_se`` method.

        Args:
            frequency: Frequency at which to meet the target (Hz).
            target_se: Desired shielding effectiveness (dB).
            max_total_thickness: Not currently enforced; reserved for future
                constrained optimization (meters).

        Returns:
            Dictionary with keys:
                ``optimal_thicknesses`` -- list of per-layer thicknesses (m)
                ``total_thickness``     -- sum of all layer thicknesses (m)
                ``achieved_se``         -- SE at the optimal configuration (dB)
                ``target_se``           -- the requested target (dB)
                ``scale_factor``        -- multiplier applied to original thicknesses
        """
        from scipy.optimize import minimize_scalar

        if not self.layers:
            raise ValueError("No layers have been added to the shield")

        original_thicknesses = [layer.thickness for layer in self.layers]

        def objective(scale: float) -> float:
            for i, layer in enumerate(self.layers):
                layer.thickness = original_thicknesses[i] * scale
            result = self.calculate_se(frequency)
            return abs(result['total_se'] - target_se)

        opt = minimize_scalar(objective, bounds=(0.01, 100.0), method='bounded')

        optimal_thicknesses = [t * opt.x for t in original_thicknesses]
        total_thickness = sum(optimal_thicknesses)

        # Apply optimal thicknesses and get the final result
        for i, layer in enumerate(self.layers):
            layer.thickness = optimal_thicknesses[i]

        final_result = self.calculate_se(frequency)

        return {
            'optimal_thicknesses': optimal_thicknesses,
            'total_thickness': total_thickness,
            'achieved_se': final_result['total_se'],
            'target_se': target_se,
            'scale_factor': float(opt.x),
        }
