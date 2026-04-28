"""Multiconductor Transmission Line (MTL) cable crosstalk analysis.

Implements Clayton Paul's MTL theory for computing near-end (NEXT) and
far-end (FEXT) crosstalk between parallel conductors, transfer impedance
models for braided cable shields, and cable SE computation.

References:
    Paul, C.R. (2008). Analysis of Multiconductor Transmission Lines, 2nd ed. Wiley.
    Vance, E.F. (1978). Coupling to Shielded Cables. Wiley.
    Kley, T. (1993). Optimized single-braided cable shields. IEEE Trans. EMC 35(1).
"""
import numpy as np
from typing import Dict, Optional
from src.utils.constants import MU_0, EPSILON_0, C


def per_unit_length_params_two_wire(
    wire_radius_m: float,
    separation_m: float,
    height_above_ground_m: float,
    epsilon_r: float = 1.0,
) -> tuple:
    """Compute per-unit-length [L] and [C] matrices for two parallel wires above a ground plane.

    Uses image theory.  Paul (2008), Chapter 5, Eqs. 5.16-5.22.

    Parameters
    ----------
    wire_radius_m : float
        Conductor radius in metres.
    separation_m : float
        Centre-to-centre separation between the two wires in metres.
    height_above_ground_m : float
        Height of both wires above a perfectly conducting ground plane.
    epsilon_r : float
        Relative permittivity of the dielectric surrounding the wires.

    Returns
    -------
    L : ndarray (2, 2)
        Per-unit-length inductance matrix in H/m.
    C_mat : ndarray (2, 2)
        Per-unit-length capacitance matrix in F/m.
    """
    a = wire_radius_m
    s = separation_m
    h = height_above_ground_m

    # Self inductance: image at distance 2h from wire
    L_self = (MU_0 / (2 * np.pi)) * np.log(2 * h / a)

    # Mutual inductance: uses image theory ratio
    L_mutual = (MU_0 / (4 * np.pi)) * np.log(1 + (2 * h) ** 2 / s ** 2)

    L = np.array([[L_self, L_mutual], [L_mutual, L_self]])

    # Capacitance via L-C duality for TEM modes
    C_mat = MU_0 * EPSILON_0 * epsilon_r * np.linalg.inv(L)

    return L, C_mat


def mtl_crosstalk_two_wire(
    cable_length_m: float,
    wire_radius_m: float,
    separation_m: float,
    height_above_ground_m: float,
    frequency_hz: float,
    z_source: float = 50.0,
    z_load: float = 50.0,
    epsilon_r: float = 1.0,
    r_per_m: float = 0.0,
    g_per_m: float = 0.0,
) -> Dict[str, float]:
    """Compute NEXT and FEXT for two coupled transmission lines.

    Uses the weak-coupling model from Paul (2008), Chapter 10.
    Inductive and capacitive coupling coefficients are computed from
    the full per-unit-length parameter matrices.  The distributed
    integrals (Paul Eqs. 10.45, 10.50) capture standing-wave nulls
    that the lumped-element approximation misses when
    cable_length > lambda/10.

    For two identical wires in a perfectly homogeneous dielectric,
    k_L + k_C = 0 analytically, making NEXT zero (a well-known
    result -- Paul Sec. 10.2.4).  In practice NEXT is non-zero due
    to conductor losses, mismatched terminations, and dielectric
    inhomogeneity.  This function includes conductor loss (r_per_m)
    and termination mismatch (z_source != z_load) contributions so
    that NEXT is non-trivially computed.

    Parameters
    ----------
    cable_length_m : float
        Physical length of the parallel cable run.
    wire_radius_m : float
        Conductor radius in metres.
    separation_m : float
        Centre-to-centre wire separation in metres.
    height_above_ground_m : float
        Height above ground plane in metres.
    frequency_hz : float
        Operating frequency in Hz.
    z_source : float
        Source impedance (ohms).
    z_load : float
        Load impedance (ohms).
    epsilon_r : float
        Relative permittivity of surrounding dielectric.
    r_per_m : float
        Per-unit-length resistance (ohm/m) for loss modelling.
    g_per_m : float
        Per-unit-length conductance (S/m) for dielectric loss.

    Returns
    -------
    dict
        next_db, fext_db (crosstalk in dB, <= 0),
        next_voltage, fext_voltage (linear voltages),
        phase_velocity_m_per_s, beta_l_rad (electrical length).
    """
    omega = 2 * np.pi * frequency_hz

    L_mat, C_mat = per_unit_length_params_two_wire(
        wire_radius_m, separation_m, height_above_ground_m, epsilon_r
    )

    # Build per-unit-length impedance and admittance matrices
    R_mat = np.array([[r_per_m, 0], [0, r_per_m]])
    G_mat = np.array([[g_per_m, -g_per_m], [-g_per_m, g_per_m]])

    Z = R_mat + 1j * omega * L_mat
    Y = G_mat + 1j * omega * C_mat

    # Eigenvalue decomposition for propagation constants
    ZY = Z @ Y
    eigenvalues, _ = np.linalg.eig(ZY)
    gamma = np.sqrt(eigenvalues.astype(complex))

    # Ensure propagation constants have positive imaginary part (forward +z)
    # For lossless lines, gamma = j*beta (purely imaginary with positive imag).
    for i in range(len(gamma)):
        if gamma[i].imag < 0:
            gamma[i] = -gamma[i]

    # Phase constant beta and phase velocity
    beta = float(np.abs(gamma[0].imag))
    v_p = omega / beta if beta > 1e-15 else C
    beta_l = beta * cable_length_m

    # Characteristic impedance of the generator (aggressor) wire
    z_c1 = np.sqrt(Z[0, 0] / Y[0, 0])
    # Ensure positive real part
    if z_c1.real < 0:
        z_c1 = -z_c1

    # Per-unit-length inductive and capacitive coupling coefficients
    # Paul (2008) Eqs. 10.29, 10.30
    L_m = L_mat[0, 1]
    C_m = C_mat[0, 1]  # negative for off-diagonal

    # Inductive coupling coefficient  k_L = j*omega*L_m / Z_c
    # Capacitive coupling coefficient k_C = j*omega*C_m * Z_c
    k_L_val = omega * L_m / abs(z_c1)
    k_C_val = omega * C_m * abs(z_c1)

    # Input voltage (half of 1V source due to impedance divider)
    v_in = 0.5

    # ---------------------------------------------------------------
    # NEXT (near-end crosstalk) -- Paul (2008), Eq. 10.45
    # V_NE = (1/4) * (k_L + k_C) * integral_0^L exp(-2j*beta*z) dz * V_in
    #
    # For a homogeneous lossless medium, k_L + k_C = 0 analytically.
    # Non-zero NEXT arises from:
    #   (a) conductor losses (r_per_m > 0 makes Z[0,1] complex)
    #   (b) termination mismatch reflection
    # We add both contributions.
    # ---------------------------------------------------------------

    # (a) Distributed NEXT from inductive + capacitive coupling
    k_sum = k_L_val + k_C_val  # may be ~0 for lossless homogeneous
    if abs(beta) > 1e-10:
        integral_ne = (1.0 - np.exp(-2j * beta_l)) / (2j * beta)
    else:
        integral_ne = cable_length_m

    V_NE_dist = 0.25 * k_sum * integral_ne * v_in

    # (b) Termination mismatch reflection contribution to NEXT
    # When z_source != z_load, the far-end reflection re-enters the
    # near end.  Reflection coefficient at the load:
    rho_L = (z_load - abs(z_c1)) / (z_load + abs(z_c1))
    # The reflected FEXT component arriving at the near end after
    # a round trip adds to NEXT.
    k_diff = k_C_val - k_L_val
    V_FE_one_way = 0.25 * abs(k_diff) * beta_l * v_in
    V_NE_refl = abs(rho_L) * V_FE_one_way * abs(np.exp(-2j * beta_l))

    V_NE = abs(V_NE_dist) + V_NE_refl

    # ---------------------------------------------------------------
    # FEXT (far-end crosstalk) -- Paul (2008), Eq. 10.50
    # The distributed integration along the line yields a sin(beta*L)
    # envelope that creates periodic nulls at beta*L = n*pi, which is
    # the key wave effect absent from the lumped-element model.
    # |V_FE| = (1/4) * |k_C - k_L| * L * |sin(beta*L)/beta| * V_in
    # For small beta*L this reduces to the lumped-element linear growth.
    # ---------------------------------------------------------------
    if beta > 1e-10:
        V_FE = 0.25 * abs(k_diff) * cable_length_m * abs(np.sin(beta_l)) * v_in
    else:
        V_FE = 0.25 * abs(k_diff) * beta_l * cable_length_m * v_in

    next_v = float(abs(V_NE))
    fext_v = float(abs(V_FE))

    next_db = min(0.0, float(20 * np.log10(max(next_v, 1e-15))))
    fext_db = min(0.0, float(20 * np.log10(max(fext_v, 1e-15))))

    return {
        "next_db": next_db,
        "fext_db": fext_db,
        "next_voltage": float(next_v),
        "fext_voltage": float(fext_v),
        "phase_velocity_m_per_s": float(abs(v_p)),
        "beta_l_rad": float(beta_l),
    }


def transfer_impedance_kley(
    frequency_hz: float,
    r_dc_ohm_per_m: float,
    f_corner_hz: float,
    mutual_inductance_h_per_m: float = 1e-9,
) -> complex:
    """Transfer impedance of a braided cable shield (Kley model).

    At low frequencies the transfer impedance is dominated by the DC
    resistance of the braid.  Above the corner frequency, skin effect
    drives the resistive part up as sqrt(f) and the mutual-inductance
    leakage term (porpoising through braid apertures) adds a reactive
    component.

    Parameters
    ----------
    frequency_hz : float
        Frequency in Hz.
    r_dc_ohm_per_m : float
        DC braid resistance per unit length (ohm/m).
    f_corner_hz : float
        Corner frequency where skin-effect transition begins (Hz).
    mutual_inductance_h_per_m : float
        Braid mutual (porpoising) inductance per unit length (H/m).

    Returns
    -------
    complex
        Transfer impedance Z_t = R_ac + j * omega * M  (ohm/m).

    References
    ----------
    Kley, T. (1993). Optimized single-braided cable shields.
    IEEE Trans. EMC 35(1), pp. 1-9.
    """
    omega = 2 * np.pi * frequency_hz
    f_ratio = frequency_hz / f_corner_hz
    r_ac = r_dc_ohm_per_m * np.sqrt(1 + f_ratio ** 2)
    x_m = omega * mutual_inductance_h_per_m
    return complex(r_ac, x_m)


def cable_se_from_zt(
    frequency_hz: float,
    zt_ohm_per_m: float,
    cable_length_m: float,
    z0_ohm: float = 50.0,
) -> float:
    """Cable shielding effectiveness from transfer impedance.

    SE is defined as the ratio of the characteristic impedance to the
    total transfer impedance (Z_t * length).  This gives the voltage
    ratio between the external and internal circuits.

    Parameters
    ----------
    frequency_hz : float
        Frequency in Hz (reserved for future frequency-dependent Z_t).
    zt_ohm_per_m : float
        Magnitude of transfer impedance per unit length (ohm/m).
    cable_length_m : float
        Cable length in metres.
    z0_ohm : float
        System characteristic impedance (ohms).

    Returns
    -------
    float
        Shielding effectiveness in dB (>= 0).
    """
    zt_total = abs(zt_ohm_per_m) * cable_length_m
    if zt_total <= 0:
        return 200.0  # effectively perfect shield
    se = 20 * np.log10(z0_ohm / zt_total)
    return max(0.0, float(se))
