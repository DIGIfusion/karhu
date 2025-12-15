import numpy as np
import torch
from karhu.utils_input import scale_model_input, descale_minmax

MU_0 = 4E-7 * np.pi


def convert_profiles_si_to_dimensionless(pressure, rbphi, rbdry, zbdry,
                                         radius: float, R_mag: float, eps: float, B_mag: float):
    """
    KARHU/MISHKA dimensionless normalisation from SI units
    KARHU_PRESSURE = PRESSURE_SI / (B_magaxis**2 / MU_0)
    KARHU_RBPHI    = RBPHI_SI    / (R_magaxis * B_magaxis)
    KARHU_RBDRY    = RBDRY_SI    / (radius * R_magaxis) - 1.0 / eps
    KARHU_ZBDRY    = ZBDRY_SI    / (radius * R_magaxis)
    KARHU_Q        = Q

    where
        PRESSURE_SI [N / m^2] plasma pressure          (defined on PSIN)
        RBPHI_SI    [Tm]      poloidal flux function   (defined on PSIN)
        Q           []        q-profile, safety factor (defined on PSIN)
        RBDRY_SI    [m]       R/X coordinates of plasma boundary
        ZBDRY_SI    [m]       Z/Y coordinates of plasma boundary
        B_magaxis   [T]       magnetic field strength at the magnetic axis
        R_magaxis   [m]       major radius at the magnetic axis
        eps         [-]       inverse aspect ratio of the plasma boundary
        radius      [-]       dimensionless value, used to scale R/X boundary between -1 and 1
                                is equal to eps * R_geom / R_magaxis,
                                where R_geom is the geometric axis
    """
    pressure_karhu = pressure / (B_mag**2 / MU_0)
    rbphi_karhu    = rbphi     / (R_mag * B_mag)
    rbndry_karhu   = rbdry    / (radius * R_mag) - 1.0 / eps
    zbndry_karhu   = zbdry    / (radius * R_mag)
    return pressure_karhu, rbphi_karhu, rbndry_karhu, zbndry_karhu


def forward_pass(x: list[torch.tensor], model, scaling_params) -> torch.tensor:
    x = scale_model_input(x, scaling_params)
    with torch.no_grad():
        y_pred = model(*x)
    y_pred = descale_minmax(y_pred.item(), *scaling_params["growthrate"])
    return y_pred


def get_polar_from_rz(r_vals, z_vals, r0=0.0, z0=0.0, amin=1.0, symmetric=False):
    """
    Convert (R, Z) boundary coordinates to polar coordinates (rho, theta)
    relative to the boundary center (r0, z0).

    Handles both symmetric and asymmetric boundaries:
    - If symmetric (self.symmetric=True): input contains only the top half,
        and the function mirrors it to produce a full 0-2pi contour.
    - If asymmetric: uses the full input directly.

    Parameters
    ----------
    r_vals : array_like
        R (major radius) coordinates of the boundary.
    z_vals : array_like
        Z (vertical) coordinates of the boundary.

    Returns
    -------
    rho : ndarray
        Normalized radius rho = sqrt((R - r0)^2 + (Z - z0)^2) / a_min
    theta : ndarray
        Poloidal angle theta in radians, from 0 to 2pi.
    """
    r_vals = np.asarray(r_vals)
    z_vals = np.asarray(z_vals)

    # Mirror the top half if the plasma is symmetric
    if symmetric:
        # Mirror across the midplane (z0)
        r_mirror = np.copy(r_vals[::-1])
        z_mirror = 2 * z0 - z_vals[::-1]

        # Combine top (input) and mirrored bottom
        r_vals = np.concatenate([r_vals, r_mirror[1:]])  # avoid duplicate at midplane
        z_vals = np.concatenate([z_vals, z_mirror[1:]])

    # Compute normalized radius and poloidal angle
    rho = np.sqrt((r_vals - r0)**2 + (z_vals - z0)**2) / amin
    theta = np.arctan2(z_vals - z0, r_vals - r0)

    # Convert theta range from (-pi, pi] → [0, 2pi)
    theta = np.mod(theta, 2 * np.pi)

    # Sort points by increasing theta to ensure continuous boundary
    order = np.argsort(theta)
    return rho[order], theta[order]


def get_rz_from_fourier(realfour, imagfour, r0=0.0, z0=0.0, amin=1.0):
    """
    Reconstruct the boundary (R(theta), Z(theta)) using Fourier coefficients.

    r(theta) = a0/2 + Σ [a_m cos(mtheta) + b_m sin(mtheta)],  m = 1..N/2
    R(theta) = r0 + a_min * r(theta) * cos(theta)
    Z(theta) = z0 + a_min * r(theta) * sin(theta)

    A high-resolution uniform grid in theta ∈ [0, 2pi] is used to evaluate
    the boundary.

    Parameters
    ----------
    realfour : array_like
        Real Fourier coefficients (a_m). The first element corresponds
        to the m = 0 mode.
    imagfour : array_like
        Imaginary Fourier coefficients (b_m).
    r0 : float, optional
        Radial offset of the boundary center. Default is 0.0.
    z0 : float, optional
        Vertical offset of the boundary center. Default is 0.0.
    amin : float, optional
        Minor radius. Scaling factor applied to the reconstructed radial function.
        Default is 1.0.

    Returns
    -------
    rf : ndarray
        Radial coordinate R(θ) evaluated on a fine θ grid.
    zf : ndarray
        Vertical coordinate Z(θ) evaluated on a fine θ grid.
    """

    # High-resolution grid
    thetafine = np.arange(2048) / (2048 - 1.0) * 2.0 * np.pi

    nf = len(realfour) * 2
    nph = np.matrix(range(int(nf / 2)))
    theta = np.matrix(thetafine)

    # Compute cosine and sine terms
    cm = np.cos(nph.transpose() * theta)
    sm = np.sin(nph.transpose() * theta)

    # Adjust a0 term
    realf = np.matrix(realfour)
    realf[0, 0] = realf[0, 0] / 2
    imagf = np.matrix(imagfour)

    # Reconstruct normalized radius
    rad = amin * (realf * cm + imagf * sm)

    # Convert back to Cartesian
    rf = r0 + np.asarray(rad) * np.cos(thetafine)
    zf = z0 + np.asarray(rad) * np.sin(thetafine)
    
    return rf, zf


def get_polar_from_fourier(realfour, imagfour):
    """
    Convert Fourier coefficients to polar coordinates.

    This function computes the real-space components from the given
    real and imaginary Fourier coefficients, then converts those
    components into polar coordinates.

    Parameters
    ----------
    realfour : array_like
        Real part of the Fourier coefficients.
    imagfour : array_like
        Imaginary part of the Fourier coefficients.

    Returns
    -------
    rho : array_like
        Radial magnitude corresponding to the Fourier components.
    theta : array_like
        Angular coordinate (in radians) corresponding to the Fourier components.
    """
    r, z = get_rz_from_fourier(realfour, imagfour)
    rho, theta = get_polar_from_rz(r, z)
    return rho, theta


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return(rho, theta)


def pol2cart(rho, theta):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return(x, y)
