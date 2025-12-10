import numpy as np 

mu_0 = 4E-7 * np.pi 
def convert_profiles_si_to_dimensionless(pressure, rbphi, rbdry, zbdry, 
                                         radius: float, R_mag: float, eps: float, B_mag: float): 
    """
    KARHU/MISHKA dimensionless normalisation from SI units
    KARHU_PRESSURE = PRESSURE_SI / (B_magaxis**2 / mu_0)
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
    pressure_karhu = pressure / (B_mag**2 / mu_0)
    rbphi_karhu    = rbphi     / (R_mag * B_mag)
    rbndry_karhu   = rbdry    / (radius * R_mag) - 1.0 / eps
    zbndry_karhu   = zbdry    / (radius * R_mag)
    return pressure_karhu, rbphi_karhu, rbndry_karhu, zbndry_karhu

import torch 
from karhu.utils_input import scale_model_input, descale_minmax
def forward_pass(x: list[torch.tensor], model, scaling_params) -> torch.tensor: 
    x = scale_model_input(x, scaling_params)
    with torch.no_grad(): 
        y_pred = model(*x)
    y_pred = descale_minmax(y_pred.item(), *scaling_params["growthrate"])
    return y_pred