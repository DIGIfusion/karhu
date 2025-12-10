import numpy as np 
from freeqdsk import geqdsk 
from karhu.common import convert_profiles_si_to_dimensionless 
from karhu.utils_input import interpolate_psi_profile
import torch 

def load_from_eqdsk(eqdskpath):
    with open(eqdskpath, "r") as f: 
        eqdsk = geqdsk.read(f)
    psin1d = np.linspace(0, 1.0, eqdsk.nx)
    R_mag  = eqdsk.rmagx 
    B_mag  = eqdsk.fpol[0] / eqdsk.rmagx 

    R_geom = (eqdsk.rbdry.max() + eqdsk.rbdry.min()) / 2.0
    a_geom = (eqdsk.rbdry.max() - eqdsk.rbdry.min()) / 2.0
    eps    = a_geom / R_geom 

    radius = eps * R_geom / R_mag
    pressure_karhu, rbphi_karhu, rbndry_karhu, zbndry_karhu = convert_profiles_si_to_dimensionless(eqdsk.pressure, eqdsk.fpol, eqdsk.rbdry, eqdsk.zbdry, 
                                                                                                   radius, R_mag, eps, B_mag, )
    q_karhu = eqdsk.qpsi 
    
    ninterp = 64
    KARHU_PSIN_AXIS = np.linspace(1e-5, 1.0, ninterp) ** 0.5
    KARHU_VX_AXIS   = np.linspace(-0.97, 0.97, ninterp)   # TODO/FIXME the interpolation axis is flawed here, since HELENA may not go to 0.999, 0.999... 

    pressure_karhu = interpolate_psi_profile(psin1d, pressure_karhu, KARHU_PSIN_AXIS)
    rbphi_karhu = interpolate_psi_profile(psin1d, rbphi_karhu, KARHU_PSIN_AXIS)
    q_karhu = interpolate_psi_profile(psin1d, q_karhu, KARHU_PSIN_AXIS)
    q_karhu = abs(q_karhu) # TODO/FIXME: Are the q-s normalised?  

    # FIXME: version 1.0 of the model only takes top half of the boundary
    reduced_bndry = zbndry_karhu > 0.0
    rbndry_top, zbndry_top = rbndry_karhu[reduced_bndry], zbndry_karhu[reduced_bndry]
    sorted_idx = np.argsort(rbndry_top)
    rbndry_karhu, zbndry_karhu = rbndry_top[sorted_idx], zbndry_top[sorted_idx]
    zbndry_karhu = interpolate_psi_profile(rbndry_karhu, zbndry_karhu, KARHU_VX_AXIS)

    x = [torch.tensor(pressure_karhu, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
         torch.tensor(q_karhu, dtype=torch.float32).unsqueeze(0).unsqueeze(0), 
         torch.tensor(rbphi_karhu, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
         torch.tensor(zbndry_karhu, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
         torch.tensor(B_mag, dtype=torch.float32).unsqueeze(0),
         torch.tensor(R_mag, dtype=torch.float32).unsqueeze(0),
    ]
    return x
