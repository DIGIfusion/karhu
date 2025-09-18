### v1.0.0

The `cs.npy` file contains the $\Psi$ grid onto which the pressure, safety factor, diamagnetic function ahve all been projected. The $\Psi$ grid consists 64 points that are equidistant in $\Psi^2$.

The `p0.npy` file contains the equilibrium pressure profile.

The `qs.npy` file contains the safety factor profile.

The `rbphi.npy` file contains the diamagnetic function profile.

The `vxvy.npy` file contains the the upper half of the plasma shape.

The `meta.npy` file contains the following columns:
- the maximum growth rate
- the toroidal mode number corresponding to the maximum growth rate
- the pedestal width
- the geometric radius (R_vac)
- the magnetic field at the geometric radius (B_vac)
- the magnetic radius (R_mag)
- the magnetic field at the magnetic radius (B_mag)