# KARHU - An ideal peeling-ballooning MHD stability surrogate model

KARHU is a machine learning surrogate model for the ideal peeling-ballooning MHD stability code MISHKA. This repository contains the surrogate model and helper functions for reading the necessary files. The training data will soon be available on Zenodo.


See example notebook: [run_inference.ipynb](example/run_inference.ipynb)

See publication: [https://doi.org/10.1063/5.0282085](https://doi.org/10.1063/5.0282085)


The training data is available in folder ´data´. Future datasets will be hosted on Zenodo.


In `v1`, `KARHU` takes as **ordered** inputs, i.e., `forward(P, Q, RBPHI, ZBNDRY, BMAG, RMAG)`, where `P, Q, RBPHI` are the pressure, q (safety factor) and poloidal current functions of $\psi$ defined on a uniform $\psi_N^2$ grid of `64` points. `ZBNDRY` is the Z-coordinates of the LCFS defined on a unifrom R-grid from `R_min, R_max` with also `64` points. Finally, `BMAG, RMAG` are the values of the toroidal field and major radius at the magnetic axis. 

Additionally, `P, RBPHI` are normalised such that: 

```
KARHU_PRESSURE = PRESSURE_SI / (B_magaxis**2 / mu_0)
KARHU_RBPHI    = RBPHI_SI    / (R_magaxis * B_magaxis)
KARHU_RBDRY    = RBDRY_SI    / (radius * R_magaxis) - 1.0 / eps
KARHU_ZBDRY    = ZBDRY_SI    / (radius * R_magaxis) 
KARHU_Q        = Q

where PRESSURE_SI [N / m^2] plasma pressure          (defined on PSIN)
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
```

The data must further be scaled/normalised for machine learning purposes. **ADD MORE INFO HERE** 

The output of `KARHU`, once denormalised for ML purposes, is $\gamma$, i.e., when comparing with the line `INSTABILITY = ...` in `MISHKA/fort.20` and `MISHKA/fort.22` has $\sqrt{\gamma}$, so take the square root of the output of `MISHKA`.  


## Instalation 

Easiest with the package manager [uv](https://docs.astral.sh/uv/getting-started/installation/), however, `uv` does not support python versions $\leq 3.8$. We use `uv` since it will manage dependencies in `pyproject.toml` for given python version, e.g., for `python==3.9` it finds `torch==2.8.0` and `numpy==2.0.2` while for `python==3.8` it finds `torch==2.4.1` and `numpy==1.24.4`. 

### With `uv`


0. `git clone git@github.com:DIGIfusion/karhu.git karhu && cd karhu`
1. `uv venv --python 3.8 .venv`  $\rightarrow$ create a virtual environment 
2. `source .venv/bin/activate`$\rightarrow$ activate virtual environment 
3. `uv pip install . ` $\rightarrow$ install the `karhu` package (and deps) into the virtual environment

Now you should be able to run `tests`



