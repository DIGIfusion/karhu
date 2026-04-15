# KARHU

KARHU is a surrogate model for the ideal MHD peeling-ballooning stability code MISHKA. It is based on convolutional neural networks (CNNs) and trained on a large database of MISHKA results.

Publications:
- [A.M. Bruncrona et al., "Machine learning surrogate model for ideal peeling–ballooning pedestal MHD stability", Physics of Plasmas (2025).](https://doi.org/10.1063/5.0282085)


## Installation

KARHU requires Python 3.8 or higher.

### With `uv`


```bash
git clone git@github.com:DIGIfusion/karhu.git karhu && cd karhu

uv venv --python 3.8 .venv  # create a virtual environment 
source .venv/bin/activate   # activate virtual environment 
uv pip install .            # install the `karhu` package (and deps) into the virtual environment
```

Now you should be able to run `tests`.


### Without `uv`


```bash
git clone git@github.com:DIGIfusion/karhu.git karhu && cd karhu

venv --python 3.8 .venv     # create a virtual environment 
source .venv/bin/activate   # activate virtual environment 
pip install .               # install the `karhu` package (and deps) into the virtual environment
```

Now you should be able to run `tests`.


## Usage
See tests and examples in the `examples/` directory for usage instructions.


In `KARHU v1.x.x`, the inputs to the model are **ordered**, i.e., `forward(P, Q, RBPHI, ZBNDRY, BMAG, RMAG)`, where `P` is the pressure, `Q` is the safety factor and `RBPHI` is the poloidal current functions of $\psi$ defined on a uniform $\psi_N^2$ grid of `64` points. `ZBNDRY` is the Z-coordinates of the LCFS defined on a unifrom R-grid from `R_min, R_max` with also `64` points. Finally, `BMAG, RMAG` are the values of the toroidal field and major radius at the magnetic axis. 

Additionally, `P, RBPHI` are normalised such that: 

```python
KARHU_PRESSURE = PRESSURE_SI / (B_magaxis**2 / mu_0)
KARHU_RBPHI    = RBPHI_SI    / (R_magaxis * B_magaxis)
KARHU_RBDRY    = RBDRY_SI    / (radius * R_magaxis) - 1.0 / eps
KARHU_ZBDRY    = ZBDRY_SI    / (radius * R_magaxis) 
KARHU_Q        = Q

"""
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
"""
```

The data must further be scaled/normalised for machine learning purposes. **ADD MORE INFO HERE** 

The output of `KARHU`, once denormalised for ML purposes, is the growht rate $\gamma$ normalized by the Alfén frequency $\omega_A$. Comparing with the line `INSTABILITY = ...` in `MISHKA/fort.20` and `MISHKA/fort.22`, MISHKA outputs $\gamma^2$, so take the square root of the output of `MISHKA` to find the comparable $\gamma$.  



## Citation
If you use KARHU in your research, please cite:

```bibtex
@Misc{karhu-surrogate,
  title =        {KARHU: An ideal MHD stability surrogate model},
  author =       {A.M. Bruncrona, A. Kit, A. Järvinen},
  howpublished = {Github},
  year =         {2025},
  url =          {https://github.com/DIGIfusion/karhu}
}
```

```bibtex
@article{bruncrona2025machine,
  title={Machine learning surrogate model for ideal peeling--ballooning pedestal MHD stability},
  author={Bruncrona, A.M. and others},
  journal={Physics of Plasmas},
  volume={32},
  number={9},
  year={2025},
  publisher={AIP Publishing LLC}
}
```

## Acknowledgements

The development of this framework has been support by multiple funding sources:

- Research Council of Finland project number 355460.
- EUROfusion Consortium, funded by the European Union via the Euratom Research and Training Programme (Grant Agreement No 1010522200 - EUROfusion) through the Advanced Computing Hub framework of the E-TASC program as well as dedicated machine learning projects, such as the project focused on surrogating pedestal MHD stability models.
- Multiple CSC IT Center for Science projects have provided the necessary computing resources for the development and application of the framework.
