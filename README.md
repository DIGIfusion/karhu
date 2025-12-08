# KARHU - An ideal peeling-ballooning MHD stability surrogate model

KARHU is a machine learning surrogate model for the ideal peeling-ballooning MHD stability code MISHKA. This repository contains the surrogate model and helper functions for reading the necessary files. The training data will soon be available on Zenodo.


See example notebook: [run_inference.ipynb](example/run_inference.ipynb)

See publication: [https://doi.org/10.1063/5.0282085](https://doi.org/10.1063/5.0282085)


The training data is available in folder ´data´. Future datasets will be hosted on Zenodo.


In `v1`, `KARHU` takes as **ordered** inputs: ...

## Instalation 

Easiest with the package manager [uv](https://docs.astral.sh/uv/getting-started/installation/), however, `uv` does not support python versions $\leq 3.8$. We use `uv` since it will manage dependencies in `pyproject.toml` for given python version, e.g., for `python==3.9` it finds `torch==2.8.0` and `numpy==2.0.2` while for `python==3.8` it finds `torch==2.4.1` and `numpy==1.24.4`. 

### With `uv`


0. `git clone git@github.com:DIGIfusion/karhu.git karhu && cd karhu`
1. `uv venv --python 3.8 .venv`  $\rightarrow$ create a virtual environment 
2. `source .venv/bin/activate`$\rightarrow$ activate virtual environment 
3. `uv pip install . ` $\rightarrow$ install the `karhu` package (and deps) into the virtual environment

Now you should be able to run `tests`



