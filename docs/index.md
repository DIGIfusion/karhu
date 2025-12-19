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
