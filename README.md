# Efficient MPS state-preparation for first quantized electronic structure

## Overview

This repository contains the Julia implementation of first-quantized matrix product state preparation for gaussian molecular orbitals. The code integrates Julia and Python using the `PyCall.jl` package for interfacting with the [PySCF](https://github.com/pyscf/pyscf) library for molecular electronic structure calculations. This project is associated with the following paper:

**Citation**: [William J. Huggins, Oskar Leimkuhler, Torin F. Stetina, K. Birgitta Whaley], *Efficient state preparation for the quantum simulation of molecules in first
quantization*, ArXiv Preprint, 2024, DOI: [https://doi.org/10.48550/arXiv.2407.00249].

## Table of Contents

- [Installation](#installation)
- [Dependencies](#dependencies)
- [License](#license)
- [Citation](#citation)

## Installation

To get started, first clone this repository and install the necessary Julia and Python dependencies below. 

Next, setup PyCall within julia in the following manner, so you can use a custom python environment:

```julia
julia 
ENV["PYTHON"] = "/path/to/python"
using Pkg; Pkg.build("PyCall")
```

Then you can run `julia example.jl`.


## Dependencies

Python: `scipy, numpy, pyscf`

Julia: `PyCall, ITensors, ITensorsMPS, LinearAlgebra, Random, Polynomials, SpecialPolynomials`

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Citation
If you use this software in your research, please cite:

```
@article{huggins2024efficient,
  title={Efficient state preparation for the quantum simulation of molecules in first quantization},
  author={Huggins, William J and Leimkuhler, Oskar and Stetina, Torin F and Whaley, K Birgitta},
  journal={arXiv preprint arXiv:2407.00249},
  year={2024}
}
```
