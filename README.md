# NFFTSVMipm
This repository contains an implementation of the method introduced in the paper ["A Preconditioned Interior Point Method for Support Vector Machines Usinf an ANOVA-Decomposition and NFFT-Based Matrix--Vector Products"](https://arxiv.org/)

We propose employing a NFFT-accelerated matrix--vector product using an ANOVA decomposition for the feature space within a preconditioned interior point method for Support Vector Machines. For more details, see the above-mentioned paper.

This package uses the [FastAdjacency](https://github.com/dominikbuenger/FastAdjacency) package by Dominik Bünger to perform NFFT-based fast summation to speed up kernel-vector multiplications for the ANOVA kernel.

# Installation
- This software has been tested with Python 3.8.
- This software depends on Bünger's FastAdjacency Package. We refer to https://github.com/dominikbuenger/FastAdjacency#readme for installation instructions.

# Usage
The main file class_SVMipm.py consists of the following two classes:
- `SVMipm` performs a preconditioned interior point method for Support Vector Machines.
- `RandomSearch` searches on random candidate parameter values for one of the classifiers `NFFTSVMipm` or `LIBSVM`.

It can be run via the file run_SVMipm.py.

See __ for an example.

# Datasets
All benchmark datasets used in the numerical results section can be found in the data folder of this repository.

# Citation

# References
We refer to our previous repository [NFFT4ANOVA](https://github.com/wagnertheresa/NFFT4ANOVA), where this fast NFFT-based matrix--vector product approach is applied to kernel ridge regression.


