# NFFTSVMipm
This repository contains an implementation of the method introduced in the paper ["A Preconditioned Interior Point Method for Support Vector Machines Using an ANOVA-Decomposition and NFFT-Based Matrix-Vector Products"](https://arxiv.org/)

We propose employing a NFFT-accelerated matrix-vector product using an ANOVA decomposition for the feature space within a preconditioned interior point method for support vector machines. For more details, see the above-mentioned paper.

This package uses the [FastAdjacency](https://github.com/dominikbuenger/FastAdjacency) package by Dominik Bünger to perform NFFT-based fast summation to speed up kernel-vector multiplications for the ANOVA kernel.

# Installation
- This software has been tested with Python 3.8.
- This software depends on Bünger's FastAdjacency Package. We refer to https://github.com/dominikbuenger/FastAdjacency#readme for installation instructions.

# Usage
The main file class_NFFTSVMipm.py consists of the following two classes:
- `NFFTSVMipm` performs a preconditioned interior point method for Support Vector Machines.
- `RandomSearch` searches on random candidate parameter values for one of the classifiers `NFFTSVMipm` or `LIBSVM`.

It can be run via the showcase file [run_NFFTSVMipm.py](https://github.com/wagnertheresa/NFFTSVMipm/blob/main/run_NFFTSVMipm.py).

# Datasets
The benchmark datasets used in the numerical results section can be downloaded from the following websites: [HIGGS](https://archive.ics.uci.edu/dataset/280/higgs), [SUSY](https://archive.ics.uci.edu/dataset/279/susy), [cod-rna](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html). The cod-rna data files can be found in the [data](https://github.com/wagnertheresa/NFFTSVMipm/tree/main/data) folder of this repository. The HIGGS and SUSY data files exceed the standard size limits of GitHub and should be saved locally in the data folder of this repository.

# References
We refer to our previous repository [NFFT4ANOVA](https://github.com/wagnertheresa/NFFT4ANOVA), where this fast NFFT-based matrix-vector product approach is applied to kernel ridge regression.


