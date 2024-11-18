# NFFTSVMipm
This repository contains an implementation of the method introduced in the paper ["A Preconditioned Interior Point Method for Support Vector Machines Using an ANOVA-Decomposition and NFFT-Based Matrix-Vector Products"](https://arxiv.org/)

We propose employing a NFFT-accelerated matrix-vector product using an ANOVA decomposition for the feature space within a preconditioned interior point method for support vector machines. For more details, see the above-mentioned paper.

This package uses the [FastAdjacency2.0](https://github.com/wagnertheresa/FastAdjacency2.0) package to perform NFFT-based fast summation to speed up kernel-vector multiplications for the ANOVA kernel.

## Installation

- This software has been tested with Python 3.8.
- This software depends the FastAdjacency2.0 package. We refer to https://github.com/wagnertheresa/FastAdjacency2.0#readme for installation instructions.

## Usage
The main file class_NFFTSVMipm.py consists of the following two classes:

- `NFFTSVMipm` performs a preconditioned interior point method for Support Vector Machines.
- `RandomSearch` searches on random candidate parameter values for one of the classifiers `NFFTSVMipm` or `LIBSVM`.

To reproduce the results presented in the paper, execute the following files:

* [run_test_precond_timings.py](https://github.com/wagnertheresa/NFFTSVMipm/blob/main/run_test_precond_timings.py) (Figure 2)
* [run_test_IPMGMRES_iters.py](https://github.com/wagnertheresa/NFFTSVMipm/blob/main/run_test_IPMGMRES_iters.py) (Figure 3)
* [run_test_IPMGMRES_tols.py](https://github.com/wagnertheresa/NFFTSVMipm/blob/main/run_test_IPMGMRES_tols.py) (Table 2)
* [run_test_dratio.py](https://github.com/wagnertheresa/NFFTSVMipm/blob/main/run_test_dratio.py) (Table 3)
* [run_finaltest.py](https://github.com/wagnertheresa/NFFTSVMipm/blob/main/run_finaltest.py) (Figure/Table 4)

## Data sets
The benchmark datasets used in the numerical results section can be downloaded from the following websites: [HIGGS](https://archive.ics.uci.edu/dataset/280/higgs), [SUSY](https://archive.ics.uci.edu/dataset/279/susy), [cod-rna](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html). The cod-rna data files can be found in the [data](https://github.com/wagnertheresa/NFFTSVMipm/tree/main/data) folder of this repository. The HIGGS and SUSY data files exceed the standard size limits of GitHub and should be saved locally in the data folder of this repository.

## References
We refer to our repositories [NFFT4ANOVA](https://github.com/wagnertheresa/NFFT4ANOVA) and [NFFTAddKer](https://github.com/wagnertheresa/NFFTAddKer), where this fast NFFT-based matrix-vector product approach is applied to kernel ridge regression.


