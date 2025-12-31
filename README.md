# NFFTSVMipm
This repository contains an implementation of the method introduced in the paper ["A Preconditioned Interior Point Method for Support Vector Machines Using an ANOVA-Decomposition and NFFT-Based Matrix-Vector Products"](https://arxiv.org/)

We propose employing a NFFT-accelerated matrix-vector product using an ANOVA decomposition for the feature space within a preconditioned interior point method for support vector machines. For more details, see the above-mentioned paper.

This package uses the [FastAdjacency2.0](https://github.com/wagnertheresa/FastAdjacency2.0) package to perform NFFT-based fast summation to speed up kernel-vector multiplications for the ANOVA kernel.

## Installation

- This software has been tested with Python 3.8.
- This software depends the FastAdjacency2.0 package. We refer to https://github.com/wagnertheresa/FastAdjacency2.0#readme for installation instructions.

To use this code, run `make` from the terminal. Optionally, run `make check` to test your installation.

## Usage
The main file class_NFFTSVMipm.py in the `nfftsvmipm` directory consists of the following two classes:

- `NFFTSVMipm` performs a preconditioned interior point method for Support Vector Machines.
- `RandomSearch` searches on random candidate parameter values for one of the classifiers `NFFTSVMipm` or `LIBSVM`.
-  A random seed is set as `random.seed(42)` in the class file `class_NFFTSVMipm.py` `


To reproduce the results presented in the paper, execute the following files from the `test` directory:

* [run_test_precond_timings.py](https://github.com/wagnertheresa/NFFTSVMipm/blob/main/test/run_test_precond_timings.py) (Figure 2)
* [run_test_IPMGMRES_iters.py](https://github.com/wagnertheresa/NFFTSVMipm/blob/main/test/run_test_IPMGMRES_iters.py) (Figure 3)
* [run_test_IPMGMRES_tols.py](https://github.com/wagnertheresa/NFFTSVMipm/blob/main/test/run_test_IPMGMRES_tols.py) (Table 2)
* [run_test_dratio.py](https://github.com/wagnertheresa/NFFTSVMipm/blob/main/test/run_test_dratio.py) (Table 3)
* [run_finaltest.py](https://github.com/wagnertheresa/NFFTSVMipm/blob/main/test/run_finaltest.py) (Figure/Table 4)

These Python scripts are designed to run configurable tests with various parameters, and allow users to specify these parameters via command-line arguments for maximum flexibility.

To run the scripts in sequence with the default argument values, use the following command:
```bash
python test/run_all.py
```
To run the scripts individually, use the following command:
```bash
python test/filename.py [options]
```

### `run_test_precond_timings.py`

For a given list of subset sizes for a list of data sets, the script iterates over different preconditioners and preconditioner ranks for given feature windows, length-scales and margin value.

#### Arguments

| Argument          | Type     | Choices             | Default            | Description                                                                                  |
|:------------------|:---------|:--------------------|:-------------------|:---------------------------------------------------------------------------------------------|
| `--kernel`        | `int`    | `1`, `3` | `1` | Specifies the kernel type: `1` for Gaussian kernel, `3` for Matérn(1/2) kernel. |
| `--Ndata`         | `int`    | Any positive integer or list of integers smaller or equal the size of the data set. | `[100000]` | Specifies the subset size(s). Pass a single value or a list, where `0` corresponds to the entire data set (e.g., `100000`). |
| `--prec`          | `str`    | `"chol_greedy"`, `"chol_rp"`, `"rff"`, `"nystrom"` | `["chol_greedy", "chol_rp", "rff", "nystrom"]` | Specifies the preconditioner type(s). Pass a single value or a list (e.g., `"chol_greedy" "chol_rp" "rff" "nystrom"`). |
| `--rank`          | `int`    | Any positive integer or list of integers. | `[50, 200, 1000]` | Specifies the preconditioner rank(s). Pass a single value or a list (e.g., `50 200 1000`). |

#### Output format
For every preconditioner considered on the specified data set, list of subset sizes `[n1, n2, n3]` and different preconditioner ranks `[rank1, rank2, rank3]`, the output is of the following format:

```
"preconditioner_name": {
    rank1: [preconditioner_setup_time(n1), preconditioner_setup_time(n2), preconditioner_setup_time(n3)],
    rank2: [preconditioner_setup_time(n1), preconditioner_setup_time(n2), preconditioner_setup_time(n3)],
    rank3: [preconditioner_setup_time(n1), preconditioner_setup_time(n2), preconditioner_setup_time(n3)]
}
```

### `run_test_IPMGMRES_iters.py`

The script iterates over different preconditioners and preconditioner ranks for different length-scale parameter values for a given list of subset sizes for a specified data set.

#### Arguments

| Argument          | Type     | Choices             | Default            | Description                                                                                  |
|:------------------|:---------|:--------------------|:-------------------|:---------------------------------------------------------------------------------------------|
| `--kernel`        | `int`    | `1`, `3` | `1` | Specifies the kernel type: `1` for Gaussian kernel, `3` for Matérn(1/2) kernel. |
| `--Ndata`         | `int`    | Any positive integer or list of integers smaller or equal the size of the data set. | `[0]` | Specifies the subset size(s), where `0` corresponds to the entire data set. Pass a single value or a list (e.g., `0`). |
| `--prec`          | `str`    | `"chol_greedy"`, `"chol_rp"`, `"rff"`, `"nystrom"` | `["chol_greedy", "chol_rp", "rff", "nystrom"]` | Specifies the preconditioner type(s). Pass a single value or a list (e.g., `"chol_greedy" "chol_rp" "rff" "nystrom"`). |
| `--rank`          | `int`    | Any positive integer or list of integers. | `[50, 200, 1000]` | Specifies the preconditioner rank(s). Pass a single value or a list (e.g., `50 200 1000`). |
| `--S`             | `float`  | Any positive float or list of floats. | `[1e-2, 1e-1, 1, 1e+1, 1e+2]` | Specifies the length-scale value(s). Pass a single value or a list (e.g., `1e-2 1e-1 1 1e+1 1e+2`). |
| `--C`             | `float`  | Any positive float. | `0.4` | Specifies the regularization parameter. |
| `--mis_thres`     | `float`  | Any positive float. | `0.0` | Specifies the mutual information score threshold. |
| `--window_scheme` | `str`    | `"mis"`, `"consec"`, `"random"` | `"mis"` | Specifies the window scheme. |
| `--weight_scheme` | `str`    | `"equally weighted"`, `"no weights"` | `"equally weighted"` | Specifies the weight scheme. |
| `--IPMiter`       | `int`    | Any positive integer. | `100` | Specifies the maximum number of IPM iterations. |
| `--GMRESiter`     | `int`    | Any positive integer. | `100` | Specifies the maximum number of GMRES iterations. |
| `--ipmpar`        | `float`  | List of 3 float values. | `[0.2, 1e-3, 1e-6]` | A list of three values specifying the IPM parameters: `[sigma_br, tol, Gtol]`. |
| `--dratio`        | `float`  | Any float from (0,1]. | `1.0` | Specifies the proportion of features to include. |
| `--data`          | `str`    | `"susy"`, `"cod-rna"`, `"higgs"` | `"cod-rna"` | Specifies the data set to use. |

#### Output format
For every preconditioner considered on the specified data set and subset size for different ranks `[rank1, rank2, rank3]` and length-scale parameters `[ell1, ell2, ell3, ell4, ell5]`, the output is of the following format:

```
"preconditioner_name": 
    rank1: [
        [accuracy(ell1), accuracy(ell2), accuracy(ell3), accuracy(ell4), accuracy(ell5)],
        [ipm_iterations(ell1), ipm_iterations(ell2), ipm_iterations(ell3), ipm_iterations(ell4), ipm_iterations(ell5)],
        [mean_gmres_iterations(ell1), mean_gmres_iterations(ell2), mean_gmres_iterations(ell3), mean_gmres_iterations(ell4), mean_gmres_iterations(ell5)]
    ],
    rank2: [
        [accuracy(ell1), accuracy(ell2), accuracy(ell3), accuracy(ell4), accuracy(ell5)],
        [ipm_iterations(ell1), ipm_iterations(ell2), ipm_iterations(ell3), ipm_iterations(ell4), ipm_iterations(ell5)],
        [mean_gmres_iterations(ell1), mean_gmres_iterations(ell2), mean_gmres_iterations(ell3), mean_gmres_iterations(ell4), mean_gmres_iterations(ell5)]
    ],
    rank3: [
        [accuracy(ell1), accuracy(ell2), accuracy(ell3), accuracy(ell4), accuracy(ell5)],
        [ipm_iterations(ell1), ipm_iterations(ell2), ipm_iterations(ell3), ipm_iterations(ell4), ipm_iterations(ell5)],
        [mean_gmres_iterations(ell1), mean_gmres_iterations(ell2), mean_gmres_iterations(ell3), mean_gmres_iterations(ell4), mean_gmres_iterations(ell5)]
    ]
}
```

### `run_test_IPMGMRES_tols.py`

The script iterates over different IPM and GMRES convergence tolerances for a given preconditioner and preconditioner rank and a list of subset sizes for a specified data set.

#### Arguments

| Argument          | Type     | Choices             | Default            | Description                                                                                  |
|:------------------|:---------|:--------------------|:-------------------|:---------------------------------------------------------------------------------------------|
| `--kernel`        | `int`    | `1`, `3` | `1` | Specifies the kernel type: `1` for Gaussian kernel, `3` for Matérn(1/2) kernel. |
| `--Ndata`         | `int`    | Any positive integer or list of integers smaller or equal the size of the data set. | `[0]` | Specifies the subset size(s), where `0` corresponds to the entire data set. Pass a single value or a list (e.g., `0`). |
| `--prec`          | `str`    | `"chol_greedy"`, `"chol_rp"`, `"rff"`, `"nystrom"` | `"chol_greedy"` | Specifies the preconditioner type. |
| `--rank`          | `int`    | Any positive integer. | `200` | Specifies the preconditioner rank. |
| `--iRS`           | `int`    | Any positive integer. | `25` | Specifies the number of iterations in `RandomSearch`. |
| `--mis_thres`     | `float`  | Any positive float. | `0.0` | Specifies the mutual information score threshold. |
| `--window_scheme` | `str`    | `"mis"`, `"consec"`, `"random"` | `"mis"` | Specifies the window scheme. |
| `--weight_scheme` | `str`    | `"equally weighted"`, `"no weights"` | `"equally weighted"` | Specifies the weight scheme. |
| `--IPMiter`       | `int`    | Any positive integer. | `100` | Specifies the maximum number of IPM iterations. |
| `--GMRESiter`     | `int`    | Any positive integer. | `100` | Specifies the maximum number of GMRES iterations. |
| `--sbr`           | `float`  | Any float from [0.2,0.6]. | `0.2` | Specifies the sigma barrier reduction parameter within the IPM. |
| `--dratio`        | `float`  | Any float from (0,1]. | `1.0` | Specifies the proportion of features to include. |
| `--data`          | `str`    | `"susy"`, `"cod-rna"`, `"higgs"` | `"cod-rna"` | Specifies the data set to use. |

#### Output format
For every IPM/GMRES tolerance pair `[(IPM_tol1,GMRES_tol1), (IPM_tol2,GMRES_tol2), (IPM_tol3,GMRES_tol3)]` considered on the specified preconditioner, preconditioner rank, data set and list of subset sizes `[n1, n2, n3]`, the output is of the following format:

```
"accuracy": {
    IPM_tol1: [accuracy(n1), accuracy(n2), accuracy(n3)],
    IPM_tol2: [accuracy(n1), accuracy(n2), accuracy(n3)],
    IPM_tol3: [accuracy(n1), accuracy(n2), accuracy(n3)]
}
```

In addition to accuracy, the number of IPM iterations, the mean number of GMRES iterations, the parameters, the time for fitting the model and the time for predicting from the trained model (of the random search run generating the largest accuracy) are returned.
All returned metrics follow the output format shown above.

### `run_test_dratio.py`

The script iterates over three different dimensionality reduction settings (`dratio=1`, `dratio=2/3`, `dratio=1/3`) for a given preconditioner and preconditioner rank and a list of subset sizes for a given list of data sets. Further, it runs the LIBSVM model on the entire feature space in comparison.

#### Arguments

| Argument          | Type     | Choices             | Default            | Description                                                                                  |
|:------------------|:---------|:--------------------|:-------------------|:---------------------------------------------------------------------------------------------|
| `--kernel`        | `int`    | `1`, `3` | `1` | Specifies the kernel type: `1` for Gaussian kernel, `3` for Matérn(1/2) kernel. |
| `--Ndata`         | `int`    | Any positive integer or list of integers smaller or equal the size of the data set. | `[5000, 10000, 50000]` | Specifies the subset size(s). Pass a single value or a list (e.g., `5000 10000 50000`). |
| `--prec`          | `str`    | `"chol_greedy"`, `"chol_rp"`, `"rff"`, `"nystrom"` | `"chol_greedy"` | Specifies the preconditioner type. |
| `--rank`          | `int`    | Any positive integer. | `200` | Specifies the preconditioner rank. |
| `--iRS`           | `int`    | Any positive integer. | `25` | Specifies the number of iterations in `RandomSearch`. |
| `--mis_thres`     | `float`  | Any positive float. | `0.0` | Specifies the mutual information score threshold. |
| `--window_scheme` | `str`    | `"mis"`, `"consec"`, `"random"` | `"mis"` | Specifies the window scheme. |
| `--weight_scheme` | `str`    | `"equally weighted"`, `"no weights"` | `"equally weighted"` | Specifies the weight scheme. |
| `--IPMiter`       | `int`    | Any positive integer. | `100` | Specifies the maximum number of IPM iterations. |
| `--GMRESiter`     | `int`    | Any positive integer. | `100` | Specifies the maximum number of GMRES iterations. |
| `--ipmpar`        | `float`  | List of 3 float values. | `[0.2, 1e-3, 1e-6]` | A list of three values specifying the IPM parameters: `[sigma_br, tol, Gtol]`. |
| `--dratio`        | `float`  | Any float from (0,1] or list of floats from (0,1]. | `[1/3, 2/3, 1.0]` | Specifies the proportion(s) of features to include. Pass a single value or a list (e.g., `1/3 2/3 1`). |
| `--data`          | `str`    | `"susy"`, `"cod-rna"`, `"higgs"` | `["susy", "cod-rna", "higgs"]` | Specifies the data set(s) to use. Pass a single string or a list of strings (e.g., `"susy" "cod-rna" "higgs"`). |

#### Output format
For the dimensionality reduction ratios `[dr1, dr2, dr3]` considered on the specified preconditioner, preconditioner rank, data sets `[data1, data2, data3]` and list of subset sizes `[n1, n2, n3]`, the output of the `NFFTSVMipm` model is of the following format:

```
"accuracy": {
    data1: [
        [accuracy(dr1,n1), accuracy(dr2,n1), accuracy(dr3,n1)],
        [accuracy(dr1,n2), accuracy(dr2,n2), accuracy(dr3,n2)],
        [accuracy(dr1,n3), accuracy(dr2,n3), accuracy(dr3,n3)]
    ],
    data2: [
        [accuracy(dr1,n1), accuracy(dr2,n1), accuracy(dr3,n1)],
        [accuracy(dr1,n2), accuracy(dr2,n2), accuracy(dr3,n2)],
        [accuracy(dr1,n3), accuracy(dr2,n3), accuracy(dr3,n3)]
    ],
    data3: [
        [accuracy(dr1,n1), accuracy(dr2,n1), accuracy(dr3,n1)],
        [accuracy(dr1,n2), accuracy(dr2,n2), accuracy(dr3,n2)],
        [accuracy(dr1,n3), accuracy(dr2,n3), accuracy(dr3,n3)]
    ]
}
```

In addition to accuracy, the number of IPM iterations, the mean number of GMRES iterations, the parameters, the time for fitting the model and the time for predicting from the trained model (of the `RandomSearch` run generating the largest accuracy) are returned.
For the `LIBSVM` model, the accuracy, the parameters, the time for fitting the model and the time for predicting from the trained model are returned.
All returned metrics follow the output format shown above.

### `run_finaltest.py`
The script iterates over a list of data sets and subset sizes for two different kernel_types, for a given preconditioner and preconditioner rank and fixed dimensionality reduction ratios for the data sets. Further, it runs the LIBSVM model on the entire feature space in comparison.

#### Arguments
| Argument          | Type     | Choices             | Default            | Description                                                                                  |
|:------------------|:---------|:--------------------|:-------------------|:---------------------------------------------------------------------------------------------|
| `--kernel`        | `int`    | `1`, `3` | `1` | Specifies the kernel type: `1` for Gaussian kernel, `3` for Matérn(1/2) kernel. |
| `--Ndata`         | `int`    | Any positive integer or list of integers smaller or equal the size of the data set. | - | Specifies the subset size(s). Pass a single value or a list (e.g., `1000 5000 10000 50000 100000`). |
| `--prec`          | `str`    | `"chol_greedy"`, `"chol_rp"`, `"rff"`, `"nystrom"` | `"chol_greedy"` | Specifies the preconditioner type. |
| `--rank`          | `int`    | Any positive integer. | `200` | Specifies the preconditioner rank. |
| `--iRS`           | `int`    | Any positive integer. | `25` | Specifies the number of iterations in `RandomSearch`. |
| `--mis_thres`     | `float`  | Any positive float. | `0.0` | Specifies the mutual information score threshold. |
| `--window_scheme` | `str`    | `"mis"`, `"consec"`, `"random"` | `"mis"` | Specifies the window scheme. |
| `--weight_scheme` | `str`    | `"equally weighted"`, `"no weights"` | `"equally weighted"` | Specifies the weight scheme. |
| `--IPMiter`       | `int`    | Any positive integer. | `100` | Specifies the maximum number of IPM iterations. |
| `--GMRESiter`     | `int`    | Any positive integer. | `100` | Specifies the maximum number of GMRES iterations. |
| `--ipmpar`        | `float`  | List of 3 float values. | `[0.2, 1e-3, 1e-6]` | A list of three values specifying the IPM parameters: `[sigma_br, tol, Gtol]`. |
| `--data`          | `str`    | `"susy"`, `"cod-rna"`, `"higgs"` | `["susy", "cod-rna", "higgs"]` | Specifies the data set(s) to use. Pass a single string or a list of strings (e.g., `"susy" "cod-rna" "higgs"`). |

#### Output format
For a list of data sets `[data1, data2, data3]` and subset sizes `[n1, n2, n3]` considered on the specified preconditioner, preconditioner rank and dimensionality reduction ratios for the respective data sets, the output of the `NFFTSVMipm` model is of the following format:

```
"accuracy": {
    data1: [accuracy(n1), accuracy(n2), accuracy(n3)],
    data2: [accuracy(n1), accuracy(n2), accuracy(n3)], 
    data3: [accuracy(n1), accuracy(n2), accuracy(n3)]
}
```

In addition to accuracy, the number of IPM iterations, the mean number of GMRES iterations, the parameters, the time for fitting the model and the time for predicting from the trained model (of the `RandomSearch` run generating the largest accuracy) are returned.
For the `LIBSVM` model, the accuracy, the parameters, the time for fitting the model and the time for predicting from the trained model are returned.
All returned metrics follow the output format shown above.

### Data sets
The benchmark datasets used in the numerical results section can be downloaded from the following websites: [HIGGS](https://archive.ics.uci.edu/dataset/280/higgs), [SUSY](https://archive.ics.uci.edu/dataset/279/susy), [cod-rna](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html). The cod-rna data files can be found in the [`data`](https://github.com/wagnertheresa/NFFTSVMipm/tree/main/data) directory of this repository. The HIGGS and SUSY data files exceed the standard size limits of GitHub and should be saved locally in the data folder of this repository.

### Notes
- Parameter Types:
    * Single values can be passed directly, e.g., `--rank 200`.
    * Lists should be space-separated, e.g., `--rank 50 200 1000`
- Default Values:
    * If an optional argument is omitted, its default value will be used.
- Help Menu:
    * Use `--help` to view all arguments and descriptions:
```bash
python test/runfile.py --help
```

## Examples

To reproduce the results presented in Figure 2 with the Gaussian kernel, the Cholesky (greedy) preconditioner with target rank 200 on the entire cod-rna data set, run the `run_test_precond_timings.py` script as:
```bash
python test/run_test_precond_timings.py --kernel 1 --Ndata 0 --prec "chol_greedy" --rank 200
```

## References
We refer to our repositories [NFFT4ANOVA](https://github.com/wagnertheresa/NFFT4ANOVA) and [NFFTAddKer](https://github.com/wagnertheresa/NFFTAddKer), where this fast NFFT-based matrix-vector product approach is applied to kernel ridge regression.


