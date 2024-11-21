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

#### Arguments

| Argument          | Type     | Choices             | Default            | Description                                                                                  |
|:------------------|:---------|:--------------------|:-------------------|:---------------------------------------------------------------------------------------------|
| `--kernel`        | `int`    | `1`, `3` | `1` | Specifies the kernel type: `1` for Gaussian kernel, `3` for Matérn(1/2) kernel. |
| `--Ndata`         | `int`    | Any positive integer or list of integers smaller or equal the size of the data set. | `[100000]` | Specifies the subset size(s). Pass a single value or a list, where `0` corresponds to the entire data set (e.g., `100000`). |
| `--prec`          | `str`    | `"chol_greedy"`, `"chol_rp"`, `"rff"`, `"nystrom"` | `["chol_greedy", "chol_rp", "rff", "nystrom"]` | Specifies the preconditioner type(s). Pass a single value or a list (e.g., `"chol_greedy" "chol_rp" "rff" "nystrom"`). |
| `--rank`          | `int`    | Any positive integer or list of integers. | `[50, 200, 1000]` | Specifies the preconditioner rank(s). Pass a single value or a list (e.g., `50 200 1000`). |

### `run_test_IPMGMRES_iters.py`

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

### `run_test_IPMGMRES_tols.py`

#### Arguments

| Argument          | Type     | Choices             | Default            | Description                                                                                  |
|:------------------|:---------|:--------------------|:-------------------|:---------------------------------------------------------------------------------------------|
| `--kernel`        | `int`    | `1`, `3` | `1` | Specifies the kernel type: `1` for Gaussian kernel, `3` for Matérn(1/2) kernel. |
| `--Ndata`         | `int`    | Any positive integer or list of integers smaller or equal the size of the data set. | `[0]` | Specifies the subset size(s), where `0` corresponds to the entire data set. Pass a single value or a list (e.g., `0`). |
| `--prec`          | `str`    | `"chol_greedy"`, `"chol_rp"`, `"rff"`, `"nystrom"` | `"chol_greedy"` | Specifies the preconditioner type. |
| `--rank`          | `int`    | Any positive integer. | `200` | Specifies the preconditioner rank. |
| `--iRS`           | `int`    | Any positive integer. | `25` | Specifies the number of iterations in RandomSearch. |
| `--mis_thres`     | `float`  | Any positive float. | `0.0` | Specifies the mutual information score threshold. |
| `--window_scheme` | `str`    | `"mis"`, `"consec"`, `"random"` | `"mis"` | Specifies the window scheme. |
| `--weight_scheme` | `str`    | `"equally weighted"`, `"no weights"` | `"equally weighted"` | Specifies the weight scheme. |
| `--IPMiter`       | `int`    | Any positive integer. | `100` | Specifies the maximum number of IPM iterations. |
| `--GMRESiter`     | `int`    | Any positive integer. | `100` | Specifies the maximum number of GMRES iterations. |
| `--sbr`           | `float`  | Any float from [0.2,0.6]. | `0.2` | Specifies the sigma barrier reduction parameter within the IPM. |
| `--dratio`        | `float`  | Any float from (0,1]. | `1.0` | Specifies the proportion of features to include. |
| `--data`          | `str`    | `"susy"`, `"cod-rna"`, `"higgs"` | `"cod-rna"` | Specifies the data set to use. |

### `run_test_dratio.py`

#### Arguments

| Argument          | Type     | Choices             | Default            | Description                                                                                  |
|:------------------|:---------|:--------------------|:-------------------|:---------------------------------------------------------------------------------------------|
| `--kernel`        | `int`    | `1`, `3` | `1` | Specifies the kernel type: `1` for Gaussian kernel, `3` for Matérn(1/2) kernel. |
| `--Ndata`         | `int`    | Any positive integer or list of integers smaller or equal the size of the data set. | `[5000, 10000, 50000]` | Specifies the subset size(s). Pass a single value or a list (e.g., `5000 10000 50000`). |
| `--prec`          | `str`    | `"chol_greedy"`, `"chol_rp"`, `"rff"`, `"nystrom"` | `"chol_greedy"` | Specifies the preconditioner type. |
| `--rank`          | `int`    | Any positive integer. | `200` | Specifies the preconditioner rank. |
| `--iRS`           | `int`    | Any positive integer. | `25` | Specifies the number of iterations in RandomSearch. |
| `--mis_thres`     | `float`  | Any positive float. | `0.0` | Specifies the mutual information score threshold. |
| `--window_scheme` | `str`    | `"mis"`, `"consec"`, `"random"` | `"mis"` | Specifies the window scheme. |
| `--weight_scheme` | `str`    | `"equally weighted"`, `"no weights"` | `"equally weighted"` | Specifies the weight scheme. |
| `--IPMiter`       | `int`    | Any positive integer. | `100` | Specifies the maximum number of IPM iterations. |
| `--GMRESiter`     | `int`    | Any positive integer. | `100` | Specifies the maximum number of GMRES iterations. |
| `--ipmpar`        | `float`  | List of 3 float values. | `[0.2, 1e-3, 1e-6]` | A list of three values specifying the IPM parameters: `[sigma_br, tol, Gtol]`. |
| `--dratio`        | `float`  | Any float from (0,1] or list of floats from (0,1]. | `[1/3, 2/3, 1.0]` | Specifies the proportion(s) of features to include. Pass a single value or a list (e.g., `1/3 2/3 1`). |
| `--data`          | `str`    | `"susy"`, `"cod-rna"`, `"higgs"` | `["susy", "cod-rna", "higgs"]` | Specifies the data set(s) to use. Pass a single string or a list of strings (e.g., `"susy" "cod-rna" "higgs"`). |

### `run_finaltest.py`

#### Arguments
| Argument          | Type     | Choices             | Default            | Description                                                                                  |
|:------------------|:---------|:--------------------|:-------------------|:---------------------------------------------------------------------------------------------|
| `--kernel`        | `int`    | `1`, `3` | `1` | Specifies the kernel type: `1` for Gaussian kernel, `3` for Matérn(1/2) kernel. |
| `--Ndata`         | `int`    | Any positive integer or list of integers smaller or equal the size of the data set. | - | Specifies the subset size(s). Pass a single value or a list (e.g., `1000 5000 10000 50000 100000`). |
| `--prec`          | `str`    | `"chol_greedy"`, `"chol_rp"`, `"rff"`, `"nystrom"` | `"chol_greedy"` | Specifies the preconditioner type. |
| `--rank`          | `int`    | Any positive integer. | `200` | Specifies the preconditioner rank. |
| `--iRS`           | `int`    | Any positive integer. | `25` | Specifies the number of iterations in RandomSearch. |
| `--mis_thres`     | `float`  | Any positive float. | `0.0` | Specifies the mutual information score threshold. |
| `--window_scheme` | `str`    | `"mis"`, `"consec"`, `"random"` | `"mis"` | Specifies the window scheme. |
| `--weight_scheme` | `str`    | `"equally weighted"`, `"no weights"` | `"equally weighted"` | Specifies the weight scheme. |
| `--IPMiter`       | `int`    | Any positive integer. | `100` | Specifies the maximum number of IPM iterations. |
| `--GMRESiter`     | `int`    | Any positive integer. | `100` | Specifies the maximum number of GMRES iterations. |
| `--ipmpar`        | `float`  | List of 3 float values. | `[0.2, 1e-3, 1e-6]` | A list of three values specifying the IPM parameters: `[sigma_br, tol, Gtol]`. |
| `--data`          | `str`    | `"susy"`, `"cod-rna"`, `"higgs"` | `["susy", "cod-rna", "higgs"]` | Specifies the data set(s) to use. Pass a single string or a list of strings (e.g., `"susy" "cod-rna" "higgs"`). |

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


