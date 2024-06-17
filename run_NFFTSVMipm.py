import numpy as np

from class_NFFTSVMipm import RandomSearch

####################
## CHOOSE PARAMETERS

# number of train and test data each

n = 50000

#data = "higgs"
#data = "susy"
data = "cod_rna"
####################
 
if data == "higgs":
    from data_SVMipm import higgs
    
    X_train, X_test, y_train, y_test = higgs(n)
    
elif data == "susy":
    from data_SVMipm import susy
    
    X_train, X_test, y_train, y_test = susy(n)
    
elif data == "cod_rna":
    from data_SVMipm import cod_rna
    
    X_train, X_test, y_train, y_test = cod_rna(n)
    
print("\nDataset:", data)
print("--------\nShape train data:", X_train.shape)
print("Shape test data:", X_test.shape)

#################################################################################
## RandomSearch for NFFTSVMipm

# define bounds for individual sigmas for every window
lb_sigma = np.sqrt(1/1e+2)
ub_sigma = np.sqrt(1/1e-2)

# define bounds for C
lb_C = 0.1
ub_C = 0.7

# setup RandomSearch model for SVMipm classifier
model = RandomSearch(classifier="NFFTSVMipm", lb=[lb_sigma, lb_C], ub=[ub_sigma, ub_C], mis_threshold=0.0, window_scheme="mis", D_prec=200, prec="chol_greedy", iter_ip=50)

## run classification task
results_ipm = model.tune(X_train, y_train, X_test, y_test)
print("\nRandomSearch for NFFTSVMipm")
print("Best Parameters:", results_ipm[0])
print("Best Result:", results_ipm[1])
print("Best Runtime Fit:", results_ipm[2])
print("Best Runtime Predict:", results_ipm[3])
print("Best Total Runtime:", results_ipm[2] + results_ipm[3])
print("Mean Runtime Fit:", results_ipm[4])
print("Mean Runtime Predict:", results_ipm[5])
print("Mean Total Runtime:", results_ipm[4] + results_ipm[5])

#################################################################################
## RandomSearch for LIBSVM

# define bounds for gamma
lb_gamma = 1e-2
ub_gamma = 1e+2

# define bounds for C
lb_C = 0.1
ub_C = 0.7

# setup Random Search model for LIBSVM classifier
model = RandomSearch(classifier="LIBSVM", lb=[lb_gamma, lb_C], ub=[ub_gamma, ub_C])

## run classification task
results_libsvm = model.tune(X_train, y_train, X_test, y_test)
print("\nRandomSearch for LIBSVM")
print("Best Parameters:", results_libsvm[0])
print("Best Result:", results_libsvm[1])
print("Best Runtime Fit:", results_libsvm[2])
print("Best Runtime Predict:", results_libsvm[3])
print("Best Total Runtime:", results_libsvm[2] + results_libsvm[3])
print("Mean Runtime Fit:", results_libsvm[4])
print("Mean Runtime Predict:", results_libsvm[5])
print("Mean Total Runtime:", results_libsvm[4] + results_libsvm[5])

#################################################################################

## print overall results in comparison
print("########################################################################")

print("\nDataset:", data)
print("--------\nShape train data:", X_train.shape)
print("Shape test data:", X_test.shape)
print("D_prec:", results_ipm[6])

# RandomSearch for NFFTSVMipm
print("\nRandomSearch for NFFTSVMipm")
print("Best Parameters:", results_ipm[0])
print("Best Result:", results_ipm[1])
print("Best Runtime Fit:", results_ipm[2])
print("Best Runtime Predict:", results_ipm[3])
print("Best Total Runtime:", results_ipm[2] + results_ipm[3])
print("Mean Runtime Fit:", results_ipm[4])
print("Mean Runtime Predict:", results_ipm[5])
print("Mean Total Runtime:", results_ipm[4] + results_ipm[5])
print("Best GMRESiters:", results_ipm[7])
print("Mean GMRESiters:", results_ipm[8])

# RandomSearch for LIBSVM
print("\nRandomSearch for LIBSVM")
print("Best Parameters:", results_libsvm[0])
print("Best Result:", results_libsvm[1])
print("Best Runtime Fit:", results_libsvm[2])
print("Best Runtime Predict:", results_libsvm[3])
print("Best Total Runtime:", results_libsvm[2] + results_libsvm[3])
print("Mean Runtime Fit:", results_libsvm[4])
print("Mean Runtime Predict:", results_libsvm[5])
print("Mean Total Runtime:", results_libsvm[4] + results_libsvm[5])
