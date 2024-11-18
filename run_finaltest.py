"""
Author: Theresa Wagner <theresa.wagner@mathematik.tu-chemnitz.de>

Corresponding publication:
"A Preconditioned Interior Point Method for Support Vector Machines Using an
ANOVA-Decomposition and NFFT-Based Matrix–Vector Products"
by T. Wagner, John W. Pearson, M. Stoll (2023)

Execute this file to reproduce the results presented in Figure/Table 4.
"""

import numpy as np

from class_NFFTSVMipm import RandomSearch

##################################################################################

####################
## CHOOSE PARAMETERS

# determine kernel definition
kernel = 1 # Gaussian kernel 
#kernel = 3 # Matérn(1/2) kernel 

# set number of iterations in RandomSearch
iRS = 25
# mutual information score threshold
mis_thres = 0.0
# choose window scheme
window_scheme = "mis"
#window_scheme = "consec"
#window_scheme = "random"
# choose weight scheme
weight_scheme = "equally weighted"
#weight_scheme = "no weights"
# choose target preconditioner rank
Dprec = 200
# choose precondioner
prec = "chol_greedy"
#prec = "chol_rp"
#prec = "rff"
#prec = "nystrom"
# choose maximum number of IPM iterations
iter_ip = 100
# choose maximum number of GMRES iterations
Gmaxiter = 100

# define ipm parameters: ipmpar=[sigma_br,tol,Gtol]
ipmpar = [0.2,1e-3,1e-6]

# define list of data sets
data_sets = ["susy", "cod_rna", "higgs"]

####################
# initialize dict for results
dict_acc = {d: [] for d in data_sets}
dict_ipmiters = {d: [] for d in data_sets}
dict_gmresiters = {d: [] for d in data_sets}
dict_timefastadjsetup = {d: [] for d in data_sets}
dict_bestparam = {d: [] for d in data_sets}
dict_bestfit = {d: [] for d in data_sets}
dict_bestpred = {d: [] for d in data_sets}

# initialize dict for LIBSVM results
lib_dict_acc = {d: [] for d in data_sets}
lib_dict_bestparam = {d: [] for d in data_sets}
lib_dict_bestfit = {d: [] for d in data_sets}
lib_dict_bestpred = {d: [] for d in data_sets}

####################

for data in data_sets:
    print("\n####################################################\n")
    print("############################")
    print("Solving for data = ", data)
    print("############################")

    # number of train and test data each
    if data == "higgs":
        Ndata = [1000, 5000, 10000, 50000, 100000, 250000]
    elif data == "susy":
        Ndata = [1000, 5000, 10000, 50000, 100000, 250000]
    elif data == "cod_rna":
        Ndata = [1000, 5000, 10000, 50000, 100000]
        
    for n in Ndata:
        print("\n####################################################\n")
        print("############################")
        print("Solving for n = ", n)
        print("############################")

        if data == "higgs":
            from data_SVMipm import higgs
		
            X_train, X_test, y_train, y_test = higgs(n)
            
            # define d_ratio
            dr = 1/3
		
        elif data == "susy":
            from data_SVMipm import susy
		
            X_train, X_test, y_train, y_test = susy(n)
	    
    	    # define d_ratio
            dr = 2/3
		
        elif data == "cod_rna":
            from data_SVMipm import cod_rna
		
            X_train, X_test, y_train, y_test = cod_rna(n)
            
            # define d_ratio
            dr = 1
		
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
    
        print("\n###################################################\n")
        print("############################")
        print("Solving for d_ratio = ", dr)
        print("############################")

    	# setup RandomSearch model for SVMipm classifier
        model = RandomSearch(classifier="NFFTSVMipm", kernel=kernel, lb=[lb_sigma,lb_C], ub=[ub_sigma, ub_C], max_iter_rs=iRS, mis_threshold=mis_thres, window_scheme=window_scheme, d_ratio=dr, weight_scheme=weight_scheme, sigma_br=ipmpar[0], D_prec=Dprec, prec=prec, iter_ip=iter_ip, tol=ipmpar[1], Gmaxiter=Gmaxiter, Gtol=ipmpar[2])

    	## run classification task
        results_ipm = model.tune(X_train, y_train, X_test, y_test)
        print("\nRandomSearch for NFFTSVMipm")
        print("IPM Parameters:", ipmpar)
        print("d_ratio:", dr)
        print("Best Parameters:", results_ipm[0])
        print("Best Result:", results_ipm[1])
        print("Best Runtime Fit:", results_ipm[2])
        print("Best Runtime Predict:", results_ipm[3])
        print("Best Total Runtime:", results_ipm[2] + results_ipm[3])
        print("Mean Runtime Fit:", results_ipm[4])
        print("Mean Runtime Predict:", results_ipm[5])
        print("Mean Total Runtime:", results_ipm[4] + results_ipm[5])
        print("Best IPMiter:", results_ipm[9])
        print("Best GMRESiter:", results_ipm[7])
        print("Best Time fastadjsetup:", results_ipm[10])
	    
	    # save values to dict
        dict_acc[data].append((results_ipm[1])[0])
        dict_ipmiters[data].append(results_ipm[9])
        dict_gmresiters[data].append(np.mean(results_ipm[7]))
        dict_timefastadjsetup[data].append(results_ipm[10])
        dict_bestparam[data].append(results_ipm[0])
        dict_bestfit[data].append(results_ipm[2])
        dict_bestpred[data].append(results_ipm[3])
        
        print("\nResults NFFTSVMipm:")
        print("------------------------\n")
        print("dict_acc:", dict_acc)
        print("dict_ipmiters:", dict_ipmiters)
        print("dict_gmresiters:", dict_gmresiters)
        print("dict_timefastadjsetup:", dict_timefastadjsetup)
        print("dict_bestparam", dict_bestparam)
        print("dict_bestfit", dict_bestfit)
        print("dict_bestpred", dict_bestpred)
        
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

    	# save values to dict
        lib_dict_acc[data].append((results_libsvm[1])[0])
        lib_dict_bestparam[data].append(results_libsvm[0])
        lib_dict_bestfit[data].append(results_libsvm[2])
        lib_dict_bestpred[data].append(results_libsvm[3])
        
#################################################################################
    
        ## print results in comparison
        print("########################################################################")
        
        print("\nResults NFFTSVMipm:")
        print("------------------------\n")
        print("dict_acc:", dict_acc)
        print("dict_ipmiters:", dict_ipmiters)
        print("dict_gmresiters:", dict_gmresiters)
        print("dict_timefastadjsetup:", dict_timefastadjsetup)
        print("dict_bestparam", dict_bestparam)
        print("dict_bestfit", dict_bestfit)
        print("dict_bestpred", dict_bestpred)
    
        print("\nResults LIBSVM:")
        print("------------------------\n")
        print("lib_dict_acc:", lib_dict_acc) 
        print("lib_dict_bestparam", lib_dict_bestparam)
        print("lib_dict_bestfit", lib_dict_bestfit)
        print("lib_dict_bestpred", lib_dict_bestpred)