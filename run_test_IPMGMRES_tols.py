"""
Author: Theresa Wagner <theresa.wagner@mathematik.tu-chemnitz.de>

Corresponding publication:
"A Preconditioned Interior Point Method for Support Vector Machines Using an
ANOVA-Decomposition and NFFT-Based Matrix–Vector Products"
by T. Wagner, John W. Pearson, M. Stoll (2023)

Execute this file to reproduce the results presented in Table 2.
"""

import numpy as np

from class_NFFTSVMipm import RandomSearch

##################################################################################

####################
## CHOOSE PARAMETERS

# choose kernel definition
kernel = 1 # Gaussian kernel
#kernel = 3 # Matérn(1/2) kernel

# choose data set
#data = "higgs"
#data = "susy"
data = "cod_rna"

# choose subset sizes
Ndata = [0] # 0 corresponds to entire data set

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

# define sigma_br parameter
sbr = 0.2
# define list of candidates for IPM/GMRES tolerance combination
tol_list = [[1e-1,1e-4], [1e-2,1e-5], [1e-3,1e-6], [1e-4,1e-7]]

####################
# initialize dict for results
dict_acc = {t[0]: [] for t in tol_list}
dict_ipmiters = {t[0]: [] for t in tol_list}
dict_gmresiters = {t[0]: [] for t in tol_list}
dict_bestparam = {t[0]: [] for t in tol_list}
dict_bestfit = {t[0]: [] for t in tol_list}
dict_bestpred = {t[0]: [] for t in tol_list}

####################

print("\n####################################################\n")
print("############################")
print("Solving for data = ", data)
print("############################")
 
for n in Ndata:
    print("\n####################################################\n")
    print("############################")
    print("Solving for n = ", n)
    print("############################")

    if data == "higgs":
        from data_SVMipm import higgs
            
        # define d_ratio
        dr = 1/3
            
        X_train, X_test, y_train, y_test = higgs(n)
		
    elif data == "susy":
        from data_SVMipm import susy
		
        # define d_ratio
        dr = 2/3
		
        X_train, X_test, y_train, y_test = susy(n)
		
    elif data == "cod_rna":
        from data_SVMipm import cod_rna
        
        # define d_ratio
        dr = 1
		
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

    for t in tol_list:
    
        print("\n###################################################\n")
        print("############################")
        print("Solving for tol = ", t)
        print("############################")

        # define IPM parameters: ipmpar=[sigma_br,tol,Gtol]
        ipmpar = [sbr,t[0],t[1]]

        # setup RandomSearch model for SVMipm classifier
        model = RandomSearch(classifier="NFFTSVMipm", kernel=kernel, lb=[lb_sigma,lb_C], ub=[ub_sigma, ub_C], max_iter_rs=iRS, mis_threshold=mis_thres, window_scheme=window_scheme, d_ratio=dr, weight_scheme=weight_scheme, sigma_br=ipmpar[0], D_prec=Dprec, prec=prec, iter_ip=iter_ip, tol=ipmpar[1], Gmaxiter=Gmaxiter, Gtol=ipmpar[2])

    	## run classification task
        results_ipm = model.tune(X_train, y_train, X_test, y_test)
        print("\nRandomSearch for NFFTSVMipm")
        print("IPM Parameters:", ipmpar)
        print("d_ratio:", dr)
        print("tols:", t)
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
        
	    # save values to dict
        dict_acc[t[0]].append((results_ipm[1])[0])
        dict_ipmiters[t[0]].append(results_ipm[9])
        dict_gmresiters[t[0]].append(np.mean(results_ipm[7]))
        dict_bestparam[t[0]].append(results_ipm[0])
        dict_bestfit[t[0]].append(results_ipm[2])
        dict_bestpred[t[0]].append(results_ipm[3])
        
        #################################################################################
        #################################################################################
        ## print results in comparison
        print("\n########################################################################")
        print("\nResults NFFTSVMipm:")
        print("------------------------\n")
        print("dict_acc:", dict_acc)
        print("dict_ipmiters:", dict_ipmiters)
        print("dict_gmresiters:", dict_gmresiters)
        print("dict_bestparam", dict_bestparam)
        print("dict_bestfit", dict_bestfit)
        print("dict_bestpred", dict_bestpred)
