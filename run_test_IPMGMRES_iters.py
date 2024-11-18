"""
Author: Theresa Wagner <theresa.wagner@mathematik.tu-chemnitz.de>

Corresponding publication:
"A Preconditioned Interior Point Method for Support Vector Machines Using an
ANOVA-Decomposition and NFFT-Based Matrix–Vector Products"
by T. Wagner, John W. Pearson, M. Stoll (2023)

Execute this file to reproduce the results presented in Figure 3.
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
data = "cod_rna"

# choose subset sizes
Ndata = [0] # 0 corresponds to entire data set

# define list of length-scale candidates
Slist = [1e-2, 1e-1, 1, 1e+1, 1e+2]
# define regularization parameter
C = 0.4

# set number of iterations in RandomSearch
iRS = 1 # only one run since parameters are fixed
# mutual information score threshold
mis_thres = 0.0
# choose window scheme
window_scheme = "mis"
#window_scheme = "consec"
#window_scheme = "random"
# choose weight scheme
weight_scheme = "equally weighted"
#weight_scheme = "no weights"

# define IPM parameters: ipmpar=[sigma_br,tol,Gtol]
ipmpar = [0.2,1e-3,1e-6]

# define list of preconditioner candidates
prec_list = ["chol_greedy", "chol_rp", "rff", "nystrom"]
# define list of target preconditioner rank candidates
rank_list = [50, 200, 1000]

# choose maximum number of IPM iterations
iter_ip = 100
# choose maximum number of GMRES iterations
Gmaxiter = 100

####################
# initialize dict for results
dict_cholgr = {r: [] for r in rank_list}
dict_cholrp = {r: [] for r in rank_list}
dict_rff = {r: [] for r in rank_list}
dict_nystr = {r: [] for r in rank_list}

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

    from data_SVMipm import cod_rna
		
    X_train, X_test, y_train, y_test = cod_rna(n)
            
    # define d_ratio
    dr = 1
		
    print("\nDataset:", data)
    print("--------\nShape train data:", X_train.shape)
    print("Shape test data:", X_test.shape)
    
    for prec in prec_list:
        print("Solving for preconditioner", prec)
        
        for D_prec in rank_list:
            print("Solving for rank", D_prec)
    
            acc = []
            ipmiters = []
            gmresiters = []
    
            for S in Slist:
                print("\n####################################################\n")
                print("############################")
                print("Solving for sigma = ", S)
                print("############################")
	    
                ##############################
    	        ## RandomSearch for NFFTSVMipm

    	        # define bounds for individual sigmas for every window
                lb_sigma = S
                ub_sigma = S

    	        # define bounds for C
                lb_C = C
                ub_C = C

    	        # setup RandomSearch model for SVMipm classifier
                model = RandomSearch(classifier="NFFTSVMipm", kernel=kernel, lb=[lb_sigma,lb_C], ub=[ub_sigma, ub_C], max_iter_rs=iRS, mis_threshold=mis_thres, window_scheme=window_scheme, d_ratio=dr, weight_scheme=weight_scheme, sigma_br=ipmpar[0], D_prec=D_prec, prec=prec, iter_ip=iter_ip, tol=ipmpar[1], Gmaxiter=Gmaxiter, Gtol=ipmpar[2])

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
                print("Mean GMRESiter:", results_ipm[8])
                
                acc.append(results_ipm[1][0])
                ipmiters.append(results_ipm[9])
                gmresiters.append(np.mean(results_ipm[7]))
	    
    	    # save values to dict
            if prec == "chol_greedy":
                dict_cholgr[D_prec].append([acc,ipmiters,gmresiters])
            elif prec == "chol_rp":
                dict_cholrp[D_prec].append([acc,ipmiters,gmresiters])
            elif prec == "rff":
                dict_rff[D_prec].append([acc,ipmiters,gmresiters])
            elif prec == "nystrom":
                dict_nystr[D_prec].append([acc,ipmiters,gmresiters])
        	    
            #################################################################################
            #################################################################################
            ## print results in comparison
            print("\n########################################################################")
            print("\nResults NFFTSVMipm:")
            print("------------------------\n")
            print("dict_cholgr:", dict_cholgr)
            print("dict_cholrp:", dict_cholrp)
            print("dict_rff:", dict_rff)
            print("dict_nystr:", dict_nystr)