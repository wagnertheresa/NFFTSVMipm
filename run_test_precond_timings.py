"""
Author: Theresa Wagner <theresa.wagner@mathematik.tu-chemnitz.de>

Corresponding publication:
"A Preconditioned Interior Point Method for Support Vector Machines Using an
ANOVA-Decomposition and NFFT-Based Matrix–Vector Products"
by T. Wagner, John W. Pearson, M. Stoll (2023)

Execute this file to reproduce the results presented in Figure 2.
"""

from func_prec_timings import main_precond_timings

##################################################################################

####################
## CHOOSE PARAMETERS

# choose kernel definition
kernel = 1 # Gaussian kernel
#kernel = 3 # Matérn(1/2) kernel

# choose data set
data = "cod_rna"

# choose subset sizes
Ndata = [10000, 50000, 100000, 0] # 0 corresponds to entire data set

# define list of preconditioner candidates
prec_list = ["chol_greedy", "chol_rp", "rff", "nystrom"]

# define list of target preconditioner rank candidates
rank_list = [50, 200, 1000]

####################
# initialize dict for results
dict_cholgr = {r: [] for r in rank_list}
dict_cholrp = {r: [] for r in rank_list}
dict_rff = {r: [] for r in rank_list}
dict_nystr = {r: [] for r in rank_list}

for n in Ndata:
    print("Solving for dimension", n)
    ###########################
    # work with cod-rna dataset
    from data_SVMipm import cod_rna
            
    X_train, X_test, y_train, y_test = cod_rna(n)
    
    wind_param_list = [[[0, 2, 3], [7, 6, 5], [4, 1]], [[7.199384124614643, 5.753872018666903, 6.129828390835456], 0.3407055254938164]]
    ###########################
    
    print("\nDataset:", data)
    print("--------\nShape train data:", X_train.shape)
    print("Shape test data:", X_test.shape)
    
    ###########################
    
    for prec in prec_list:
        print("Solving for preconditioner", prec)
        
        for D_prec in rank_list:
            print("Solving for rank", D_prec)
    
            # measure precond timings
            precond_time = main_precond_timings(X_train, X_test, y_train, y_test, wind_param_list, prec, D_prec, kernel)
            
            # save values to dict
            if prec == "chol_greedy":
                dict_cholgr[D_prec].append(precond_time)
            elif prec == "chol_rp":
                dict_cholrp[D_prec].append(precond_time)
            elif prec == "rff":
                dict_rff[D_prec].append(precond_time)
            elif prec == "nystrom":
                dict_nystr[D_prec].append(precond_time)
               
            print("###############################################################")
        
            print("\nDataset:", data)
            print("--------\nShape train data:", X_train.shape)
            print("Shape test data:", X_test.shape)
            print("prec:", prec)
            print("D_prec:", D_prec)
            print("results prec time:", precond_time)
            
            print("###############################################################")
        
            print("\nResults Precond Timings:")
            print("------------------------\n")
            print("dict_cholgr:", dict_cholgr)
            print("dict_cholrp:", dict_cholrp)
            print("dict_rff:", dict_rff)
            print("dict_nystr:", dict_nystr)
