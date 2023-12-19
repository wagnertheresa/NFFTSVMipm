import numpy as np
import pandas as pd
import random
import time
import scipy

import fastadj
# import libsvm.svmutil for comparison 
from libsvm import svmutil

# import auxiliary functions
from svm_ipm import svm_ipm
from precond import pivoted_chol_rp
from svm_predict_fastsum import svm_predict_fastsum
from data_preprocessing import data_preprocess

from sklearn.feature_selection import mutual_info_classif

class NFFTSVMipm:
    """
    Preconditioned Interior-Point Method for Support Vector Machines
    
    Parameters
    ----------
    sigma : float, default=1.0
         Sigma parameter for the Gaussian kernel.
    C : float, default=1.0
        The regularization parameter controlling the amount of misclassification.
        The relative weight of error vs. margin, such that 0 <= alpha <= C.
    indiv_sig : bool, default=True
        Whether the kernel consists of a sum of kernels, i.e. several sigmas are needed.
        If classifier="NFFTSVMipm" indiv_sig=True if len(windows)>1, indiv_sig=False else.
        If classifier="LIBSVM" indiv_sig=False.
    D_prec : int, default=200
        The desired rank of the preconditioner.
    sigma_br : float, default=0.6
        The barrier reduction parameter.
    windows : list, default=[]
        The list of windows determining the feature grouping.
    weights : float, default=1.0
        The weight for the weighted sum of kernels.
    fastadj_setup : str, default="default"
        Defines the desired approximation accuracy of the NFFT fastsum method. It is one of the strings 'fine', 'default' or 'rough'.
        
    Attributes
    ----------
    
    alpha_fast : ndarray
        The dual-variable for the SVM-Model.
    Xtrain : ndarray
        The training data used to fit the model.
    ytrain : ndarray
        The corresponding target vector.
    
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import train_test_split
    >>> N, d = 25000, 15
    >>> rng = np.random.RandomState(0)
    >>> X = rng.randn(N, d)
    >>> y = np.sign(rng.randn(N))
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=42)
    >>> clf = NFFTSVMipm
    >>> clf.fit(X_train, y_train)
    >>> clf.predict(X_test)
    """
    
    def __init__(self, sigma=1, C=1, indiv_sig=True, D_prec=200, sigma_br=0.6, windows=[], weights=1.0, fastadj_setup="default"):
        self.sigma = sigma
        self.C = C
        self.indiv_sig = indiv_sig
        self.D_prec = D_prec
        self.sigma_br = sigma_br
        self.windows = windows
        self.weights = weights
        self.fastadj_setup = fastadj_setup
        
    ############################################################################
        
    def init_fast_matvec(self, X_train):
        """
        Set up computations with the adjacency matrix and create adjacency matrix object.
            
        Parameters
        ----------
        X_train : ndarray
            The train data.
        
        Returns
        -------
        adj_mats : object
            The adjacency matrix object.
        """
        ## setup computations with the adjacency matrices
        if self.indiv_sig == True:
            adj_mats = [fastadj.AdjacencyMatrix(X_train[:,self.windows[l]], self.sigma[l], setup=self.fastadj_setup, diagonal=1.0) for l in range(len(self.windows))]
        else:
            adj_mats = [fastadj.AdjacencyMatrix(X_train[:,self.windows[l]], self.sigma, setup=self.fastadj_setup, diagonal=1.0) for l in range(len(self.windows))]
        
        return adj_mats
        
    ############################################################################  
    
    def fast_matvec(self, adj_mats, p):
        """
        Approximate matrix-vector product A*p, where A = K1 + ... + KP, with equal weights 
        
        Parameters
        ----------
        adj_mats : object
            The adjacency matrix object.
        p : ndarray
            The vector, whose product A*p with the matrix A shall be approximated.

        Returns
        -------
        Ap : ndarray
            The approximated matrix-vector product A*p.
        """    
        if self.windows == None:

            Ap = self.weights * adj_mats.apply(p)
        
        else:
            
            # perform kernel-vector multiplication
            Ap_i = np.asarray([adj_mats[l].apply(p) for l in range(len(self.windows))])
            
            # add weights and sum weighted sub-kernels up
            Ap = self.weights * np.sum(Ap_i, axis=0)
            
        return Ap
        
    ##############################################################################
    
    def fit(self, X_train, y_train, prec, iter_ip):
        """
        Perform the IPM for training the SVM on the train data.
        
        Parameters
        ----------
        X_train : ndarray
            The train data.
        y_train : ndarray
            The corresponding target vector.
        prec : str
            The preconditioner that shall be used for the IPM.
        iter_ip : int
            The maximum number of interior point iterations.

        Returns
        -------
        alpha_fast : ndarray
            The learned classifier parameter.
        GMRESiter_fast : list
            Number of GMRES iterations within the interior points iterations.
        """            
        # set up computations with adjacency matrix
        adj_mats = self.init_fast_matvec(X_train)
        
        # define LinearOperator for fast matrix-vector multiplications
        KER_fast = lambda p: self.fast_matvec(adj_mats, p)

        #####################
        ## PRECONDITIONING
        #####################
        # preconditioning with cholesky --> see svm_ipm.py
        if prec == "chol_greedy":
            ## GREEDY-BASED PIVOTED CHOLESKY APPROACH      
            MM = self.D_prec
            n = len(y_train)
            Ldec = pivoted_chol_rp(MM,KER_fast,n,"greedy")
                
        ########################
        elif prec == "chol_rp":
            ## RANDOMIZED PIVOTED CHOLESKY APPROACH
            MM = self.D_prec
            n = len(y_train)
            Ldec = pivoted_chol_rp(MM,KER_fast,n,"rp")
            
        ######################
        # preconditioning with random fourier features
        elif prec == "rff":
            ## RANDOM FOURIER FEATURES APPROACH
            # https://github.com/hichamjanati/srf/blob/master/RFF-I.ipynb
            
            # initialize array of decompositions
            Ldec = []
            
            # 1 rff decomposition per window
            for l in range(len(self.windows)):
                
                # generate D_prec iid samples from p(w)
                if self.indiv_sig == True:
                    W = np.sqrt(2/(self.sigma[l]**2))*np.random.normal(size=(self.D_prec,(X_train[:,self.windows[l]]).shape[1]))
                else:
                    W = np.sqrt(2/(self.sigma**2))*np.random.normal(size=(self.D_prec,(X_train[:,self.windows[l]]).shape[1]))
                # generate D_prec iid samples from Uniform(0,2*pi)
                b = 2*np.pi*np.random.rand(self.D_prec)
                
                Zl = np.sqrt(2/self.D_prec) * np.cos(((X_train[:,self.windows[l]]).dot(W.conj().T) + b[np.newaxis,:]))
            
                Ldec.append(Zl)
            
            Ldec = np.concatenate(Ldec, axis=1)
        
        ###########################
        elif prec == "nystrom":
            ## Nyström APPROACH
            
            # setup Nyström decomposition
            k = self.D_prec
            G = np.random.randn(X_train.shape[0],k)
            
            Y_ny = np.zeros((G.shape))
            for i in range(k):
                Y_ny[:,i] = KER_fast(G[:,i])
            Q = np.linalg.qr(Y_ny)[0]
            
            AQ = np.zeros((Q.shape))
            for j in range(k):
                AQ[:,j] = KER_fast(Q[:,j])    
                
            QaAQ = Q.T @ AQ
            
            # compute LDL^T decomposition
            LL, D, per = scipy.linalg.ldl(QaAQ)
            D = D.clip(min = 1e-2)
            L = LL@scipy.linalg.sqrtm(D)
            
            Ldec = np.zeros((X_train.shape[0],k))
            Ldec = np.linalg.solve(L.T,AQ.T)
            Ldec = Ldec.T
       
        ########################

        [alpha_fast, GMRESiter_fast] = svm_ipm(KER_fast,y_train,self.C,iter_ip,1e-1,self.sigma_br,100,1e-3,prec,Ldec)
        
        print("GMRES-iterations in Fastsum:", GMRESiter_fast)
        
        self.alpha_fast = alpha_fast
        self.Xtrain = X_train
        self.ytrain = y_train
        
        return alpha_fast, GMRESiter_fast
    
    ##############################################################################
    
    def predict(self, X_test):
        """
        Predict class affiliations for the test data.
        
        Parameters
        ----------
        X_test : ndarray
            The test data.

        Returns
        -------
        yhat_fast : ndarray
            The predicted class affiliations for the test data.
        """        
        ## NFFT-BASED FAST SUMMATION APPROACH
        yhat_fast = svm_predict_fastsum(X_test,self.alpha_fast,self.ytrain,self.Xtrain,self.sigma,self.windows,self.weights,self.fastadj_setup)
        
        return yhat_fast

#################################################################################
    
#################################################################################

class RandomSearch:
    """
    https://medium.com/analytics-vidhya/how-does-random-search-algorithm-work-python-implementation-b69e779656d6
    
    Hyperparameter optimization based on RandomSearch.
    
    Parameters
    ----------
    classifier : str, default="NFFTSVMipm"
        The classifier parameter determines, for which classifier RandomSearch shall be performed.
        It is either "NFTTSVMipm" or "LIBSV;".
    lb : list
        List of lower bounds for the parameters sigma/gamma and C.
    ub : list
        List of upper bounds for the parameters sigma/gamma and C.
    max_iters_rs : int, default=25
        Maximum number of iterations in RandomSearch.
    mis_threshold : float, default=0.0
        Mutual information score threshold determining, which features to include in the kernel. All features with a score below this threshold are dropped, the others are included.
    window_scheme : str, default="mis"
        The window-scheme argument determining how the windows shall be built.
        If "mis" is passed, the features are seperated up into windows following their mutual information scores in descending order.
        If "consec", the windows are built following the feature indices in ascending order.
        If "random", the windows of features are built randomly.
    weight_scheme : str, default="no weights"
        The weighting-scheme determining how the weights in the weighted sum of kernels are built.
        If weight_scheme="equally weighted", all weights are equal, so that they sum up to 1.
        If weight_scheme="no weights", all weights are 1.
    sigma_br : float, default=0.6
        Barrier reduction parameter used in the IPM.
    D_prec : int, default=50
        Rank of the low-rank decomposition based preconditioner for the IPM.
    prec : str, default="chol"
        The preconditioner that shall be used in the IPM.
    iter_ip : int, default=25
        Maximum number of interior point iterations.
    scoring : str, default="accuracy"
        The scoring parameter determines, which evaluation metric shall be used for measuring the prediction quality.
        It is either "accuracy", "precision" or "recall".

    Attributes
    ----------
    windows : list
        The list of windows determining the groups of features.
    weights : float
        The weight used for weighted sum of kernels.
    indiv_sig : bool
        Whether the kernel consists of a sum of kernels, i.e. several sigmas are needed.
        If classifier="NFFTSVMipm" indiv_sig=True if len(windows)>1, indiv_sig=False else.
        If classifier="LIBSVM" indiv_sig=False.
    lb_rs : list
        List of lower bounds for the parameters RandomSearch is performed on.
        If classifier="NFFTSVMipm" the number of sigma parameters equals the number of windows/kernels.
        If classifier="LIBSVM" the parameters gamma and C only exist once each.
    ub_rs : list
        List of upper bounds for the parameters RandomSearch is performed on.
        If classifier="NFFTSVMipm" the number of sigma parameters equals the number of windows/kernels.
        If classifier="LIBSVM" the parameters gamma and C only exist once each.
    
    Examples
    --------

    """
    
    def __init__(self, classifier, lb, ub, max_iter_rs=25, mis_threshold=0.0, window_scheme="mis", weight_scheme="no weights", sigma_br=0.6, D_prec=50, prec="chol", iter_ip=25, scoring="accuracy"):
        
        self.classifier = classifier
        self.lb = lb
        self.ub = ub
        self.max_iter_rs = max_iter_rs
        self.mis_threshold = mis_threshold
        self.window_scheme = window_scheme
        self.weight_scheme = weight_scheme
        self.sigma_br = sigma_br
        self.D_prec = D_prec
        self.prec = prec
        self.iter_ip = iter_ip
        self.scoring = scoring
        
    #############################################################################
        
    def evaluation_metrics(self, Y, YPred):
        """
        Evaluate the quality of a prediction.
            
        Parameters
        ----------
        Y : ndarray
            The target vector incorporating the true labels.
        YPred : ndarray
            The predicted class affiliations.
            
        Returns
        -------
        accuracy : float
            Share of correct predictions in all predictions.
        precision : float
            Share of true positives in all positive predictions.
        recall : float
            Share of true positives in all positive values.
        """
        # initialize TP, TN, FP, FN (true positive, true negative, false positive, false negative)
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for j in range(len(Y)):
            #print(Y[j], YPred[j])
            if Y[j]==1.0:
                if YPred[j]==1.0:
                    TP += 1
                elif YPred[j]==-1.0:
                    FN += 1
                else:
                    print("neither predicted class 1 nor -1 for test_sample:", j)
                    print("YPred[j]", YPred[j])

            elif Y[j]==-1.0:
                if YPred[j]==1.0:
                    FP += 1
                elif YPred[j]==-1.0:
                    TN += 1
                else:
                    print("neither predicted class 1 nor -1 for test_sample:", j)
                    print("YPred[j]", YPred[j])
                
        if (TP+TN) == 0:
            accuracy = 0
        else:
            accuracy = np.divide((TP+TN), len(Y))
        if TP == 0:
            precision = 0
            recall = 0
        else:
            precision = np.divide(TP, (TP+FP))
            recall = np.divide(TP, (TP+FN))
            
        # return evaluation metrics
        return [accuracy, precision, recall]
    
    #############################################################################
    
    def make_mi_scores(self, X, y):
        """
        Compute the mutual information scores.
            
        Parameters
        ----------
        X : ndarray
            The data matrix.
        y : ndarray
            The target vector incorporating the labels.
            
        Returns
        -------
        res_idx : list
            List of feature indices following their mutual information scores in descending order.   
        """
        threshold = self.mis_threshold
        
        mi_scores = mutual_info_classif(X, y, random_state=0)
        mi_scores = pd.Series(mi_scores, name="MI Scores")
        mi_scores = mi_scores.tolist()
        
        # sort scores in descending order and covert np arrays to list
        sorted_scores = (np.sort(mi_scores)[::-1]).tolist()
        sorted_idx = (np.argsort(mi_scores)[::-1]).tolist()
        
        # adjust threshold, if not enough features have a score above threshold
        while len([i for i in sorted_scores if i >= threshold]) < 3:
            print("Too many features are discarded with the chosen MIS-threshold. The threshold will be halved in the following.")
            threshold = threshold * 0.5
        
        # drop features with mi_score below threshold
        res_scores = [i for i in sorted_scores if i >= threshold]
        res_idx = sorted_idx[:len(res_scores)]
        
        return res_idx
        
    ##############################################################################
    
    def get_windows(self, mi_idx):
        """
        Construct a list of windows of features based on MIS-ranking.
            
        Parameters
        ----------
        windows : list
            The corresponding list of feature windows.
        
        Returns
        -------
        windows : list
            List of windows of features.
        """
        # number of features
        d = len(mi_idx)
            
        # create windows of length 3
        windows = [mi_idx[(l*3):(l*3)+3] for l in range(d//3)]
        
        # if |d| is not divisible by 3, the last window contains only 1 or 2 indices
        if d%3 != 0:
            windows.append([mi_idx[i] for i in range(d - d%3,d)])
            
        return windows
    
    ###############################################################################
    
    def preprocess(self, X_train, y_train, X_test):
        """
        Balance train and z-score normalize train and test data and determine windows and weights for the sum of kernels.
            
        Parameters
        ----------
        X_train : ndarray
            The training data.
        y_train : ndarray
            The target vector incorporating the true labels for the training data.
        X_test : ndarray
            The test data.
            
        Returns
        -------
        X_train : ndarray
            The balanced and z-score normalized train data.
        y_train : ndarray
            The corresponding target vector to the balanced train data.
        X_test : ndarray
            The z-score normalized test data.
        """
        ############################################################
        # preprocess data: balance train data and z-score normalize
        X_train, y_train, X_test = data_preprocess(X_train, y_train, X_test, balance=True)
        
        if self.classifier == "NFFTSVMipm":
            #######################
            ## determine windows of features
            # determine windows of features by their mis
            if self.window_scheme == "mis":
                if X_train.shape[1] > 3:
                    res_idx = self.make_mi_scores(X_train, y_train)
                    self.windows = self.get_windows(res_idx)
                else:
                    self.windows = [list(range(X_train.shape[1]))]
                    
            # windows are built following the feature indices in ascending order
            elif self.window_scheme == "consec":
                
                d = X_train.shape[1]
            
                # add windows of length 3                
                wind = [list(range((l*3),(l*3) + 3)) for l in range(d//3)]
            
                # if |d| is not divisible by 3, the last window contains only 1 or 2 indices                
                if d%3 != 0:
                    wind.append([l for l in range(d - d%3, d)])
                
                self.windows = wind
                
            # create the windows randomly
            elif self.window_scheme == "random":
                d = list(range(X_train.shape[1]))
                
                idx_list = random.choices(d, k=X_train.shape[1])
                
                self.windows = self.get_windows(idx_list)
            
            ##################################
            
            ## compute kernel weights
                    
            # equally weighted kernels, so that weights sum up to 1
            if self.weight_scheme == "equally weighted":
                kweights = 1.0/len(self.windows)
            # no weighting, all weights are 1
            else:
                kweights = 1.0
            
            self.weights = kweights
    
            print("Windows:", self.windows)
            print("Weights:", self.weights)
        
        return X_train, y_train, X_test
    
    ############################################################################
    
    def optimize(self, params, X_train, y_train, X_test, y_test):
        """
        Fit and train the model, make predictions for unseen data points an measure the runtime.
            
        Parameters
        ----------
        params : list
            Parameter combination the model shall be performed on.
        X_train : ndarray
            The training data.
        y_train : ndarray
            The target vector incorporating the true labels for the training data.
        X_test : ndarray
            The test data.
        y_test : ndarray
            The target vector incorporating the true labels for the test data.
            
        Returns
        -------
        time_fit : float
            Fitting time for one run of RandomSearch on the parameter combination params.
        time_pred : float
            Predition time for one run of RandomSearch on the parameter combination params.
        result : list
            The evaluation metrics (accuracy, precision, recall) obtained by comparing the preditions for the unseen data with the true target values.
        """
        if self.classifier == "NFFTSVMipm":
            # measure time needed for fitting
            start_fit = time.time()
            
            clf = NFFTSVMipm(sigma=params[0], C=params[1], indiv_sig=self.indiv_sig, D_prec=self.D_prec, sigma_br=self.sigma_br, windows=self.windows, weights=self.weights)
        
            alpha, GMRESiter = clf.fit(X_train, y_train, self.prec, self.iter_ip)
            
            time_fit = time.time() - start_fit
            
            # measure time needed for predicting
            start_predict = time.time()
            
            #evaluation, evaluation_minimize = clf.predict(X_test)
            evaluation = clf.predict(X_test)
            
            time_pred = time.time() - start_predict
            
            # compute prediction result
            result = self.evaluation_metrics(y_test,evaluation)
                
            return time_fit, time_pred, result, GMRESiter
            
        elif self.classifier == "LIBSVM":
            
            param = svmutil.svm_parameter("-q")
            param.svm_type = 0 # C-SVC
            param.kernel_type = 2 # rbf kernel
            param.shrinking = 0
            param.nu = 0.5
            param.cost = 1
            param.cross_validation = False
            
            param.gamma = params[0]
            param.C = params[1]
        
            # measure time needed for fitting
            start_fit = time.time()
        
            problem = svmutil.svm_problem(y_train, X_train)

            train = svmutil.svm_train(problem, param)
            
            time_fit = time.time() - start_fit
            
            # measure time needed for predicting
            start_predict = time.time()
            
            pred_lbl, pred_acc, pred_val = svmutil.svm_predict(y_test, X_test, train, "-q")
        
            time_pred = time.time() - start_predict
        
            evaluation = pred_lbl
    
            # compute prediction result
            result = self.evaluation_metrics(y_test,evaluation)
                
            return time_fit, time_pred, result
            
    #############################################################################    
    
    def tune(self, X_train, y_train, X_test, y_test):
        """
        Optimize the hyperparameters using random search.
        
        Parameters
        ----------
        X_train : ndarray
            The training data.
        y_train : ndarray
            The corresponding labels for the training data.
        X_test : ndarray
            The test data.
        y_test : ndarray
            The corresponding labels for the test data.
            
        Returns
        -------
        best_params : list
            List of the parameters, which yield the highest value for the chosen scoring-parameter (accuracy, precision or recall).
        best_result : list
            List of the best results, where the chosen scoring-parameter is crucial.
        best_time_fit : float
            Fitting time of the run, that yielded the best result.
        best_time_pred : float
            Prediction time of the run, that yielded the best result.
        mean_total_time_fit : float
            Mean value over the fitting times of all candidate parameters.
        mean_total_time_pred : float
            Mean value over the prediction times of all candidate parameters.
        (D_prec : int
            Rank of the low-rank decomposition based preconditioner for the IPM.)
        best_GMRESiter : list
            Number of GMRES iterations at each IPM step for the run yielding the best prediction quality.
        mean_GMRESiter : list
            List of the mean number of GMRES iterations at each IPM step for all runs.
        
        """
        total_time_fit = []
        total_time_pred = []
        total_acc = []
        if self.classifier == "NFFTSVMipm":
            mean_GMRESiter = []
        
        # compute feature windows and weights and normalize data points
        X_train, y_train, X_test = self.preprocess(X_train, y_train, X_test)
        
        if self.classifier == "NFFTSVMipm":

            if len(self.windows) > 1 and all([type(self.windows[i])==list for i in range(len(self.windows))]):
                self.indiv_sig = True
                
                self.lb_rs = [np.array([self.lb[0]] * len(self.windows)), self.lb[1]]
                self.ub_rs = [np.array([self.ub[0]] * len(self.windows)), self.ub[1]]
                  
            else:
                self.indiv_sig = False
                
                self.lb_rs = self.lb
                self.ub_rs = self.ub
        
        elif self.classifier == "LIBSVM":
            self.indiv_sig = False
            self.lb_rs = self.lb
            self.ub_rs = self.ub
        
        dim_rs = len(self.lb_rs)

        best_params = [0] * dim_rs
        if self.indiv_sig == True:
            best_params[0] = [0] * len(self.windows)
        new_params = best_params.copy()
        best_result = [0,0,0]
        
        if self.indiv_sig == True:
            p = []
            for j in range(len(best_params[0])):
                p.append(random.uniform((self.lb_rs[0])[j], (self.ub_rs[0]))[j])
            best_params[0] = p
        else:
            best_params[0] = random.uniform(self.lb_rs[0], self.ub_rs[0])
        best_params[1] = random.uniform(self.lb_rs[1], self.ub_rs[1])
        
        if self.classifier == "NFFTSVMipm":
            best_time_fit, best_time_pred, best_result, best_GMRESiter = self.optimize(best_params, X_train, y_train, X_test, y_test)
        elif self.classifier == "LIBSVM":
            best_time_fit, best_time_pred, best_result = self.optimize(best_params, X_train, y_train, X_test, y_test)
            
        print("\nFirst Parameter:", best_params)
        print("First Result:", best_result)
        print("Time Fit:", best_time_fit)
        if self.classifier == "NFFTSVMipm":
            print("GMRESiters:", best_GMRESiter)
        
        total_time_fit.append(best_time_fit)
        total_time_pred.append(best_time_pred)
        total_acc.append(best_result[0])
        if self.classifier == "NFFTSVMipm":
            mean_GMRESiter.append(np.mean(best_GMRESiter))
        
        for _ in range(self.max_iter_rs-1):
            
            if self.indiv_sig == True:
                new_p = []
                for j in range(len(new_params[0])):
                    new_p.append(random.uniform((self.lb_rs[0])[j], (self.ub_rs[0]))[j])
                new_params[0] = new_p
            else:
                new_params[0] = self.lb_rs[0] + random.random() * (self.ub_rs[0] - self.lb_rs[0])
            new_params[1] = self.lb_rs[1] + random.random() * (self.ub_rs[1] - self.lb_rs[1])
    
            if np.greater_equal(new_params[0], self.lb_rs[0]).all() and np.greater_equal(new_params[1], self.lb_rs[1]).all() and np.less_equal(new_params[0], self.ub_rs[0]).all() and np.less_equal(new_params[1], self.ub_rs[1]).all():
                if self.classifier == "NFFTSVMipm":
                    new_time_fit, new_time_pred, new_result, new_GMRESiter = self.optimize(new_params, X_train, y_train, X_test, y_test)
                elif self.classifier == "LIBSVM":
                    new_time_fit, new_time_pred, new_result = self.optimize(new_params, X_train, y_train, X_test, y_test)
            
                total_time_fit.append(new_time_fit)
                total_time_pred.append(new_time_pred)
                total_acc.append(new_result[0])
                if self.classifier == "NFFTSVMipm":
                    mean_GMRESiter.append(np.mean(new_GMRESiter))
                
            else:
                new_result = [0,0,0]
            
            print("\nNew Parameter:", new_params)
            print("New Result:", new_result)
            print("Time Fit:", new_time_fit)
            if self.classifier == "NFFTSVMipm":
                print("GMRESiter:", new_GMRESiter)
            
            if self.scoring == "accuracy":
                if new_result[0] > best_result[0]:
                    best_params = new_params
                    best_result = new_result
                    best_time_fit = new_time_fit
                    best_time_pred = new_time_pred
                    if self.classifier == "NFFTSVMipm":
                        best_GMRESiter = new_GMRESiter

            elif self.scoring == "precision":
                if new_result[1] > best_result[1]:
                    best_params = new_params
                    best_result = new_result
                    best_time_fit = new_time_fit
                    best_time_pred = new_time_pred
                    if self.classifier == "NFFTSVMipm":
                        best_GMRESiter = new_GMRESiter

            elif self.scoring == "recall":
                if new_result[2] > best_result[2]:
                    best_params = new_params
                    best_result = new_result
                    best_time_fit = new_time_fit
                    best_time_pred = new_time_pred
                    if self.classifier == "NFFTSVMipm":
                        best_GMRESiter = new_GMRESiter
            
        if self.classifier == "NFFTSVMipm":
            return best_params, best_result, best_time_fit, best_time_pred, np.mean(total_time_fit), np.mean(total_time_pred), self.D_prec, best_GMRESiter, mean_GMRESiter
        elif self.classifier == "LIBSVM":
            return best_params, best_result, best_time_fit, best_time_pred, np.mean(total_time_fit), np.mean(total_time_pred)
