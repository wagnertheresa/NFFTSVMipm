import numpy as np
import fastadj
from sklearn.metrics import pairwise_distances

def svm_predict_fastsum(X_test, alpha, y_train, X_train, sigma, windows, weights, fastadj_setup):
    """
    Predict class affiliations for the test data.
            
    Parameters
    ----------
    X_test : ndarray
        The test data.
    alpha : ndarray
        The learned classifier parameter.
    y_train : ndarray
        The target vector incorporating the true labels for the training data.
    X_train : ndarray
        The training data.
    sigma : float
        Sigma parameter for the Gaussian kernel.
    windows : list
        The list of windows determining the feature grouping.
    weights : float
        The weight for the weighted sum of kernels.
    fastadj_setup : str
        Defines the desired approximation accuracy of the NFFT fastsum method. It is one of the strings 'fine', 'default' or 'rough'.
        
    Returns
    -------
    YPred : ndarray
        The predicted class affiliations for tha test data.
    """
    N_sum = X_test.shape[0] + X_train.shape[0]
    arr = np.append(X_train, X_test, axis=None).reshape(N_sum,X_train.shape[1])
    
    p = np.append(alpha*y_train, np.zeros(X_test.shape[0]), axis=None).reshape(N_sum,)
    
    if isinstance(sigma, float):
        adj_vals = [fastadj.AdjacencyMatrix(arr[:,windows[l]], sigma, setup=fastadj_setup, diagonal=1.0) for l in range(len(windows))]
    else:
        adj_vals = [fastadj.AdjacencyMatrix(arr[:,windows[l]], sigma[l], setup=fastadj_setup, diagonal=1.0) for l in range(len(windows))]

    ## predict responses
    # perform kernel-vector multiplication
    vals_i = np.asarray([adj_vals[l].apply(p) for l in range(len(windows))])
    # add weights and sum weighted sub-kernels up
    vals = weights * np.sum(vals_i, axis=0)
    
    # select predicted responses for test data
    YPred = np.sign(vals[-X_test.shape[0]:])
    
    return YPred
        

def kernelfun(X_test, X_train, sigma):
    """
    Evaluate the kernel function on the test and train data.
            
    Parameters
    ----------
    X_test : ndarray
        The test data.
    X_train : ndarray
        The training data.
    sigma : float
        Sigma parameter for the Gaussian kernel.
        
    Returns
    -------
    k : ndarray
        The kernel function evaluated on the test and train data.
    """
    k = np.exp(-(pairwise_distances(X_test, X_train))**2/(sigma**2))    
    
    return k