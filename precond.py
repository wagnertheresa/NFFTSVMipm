import numpy as np

#################################################################################

def pivoted_chol_rp(k, KER, n, alg):
    """
    Computing multiplications with the kernel matrix of the form YKY*inx quickly.
            
    Parameters
    ----------
    k : int
        Desired preconditioner rank.
    KER : object
        LinearOperator for approximating a kernel vector product.
    n : int
        Number of training data points.
    alg : str
        The Cholesky preconditioner that shall be used.
        If alg=="rp", the randomized pivoted Cholesky preconditioner is chosen.
        If alg=="greedy" the greedy-based Cholesky preconditioner is chosen.   
        
    Returns
    -------
    Ldec : ndarray
        Cholesky preconditioner decomposition matrix.
    """
    diags =  np.copy(get_diag(KER,n))

    # row ordering, is much faster for large scale problems
    F = np.zeros((k,n))
    rows = np.zeros((k,n))
    rng = np.random.default_rng()
    arr_idx = []
    for i in range(k):
        if alg == "rp":
            idx = rng.choice(range(n), p = diags / sum(diags))
        elif alg == "greedy":
            idx = np.argmax(diags)
        arr_idx.append(idx)
        rows[i,:] = get_row(idx,KER,n)
        F[i,:] = (rows[i,:] - F[:i,idx].T @ F[:i,:]) / np.sqrt(diags[idx])
        diags -= F[i,:]**2
        diags = diags.clip(min = 0)
        
    Ldec = np.transpose(F)
        
    return Ldec

##################################################################################

def SMW_prec(inx, n, Lp, y, dhm, test):
    """
    Linear operator for the Sherman-Morrison-Woodbury preconditioner.
            
    Parameters
    ----------
    inx : ndarray
        Array the SMW preconditioner shall be applied to.
    n : int
        Number of training data points.
    Lp : ndarray
        The low-rank decomposition matrix for the preconditioner.
    y : ndarray
        The training target vector.
    dhm : ndarray
        Diagonal barrier matrix theta from the paper.
    test : ndarray
        Matrix I_k + ZD^-1Z^T from the paper.    
    
    Returns
    -------
    out : ndarray
        The result of the SMW preconditioner applied to a vector.
    """
    # (1,1) block
    temp = inx[:n]/dhm # theta^{-1}
    z = y*temp # Y 
    z = Lp.conj().T@ z
    z = np.linalg.solve(test, z)
    z = Lp@z
    z = y*z
    z = z/dhm
    z = temp - z
    out = z

    # (2,2) block 
    out = np.append(out,inx[n])    
    
    return out

##################################################################################
    
def get_diag(KER, n):
    """
    Get diagonal of matrix approximated by operator KER.
    
    Note: Diagonal entries of kernel matrix are all equal.
            
    Parameters
    ----------
    KER : object
        LinearOperator for approximating a kernel vector product.
    n : int
        Shape parameter of square matrix approximated by operator KER.
    
    Returns
    -------
    out : ndarray
        Diagonal of matrix approximated by operator KER.
    """
    ei = np.zeros(n,)
    ei[0] = 1
    a = np.transpose(KER(ei))
    
    out = a[0]*np.ones((n,))
    return out

##################################################################################

def get_row(i, KER, n):
    """
    Get i-th row of matrix approximated by operator KER.
            
    Parameters
    ----------
    i : int
        Index of row that shall be returned.
    KER : object
        LinearOperator for approximating a kernel vector product.
    n : int
        Shape parameter of square matrix approximated by operator KER.
        
    Returns
    -------
    row : ndarray
       I-th row of matrix approximated by operator KER. 
    """
    ei = np.zeros(n,)
    ei[i] = 1
    
    row = np.transpose(KER(ei))
    return row
    
