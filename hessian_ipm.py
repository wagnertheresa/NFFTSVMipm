import numpy as np

def hessian_ipm(inx, KER, y, n, C, alpha, mu):
    """
    Solving the linear system which must be solved to determine the direction of the Newton step.
            
    Note
    ----
    See equation (2.1) in paper.
    
    Parameters
    ----------
    inx : ndarray
        The vector that is multiplied to YKY from the right.
    KER : object
        LinearOperator for approximating a kernel vector product.
    y : ndarray
        The vector that is multiplied to the kernel matrix form both sides.
    n : int
        Number of training data.
    C : float
        The regularization parameter controlling the amount of misclassification.
        The relative weight of error vs. margin, such that 0 <= alpha <= C.
    alpha : ndarray
        Langrange multipliers and learned classifier in the end.
    mu : float
        The barrier parameter being progressively reduced towards zero within ithe IPM routine.
        
    Returns
    -------
    out : ndarray
        The resulting vector.
    """
    out = np.zeros((n+1,))
    
    # (1,1) block
    out[:n] = y * inx[:n]
    out[:n] = KER(out[:n])
    out[:n] = y * out[:n]
    out[:n] = out[:n] + mu*(1/(C-alpha)**2 + 1/alpha**2) * inx[:n]
    
    # (1,2) block
    out[:n] = out[:n] - y * inx[n]
    
    # (2,1) block 
    out[n] = -y.conj().T @ inx[:n]
    
    return out

def hessian_ipm_pd(inx, KER, DD,y, n, C, alpha, mu):
    """
    Solving the linear system which must be solved to determine the direction of the Newton step.

    Note
    ----
    See equation (2.1) in paper.

    Parameters
    ----------
    inx : ndarray
        The vector that is multiplied to YKY from the right.
    KER : object
        LinearOperator for approximating a kernel vector product.
    y : ndarray
        The vector that is multiplied to the kernel matrix form both sides.
    n : int
        Number of training data.
    C : float
        The regularization parameter controlling the amount of misclassification.
        The relative weight of error vs. margin, such that 0 <= alpha <= C.
    alpha : ndarray
        Langrange multipliers and learned classifier in the end.
    mu : float
        The barrier parameter being progressively reduced towards zero within ithe IPM routine.

    Returns
    -------
    out : ndarray
        The resulting vector.
    """
    out = np.zeros((n+1,))

    # (1,1) block
    out[:n] = y * inx[:n]
    out[:n] = KER(out[:n])
    out[:n] = y * out[:n]
    out[:n] = out[:n] + DD*inx[:n]

    # (1,2) block
    out[:n] = out[:n] - y * inx[n]

    # (2,1) block
    out[n] = -y.conj().T @ inx[:n]

    return out
