def kernel_matvec(inx, KER, y):
    """
    Computing multiplications with the kernel matrix of the form YKY*inx quickly.
            
    Parameters
    ----------
    inx : ndarray
        The vector that is multiplied to YKY from the right.
    KER : object
        LinearOperator for approximating a kernel vector product.
    y : ndarray
        The vector that is multiplied to the kernel matrix form both sides.
        
    Returns
    -------
    out : ndarray
        The resulting vector.
    """
    out = y * inx
    out = KER(out)
    out = y * out
    
    return out
