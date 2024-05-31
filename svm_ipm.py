import numpy as np
import scipy
from scipy.sparse.linalg import LinearOperator
import warnings

from kernel_matvec import kernel_matvec
from hessian_ipm import hessian_ipm,hessian_ipm_pd
from precond import SMW_prec

def svm_ipm(KER, y, C, iter_ip, tol, sigma_br, Gmaxiter, Gtol, prec, Ldec=[]):
    """
    Interior point method for training a support vector machine.
    
    Parameters
    ----------  
    KER : LinearOperator
        The routine for multiplying the kernel matrix by a vector.
    y : ndarray
        The training target vector, containing entries 1 or -1.
    C : float
        The relative weight of error vs. margin, such that 0 <= alpha <= C.
    iter_ip : int
        The maximum number of interior point iterations.
    tol : float
        The interior point method convergence tolerance.
    sigma_br : float
        The barrier reduction parameter.
    Gmaxiter : int
        The maximum number of GMRES iterations.
    Gtol : float
        The GMRES convergence tolerance.
    prec : str
        The preconditioner that shall be used for the GMRES within the IPM.
    Ldec : ndarray, default=[]
        The low-rank decomposition matrix for the preconditioner.
        
    Returns
    -------
    alpha : ndarray
        The learned classifier parameter.
    GMRESiter : list
        List of the GMRES iterations at each IPM step.
    """
    # Set parameters
    gap = 0.99995
    n = len(y)
    GMRESiter = []
    
    # Initialize alpha: components of alpha corresponding to y=1 equal to
    # C*(1-p), components corresponding to y=-1 equal to C*p, with p fraction
    # of components with y=1.
    p = np.sum(y>0)/n
    alpha = np.tile(C*p, (n,))
    alpha[y>0] = C * (1-p)
    
    print(type(alpha))
    # Initialize barrier parameter mu and dual variable lambda_svm
    mu = 1.0
    mu_vec = [mu]
    lambda_svm = 0
    
    # Specify function for matrix-vector product
    MAT = lambda x: kernel_matvec(x, KER, y)
    
    # Compute initial infeasibilities, norms & primal/dual tolerances required
    xid = 1 - MAT(alpha) + lambda_svm * y + mu * (C - 2*alpha)/(alpha*(C-alpha))
    xip = alpha.conj().T @ y
    nrmxid = np.linalg.norm(xid)
    dtol = tol * (1 + np.sqrt(n)) # 1+||c|| in 25 years paper
    nrmxip = np.linalg.norm(xip)
    ptol = tol * (1+0) # 1+||b|| in 25 years paper
    
    # Set iteration counter
    i = 0
        
    # Iteration loop for interior point method
    while (i < iter_ip) & (mu > tol or nrmxip > ptol or nrmxid > dtol):
              
        i = i+1
        # Solve Newton system using MINRES to obtain search direction
        rhs = np.append(xid, xip)
        rhs = np.reshape(rhs,(len(rhs),))
                            
        # initialize counter for GMRES iterations taken
        Giter = 0
        # function to count number of iterations needed in gmres
        def callback_gmres(rk):
            nonlocal Giter
            Giter += 1
        
# =============================================================================
         # A as matrix -> see left side of (2) in paper
        Ax = lambda x: hessian_ipm(x,KER,y,n,C,alpha,mu)
        A = LinearOperator((n+1,n+1), Ax)
# =============================================================================
        
        # prepare usage of preconditioner
        dh = 1/(C-alpha)**2 + 1/alpha**2
        dhm = mu * dh
            
        # setup SMW preconditioner
        dinv = scipy.sparse.spdiags(1.0/dhm, 0, len(dhm), len(dhm))
        bb = Ldec.conj().T @ (dinv@Ldec)
        aa = np.eye(Ldec.shape[1])
        test = aa+bb
        PCx= lambda x : SMW_prec(x, n, Ldec, y, dhm, test)
        M = LinearOperator((n+1,n+1), PCx)
            
        [dx,flag] = scipy.sparse.linalg.gmres(A=A, b=rhs, tol=Gtol, restart=Gmaxiter, maxiter=1, M=M, callback=callback_gmres, callback_type='pr_norm')        
              
        # break if the GMRES iterations are too large
        # Store number of Gmres iterations taken
        GMRESiter.append(Giter)
        print("GMRESiters:", GMRESiter, "IPMiters:", i)
        if Giter>99:
            break;
        
        # Compute largest possible step size such that alpha remains in [0,C]
        dalpha = dx[:-1]
        stepL = -1/np.minimum(np.min(dalpha/(alpha+1e-15)),-1)
        stepU = -1/np.minimum(np.min(-dalpha/(C-alpha+1e-15)),-1)
        step = gap * np.minimum(stepL,stepU)
        
        # Update alpha and lambda
        alpha = alpha + step*dalpha
        lambda_svm = lambda_svm + step * dx[-1]
        
        # Update barrier parameter mu
        mu = sigma_br * mu
        
        # Set lower bound for mu to avoid getting too close to zero barrier
        if (mu < tol/10):
            mu = tol/10
        mu_vec.append(mu)
        
        # Compute infeasibilities and norms
        xid = 1 - MAT(alpha) + lambda_svm * y + mu * (C - 2*alpha)/(alpha*(C-alpha))
        nrmxid = np.linalg.norm(xid)
        xip = alpha.conj().T @ y 
        nrmxip = np.linalg.norm(xip)
        print("norm of stopping mu:",mu)
        print("Norm of the xip:", nrmxip, "norm of xid:", nrmxid)
        ########################################################################
        # STOPPING CRITERION when bad convergence in IPM
        if i == 1:
            # record first dual infeasibility
            di0 = nrmxid
        elif i == 4:
            # record previous infeasibilities for 5th iteration
            di = nrmxid
            pi = nrmxip
            # counter for whether the dual infeasibility has increased from the previous iteration
            di_incr = 0
        # check stopping criteria after 5 iterations
        elif i > 5:
            # record previous and current dual infeasibilities
            di_prev = di
            di = nrmxid
            # record previous and current primal infeasibilities
            pi_prev = pi
            pi = nrmxip
            if di_prev < di:
                # add 1 to counter if di has increased from previous iteration
                di_incr = di_incr + 1
            else:
                di_incr = 0
                
            # criterion on whether to stop the IPM due to bad convergence behavior
            if di_incr == 2 and di0 < di and pi_prev < pi:
                print("dual infeasibility increased 2 times in a row and primal one time and is larger than first one: break in IPM")
                break
            elif di_incr > 2 and pi_prev < pi:
                print("dual infeasibility increased 3 times in a row and primal one time: break in IPM")
                break
        ########################################################################
            
    # Warning if IPM has not converged; plot GMRES iterations at each IPM step
    if (mu > tol or nrmxip > ptol or nrmxid > dtol):
        warnings.warn("IPM has not converged")
        print("\nIPM has not converged!\n")
    
    return [alpha, GMRESiter]



def svm_ipm_pd(KER, ytrain, C, iter_ip, tol, sigma_br, Gmaxiter, Gtol, prec, Ldec=[]):
    """
    Interior point method for training a support vector machine.

    Parameters
    ----------
    KER : LinearOperator
        The routine for multiplying the kernel matrix by a vector.
    y : ndarray
        The training target vector, containing entries 1 or -1.
    C : float
        The relative weight of error vs. margin, such that 0 <= alpha <= C.
    iter_ip : int
        The maximum number of interior point iterations.
    tol : float
        The interior point method convergence tolerance.
    sigma_br : float
        The barrier reduction parameter.
    Gmaxiter : int
        The maximum number of GMRES iterations.
    Gtol : float
        The GMRES convergence tolerance.
    prec : str
        The preconditioner that shall be used for the GMRES within the IPM.
    Ldec : ndarray, default=[]
        The low-rank decomposition matrix for the preconditioner.

    Returns
    -------
    alpha : ndarray
        The learned classifier parameter.
    GMRESiter : list
        List of the GMRES iterations at each IPM step.
    """
    # Set parameters
    gap = 0.99995
    n = len(ytrain)
    GMRESiter = []

    # Initialize alpha: components of alpha corresponding to y=1 equal to
    # C*(1-p), components corresponding to y=-1 equal to C*p, with p fraction
    # of components with y=1.
    p = np.sum(ytrain>0)/n
    alpha = np.tile(C*p, (n,))
    alpha[ytrain>0] = C * (1-p)
    eta = np.ones(n)
    xi = np.ones(n)
    lambda_svm = 1



    # Initialize barrier parameter mu and dual variable lambda_svm
    mu = 1.0
    mu_vec = [mu]
    lambda_svm = 0

    # Specify function for matrix-vector product YKY
    MAT = lambda x: kernel_matvec(x, KER, ytrain)


    # Compute initial infeasibilities, norms & primal/dual tolerances required
    xid = 1 - MAT(alpha) + lambda_svm * ytrain + mu*1/alpha-mu*1/(C-alpha) # Right hand side first component
    xip = alpha.conj().T @ ytrain # Right hand side second component
    nrmxid = np.linalg.norm(xid)
    dtol = tol * (1 + np.sqrt(n)) # 1+||c|| in 25 years paper
    nrmxip = np.linalg.norm(xip)
    ptol = tol * (1+0) # 1+||b|| in 25 years paper

    # Set iteration counter
    i = 0
    # Iteration loop for interior point method
    while (i < iter_ip) & (mu > tol or nrmxip > ptol or nrmxid > dtol):

        i = i+1
        # Solve Newton system using MINRES to obtain search direction
        rhs = np.append(xid, xip)
        rhs = np.reshape(rhs,(len(rhs),))

        # initialize counter for GMRES iterations taken
        Giter = 0
        # function to count number of iterations needed in gmres
        def callback_gmres(rk):
            nonlocal Giter
            Giter += 1

        # Prepare the diagonal term
        DD = 1/alpha*eta+1/(C-alpha)*xi

        # =============================================================================
         # A as matrix -> see left side of (2) in paper
        Ax = lambda x: hessian_ipm_pd(x,KER,DD,ytrain,n,C,alpha,mu)
        A = LinearOperator((n+1,n+1), Ax)
        # =============================================================================

        # setup SMW preconditioner
        dinv = scipy.sparse.spdiags(1.0/DD, 0, len(DD), len(DD))
        bb = Ldec.conj().T @ (dinv@Ldec)
        aa = np.eye(Ldec.shape[1])
        test = aa+bb
        PCx= lambda x : SMW_prec(x, n, Ldec, ytrain,DD, test)
        M = LinearOperator((n+1,n+1), PCx)

        [dx,flag] = scipy.sparse.linalg.gmres(A=A, b=rhs, tol=Gtol, restart=Gmaxiter, maxiter=1, M=M, callback=callback_gmres, callback_type='pr_norm')

        # break if the GMRES iterations are too large
        # Store number of Gmres iterations taken
        GMRESiter.append(Giter)
        print("GMRESiters:", GMRESiter, "IPMiters:", i)
        if Giter>99:
            break;

        # Compute largest possible step size such that alpha remains in [0,C]
        dalpha = dx[:-1]
        dlambda = dx[-1]
        deta = mu/alpha-eta-eta/alpha*dalpha
        dxi = mu*1/(C-alpha)-xi+1/(C-alpha)*xi*dalpha


        # Update alpha and lambda
        stepL = -1/np.minimum(np.min(dalpha/(alpha+1e-15)),-1)
        stepU = -1/np.minimum(np.min(-dalpha/(C-alpha+1e-15)),-1)
        stepP = gap * np.minimum(stepL,stepU)
        alpha = alpha + stepP*dalpha

        # Update eta and xi
        stepEta = -1/np.minimum(np.min(deta/(eta+1e-15)),-1)
        stepXi = -1/np.minimum(np.min(dxi/(xi+1e-15)),-1)
        stepD = gap * np.minimum(stepEta,stepXi)
        eta = eta + stepD*deta
        xi = xi + stepD*dxi
        lambda_svm = lambda_svm + stepD * dlambda

        # Update barrier parameter mu
        mu = sigma_br * mu

        # Set lower bound for mu to avoid getting too close to zero barrier
        if (mu < tol/10):
            mu = tol/10
        mu_vec.append(mu)

        # Compute infeasibilities and norms
        # xid = 1 - MAT(alpha) + lambda_svm * ytrain + mu * (C - 2*alpha)/(alpha*(C-alpha))
        # nrmxid = np.linalg.norm(xid)
        # xip = alpha.conj().T @ ytrain
        # nrmxip = np.linalg.norm(xip)

        xid = 1 - MAT(alpha) + lambda_svm * ytrain + mu*1/alpha-mu*1/(C-alpha) # Right hand side first component
        xip = alpha.conj().T @ ytrain # Right hand side second component
        nrmxip = np.linalg.norm(xip)
        nrmxid = np.linalg.norm(xid)
        print("norm of stopping mu:",mu)
        print("Norm of the xip:", nrmxip, "norm of xid:", nrmxid)
        ########################################################################
        # STOPPING CRITERION when bad convergence in IPM
        ########################################################################

        # Warning if IPM has not converged; plot GMRES iterations at each IPM step
        if (mu > tol or nrmxip > ptol or nrmxid > dtol):
            warnings.warn("IPM has not converged")
            print("\nIPM has not converged!\n")

    return [alpha, GMRESiter]
