import numpy as np
import cvxpy as cp


def ctrl_design_LQR(D, n, m, Q, R, gamma=1e-4):
    T = D.shape[1]
    Z = D[:n]
    X = D[n:2*n]
    U = D[2*n:2*n+m]
    W0 = np.vstack((X, U))
    Rs = np.linalg.cholesky(R)
    S = cp.Variable((m, m), PSD=True)
    L = cp.Variable((T, n))
    P = cp.Variable((n, n), PSD=True)
    Pi = np.eye(T) - np.linalg.pinv(W0) @ W0
    objective = cp.trace(Q @ X @ L) + cp.trace(S) + gamma * cp.sum_squares(Pi @ L)
    constraints = [
        cp.bmat([[S, Rs @ U @ L], [(Rs @ U @ L).T, P]]) >> 0,
        cp.bmat([[P - np.eye(n), Z @ L], [(Z @ L).T, P]]) >> 0,
        P == X @ L
    ]
    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(solver=cp.SCS, verbose=False)
    if L.value is None:
        raise ValueError("cvxpy failed")
    Lval = L.value
    Xval = X
    Uval = U
    K = Uval @ Lval @ np.linalg.inv(Xval @ Lval)
    return K
