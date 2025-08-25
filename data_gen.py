import numpy as np
from typing import Tuple


def gen_sig_exp_rep(n: int, m: int, T: int, w: float, v: float, A: np.ndarray, B: np.ndarray):
    """Generate input/state data for system x_{k+1}=A x_k + B u_k with noise."""
    x = np.zeros((n, T + 1))
    u = np.random.randn(m, T)
    for t in range(T):
        w_t = w * np.random.randn(n)
        x[:, t + 1] = A @ x[:, t] + B @ u[:, t] + w_t
    x_meas = x[:, :T] + v * np.random.randn(n, T)
    z_meas = x[:, 1:] + v * np.random.randn(n, T)
    return u, x_meas, z_meas


def data_gen(n, m, T, w, v, A, B, pert_norm='max'):
    U, X, Z = gen_sig_exp_rep(n, m, T, w, v, A, B)
    D = np.vstack((Z, X, U))
    if pert_norm == 'Fro':
        Z_nor = np.linalg.norm(Z, 'fro')
        X_nor = np.linalg.norm(X, 'fro')
        U_nor = np.linalg.norm(U, 'fro')
    else:
        Z_nor = np.max(np.abs(Z))
        X_nor = np.max(np.abs(X))
        U_nor = np.max(np.abs(U))
    D_norms = (Z_nor, X_nor, U_nor)
    return D, D_norms, U, X, Z
