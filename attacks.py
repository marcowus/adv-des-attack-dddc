import numpy as np
from lqr import ctrl_design_LQR
from tools import proj_D


def finite_diff_grad(D, n, m, Q, R, gamma, rho, eps=1e-6):
    grad = np.zeros_like(D)
    for idx in np.ndindex(D.shape):
        perturb = np.zeros_like(D)
        perturb[idx] = eps
        Kp = ctrl_design_LQR(D + perturb, n, m, Q, R, gamma)
        Km = ctrl_design_LQR(D - perturb, n, m, Q, R, gamma)
        grad[idx] = (rho(Kp) - rho(Km)) / (2 * eps)
    return grad


def OptDelta_GradProj(D, n, m, Q, R, gamma, pert_size, pert_norm, D_norms, step_size,
                       rho, Ite_num=10):
    Delta = np.zeros_like(D)
    for _ in range(Ite_num):
        D_Del = D + Delta
        K = ctrl_design_LQR(D_Del, n, m, Q, R, gamma)
        if rho(K) > 1:
            return Delta, True
        grad = finite_diff_grad(D_Del, n, m, Q, R, gamma, rho)
        Delta += step_size * grad
        Delta = proj_D(Delta, D_norms, pert_size, pert_norm, n, m)
    return Delta, False


def RandDelta_Uniform(D, n, m, pert_size, pert_norm, D_norms):
    Delta = np.random.uniform(-1.0, 1.0, D.shape)
    Delta = proj_D(Delta, D_norms, pert_size, pert_norm, n, m)
    return Delta
