import numpy as np
from systems import build_sys
from data_gen import data_gen
from lqr import ctrl_design_LQR
from attacks import OptDelta_GradProj, RandDelta_Uniform


def demonstration():
    np.random.seed(3)
    pert_norm = 'max'
    w = 0.0
    v = 0.0
    q = 1.0
    r = 1e-2
    gamma = 1e-4
    Ite_num = 5  # fewer iterations for speed
    T_str = 'medium'
    pert_size = 5e-3

    A, B, Q, R, n, m, T, rho = build_sys('TT', q, r, T_str)
    D, D_norms, U, X, Z = data_gen(n, m, T, w, v, A, B, pert_norm)

    K_dd = ctrl_design_LQR(D, n, m, Q, R, gamma)
    step_size = max(D_norms) * pert_size
    Delta, flag_des = OptDelta_GradProj(D, n, m, Q, R, gamma, pert_size, pert_norm,
                                        D_norms, step_size, rho, Ite_num)
    D_Del = D + Delta
    K_Del = ctrl_design_LQR(D_Del, n, m, Q, R, gamma)

    Delta_rand = RandDelta_Uniform(D, n, m, pert_size, pert_norm, D_norms)
    K_Del_rand = ctrl_design_LQR(D + Delta_rand, n, m, Q, R, gamma)

    print("rho(K_dd)=", rho(K_dd))
    print("rho(K_Del)=", rho(K_Del))
    print("rho(K_Del_rand)=", rho(K_Del_rand))

if __name__ == "__main__":
    demonstration()
