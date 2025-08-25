import numpy as np


def build_sys(sys_flag="TT", q=1.0, r=1e-2, T_str="medium"):
    """Construct example systems similar to build_sys.m. Currently supports 'TT'."""
    if sys_flag == "TT":
        A = np.array([[0.96, 0.0, 0.0],
                      [0.04, 0.97, 0.0],
                      [-0.04, 0.0, 0.9]])
        B = np.array([[8.8, -2.3, 0.0],
                      [0.2, 2.2, 4.9],
                      [-0.21, -2.2, 1.9]])
    else:
        raise NotImplementedError("Only 'TT' system is implemented in Python port.")

    n, m = B.shape
    Q = q * np.eye(n)
    R = r * np.eye(m)
    if T_str == 'short':
        T = n + m + 1
    elif T_str == 'medium':
        T = 2 * (n + m)
    elif T_str == 'long':
        T = 3 * (n + m)
    else:
        raise ValueError('unknown T_str')
    rho = lambda K: np.max(np.abs(np.linalg.eigvals(A + B @ K)))
    return A, B, Q, R, n, m, T, rho
