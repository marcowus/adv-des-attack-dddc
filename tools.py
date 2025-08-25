import numpy as np


def spec_rad_clsys(A: np.ndarray, B: np.ndarray, K: np.ndarray) -> float:
    """Return spectral radius of closed loop system A + B K."""
    eigvals = np.linalg.eigvals(A + B @ K)
    return float(np.max(np.abs(eigvals)))


def sep_ZXU(D: np.ndarray, n: int, m: int):
    """Separate data matrix D into Z,X,U blocks."""
    Z = D[:n]
    X = D[n:2*n]
    U = D[2*n:2*n+m]
    return Z, X, U


def proj_block(Del, V_norm, pert_size, pert_norm):
    tmp = pert_size * V_norm
    if pert_norm == 'Fro':
        norm = np.linalg.norm(Del, 'fro')
        if norm >= tmp:
            return tmp * Del / norm
        return Del
    elif pert_norm == 'max':
        return np.clip(Del, -tmp, tmp)
    else:
        raise ValueError('unknown norm')


def proj_D(Del, D_norms, pert_size, pert_norm, n, m):
    Del_Z, Del_X, Del_U = sep_ZXU(Del, n, m)
    Z_nor, X_nor, U_nor = D_norms
    Del_Z = proj_block(Del_Z, Z_nor, pert_size, pert_norm)
    Del_X = proj_block(Del_X, X_nor, pert_size, pert_norm)
    Del_U = proj_block(Del_U, U_nor, pert_size, pert_norm)
    return np.vstack((Del_Z, Del_X, Del_U))
