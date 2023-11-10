# this is the first time that I work with 2D finite difference matrices
# I number my nodes in the 2D box from left to right and bottom to top

from scipy.sparse import eye, diags, kron
from constructions1D import *
import numpy as np


def helmholtz2D(n, sigma):
    A_1D = n**2 * diags([-1, 2, -1], [-1, 0, 1], shape=(n-1, n-1))
    A_2D = kron(eye(n-1), A_1D) + kron(A_1D, eye(n-1))
    return A_2D + sigma * eye((n-1)**2)


def simple_restrict_matrix2D(n):
    if n % 2 != 0:
        raise ValueError("restrict matrix needs n even")
    R_1D = simple_restrict_matrix(n)
    return kron(R_1D, R_1D)


def simple_interpolate_matrix2D(n):
    I_1D = simple_interpolate_matrix(n)
    return kron(I_1D, I_1D)


def wave_basis_2D(n, k, l):
    return np.outer(wave_basis_1D(n, k), wave_basis_1D(n, l)).flatten()


def wave_basis_2Dx(n, k):
    return np.outer(np.ones(n-1), wave_basis_1D(n, k)).flatten()


def wave_basis_2Dy(n, k):
    return np.outer(wave_basis_1D(n, k), np.ones(n-1)).flatten()
