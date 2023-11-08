from scipy.sparse.linalg import spsolve, inv
from scipy.sparse import spdiags, tril, triu
import numpy as np


def wjacobi(A, f, u, omega):
    n = len(u)+1
    Dinv = inv(spdiags(A.diagonal(), [0], (n-1, n-1)))
    U, L = triu(A, 1), tril(A, -1)
    return (1-omega)*u + omega*Dinv @ (f-(U+L)@u)


def isPowerOf2(n): return (n & (n - 1)) == 0


def geoVcycle1D(mat, f, u, recursion_depth):

    if not (isPowerOf2(len(u))):
        raise ValueError("Length of u must be power of 2")

    if np.log2(len(u)) < recursion_depth:
        raise ValueError("to big recursion_depth")

    n = len(u) + 1
    A = mat(n)
    if recursion_depth == 0:
        return spsolve(A, f)

    u = relax(A, f, u, omega=2/3)
    r = A@u - f
    r_coarse, f_coarse = restrict(r), restrict(f)
    e_coarse = geoVcycle1D(mat, f_coarse, r_coarse, recursion_depth-1)
    e = interpolate(e)
    u += e
    u = relax(A, f, u, omega=2/3)
    return u
