from constructions import simple_restrict_matrix, simple_interpolate_matrix
from plt_utils import *
from scipy.sparse.linalg import spsolve
from scipy.sparse import spdiags, tril, triu


def wjacobi(A, f, u, omega):
    n = len(u)+1
    Dinv = spdiags(1/A.diagonal(), [0], (n-1, n-1))
    U, L = triu(A, 1), tril(A, -1)
    return (1-omega)*u + omega*Dinv @ (f-(U+L)@u)


def simple_restrict(v):
    n = len(v)+1
    return simple_restrict_matrix(n)@v


def simple_interpolate(v):
    n = len(v)+1
    return simple_interpolate_matrix(n)@v


def geoVcycle1D(mat, f, u, nu1, nu2, relax, restrict, interpolate, recursion_depth):
    n = len(u) + 1
    if not (n % 2**recursion_depth == 0):
        raise ValueError("Length of u must be divisible by 2**recursion depth")
    A = mat(n)
    if recursion_depth == 0:
        return spsolve(A, f)

    for _ in range(nu1):
        u = relax(A, f, u)

    r_coarse = restrict(f - A@u)
    e_coarse = geoVcycle1D(mat, r_coarse, np.zeros(n//2 - 1), nu1, nu2,  relax,
                           restrict, interpolate, recursion_depth-1)
    e = interpolate(e_coarse)
    u += e

    for _ in range(nu2):
        u = relax(A, f, u)
    return u
