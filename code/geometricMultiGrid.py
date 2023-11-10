from constructions1D import simple_restrict_matrix, simple_interpolate_matrix
from constructions2D import simple_restrict_matrix2D, simple_interpolate_matrix2D
from plt_utils import *
from scipy.sparse.linalg import spsolve
from scipy.sparse import spdiags, tril, triu


def get_n(v, dimensions):
    return int(np.power(len(v), 1/dimensions)) + 1


def wjacobi(A, f, u, omega):
    Dinv = spdiags(1/A.diagonal(), [0], (A.shape[0], A.shape[0]))
    U, L = triu(A, 1), tril(A, -1)
    return (1-omega)*u + omega*Dinv @ (f-(U+L)@u)


def simple_restrict(v):
    return simple_restrict_matrix(get_n(v, 1))@v


def simple_interpolate(v):
    return simple_interpolate_matrix(get_n(v, 1))@v


def simple_restrict2D(v):
    return simple_restrict_matrix2D(get_n(v, 2))@v


def simple_interpolate2D(v):
    return simple_interpolate_matrix2D(get_n(v, 2))@v


def geoVcycle(mat, f, u, nu1, nu2, relax, restrict, interpolate, recursion_depth, dimensions=1):
    n = get_n(u, dimensions)

    if not (n % 2**(recursion_depth*dimensions) == 0):
        raise ValueError(
            "Length of u must be divisible by 2**(recursion_depth*dimensions)")

    A = mat(n)
    if recursion_depth == 0:
        return spsolve(A, f)

    for _ in range(nu1):
        u = relax(A, f, u)

    r_coarse = restrict(f - A@u)
    e_coarse = geoVcycle(mat, r_coarse, np.zeros(len(r_coarse)), nu1, nu2,  relax,
                         restrict, interpolate, recursion_depth-1, dimensions)
    e = interpolate(e_coarse)
    u += e

    for _ in range(nu2):
        u = relax(A, f, u)
    return u
