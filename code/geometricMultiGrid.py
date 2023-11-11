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


def simple_interpolate(v, lin_boundary=False):
    return simple_interpolate_matrix(get_n(v, 1), lin_boundary)@v


def simple_restrict2D(v):
    return simple_restrict_matrix2D(get_n(v, 2))@v


def simple_interpolate2D(v, lin_boundary=False):
    return simple_interpolate_matrix2D(get_n(v, 2), lin_boundary)@v


def geoVcycle(mat, f, u, nu1, nu2, relax, restrict, interpolate, recursion_depth, dimensions=1):
    """
    Applies 1 V-cylce.

    Parameters:
    ----------
    mat : function
        A function that generates the matrix A for a given grid size.
        See in constructions1D for example.
    f : numpy.ndarray
        The right-hand side vector of the linear system Au = f.
    u : numpy.ndarray
        The initial guess for the solution vector u. In general
        should contain the geometric information of u.
    nu1 : int
        The number of relaxation iterations before the V-cycle.
    nu2 : int
        The number of relaxation iterations after the V-cycle.
    relax : function
        A relaxation method to smooth the error in each iteration.
        Should be similar to wjacobi. 
    restrict : function
        A restriction operator that maps a fine grid residual to a coarse grid.
        Should be similar to simple_restrict.
    interpolate : function
        An interpolation operator that maps a coarse grid correction to a fine grid.
        Should be similar to interpolate.
    recursion_depth : int
        The recursion depth, how many recursion to be done
        after solving directly with spsolve.
    dimensions : int, optional
        The number of dimensions of the grid. 

    Returns:
    -------
    numpy.ndarray
        The solution vector u after the V-cycle iterations.
    ------
    """

    n = get_n(u, dimensions)  # geometric/grid information
    A = mat(n)

    # Recursion stops when the grid size is less then 3 or when the specified recursion depth is reached.
    if n <= 3 or recursion_depth == 0:
        return spsolve(A, f)

    for _ in range(nu1):
        u = relax(A, f, u)

    r_coarse = restrict(f - A@u)
    e_coarse = geoVcycle(
        mat,  r_coarse, np.zeros(len(r_coarse)),
        nu1, nu2, relax, restrict, interpolate, recursion_depth-1, dimensions)
    e = interpolate(e_coarse)
    u += e

    for _ in range(nu2):
        u = relax(A, f, u)
    return u
