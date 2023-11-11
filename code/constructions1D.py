from scipy.sparse import eye, diags, dok_matrix, spdiags
import numpy as np


def helmholtz1D(n, sigma):
    A = n**2*diags([-1, 2, -1], [-1, 0, 1],
                   shape=(n-1, n-1))
    return A + sigma*eye(n-1)


def eigen_helmhotz1D(n, k, sigma):
    return 4*np.sin(k*np.pi/(2*n))**2*n**2 + sigma


def Romega(n, sigma, omega):
    H = helmholtz1D(n, sigma)
    Dinv = spdiags(1/H.diagonal(), [0], (n-1, n-1))
    return eye(n-1)-omega*Dinv@H


def simple_restrict_matrix(n):
    if n % 2 != 0:
        raise ValueError("restrict matrix needs n even")
    R = dok_matrix((n//2 - 1, n-1))
    row0 = np.array([1, 2, 1])
    for i in range(R.shape[0]):
        R[i, 2*i:2*i+3] = row0/4
    return R


def simple_interpolate_matrix(n, lin_boundary=False):
    I = dok_matrix((2*n - 1, n-1))
    col0 = np.array([1, 2, 1])
    for j in range(I.shape[1]):
        I[2*j:2*j+3, j] = col0/2
    if lin_boundary:  # @boundaries linear extrapolation
        I[0, 0] = 3/2
        I[0, 1] = -1/2
        I[-1, -1] = 3/2
        I[-1, -2] = -1/2
    return I


def wave_basis_1D(n, k):
    return np.array([np.sin((j+1)*k*np.pi/n) for j in range(n-1)])


def wave_basis_1D_complex(n, k):
    return np.array([-complex(0, 1)*np.exp(complex(0, (j+1)*k*np.pi/n)) for j in range(n-1)])


def pointsource_half(n):
    return np.array([n if j == int(n/2) else 0 for j in range(n-1)])


def guessv0(n):
    return np.array([0.5*(np.sin(j*16 * np.pi / n)+np.sin(j*40 * np.pi / n))for j in range(n-1)])
