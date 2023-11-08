from scipy.sparse import eye, diags
import numpy as np


def helmholtz1D(n, sigma):
    H = n**2*diags([-1, 2, -1], [-1, 0, 1],
                   shape=(n-1, n-1))
    return H + sigma*eye(n-1)


def pointsource_half(n):
    return np.array([n**2 if j == int(n/2) else 0 for j in range(n-1)])


def guessv0(n):
    return np.array([0.5*(np.sin(j*16 * np.pi / n)+np.sin(j*40 * np.pi / n))for j in range(n-1)])
