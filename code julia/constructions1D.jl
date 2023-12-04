using SparseArrays, LinearAlgebra

function helmholtz1D(n, sigma)
    A = n^2 * spdiagm(-1 => -ones(n - 2),
        0 => 2 * ones(n - 1), 1 => -ones(n - 2))
    return A + sigma * I
end

function eigen_helmhotz1D(n, k, sigma)
    return 4 * sin(k * π / (2 * n))^2 * n^2 + sigma
end

function Romega(n, sigma, omega)
    H = helmholtz1D(n, sigma)
    Dinv = spdiagm(0 => 1 ./ diag(H))
    return I - omega * Dinv * H
end

function simple_restrict_matrix(n)
    if n % 2 != 0
        error("restrict matrix needs n even")
    end
    R = spzeros(n ÷ 2 - 1, n - 1)
    row0 = [1, 2, 1] / 4
    # need to check indexing
    for i in 1:size(R, 1)
        R[i, 2*i-1:2*i+1] = row0
    end
    return R
end

function simple_interpolate_matrix(n, lin_boundary=false)
    P = spzeros(2 * n - 1, n - 1)
    col0 = [1, 2, 1] / 2
    for j in 1:size(P, 2)
        P[2*j-1:2*j+1, j] = col0
    end
    if lin_boundary
        P[1, 1] = 3 / 2
        P[1, 2] = -1 / 2
        P[end, end] = 3 / 2
        P[end, end-1] = -1 / 2
    end
    return P
end

function wave_basis_1D(n, k)
    return [sin(j * k * π / n) for j in 1:n-1]
end

function wave_basis_1D_complex(n, k)
    return [-im * exp(im * j * k * π / n) for j in 1:n-1]
end

function pointsource_half(n)
    return [ifelse(j == n ÷ 2, n, 0) for j in 1:n-1]
end

# maybe i need to shift j
function guessv0(n)
    return [0.5 * (sin(j * 16 * π / n) + sin(j * 40 * π / n)) for j in 1:n-1]
end