using SparseArrays, LinearAlgebra
include("constructions1D.jl")

function helmholtz2D(n, σ)
    A_1D = n^2 * spdiagm(-1 => -ones(n - 2), 0 => 2 * ones(n - 1), 1 => -ones(n - 2))
    A_2D = kron(sparse(I, n - 1, n - 1), A_1D) .+ kron(A_1D, sparse(I, n - 1, n - 1))
    return A_2D .+ σ * sparse(I, (n - 1)^2, (n - 1)^2)
end

function simple_restrict_matrix2D(n)
    if n % 2 != 0
        error("restrict matrix needs n even")
    end
    R_1D = simple_restrict_matrix(n)
    return kron(R_1D, R_1D)
end

function simple_interpolate_matrix2D(n, lin_boundary=false)
    P_1D = simple_interpolate_matrix(n, lin_boundary)
    return kron(P_1D, P_1D)
end

function wave_basis_2D(n, k, l)
    return vec(kron(wave_basis_1D(n, k), wave_basis_1D(n, l)))
end

function wave_basis_2D_complex(n, k, l)
    return vec(kron(wave_basis_1D_complex(n, k), wave_basis_1D_complex(n, l)))
end

function wave_basis_2Dx(n, k)
    return vec(kron(ones(n - 1), wave_basis_1D(n, k)))
end

function wave_basis_2Dy(n, k)
    return vec(kron(wave_basis_1D(n, k), ones(n - 1)))
end

function pointsource_half2D(n)
    return vec(kron(pointsource_half(n), pointsource_half(n)))
end