using SparseArrays, LinearAlgebra

function get_n(v, dimensions)
    return Int(floor(length(v)^(1 / dimensions))) + 1
end

function wjacobi(A, f, u, omega)
    Dinv = spdiagm(0 => 1 ./ diag(A))
    U, L = triu(A, 1), tril(A, -1)
    return (1 - omega) * u + omega * Dinv * (f - (U + L) * u)
end

function simple_restrict(v)
    return simple_restrict_matrix(get_n(v, 1)) * v
end

function simple_interpolate(v, lin_boundary=false)
    return simple_interpolate_matrix(get_n(v, 1), lin_boundary) * v
end

function simple_restrict2D(v)
    return simple_restrict_matrix2D(get_n(v, 2)) * v
end

function simple_interpolate2D(v)
    return simple_interpolate_matrix2D(get_n(v, 2)) * v
end


function geoVcycle(; mat, f, u, nu1, nu2, relax, restrict, interpolate, recursion_depth, dimensions=1)
    n = get_n(u, dimensions)  # geometric/grid information
    A = mat(n)

    # Recursion stops when the grid size is less then 3 or 
    # when the specified recursion depth is reached.
    if n <= 3 || recursion_depth == 0
        return A \ f
    end

    for _ in 1:nu1
        u = relax(A, f, u)
    end

    r_coarse = restrict(f - A * u)
    e_coarse = geoVcycle(
        mat, r_coarse, zeros(length(r_coarse)),
        nu1, nu2, relax, restrict, interpolate, recursion_depth - 1, dimensions)
    e = interpolate(e_coarse)
    u += e

    for _ in 1:nu2
        u = relax(A, f, u)
    end
    return u
end

