using Plots, SparseArrays, LinearAlgebra

include("geometricMultiGrid.jl")
include("constructions1D.jl")
include("constructions2D.jl")


begin
    function test_relax_solving(relax::Function; n, iterations)
        A = spdiagm(-1 => -ones(n - 1), 0 => 4 * ones(n), 1 => -ones(n - 1))
        f = collect(0:n-1)
        u = zeros(n)
        for _ in 1:iterations
            u = relax(A, f, u)
        end
        usol = A \ f
        p1 = bar(u - usol, title="error")
        p2 = bar(u, title="sol")
        display(plot(p1, p2, layout=(1, 2)))
    end

    function test_relax_smoothing(relax::Function; n, iterations)
        A = spdiagm(-1 => -ones(n - 1), 0 => 4 * ones(n), 1 => -ones(n - 1))
        f = collect(0:n-1)
        usol = A \ f
        u = 20 * wave_basis_1D(n + 1, n ÷ 2)
        bar(u - usol, title="Error at iteration 0")
        @gif for i in 1:iterations
            u = relax(A, f, u)
            bar(u - usol, title="Error at iteration $i")
        end fps = 5
    end

    omega = 2 / 3
    wjacobi_omega = (A, f, u) -> wjacobi(A, f, u, omega)
    test_relax_solving(wjacobi_omega, n=20, iterations=60)
    test_relax_smoothing(wjacobi_omega, n=20, iterations=60)
end


begin
    function test_recursive_restriction(depth, v, restrict)
        anim = @gif for i in 1:depth
            v = restrict(v)
            bar(v, title="Iteration $i")
        end fps = 2
        display(anim)
    end

    function test_recursive_restriction2D(depth, v, restrict)
        anim = @gif for i in 1:depth
            v = restrict(v)
            n = get_n(v, 2)
            heatmap(reshape(v, (n - 1, n - 1)), title="Iteration $i")
        end fps = 1
        display(anim)
    end

    test_recursive_restriction(8, wave_basis_1D(2^8, 2), simple_restrict)
    test_recursive_restriction2D(7, wave_basis_2D(2^8, 1, 1), simple_restrict2D)
end


begin
    function laplace1D_test(n=32, iterations=1, recursion_depth=3)
        A = helmholtz1D(n, 0)
        u_exact = wave_basis_1D(n, 1)
        f = A * u_exact
        u_approx = zeros(n - 1)

        helmholtz1D_sigma(x) = helmholtz1D(x, 0)
        wjacobi_omega(A, f, u) = wjacobi(A, f, u, 2 / 3)

        anim = @gif for i in 1:iterations
            u_approx = geoVcycle(
                mat=helmholtz1D_sigma,
                f=f,
                u=u_approx,
                nu1=1,
                nu2=1,
                relax=wjacobi_omega,
                restrict=simple_restrict,
                interpolate=simple_interpolate,
                recursion_depth=recursion_depth
            )
            error = u_exact - u_approx
            bar(error, title="Error after $(i-1) vcycles")
        end fps = 1
        display(anim)
    end

    laplace1D_test(2^7, 20, 6)
end

begin
    function laplace2D_test(n=32, iterations=1, recursion_depth=3)
        A = helmholtz2D(n, 0)
        u_exact = wave_basis_2D(n, 1, 1)
        f = A * u_exact
        u_approx = zeros((n - 1)^2)

        helmholtz2D_sigma(x) = helmholtz2D(x, 0)
        wjacobi_omega(A, f, u) = wjacobi(A, f, u, 2 / 3)
        anim = @gif for i in 1:iterations
            u_approx = geoVcycle(
                mat=helmholtz2D_sigma,
                f=f,
                u=u_approx,
                nu1=1,
                nu2=1,
                relax=wjacobi_omega,
                restrict=simple_restrict2D,
                interpolate=simple_interpolate2D,
                recursion_depth=recursion_depth,
                dimensions=2
            )
            error = u_exact - u_approx
            heatmap(reshape(error, n - 1, n - 1), title="Error after $(i-1) vcycles")
        end fps = 1
        display(anim)
    end

    laplace2D_test(2^(2 * 4), 10, 4)

end