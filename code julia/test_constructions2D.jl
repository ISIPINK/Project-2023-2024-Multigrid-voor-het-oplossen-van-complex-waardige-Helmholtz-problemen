using Plots
include("constructions2D.jl")


begin
    n = 10
    σ = 600
    H = helmholtz2D(n, σ)
    R = simple_restrict_matrix2D(n)
    P = simple_interpolate_matrix2D(n)
    mats = [H, R, P]
    titles = ["H", "R", "P"]

    for (mat, title) in zip(mats, titles)
        p = heatmap(mat, color=:viridis, title=title)
        display(p)
    end

end


begin
    n = 10
    k = 1
    l = 1
    wave_2D = reshape(wave_basis_2D(n, k, l), n - 1, n - 1)
    heatmap(wave_2D, color=:viridis, title="wave basis 2D")
end

begin
    n = 64
    k = 9
    wave_2D = reshape(wave_basis_2Dx(n, k), n - 1, n - 1)
    heatmap(wave_2D, color=:viridis, title="wave basisx 2D")
end

begin
    n = 64
    p = plot()  # Initialize an empty plot
    @gif for k in 1:n-1
        wave_2D = reshape(wave_basis_2Dy(n, k), n - 1, n - 1)
        heatmap!(p, wave_2D, color=:viridis, title="wave basisy 2D, k = $k")
    end fps = 1
end

begin
    n = 10
    k = 1
    l = 1
    wave_2D = reshape(wave_basis_2D_complex(n, k, l), n - 1, n - 1)
    p1 = heatmap(real(wave_2D), color=:viridis, title="Re w com")
    p2 = heatmap(imag(wave_2D), color=:viridis, title="Im w com")
    plot(p1, p2, layout=(1, 2))
end

begin
    n = 64
    ps = reshape(pointsource_half2D(n), n - 1, n - 1)
    plot(heatmap(ps, color=:viridis, title="Re w com"))
end