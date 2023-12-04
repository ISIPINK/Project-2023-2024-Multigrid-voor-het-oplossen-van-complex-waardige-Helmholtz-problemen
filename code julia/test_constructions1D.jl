using Plots

include("constructions1D.jl")



begin
    n = 10
    σ = 600
    H = helmholtz1D(n, σ)
    Rw = Romega(n, σ, 2 / 3)
    R = simple_restrict_matrix(n)
    P = simple_interpolate_matrix(n)
    mats = [H, Rw, R, P]
    titles = ["H", "Rw", "R", "P"]

    for (mat, title) in zip(mats, titles)
        p = heatmap(mat, color=:viridis, title=title)
        display(p)
    end

end

begin
    n = 100
    f = pointsource_half(n)
    w = wave_basis_1D(n, 1)
    v0 = guessv0(n)
    titles = ["\$f, n = $n\$", "\$w, n = $n\$", "\$v_0,  n = $n\$"]

    for (vec, title) in zip([f, w, v0], titles)
        p = bar(vec, title=title)
        display(p)
    end
end




