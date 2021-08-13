# compute the BSE in the u-channel for a frequency tuple (w1, w2, w3) on all lattice sites
function compute_channel_u_BSE!(
    Λ     :: Float64,
    w1    :: Int64,
    w2    :: Int64,
    w3    :: Int64,
    r     :: Reduced_lattice,
    m     :: Mesh,
    a1    :: Action,
    a2    :: Action,
    tbuff :: NTuple{2, Matrix{Float64}},
    temp  :: Array{Float64, 3},
    eval  :: Int64,
    Γ_tol :: NTuple{2, Float64}
    )     :: Nothing

    # get frequency arguments
    u, vu, vup = m.Ωs[w1], m.νs[w2], m.νs[w3]

    # define reference frequency
    ref = Λ + 0.5 * u

    # define integrand
    integrand!(b, v, dv) = compute_u_BSE!(Λ, b, v, dv, u, vu, vup, r, m, a1, temp)

    # compute inner part and parse result
    trapz!((b, v, dv) -> integrand!(b, v, dv), tbuff[1], tbuff[2], -2.0 * ref, 2.0 * ref, eval, Γ_tol[1], Γ_tol[2], 1000)

    for i in eachindex(a2.Γ)
        @turbo a2.Γ[i].ch_u.q3[:, w1, w2, w3] .= view(tbuff[1], i, :)
    end

    # compute left tail and add result
    trapz!((b, v, dv) -> integrand!(b, v, dv), tbuff[1], tbuff[2], -50.0 * ref, -2.0 * ref, eval, Γ_tol[1], Γ_tol[2], 1000)

    for i in eachindex(a2.Γ)
        @turbo a2.Γ[i].ch_u.q3[:, w1, w2, w3] .+= view(tbuff[1], i, :)
    end

    # compute right tail and add result
    trapz!((b, v, dv) -> integrand!(b, v, dv), tbuff[1], tbuff[2], 2.0 * ref, 50.0 * ref, eval, Γ_tol[1], Γ_tol[2], 1000)

    for i in eachindex(a2.Γ)
        @turbo a2.Γ[i].ch_u.q3[:, w1, w2, w3] .+= view(tbuff[1], i, :)
    end

    return nothing
end