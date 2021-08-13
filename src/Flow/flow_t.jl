# compute the Katanin truncated flow equations in the t-channel for a frequency tuple (w1, w2, w3) on all lattice sites
function compute_channel_t_kat!(
    Λ     :: Float64,
    w1    :: Int64,
    w2    :: Int64,
    w3    :: Int64,
    r     :: Reduced_lattice,
    m     :: Mesh,
    a     :: Action,
    da    :: Action,
    tbuff :: NTuple{2, Matrix{Float64}},
    temp  :: Array{Float64, 3},
    eval  :: Int64,
    Γ_tol :: NTuple{2, Float64}
    )     :: Nothing

    # get frequency arguments
    t, vt, vtp = m.Ωs[w1], m.νs[w2], m.νs[w3]

    # define reference frequency
    ref = Λ + 0.5 * t

    # define integrand
    integrand!(b, v, dv) = compute_t_kat!(Λ, b, v, dv, t, vt, vtp, r, m, a, da, temp)

    # compute inner part and parse result
    trapz!((b, v, dv) -> integrand!(b, v, dv), tbuff[1], tbuff[2], -2.0 * ref, 2.0 * ref, eval, Γ_tol[1], Γ_tol[2], 1000)

    for i in eachindex(da.Γ)
        @turbo da.Γ[i].ch_t.q3[:, w1, w2, w3] .= view(tbuff[1], i, :)
    end

    # compute left tail and add result
    trapz!((b, v, dv) -> integrand!(b, v, dv), tbuff[1], tbuff[2], -25.0 * ref, -2.0 * ref, eval, Γ_tol[1], Γ_tol[2], 1000)

    for i in eachindex(da.Γ)
        @turbo da.Γ[i].ch_t.q3[:, w1, w2, w3] .+= view(tbuff[1], i, :)
    end

    # compute right tail and add result
    trapz!((b, v, dv) -> integrand!(b, v, dv), tbuff[1], tbuff[2], 2.0 * ref, 25.0 * ref, eval, Γ_tol[1], Γ_tol[2], 1000)

    for i in eachindex(da.Γ)
        @turbo da.Γ[i].ch_t.q3[:, w1, w2, w3] .+= view(tbuff[1], i, :)
    end

    return nothing
end





# compute the left part of the flow equations in the t-channel for a frequency tuple (w1, w2, w3) on all lattice sites
function compute_channel_t_left!(
    Λ     :: Float64,
    w1    :: Int64,
    w2    :: Int64,
    w3    :: Int64,
    r     :: Reduced_lattice,
    m     :: Mesh,
    a     :: Action,
    da    :: Action,
    da_l  :: Action,
    tbuff :: NTuple{2, Matrix{Float64}},
    temp  :: Array{Float64, 3},
    eval  :: Int64,
    Γ_tol :: NTuple{2, Float64}
    )     :: Nothing

    # get frequency arguments
    t, vt, vtp = m.Ωs[w1], m.νs[w2], m.νs[w3]

    # define reference frequency
    ref = Λ + 0.5 * t

    # define integrand
    integrand!(b, v, dv) = compute_t_left!(Λ, b, v, dv, t, vt, vtp, r, m, a, da, temp)

    # compute inner part and parse result
    trapz!((b, v, dv) -> integrand!(b, v, dv), tbuff[1], tbuff[2], -2.0 * ref, 2.0 * ref, eval, Γ_tol[1], Γ_tol[2], 1000)

    for i in eachindex(da_l.Γ)
        @turbo da_l.Γ[i].ch_t.q3[:, w1, w2, w3] .= view(tbuff[1], i, :)
    end

    # compute left tail and add result
    trapz!((b, v, dv) -> integrand!(b, v, dv), tbuff[1], tbuff[2], -50.0 * ref, -2.0 * ref, eval, Γ_tol[1], Γ_tol[2], 1000)

    for i in eachindex(da_l.Γ)
        @turbo da_l.Γ[i].ch_t.q3[:, w1, w2, w3] .+= view(tbuff[1], i, :)
    end

    # compute right tail and add result
    trapz!((b, v, dv) -> integrand!(b, v, dv), tbuff[1], tbuff[2], 2.0 * ref, 50.0 * ref, eval, Γ_tol[1], Γ_tol[2], 1000)

    for i in eachindex(da_l.Γ)
        @turbo da_l.Γ[i].ch_t.q3[:, w1, w2, w3] .+= view(tbuff[1], i, :)
    end

    return nothing
end





# compute the central part of the flow equations in the t-channel for a frequency tuple (w1, w2, w3) on all lattice sites
function compute_channel_t_central!(
    Λ     :: Float64,
    w1    :: Int64,
    w2    :: Int64,
    w3    :: Int64,
    r     :: Reduced_lattice,
    m     :: Mesh,
    a     :: Action,
    da_l  :: Action,
    da_c  :: Action,
    tbuff :: NTuple{2, Matrix{Float64}},
    temp  :: Array{Float64, 3},
    eval  :: Int64,
    Γ_tol :: NTuple{2, Float64}
    )     :: Nothing

    # get frequency arguments
    t, vt, vtp = m.Ωs[w1], m.νs[w2], m.νs[w3]

    # define reference frequency
    ref = Λ + 0.5 * t

    # define integrand
    integrand!(b, v, dv) = compute_t_central!(Λ, b, v, dv, t, vt, vtp, r, m, a, da_l, temp)

    # compute inner part and parse result
    trapz!((b, v, dv) -> integrand!(b, v, dv), tbuff[1], tbuff[2], -2.0 * ref, 2.0 * ref, eval, Γ_tol[1], Γ_tol[2], 1000)

    for i in eachindex(da_c.Γ)
        @turbo da_c.Γ[i].ch_t.q3[:, w1, w2, w3] .= view(tbuff[1], i, :)
    end

    # compute left tail and add result
    trapz!((b, v, dv) -> integrand!(b, v, dv), tbuff[1], tbuff[2], -50.0 * ref, -2.0 * ref, eval, Γ_tol[1], Γ_tol[2], 1000)

    for i in eachindex(da_c.Γ)
        @turbo da_c.Γ[i].ch_t.q3[:, w1, w2, w3] .+= view(tbuff[1], i, :)
    end

    # compute right tail and add result
    trapz!((b, v, dv) -> integrand!(b, v, dv), tbuff[1], tbuff[2], 2.0 * ref, 50.0 * ref, eval, Γ_tol[1], Γ_tol[2], 1000)

    for i in eachindex(da_c.Γ)
        @turbo da_c.Γ[i].ch_t.q3[:, w1, w2, w3] .+= view(tbuff[1], i, :)
    end

    return nothing
end