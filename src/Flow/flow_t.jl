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
    tbuff :: NTuple{3, Vector{Float64}},
    temp  :: Array{Float64, 3},
    eval  :: Int64,
    Γ_tol :: NTuple{2, Float64}
    )     :: Nothing

    for comp in eachindex(da.Γ)
        # reset buffer
        @turbo tbuff[1] .= 0.0

        # get frequency arguments
        t, vt, vtp = m.Ωs[comp][w1], m.νs[comp][w2], m.νs[comp][w3]

        # define integrand
        integrand!(b, v, dv) = compute_t_kat!(Λ, comp, b, v, dv, t, vt, vtp, r, m, a, da, temp)

        # compute integral
        ref = Λ + 0.5 * t
        integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  2.0 * ref, 75.0 * ref, eval, Γ_tol[1], Γ_tol[2], sgn = -1.0)
        integrate_lin!((b, v, dv) -> integrand!(b, v, dv), tbuff, -2.0 * ref,  2.0 * ref, eval, Γ_tol[1], Γ_tol[2])
        integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  2.0 * ref, 75.0 * ref, eval, Γ_tol[1], Γ_tol[2])

        # parse result
        @turbo da.Γ[comp].ch_t.q3[:, w1, w2, w3] .= tbuff[1]
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
    tbuff :: NTuple{3, Vector{Float64}},
    temp  :: Array{Float64, 3},
    eval  :: Int64,
    Γ_tol :: NTuple{2, Float64}
    )     :: Nothing

    for comp in eachindex(da_l.Γ)
        # reset buffer
        @turbo tbuff[1] .= 0.0

        # get frequency arguments
        t, vt, vtp = m.Ωs[comp][w1], m.νs[comp][w2], m.νs[comp][w3]

        # define integrand
        integrand!(b, v, dv) = compute_t_left!(Λ, comp, b, v, dv, t, vt, vtp, r, m, a, da, temp)

        # compute integral
        ref = Λ + 0.5 * t
        integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  2.0 * ref, 75.0 * ref, eval, Γ_tol[1], Γ_tol[2], sgn = -1.0)
        integrate_lin!((b, v, dv) -> integrand!(b, v, dv), tbuff, -2.0 * ref,  2.0 * ref, eval, Γ_tol[1], Γ_tol[2])
        integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  2.0 * ref, 75.0 * ref, eval, Γ_tol[1], Γ_tol[2])

        # parse result
        @turbo da_l.Γ[comp].ch_t.q3[:, w1, w2, w3] .= tbuff[1]
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
    tbuff :: NTuple{3, Vector{Float64}},
    temp  :: Array{Float64, 3},
    eval  :: Int64,
    Γ_tol :: NTuple{2, Float64}
    )     :: Nothing

    for comp in eachindex(da_c.Γ)
        # reset buffer
        @turbo tbuff[1] .= 0.0

        # get frequency arguments
        t, vt, vtp = m.Ωs[comp][w1], m.νs[comp][w2], m.νs[comp][w3]

        # define integrand
        integrand!(b, v, dv) = compute_t_central!(Λ, comp, b, v, dv, t, vt, vtp, r, m, a, da_l, temp)

        # compute integral
        ref = Λ + 0.5 * t
        integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  2.0 * ref, 75.0 * ref, eval, Γ_tol[1], Γ_tol[2], sgn = -1.0)
        integrate_lin!((b, v, dv) -> integrand!(b, v, dv), tbuff, -2.0 * ref,  2.0 * ref, eval, Γ_tol[1], Γ_tol[2])
        integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  2.0 * ref, 75.0 * ref, eval, Γ_tol[1], Γ_tol[2])

        # parse result
        @turbo da_c.Γ[comp].ch_t.q3[:, w1, w2, w3] .= tbuff[1]
    end

    return nothing
end