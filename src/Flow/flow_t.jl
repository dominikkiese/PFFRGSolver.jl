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
    tbuff :: Matrix{Float64},
    temp  :: Array{Float64, 3},
    eval  :: Int64
    )     :: Nothing

    # reset buffer
    @turbo tbuff .= 0.0

    # get frequency arguments
    t, vt, vtp = m.Ωt[w1], m.νt[w2], m.νt[w3]

    # define integrand
    integrand!(b, v, dv) = compute_t_kat!(Λ, b, v, dv, t, vt, vtp, r, m, a, da, temp)

    # compute integral
    ref = Λ + 0.5 * t
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  1.2 * ref, 25.0 * ref, eval, sgn = -1.0)
    integrate_lin!((b, v, dv) -> integrand!(b, v, dv), tbuff, -1.2 * ref,  1.2 * ref, eval)
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  1.2 * ref, 25.0 * ref, eval)

    # parse result
    for i in eachindex(da.Γ)
        da.Γ[i].ch_t.q3[:, w1, w2, w3] .= view(tbuff, i, :)
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
    tbuff :: Matrix{Float64},
    temp  :: Array{Float64, 3},
    eval  :: Int64
    )     :: Nothing

    # reset buffer
    @turbo tbuff .= 0.0

    # get frequency arguments
    t, vt, vtp = m.Ωt[w1], m.νt[w2], m.νt[w3]

    # define integrand
    integrand!(b, v, dv) = compute_t_left!(Λ, b, v, dv, t, vt, vtp, r, m, a, da, temp)

    # compute integral
    ref = Λ + 0.5 * t
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  1.2 * ref, 50.0 * ref, eval, sgn = -1.0)
    integrate_lin!((b, v, dv) -> integrand!(b, v, dv), tbuff, -1.2 * ref,  1.2 * ref, eval)
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  1.2 * ref, 50.0 * ref, eval)

    # parse result
    for i in eachindex(da_l.Γ)
        da_l.Γ[i].ch_t.q3[:, w1, w2, w3] .= view(tbuff, i, :)
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
    tbuff :: Matrix{Float64},
    temp  :: Array{Float64, 3},
    eval  :: Int64
    )     :: Nothing

    # reset buffer
    @turbo tbuff .= 0.0

    # get frequency arguments
    t, vt, vtp = m.Ωt[w1], m.νt[w2], m.νt[w3]

    # define integrand
    integrand!(b, v, dv) = compute_t_central!(Λ, b, v, dv, t, vt, vtp, r, m, a, da_l, temp)

    # compute integral
    ref = Λ + 0.5 * t
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  1.2 * ref, 50.0 * ref, eval, sgn = -1.0)
    integrate_lin!((b, v, dv) -> integrand!(b, v, dv), tbuff, -1.2 * ref,  1.2 * ref, eval)
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  1.2 * ref, 50.0 * ref, eval)

    # parse result
    for i in eachindex(da_c.Γ)
        da_c.Γ[i].ch_t.q3[:, w1, w2, w3] .= view(tbuff, i, :)
    end

    return nothing
end