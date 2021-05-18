# compute the Katanin truncated flow equations in the s-channel for a frequency tuple (w1, w2, w3) on all lattice sites
function compute_channel_s_kat!(
    Λ     :: Float64,
    w1    :: Int64,
    w2    :: Int64,
    w3    :: Int64,
    r     :: Reduced_lattice,
    m     :: Mesh,
    a     :: Action,
    da    :: Action,
    tbuff :: NTuple{3, Matrix{Float64}},
    temp  :: Array{Float64, 3},
    eval  :: Int64
    )     :: Nothing

    # reset buffer
    tbuff[1] .= 0.0

    # get frequency arguments
    s, vs, vsp = m.Ωs[w1], m.νs[w2], m.νs[w3]

    # define integrand
    integrand!(b, v, dv) = compute_s_kat!(Λ, b, v, dv, s, vs, vsp, r, m, a, da, temp)

    # compute integral
    ref = Λ + 0.5 * s
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  2.0 * ref, 100.0 * ref, eval, sgn = -1.0)
    integrate_lin!((b, v, dv) -> integrand!(b, v, dv), tbuff, -2.0 * ref,   0.0 * ref, eval)
    integrate_lin!((b, v, dv) -> integrand!(b, v, dv), tbuff,  0.0 * ref,   2.0 * ref, eval)
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  2.0 * ref, 100.0 * ref, eval)

    # parse result
    for i in eachindex(da.Γ)
        da.Γ[i].ch_s.q3[:, w1, w2, w3] .= view(tbuff[1], i, :)
    end

    return nothing
end





# compute the left part of the flow equations in the s-channel for a frequency tuple (w1, w2, w3) on all lattice sites
function compute_channel_s_left!(
    Λ     :: Float64,
    w1    :: Int64,
    w2    :: Int64,
    w3    :: Int64,
    r     :: Reduced_lattice,
    m     :: Mesh,
    a     :: Action,
    da    :: Action,
    da_l  :: Action,
    tbuff :: NTuple{3, Matrix{Float64}},
    temp  :: Array{Float64, 3},
    eval  :: Int64
    )     :: Nothing

    # reset buffer
    tbuff[1] .= 0.0

    # get frequency arguments
    s, vs, vsp = m.Ωs[w1], m.νs[w2], m.νs[w3]

    # define integrand
    integrand!(b, v, dv) = compute_s_left!(Λ, b, v, dv, s, vs, vsp, r, m, a, da, temp)

    # compute integral
    ref = Λ + 0.5 * s
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  2.0 * ref, 100.0 * ref, eval, sgn = -1.0)
    integrate_lin!((b, v, dv) -> integrand!(b, v, dv), tbuff, -2.0 * ref,   0.0 * ref, eval)
    integrate_lin!((b, v, dv) -> integrand!(b, v, dv), tbuff,  0.0 * ref,   2.0 * ref, eval)
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  2.0 * ref, 100.0 * ref, eval)

    # parse result
    for i in eachindex(da_l.Γ)
        da_l.Γ[i].ch_s.q3[:, w1, w2, w3] .= view(tbuff[1], i, :)
    end

    return nothing
end





# compute the central part of the flow equations in the s-channel for a frequency tuple (w1, w2, w3) on all lattice sites
function compute_channel_s_central!(
    Λ     :: Float64,
    w1    :: Int64,
    w2    :: Int64,
    w3    :: Int64,
    r     :: Reduced_lattice,
    m     :: Mesh,
    a     :: Action,
    da_l  :: Action,
    da_c  :: Action,
    tbuff :: NTuple{3, Matrix{Float64}},
    temp  :: Array{Float64, 3},
    eval  :: Int64
    )     :: Nothing

    # reset buffer
    tbuff[1] .= 0.0

    # get frequency arguments
    s, vs, vsp = m.Ωs[w1], m.νs[w2], m.νs[w3]

    # define integrand
    integrand!(b, v, dv) = compute_s_central!(Λ, b, v, dv, s, vs, vsp, r, m, a, da_l, temp)

    # compute integral
    ref = Λ + 0.5 * s
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  2.0 * ref, 100.0 * ref, eval, sgn = -1.0)
    integrate_lin!((b, v, dv) -> integrand!(b, v, dv), tbuff, -2.0 * ref,   0.0 * ref, eval)
    integrate_lin!((b, v, dv) -> integrand!(b, v, dv), tbuff,  0.0 * ref,   2.0 * ref, eval)
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  2.0 * ref, 100.0 * ref, eval)

    # parse result
    for i in eachindex(da_c.Γ)
        da_c.Γ[i].ch_s.q3[:, w1, w2, w3] .= view(tbuff[1], i, :)
    end

    return nothing
end