"""
    compute_channel_u_kat!(
        Λ     :: Float64,
        w1    :: Int64,
        w2    :: Int64,
        w3    :: Int64,
        r     :: reduced_lattice,
        m     :: mesh,
        a     :: action,
        da    :: action,
        tbuff :: NTuple{3, Matrix{Float64}},
        temp  :: Array{Float64, 3},
        eval  :: Int64
        )     :: Nothing

Compute the right side of the Katanin truncated flow equations for the u channel for a frequency tuple (w1, w2, w3) on all lattice sites.
tbuff saves intermediate results during the frequency integration, temp buffers interpolated vertices and eval sets the number of discretized subdomains during the integration.
"""
function compute_channel_u_kat!(
    Λ     :: Float64,
    w1    :: Int64,
    w2    :: Int64,
    w3    :: Int64,
    r     :: reduced_lattice,
    m     :: mesh,
    a     :: action,
    da    :: action,
    tbuff :: NTuple{3, Matrix{Float64}},
    temp  :: Array{Float64, 3},
    eval  :: Int64
    )     :: Nothing

    # reset buffer 
    tbuff[1] .= 0.0

    # get frequency arguments
    u, vu, vup = m.Ωu[w1], m.νu[w2], m.νu[w3]

    # define integrand
    integrand!(b, v, dv) = compute_u_kat!(Λ, b, v, dv, u, vu, vup, r, m, a, da, temp)

    # compute integral
    ref = Λ + 0.5 * u
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  2.0 * ref, 100.0 * ref, eval, sgn = -1.0)
    integrate_lin!((b, v, dv) -> integrand!(b, v, dv), tbuff, -2.0 * ref,   0.0 * ref, eval)
    integrate_lin!((b, v, dv) -> integrand!(b, v, dv), tbuff,  0.0 * ref,   2.0 * ref, eval)
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  2.0 * ref, 100.0 * ref, eval)

    # parse result
    for i in eachindex(da.Γ)
        da.Γ[i].ch_u.q3[:, w1, w2, w3] .= view(tbuff[1], i, :)
    end

    return nothing
end





"""
    compute_channel_u_left!(
        Λ     :: Float64,
        w1    :: Int64,
        w2    :: Int64,
        w3    :: Int64,
        r     :: reduced_lattice,
        m     :: mesh,
        a     :: action,
        da    :: action,
        da_l  :: action,
        tbuff :: NTuple{3, Matrix{Float64}},
        temp  :: Array{Float64, 3},
        eval  :: Int64
        )     :: Nothing

Compute the left part of the flow equations for the u channel for a frequency tuple (w1, w2, w3) on all lattice sites.
tbuff saves intermediate results during the frequency integration, temp buffers interpolated vertices and eval sets the number of discretized subdomains during the integration.
"""
function compute_channel_u_left!(
    Λ     :: Float64,
    w1    :: Int64,
    w2    :: Int64,
    w3    :: Int64,
    r     :: reduced_lattice,
    m     :: mesh,
    a     :: action,
    da    :: action,
    da_l  :: action,
    tbuff :: NTuple{3, Matrix{Float64}},
    temp  :: Array{Float64, 3},
    eval  :: Int64
    )     :: Nothing

    # reset buffer 
    tbuff[1] .= 0.0

    # get frequency arguments
    u, vu, vup = m.Ωu[w1], m.νu[w2], m.νu[w3]

    # define integrand
    integrand!(b, v, dv) = compute_u_left!(Λ, b, v, dv, u, vu, vup, r, m, a, da, temp)

    # compute integral
    ref = Λ + 0.5 * u
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  2.0 * ref, 100.0 * ref, eval, sgn = -1.0)
    integrate_lin!((b, v, dv) -> integrand!(b, v, dv), tbuff, -2.0 * ref,   0.0 * ref, eval)
    integrate_lin!((b, v, dv) -> integrand!(b, v, dv), tbuff,  0.0 * ref,   2.0 * ref, eval)
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  2.0 * ref, 100.0 * ref, eval)

    # parse result
    for i in eachindex(da_l.Γ)
        da_l.Γ[i].ch_u.q3[:, w1, w2, w3] .= view(tbuff[1], i, :)
    end

    return nothing
end





"""
    compute_channel_u_central!(
        Λ     :: Float64,
        w1    :: Int64,
        w2    :: Int64,
        w3    :: Int64,
        r     :: reduced_lattice,
        m     :: mesh,
        a     :: action,
        da_l  :: action,
        da_c  :: action,
        tbuff :: NTuple{3, Matrix{Float64}},
        temp  :: Array{Float64, 3},
        eval  :: Int64
        )     :: Nothing

Compute the central part of the flow equations for the u channel for a frequency tuple (w1, w2, w3) on all lattice sites.
tbuff saves intermediate results during the frequency integration, temp buffers interpolated vertices and eval sets the number of discretized subdomains during the integration.
"""
function compute_channel_u_central!(
    Λ     :: Float64,
    w1    :: Int64,
    w2    :: Int64,
    w3    :: Int64,
    r     :: reduced_lattice,
    m     :: mesh,
    a     :: action,
    da_l  :: action,
    da_c  :: action,
    tbuff :: NTuple{3, Matrix{Float64}},
    temp  :: Array{Float64, 3},
    eval  :: Int64
    )     :: Nothing

    # reset buffer 
    tbuff[1] .= 0.0

    # get frequency arguments
    u, vu, vup = m.Ωu[w1], m.νu[w2], m.νu[w3]
    
    # define integrand
    integrand!(b, v, dv) = compute_u_central!(Λ, b, v, dv, u, vu, vup, r, m, a, da_l, temp)

    # compute integral
    ref = Λ + 0.5 * u
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  2.0 * ref, 100.0 * ref, eval, sgn = -1.0)
    integrate_lin!((b, v, dv) -> integrand!(b, v, dv), tbuff, -2.0 * ref,   0.0 * ref, eval)
    integrate_lin!((b, v, dv) -> integrand!(b, v, dv), tbuff,  0.0 * ref,   2.0 * ref, eval)
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  2.0 * ref, 100.0 * ref, eval)

    # parse result
    for i in eachindex(da_c.Γ)
        da_c.Γ[i].ch_u.q3[:, w1, w2, w3] .= view(tbuff[1], i, :)
    end

    return nothing
end