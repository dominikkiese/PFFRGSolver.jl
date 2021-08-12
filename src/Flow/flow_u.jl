# compute the Katanin truncated flow equations in the u-channel for a frequency tuple (w1, w2, w3) on all lattice sites
function compute_channel_u_kat!(
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
    u, vu, vup = m.Ωu[w1], m.νu[w2], m.νu[w3]

    # define integrand
    integrand!(b, v, dv) = compute_u_kat!(Λ, b, v, dv, u, vu, vup, r, m, a, da, temp)

    # define reference frequency
    ref = Λ + 0.5 * u

    # compute integral over tails
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff, ref, 25.0 * ref, eval, sgn = -1.0)
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff, ref, 25.0 * ref, eval)

    # compute integral around origin
    integrate_lin!((b, v, dv) -> integrand!(b, v, dv), tbuff, -ref, 0.0, 2 * eval)
    integrate_lin!((b, v, dv) -> integrand!(b, v, dv), tbuff,  0.0, ref, 2 * eval)

    # parse result
    for i in eachindex(da.Γ)
        da.Γ[i].ch_u.q3[:, w1, w2, w3] .= view(tbuff, i, :)
    end

    return nothing
end





# compute the left part of the flow equations in the u-channel for a frequency tuple (w1, w2, w3) on all lattice sites
function compute_channel_u_left!(
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
    u, vu, vup = m.Ωu[w1], m.νu[w2], m.νu[w3]

    # define integrand
    integrand!(b, v, dv) = compute_u_left!(Λ, b, v, dv, u, vu, vup, r, m, a, da, temp)

    # define reference frequency
    ref = Λ + 0.5 * u

    # compute integral over tails
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff, ref, 50.0 * ref, eval, sgn = -1.0)
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff, ref, 50.0 * ref, eval)

    # compute integral around origin
    integrate_lin!((b, v, dv) -> integrand!(b, v, dv), tbuff, -ref, 0.0, 2 * eval)
    integrate_lin!((b, v, dv) -> integrand!(b, v, dv), tbuff,  0.0, ref, 2 * eval)

    # parse result
    for i in eachindex(da_l.Γ)
        da_l.Γ[i].ch_u.q3[:, w1, w2, w3] .= view(tbuff, i, :)
    end

    return nothing
end





# compute the central part of the flow equations in the u-channel for a frequency tuple (w1, w2, w3) on all lattice sites
function compute_channel_u_central!(
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
    u, vu, vup = m.Ωu[w1], m.νu[w2], m.νu[w3]

    # define integrand
    integrand!(b, v, dv) = compute_u_central!(Λ, b, v, dv, u, vu, vup, r, m, a, da_l, temp)

    # define reference frequency
    ref = Λ + 0.5 * u

    # compute integral over tails
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff, ref, 50.0 * ref, eval, sgn = -1.0)
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff, ref, 50.0 * ref, eval)

    # compute integral around origin
    integrate_lin!((b, v, dv) -> integrand!(b, v, dv), tbuff, -ref, 0.0, 2 * eval)
    integrate_lin!((b, v, dv) -> integrand!(b, v, dv), tbuff,  0.0, ref, 2 * eval)

    # parse result
    for i in eachindex(da_c.Γ)
        da_c.Γ[i].ch_u.q3[:, w1, w2, w3] .= view(tbuff, i, :)
    end

    return nothing
end