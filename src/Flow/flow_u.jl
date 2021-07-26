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
    tbuff :: NTuple{3, Vector{Float64}},
    temp  :: Array{Float64, 3},
    eval  :: Int64,
    Γ_tol :: NTuple{2, Float64}
    )     :: Nothing

    for comp in eachindex(a.Γ)
        # get frequency arguments
        u, vu, vup = m.Ωu[comp][w1], m.νu[comp][w2], m.νu[comp][w3]
        ref        = Λ + 0.5 * u

        # define integrand
        integrand!(b, v, dv) = compute_u_kat!(Λ, comp, b, v, dv, u, vu, vup, r, m, a, da, temp)

        # compute diagrams and parse result
        tbuff[1] .= 0.0
        integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  1.2 * ref, 25.0 * ref, eval, Γ_tol[1], Γ_tol[2], sgn = -1.0)
        integrate_lin!((b, v, dv) -> integrand!(b, v, dv), tbuff, -1.2 * ref,  1.2 * ref, eval, Γ_tol[1], Γ_tol[2])
        integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  1.2 * ref, 25.0 * ref, eval, Γ_tol[1], Γ_tol[2])
        da.Γ[comp].ch_u.q3[:, w1, w2, w3] .= tbuff[1]
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
    tbuff :: NTuple{3, Matrix{Float64}},
    temp  :: Array{Float64, 3},
    eval  :: Int64,
    Γ_tol :: NTuple{2, Float64}
    )     :: Nothing

    # reset buffer
    tbuff[1] .= 0.0

    # get frequency arguments
    u, vu, vup = m.Ωu[w1], m.νu[w2], m.νu[w3]

    # define integrand
    integrand!(b, v, dv) = compute_u_left!(Λ, b, v, dv, u, vu, vup, r, m, a, da, temp)

    # compute integral
    ref = Λ + 0.5 * u
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  1.2 * ref, 50.0 * ref, eval, Γ_tol[1], Γ_tol[2], sgn = -1.0)
    integrate_lin!((b, v, dv) -> integrand!(b, v, dv), tbuff, -1.2 * ref,  1.2 * ref, eval, Γ_tol[1], Γ_tol[2])
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  1.2 * ref, 50.0 * ref, eval, Γ_tol[1], Γ_tol[2])

    # parse result
    for i in eachindex(da_l.Γ)
        da_l.Γ[i].ch_u.q3[:, w1, w2, w3] .= view(tbuff[1], i, :)
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
    tbuff :: NTuple{3, Matrix{Float64}},
    temp  :: Array{Float64, 3},
    eval  :: Int64,
    Γ_tol :: NTuple{2, Float64}
    )     :: Nothing

    # reset buffer
    tbuff[1] .= 0.0

    # get frequency arguments
    u, vu, vup = m.Ωu[w1], m.νu[w2], m.νu[w3]

    # define integrand
    integrand!(b, v, dv) = compute_u_central!(Λ, b, v, dv, u, vu, vup, r, m, a, da_l, temp)

    # compute integral
    ref = Λ + 0.5 * u
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  1.2 * ref, 50.0 * ref, eval, Γ_tol[1], Γ_tol[2], sgn = -1.0)
    integrate_lin!((b, v, dv) -> integrand!(b, v, dv), tbuff, -1.2 * ref,  1.2 * ref, eval, Γ_tol[1], Γ_tol[2])
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  1.2 * ref, 50.0 * ref, eval, Γ_tol[1], Γ_tol[2])

    # parse result
    for i in eachindex(da_c.Γ)
        da_c.Γ[i].ch_u.q3[:, w1, w2, w3] .= view(tbuff[1], i, :)
    end

    return nothing
end