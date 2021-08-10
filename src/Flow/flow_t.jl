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

    for comp in eachindex(a.Γ)
        # get frequency arguments
        t, vt, vtp = m.Ωt[comp][w1], m.νt[comp][w2], m.νt[comp][w3]
        ref        = Λ + 0.5 * t

        # define integrand for chalice diagrams
        integrand_chalice!(b, v, dv) = compute_t_chalice_kat!(Λ, comp, b, v, dv, t, vt, vtp, r, m, a, da, temp)

        # define integrand for RPA diagrams
        integrand_RPA!(b, v, dv) = compute_t_RPA_kat!(Λ, comp, b, v, dv, t, vt, vtp, r, m, a, da, temp)

        # compute chalice diagrams and parse result
        @turbo tbuff[1] .= 0.0
        integrate_log!((b, v, dv) -> integrand_chalice!(b, v, dv), tbuff,  1.2 * ref, 25.0 * ref, eval, Γ_tol[1], Γ_tol[2], sgn = -1.0)
        integrate_lin!((b, v, dv) -> integrand_chalice!(b, v, dv), tbuff, -1.2 * ref,  1.2 * ref, eval, Γ_tol[1], Γ_tol[2])
        integrate_log!((b, v, dv) -> integrand_chalice!(b, v, dv), tbuff,  1.2 * ref, 25.0 * ref, eval, Γ_tol[1], Γ_tol[2])
        @turbo da.Γ[comp].ch_t.q3[:, w1, w2, w3] .= tbuff[1]

        # compute RPA diagrams and parse result
        @turbo tbuff[1] .= 0.0
        integrate_log!((b, v, dv) -> integrand_RPA!(b, v, dv), tbuff,  1.2 * ref, 25.0 * ref, eval, Γ_tol[1], Γ_tol[2], sgn = -1.0)
        integrate_lin!((b, v, dv) -> integrand_RPA!(b, v, dv), tbuff, -1.2 * ref,  1.2 * ref, eval, Γ_tol[1], Γ_tol[2])
        integrate_log!((b, v, dv) -> integrand_RPA!(b, v, dv), tbuff,  1.2 * ref, 25.0 * ref, eval, Γ_tol[1], Γ_tol[2])
        @turbo da.Γ[comp].ch_t.q3[:, w1, w2, w3] .+= tbuff[1]
    end

    return nothing
end





#==
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
    tbuff :: NTuple{3, Matrix{Float64}},
    temp  :: Array{Float64, 3},
    eval  :: Int64,
    Γ_tol :: NTuple{2, Float64}
    )     :: Nothing

    # reset buffer
    tbuff[1] .= 0.0

    # get frequency arguments
    t, vt, vtp = m.Ωt[w1], m.νt[w2], m.νt[w3]

    # define integrand
    integrand!(b, v, dv) = compute_t_left!(Λ, b, v, dv, t, vt, vtp, r, m, a, da, temp)

    # compute integral
    ref = Λ + 0.5 * t
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  1.2 * ref, 50.0 * ref, eval, Γ_tol[1], Γ_tol[2], sgn = -1.0)
    integrate_lin!((b, v, dv) -> integrand!(b, v, dv), tbuff, -1.2 * ref,  1.2 * ref, eval, Γ_tol[1], Γ_tol[2])
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  1.2 * ref, 50.0 * ref, eval, Γ_tol[1], Γ_tol[2])

    # parse result
    for i in eachindex(da_l.Γ)
        da_l.Γ[i].ch_t.q3[:, w1, w2, w3] .= view(tbuff[1], i, :)
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
    tbuff :: NTuple{3, Matrix{Float64}},
    temp  :: Array{Float64, 3},
    eval  :: Int64,
    Γ_tol :: NTuple{2, Float64}
    )     :: Nothing

    # reset buffer
    tbuff[1] .= 0.0

    # get frequency arguments
    t, vt, vtp = m.Ωt[w1], m.νt[w2], m.νt[w3]

    # define integrand
    integrand!(b, v, dv) = compute_t_central!(Λ, b, v, dv, t, vt, vtp, r, m, a, da_l, temp)

    # compute integral
    ref = Λ + 0.5 * t
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  1.2 * ref, 50.0 * ref, eval, Γ_tol[1], Γ_tol[2], sgn = -1.0)
    integrate_lin!((b, v, dv) -> integrand!(b, v, dv), tbuff, -1.2 * ref,  1.2 * ref, eval, Γ_tol[1], Γ_tol[2])
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  1.2 * ref, 50.0 * ref, eval, Γ_tol[1], Γ_tol[2])

    # parse result
    for i in eachindex(da_c.Γ)
        da_c.Γ[i].ch_t.q3[:, w1, w2, w3] .= view(tbuff[1], i, :)
    end

    return nothing
end
==#