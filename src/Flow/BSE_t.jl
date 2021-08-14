# compute the BSE in the t-channel for a frequency tuple (w1, w2, w3) on all lattice sites
function compute_channel_t_BSE!(
    Λ     :: Float64,
    w1    :: Int64,
    w2    :: Int64,
    w3    :: Int64,
    r     :: Reduced_lattice,
    m     :: Mesh,
    a1    :: Action,
    a2    :: Action,
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
    integrand!(b, v, dv) = compute_t_BSE!(Λ, b, v, dv, t, vt, vtp, r, m, a1, temp)

    # compute integral
    ref = Λ + 0.5 * t
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  1.2 * ref, 50.0 * ref, eval, Γ_tol[1], Γ_tol[2], sgn = -1.0)
    integrate_lin!((b, v, dv) -> integrand!(b, v, dv), tbuff, -1.2 * ref,  1.2 * ref, eval, Γ_tol[1], Γ_tol[2])
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  1.2 * ref, 50.0 * ref, eval, Γ_tol[1], Γ_tol[2])

    # parse result
    for i in eachindex(a2.Γ)
        a2.Γ[i].ch_t.q3[:, w1, w2, w3] .= view(tbuff[1], i, :)
    end

    return nothing
end