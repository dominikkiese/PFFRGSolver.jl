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
    tbuff :: Matrix{Float64},
    temp  :: Array{Float64, 3},
    eval  :: Int64
    )     :: Nothing

    # reset buffer
    @turbo tbuff .= 0.0

    # get frequency arguments
    t, vt, vtp = m.Ωt[w1], m.νt[w2], m.νt[w3]

    # define integrand
    integrand!(b, v, dv) = compute_t_BSE!(Λ, b, v, dv, t, vt, vtp, r, m, a1, temp)

    # define reference frequency
    ref = Λ + 0.5 * t

    # compute integral over tails
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff, ref, 50.0 * ref, eval, sgn = -1.0)
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff, ref, 50.0 * ref, eval)

    # compute integral around origin
    integrate_lin!((b, v, dv) -> integrand!(b, v, dv), tbuff, -ref, 0.0, 2 * eval)
    integrate_lin!((b, v, dv) -> integrand!(b, v, dv), tbuff,  0.0, ref, 2 * eval)

    # parse result
    for i in eachindex(a2.Γ)
        a2.Γ[i].ch_t.q3[:, w1, w2, w3] .= view(tbuff, i, :)
    end

    return nothing
end