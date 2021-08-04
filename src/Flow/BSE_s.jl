# compute the BSE in the s-channel for a frequency tuple (w1, w2, w3) on all lattice sites
function compute_channel_s_BSE!(
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
    s, vs, vsp = m.Ωs[w1], m.νs[w2], m.νs[w3]

    # define integrand
    integrand!(b, v, dv) = compute_s_BSE!(Λ, b, v, dv, s, vs, vsp, r, m, a1, temp)

    # compute integral
    ref = Λ + 0.5 * s
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  1.2 * ref, 50.0 * ref, eval, sgn = -1.0)
    integrate_lin!((b, v, dv) -> integrand!(b, v, dv), tbuff, -1.2 * ref,  1.2 * ref, eval)
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  1.2 * ref, 50.0 * ref, eval)

    # parse result
    for i in eachindex(a2.Γ)
        a2.Γ[i].ch_s.q3[:, w1, w2, w3] .= view(tbuff, i, :)
    end

    return nothing
end