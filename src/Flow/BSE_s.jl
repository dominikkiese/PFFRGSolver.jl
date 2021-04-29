# compute the right side of the BSE in the s-channel for a frequency tuple (w1, w2, w3) on all lattice sites
function compute_channel_s_BSE!(
    Λ     :: Float64,
    w1    :: Int64,
    w2    :: Int64,
    w3    :: Int64,
    r     :: reduced_lattice,
    m     :: mesh,
    a1    :: action,
    a2    :: action,
    tbuff :: NTuple{3, Matrix{Float64}},
    temp  :: Array{Float64, 3},
    eval  :: Int64
    )     :: Nothing

    # reset buffer 
    tbuff[1] .= 0.0

    # get frequency arguments
    s, vs, vsp = m.Ωs[w1], m.νs[w2], m.νs[w3]

    # define integrand
    integrand!(b, v, dv) = compute_s_BSE!(Λ, b, v, dv, s, vs, vsp, r, m, a1, temp)

    # compute integral
    ref = Λ + 0.5 * s
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  2.0 * ref, 100.0 * ref, eval, sgn = -1.0)
    integrate_lin!((b, v, dv) -> integrand!(b, v, dv), tbuff, -2.0 * ref,   0.0 * ref, eval)
    integrate_lin!((b, v, dv) -> integrand!(b, v, dv), tbuff,  0.0 * ref,   2.0 * ref, eval)
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  2.0 * ref, 100.0 * ref, eval)

    # parse result
    for i in eachindex(a2.Γ)
        a2.Γ[i].ch_s.q3[:, w1, w2, w3] .= view(tbuff[1], i, :)
    end

    return nothing
end
