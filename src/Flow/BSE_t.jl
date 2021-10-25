# compute the BSE in the t-channel for a frequency tuple (w1, w2, w3) and a given kernel on all lattice sites
function compute_channel_t_BSE!(
    Λ      :: Float64,
    kernel :: Int64,
    w1     :: Int64,
    w2     :: Int64,
    w3     :: Int64,
    r      :: Reduced_lattice,
    m      :: Mesh,
    a1     :: Action,
    a2     :: Action,
    tbuff  :: NTuple{3, Matrix{Float64}},
    temp   :: Array{Float64, 3},
    eval   :: Int64,
    Γ_tol  :: NTuple{2, Float64}
    )      :: Nothing

    # reset buffer
    tbuff[1] .= 0.0

    # get frequency arguments
    t, vt, vtp = get_kernel_args(2, kernel, w1, w2, w3, m)

    # define integrand
    integrand!(b, v, dv) = compute_t_BSE!(Λ, b, v, dv, t, vt, vtp, r, m, a1, temp)

    # compute integral
    ref = Λ + 0.5 * t
    val = max(2.0 * m.Ωt[end], m.νt[end])
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  2.0 * ref, 10.0 * val, eval, Γ_tol[1], Γ_tol[2], sgn = -1.0)
    integrate_lin!((b, v, dv) -> integrand!(b, v, dv), tbuff, -2.0 * ref,  0.0 * ref, eval, Γ_tol[1], Γ_tol[2])
    integrate_lin!((b, v, dv) -> integrand!(b, v, dv), tbuff,  0.0 * ref,  2.0 * ref, eval, Γ_tol[1], Γ_tol[2])
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  2.0 * ref, 10.0 * val, eval, Γ_tol[1], Γ_tol[2])

    # parse result
    for i in eachindex(a2.Γ)
        if kernel == 1
            @turbo a2.Γ[i].ch_t.q1[:, w1] .= view(tbuff[1], i, :)
        elseif kernel == 2
            @turbo a2.Γ[i].ch_t.q2_1[:, w1, w2] .= view(tbuff[1], i, :)
        elseif kernel == 3
            @turbo a2.Γ[i].ch_t.q2_2[:, w1, w3] .= view(tbuff[1], i, :)
        else
            @turbo a2.Γ[i].ch_t.q3[:, w1, w2, w3] .= view(tbuff[1], i, :)
        end
    end

    return nothing
end