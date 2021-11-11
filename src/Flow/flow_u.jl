# compute the Katanin truncated flow equations in the u-channel for a frequency tuple (w1, w2, w3) and a given kernel on all lattice sites
function compute_channel_u_kat!(
    Λ      :: Float64,
    kernel :: Int64,
    w1     :: Int64,
    w2     :: Int64,
    w3     :: Int64,
    r      :: Reduced_lattice,
    m      :: Mesh,
    a      :: Action,
    da     :: Action,
    tbuff  :: NTuple{3, Matrix{Float64}},
    temp   :: Array{Float64, 3},
    eval   :: Int64,
    Γ_tol  :: NTuple{2, Float64}
    )      :: Nothing

    # reset buffer
    tbuff[1] .= 0.0

    # get frequency arguments
    u, vu, vup = get_kernel_args(3, kernel, w1, w2, w3, m)

    # define integrand
    integrand!(b, v, dv) = compute_u_kat!(Λ, b, v, dv, u, vu, vup, r, m, a, da, temp)

    # compute integral
    ref = Λ + 0.5 * u
    val = max(2.0 * m.Ωu[end], m.νu[end], 5.0 * ref)
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  2.0 * ref, 20.0 * val, eval, Γ_tol[1], Γ_tol[2], sgn = -1.0)
    integrate_lin!((b, v, dv) -> integrand!(b, v, dv), tbuff, -2.0 * ref,  0.0 * ref, eval, Γ_tol[1], Γ_tol[2])
    integrate_lin!((b, v, dv) -> integrand!(b, v, dv), tbuff,  0.0 * ref,  2.0 * ref, eval, Γ_tol[1], Γ_tol[2])
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  2.0 * ref, 20.0 * val, eval, Γ_tol[1], Γ_tol[2])

    # parse result
    for i in eachindex(da.Γ)
        if kernel == 1
            @turbo da.Γ[i].ch_u.q1[:, w1] .= view(tbuff[1], i, :)
        elseif kernel == 2
            @turbo da.Γ[i].ch_u.q2_1[:, w1, w2] .= view(tbuff[1], i, :)
        elseif kernel == 3
            @turbo da.Γ[i].ch_u.q2_2[:, w1, w3] .= view(tbuff[1], i, :)
        else
            @turbo da.Γ[i].ch_u.q3[:, w1, w2, w3] .= view(tbuff[1], i, :)
        end
    end

    return nothing
end





# compute the left part of the flow equations in the u-channel for a frequency tuple (w1, w2, w3) and a given kernel on all lattice sites
function compute_channel_u_left!(
    Λ      :: Float64,
    kernel :: Int64,
    w1     :: Int64,
    w2     :: Int64,
    w3     :: Int64,
    r      :: Reduced_lattice,
    m      :: Mesh,
    a      :: Action,
    da     :: Action,
    da_l   :: Action,
    tbuff  :: NTuple{3, Matrix{Float64}},
    temp   :: Array{Float64, 3},
    eval   :: Int64,
    Γ_tol  :: NTuple{2, Float64}
    )      :: Nothing

    # reset buffer
    tbuff[1] .= 0.0

    # get frequency arguments
    u, vu, vup = get_kernel_args(3, kernel, w1, w2, w3, m)

    # define integrand
    integrand!(b, v, dv) = compute_u_left!(Λ, b, v, dv, u, vu, vup, r, m, a, da, temp)

    # compute integral
    ref = Λ + 0.5 * u
    val = max(2.0 * m.Ωu[end], m.νu[end], 5.0 * ref)
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  2.0 * ref, 20.0 * val, eval, Γ_tol[1], Γ_tol[2], sgn = -1.0)
    integrate_lin!((b, v, dv) -> integrand!(b, v, dv), tbuff, -2.0 * ref,  0.0 * ref, eval, Γ_tol[1], Γ_tol[2])
    integrate_lin!((b, v, dv) -> integrand!(b, v, dv), tbuff,  0.0 * ref,  2.0 * ref, eval, Γ_tol[1], Γ_tol[2])
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  2.0 * ref, 20.0 * val, eval, Γ_tol[1], Γ_tol[2])

    # parse result
    for i in eachindex(da_l.Γ)
        if kernel == 1
            @turbo da_l.Γ[i].ch_u.q1[:, w1] .= view(tbuff[1], i, :)
        elseif kernel == 2
            @turbo da_l.Γ[i].ch_u.q2_1[:, w1, w2] .= view(tbuff[1], i, :)
        elseif kernel == 3
            @turbo da_l.Γ[i].ch_u.q2_2[:, w1, w3] .= view(tbuff[1], i, :)
        else
            @turbo da_l.Γ[i].ch_u.q3[:, w1, w2, w3] .= view(tbuff[1], i, :)
        end
    end

    return nothing
end





# compute the central part of the flow equations in the u-channel for a frequency tuple (w1, w2, w3) and a given kernel on all lattice sites
function compute_channel_u_central!(
    Λ      :: Float64,
    kernel :: Int64,
    w1     :: Int64,
    w2     :: Int64,
    w3     :: Int64,
    r      :: Reduced_lattice,
    m      :: Mesh,
    a      :: Action,
    da_l   :: Action,
    da_c   :: Action,
    tbuff  :: NTuple{3, Matrix{Float64}},
    temp   :: Array{Float64, 3},
    eval   :: Int64,
    Γ_tol  :: NTuple{2, Float64}
    )      :: Nothing

    # reset buffer
    tbuff[1] .= 0.0

    # get frequency arguments
    u, vu, vup = get_kernel_args(3, kernel, w1, w2, w3, m)

    # define integrand
    integrand!(b, v, dv) = compute_u_central!(Λ, b, v, dv, u, vu, vup, r, m, a, da_l, temp)

    # compute integral
    ref = Λ + 0.5 * u
    val = max(2.0 * m.Ωu[end], m.νu[end], 5.0 * ref)
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  2.0 * ref, 20.0 * val, eval, Γ_tol[1], Γ_tol[2], sgn = -1.0)
    integrate_lin!((b, v, dv) -> integrand!(b, v, dv), tbuff, -2.0 * ref,  0.0 * ref, eval, Γ_tol[1], Γ_tol[2])
    integrate_lin!((b, v, dv) -> integrand!(b, v, dv), tbuff,  0.0 * ref,  2.0 * ref, eval, Γ_tol[1], Γ_tol[2])
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  2.0 * ref, 20.0 * val, eval, Γ_tol[1], Γ_tol[2])

    # parse result
    for i in eachindex(da_c.Γ)
        if kernel == 1
            @turbo da_c.Γ[i].ch_u.q1[:, w1] .= view(tbuff[1], i, :)
        elseif kernel == 2
            @turbo da_c.Γ[i].ch_u.q2_1[:, w1, w2] .= view(tbuff[1], i, :)
        elseif kernel == 3
            @turbo da_c.Γ[i].ch_u.q2_2[:, w1, w3] .= view(tbuff[1], i, :)
        else
            @turbo da_c.Γ[i].ch_u.q3[:, w1, w2, w3] .= view(tbuff[1], i, :)
        end
    end

    return nothing
end