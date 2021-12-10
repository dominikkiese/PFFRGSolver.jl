# compute reduced s bubble
function compute_channel_s_reduced!(
    Λ      :: Float64,
    kernel :: Int64,
    w1     :: Int64,
    w3     :: Int64,
    r      :: Reduced_lattice,
    m      :: Mesh,
    a1     :: Action,
    a2     :: Action,
    tbuff  :: NTuple{3, Matrix{Float64}},
    temp   :: Array{Float64, 3},
    corrs  :: Array{Float64, 3},
    eval   :: Int64,
    Γ_tol  :: NTuple{2, Float64}
    )      :: Nothing

    # reset buffer
    tbuff[1] .= 0.0

    # get frequency arguments
    s, vsp = 0.0, 0.0

    if kernel == 1 
        s, vsp = m.Ωs[w1], Inf 
    else
        s, vsp = m.Ωs[w1], m.νs[w3]
    end

    # define integrand
    integrand!(b, v, dv) = compute_s_reduced!(Λ, b, v, dv, s, vsp, r, m, a1, temp)

    # compute integrals
    ref = Λ + 0.5 * s
    val = m.Ωs[end] + m.νs[end]
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  2.0 * ref, 1.0 * val, eval, Γ_tol[1], Γ_tol[2], sgn = -1.0)
    integrate_lin!((b, v, dv) -> integrand!(b, v, dv), tbuff, -2.0 * ref, 0.0 * ref, eval, Γ_tol[1], Γ_tol[2])
    integrate_lin!((b, v, dv) -> integrand!(b, v, dv), tbuff,  0.0 * ref, 2.0 * ref, eval, Γ_tol[1], Γ_tol[2])
    integrate_log!((b, v, dv) -> integrand!(b, v, dv), tbuff,  2.0 * ref, 1.0 * val, eval, Γ_tol[1], Γ_tol[2])

    # correct boundaries
    integrand!(tbuff[1], -val, corrs[1, 1, w1])
    integrand!(tbuff[1],  val, corrs[2, 1, w1])

    # parse result 
    for i in eachindex(a2.Γ)
        if kernel == 1
            @turbo a2.Γ[i].ch_s.q1[:, w1] .= view(tbuff[1], i, :)
        else
            @turbo a2.Γ[i].ch_s.q2_2[:, w1, w3] .= view(tbuff[1], i, :)
        end
    end

    return nothing 
end

# compute self energy from SDE
function compute_Σ!(
    Λ      :: Float64,
    r      :: Reduced_lattice,
    m      :: Mesh,
    a1     :: Action,
    a2     :: Action,
    tbuffs :: Vector{NTuple{3, Matrix{Float64}}},
    temps  :: Vector{Array{Float64, 3}},
    corrs  :: Array{Float64, 3},
    eval   :: Int64,
    Γ_tol  :: NTuple{2, Float64},
    Σ_tol  :: NTuple{2, Float64}
    )      :: Nothing

    # compute boundary corrections
    compute_corrs!(Λ, m, a1, corrs, Γ_tol)

    # compute reduced s bubble in q1 and q2_2
    @sync begin 
        for w1 in 1 : m.num_Ω 
            # compute q1
            Threads.@spawn compute_channel_s_reduced!(Λ, 1, w1, -1, r, m, a1, a2, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)

            for w3 in 1 : m.num_ν
                # compute q2_2
                Threads.@spawn compute_channel_s_reduced!(Λ, 3, w1, w3, r, m, a1, a2, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
            end 
        end 
    end

    # compute self energy for all frequencies
    @sync for i in 2 : length(m.σ)
        Threads.@spawn begin
            integrand = v -> compute_Σ_kernel(Λ, m.σ[i], v, r, m, a1, a2)
            a2.Σ[i]   = quadgk(integrand, -Inf, -2.0 * Λ, 0.0, 2.0 * Λ, Inf, atol = Σ_tol[1], rtol = Σ_tol[2])[1]
        end
    end

    return nothing
end