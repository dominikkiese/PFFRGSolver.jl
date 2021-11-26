# get bare propagator
function get_G_bare(
    Λ :: Float64,
    w :: Float64
    ) :: Float64 

    val = 0.0

    if abs(w) > 1e-8
        val = -expm1(-w^2 / Λ^2) / w
    end

    return val
end

# get dressed propagator
function get_G(
    Λ :: Float64,
    w :: Float64,
    m :: Mesh,
    a :: Action
    ) :: Float64

    val = 0.0 

    if abs(w) > 1e-8
        Σ     = get_Σ(w, m, a)
        G0    = get_G_bare(Λ, w)
        denom = 0.0

        # ensure causality
        if w * Σ >= 0.0
            denom = 1.0 / G0 + Σ
        else
            denom = 1.0 / G0
        end 

        val = 1.0 / denom
    end

    return val 
end

# get single scale propagator
function get_S(
    Λ :: Float64,
    w :: Float64,
    m :: Mesh,
    a :: Action
    ) :: Float64

    val = 0.0 

    if abs(w) > 1e-8
        G   = get_G(Λ, w, m, a)
        G0  = get_G_bare(Λ, w)
        val = (G / G0)^2 * exp(-w^2 / Λ^2) * 2.0 * w / Λ^3 
    end 

    return val 
end

# get differentiated propagator bubble
function get_propagator_kat(
    Λ  :: Float64,
    w1 :: Float64,
    w2 :: Float64,
    m  :: Mesh,
    a  :: Action,
    da :: Action
    )  :: Float64 

    dΣ  = get_Σ(w1, m, da)
    S1  = get_S(Λ, w1, m, a)
    G1  = get_G(Λ, w1, m, a)
    G2  = get_G(Λ, w2, m, a)
    val = (S1 + G1^2 * dΣ) * G2 / (2.0 * pi)

    return val 
end

# get undifferentiated propagator bubble
function get_propagator(
    Λ  :: Float64,
    w1 :: Float64,
    w2 :: Float64,
    m  :: Mesh,
    a  :: Action
    )  :: Float64

    G1  = get_G(Λ, w1, m, a)
    G2  = get_G(Λ, w2, m, a)
    val = G1 * G2 / (2.0 * pi)

    return val
end

# compute boundary corrections for vertex integrals 
function compute_corrs_kat!(
    Λ     :: Float64,
    m     :: Mesh,
    a     :: Action,
    da    :: Action,
    corrs :: Array{Float64, 3},
    Γ_tol :: NTuple{2, Float64}
    )     :: Nothing 

    # reset buffer
    @turbo corrs .= 0.0

    # compute boundary corrections
    @sync for i in size(corrs, 3)
        Threads.@spawn begin
            # compute boundary corrections for s channel
            s_propagator  = v -> get_propagator_kat(Λ, v + 0.5 * m.Ωs[i], 0.5 * m.Ωs[i] - v, m, a, da) + get_propagator_kat(Λ, 0.5 * m.Ωs[i] - v, v + 0.5 * m.Ωs[i], m, a, da)
            s_val         = m.Ωs[end] + m.νs[end]
            s_bound_minus = s_propagator(-s_val)
            s_bound_plus  = s_propagator( s_val)

            if abs(s_bound_minus) > 1e-8
                corrs[1, 1, i] = quadgk(s_propagator, Inf, -s_val, atol = Γ_tol[1], rtol = Γ_tol[2], order = 10)[1] / s_bound_minus
            end 

            if abs(s_bound_plus) > 1e-8
                corrs[2, 1, i] = quadgk(s_propagator, s_val, Inf, atol = Γ_tol[1], rtol = Γ_tol[2], order = 10)[1] / s_bound_plus
            end

            # compute boundary corrections for t channel
            t_propagator  = v -> get_propagator_kat(Λ, v + 0.5 * m.Ωt[i], v - 0.5 * m.Ωt[i], m, a, da) + get_propagator_kat(Λ, v - 0.5 * m.Ωt[i], v + 0.5 * m.Ωt[i], m, a, da)
            t_val         = m.Ωt[end] + m.νt[end]
            t_bound_minus = t_propagator(-t_val)
            t_bound_plus  = t_propagator( t_val)

            if abs(t_bound_minus) > 1e-8
                corrs[1, 2, i] = quadgk(t_propagator, -Inf, -t_val, atol = Γ_tol[1], rtol = Γ_tol[2], order = 10)[1] / t_bound_minus
            end

            if abs(t_bound_plus) > 1e-8
                corrs[2, 2, i] = quadgk(t_propagator, t_val, Inf, atol = Γ_tol[1], rtol = Γ_tol[2], order = 10)[1] / t_bound_plus
            end

            # compute boundary corrections for u channel
            u_propagator  = v -> get_propagator_kat(Λ, v - 0.5 * m.Ωu[i], v + 0.5 * m.Ωu[i], m, a, da) + get_propagator_kat(Λ, v + 0.5 * m.Ωu[i], v - 0.5 * m.Ωu[i], m, a, da)
            u_val         = m.Ωu[end] + m.νu[end]
            u_bound_minus = u_propagator(-u_val)
            u_bound_plus  = u_propagator( u_val)

            if abs(u_bound_minus) > 1e-8
                corrs[1, 3, i] = quadgk(u_propagator, -Inf, -u_val, atol = Γ_tol[1], rtol = Γ_tol[2], order = 10)[1] / u_bound_minus
            end
            
            if abs(u_bound_plus) > 1e-8
                corrs[2, 3, i] = quadgk(u_propagator, u_val, Inf, atol = Γ_tol[1], rtol = Γ_tol[2], order = 10)[1] / u_bound_plus
            end
        end
    end

    return nothing 
end

# compute boundary corrections for vertex integrals 
function compute_corrs!(
    Λ     :: Float64,
    m     :: Mesh,
    a     :: Action,
    corrs :: Array{Float64, 3},
    Γ_tol :: NTuple{2, Float64}
    )     :: Nothing 

    # reset buffer
    @turbo corrs .= 0.0

    # compute boundary corrections
    @sync for i in size(corrs, 3)
        Threads.@spawn begin
            # compute boundary corrections for s channel
            s_propagator  = v -> -get_propagator(Λ, v + 0.5 * m.Ωs[i], 0.5 * m.Ωs[i] - v, m, a) 
            s_val         = m.Ωs[end] + m.νs[end]
            s_bound_minus = s_propagator(-s_val)
            s_bound_plus  = s_propagator( s_val)

            if abs(s_bound_minus) > 1e-8
                corrs[1, 1, i] = quadgk(s_propagator, -Inf, -s_val, atol = Γ_tol[1], rtol = Γ_tol[2], order = 10)[1] / s_bound_minus
            end 

            if abs(s_bound_plus) > 1e-8
                corrs[2, 1, i] = quadgk(s_propagator, s_val, Inf, atol = Γ_tol[1], rtol = Γ_tol[2], order = 10)[1] / s_bound_plus
            end

            # compute boundary corrections for t channel
            t_propagator  = v -> -get_propagator(Λ, v + 0.5 * m.Ωt[i], v - 0.5 * m.Ωt[i], m, a)
            t_val         = m.Ωt[end] + m.νt[end]
            t_bound_minus = t_propagator(-t_val)
            t_bound_plus  = t_propagator( t_val)

            if abs(t_bound_minus) > 1e-8
                corrs[1, 2, i] = quadgk(t_propagator, -Inf, -t_val, atol = Γ_tol[1], rtol = Γ_tol[2], order = 10)[1] / t_bound_minus
            end

            if abs(t_bound_plus) > 1e-8
                corrs[2, 2, i] = quadgk(t_propagator, t_val, Inf, atol = Γ_tol[1], rtol = Γ_tol[2], order = 10)[1] / t_bound_plus
            end

            # compute boundary corrections for u channel
            u_propagator  = v -> -get_propagator(Λ, v - 0.5 * m.Ωu[i], v + 0.5 * m.Ωu[i], m, a)
            u_val         = m.Ωu[end] + m.νu[end]
            u_bound_minus = u_propagator(-u_val)
            u_bound_plus  = u_propagator( u_val)

            if abs(u_bound_minus) > 1e-8
                corrs[1, 3, i] = quadgk(u_propagator, -Inf, -u_val, atol = Γ_tol[1], rtol = Γ_tol[2], order = 10)[1] / u_bound_minus
            end
            
            if abs(u_bound_plus) > 1e-8
                corrs[2, 3, i] = quadgk(u_propagator, u_val, Inf, atol = Γ_tol[1], rtol = Γ_tol[2], order = 10)[1] / u_bound_plus
            end
        end
    end

    return nothing 
end