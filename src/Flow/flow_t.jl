# computation of t channel flow (Katanin) for frequency tuple (w1, w2, w3) on all lattice sites
function compute_channel_t_kat!(
    Λ     :: Float64,
    w1    :: Int64,
    w2    :: Int64,
    w3    :: Int64,
    r     :: reduced_lattice,
    m     :: mesh,
    a     :: action,
    da    :: action,
    tbuff :: NTuple{3, Matrix{Float64}},
    temp  :: Array{Float64, 3},
    eval  :: Int64
    )     :: Nothing

    # reset buffer 
    tbuff[1] .= 0.0

    # get frequency arguments
    t, vt, vtp = m.Ω[w1], m.ν[w2], m.ν[w3]

    # define integrand
    integrand!(b, v, dv) = compute_t_kat!(Λ, b, v, dv, t, vt, vtp, r, m, a, da, temp)

    # compute integral
    ref = Λ + 0.5 * t
    integrate!(integrand!, tbuff, -100.0 * ref,  -2.0 * ref, eval)
    integrate!(integrand!, tbuff,   -2.0 * ref,   2.0 * ref, eval)
    integrate!(integrand!, tbuff,    2.0 * ref, 100.0 * ref, eval)

    # parse result
    for i in eachindex(da.Γ)
        da.Γ[i].ch_t.q3[:, w1, w2, w3] .= view(tbuff[1], i, :)
    end

    return nothing
end





# computation of t channel flow (left part, right part by symmetry) for frequency tuple (w1, w2, w3) on all lattice sites
function compute_channel_t_left!(
    Λ     :: Float64,
    w1    :: Int64,
    w2    :: Int64,
    w3    :: Int64,
    r     :: reduced_lattice,
    m     :: mesh,
    a     :: action,
    da    :: action,
    da_l  :: action,
    tbuff :: NTuple{3, Matrix{Float64}},
    temp  :: Array{Float64, 3},
    eval  :: Int64
    )     :: Nothing

    # reset buffer 
    tbuff[1] .= 0.0

    # get frequency arguments
    t, vt, vtp = m.Ω[w1], m.ν[w2], m.ν[w3]

    # define integrand
    integrand!(b, v, dv) = compute_t_left!(Λ, b, v, dv, t, vt, vtp, r, m, a, da, temp)

    # compute integral
    ref = Λ + 0.5 * t
    integrate!(integrand!, tbuff, -100.0 * ref,  -2.0 * ref, eval)
    integrate!(integrand!, tbuff,   -2.0 * ref,   2.0 * ref, eval)
    integrate!(integrand!, tbuff,    2.0 * ref, 100.0 * ref, eval)

    # parse result
    for i in eachindex(da_l.Γ)
        da_l.Γ[i].ch_t.q3[:, w1, w2, w3] .= view(tbuff[1], i, :)
    end

    return nothing
end





# computation of t channel flow (central) for frequency tuple (w1, w2, w3) on all lattice sites
function compute_channel_t_central!(
    Λ     :: Float64,
    w1    :: Int64,
    w2    :: Int64,
    w3    :: Int64,
    r     :: reduced_lattice,
    m     :: mesh,
    a     :: action,
    da_l  :: action,
    da_c  :: action,
    tbuff :: NTuple{3, Matrix{Float64}},
    temp  :: Array{Float64, 3},
    eval  :: Int64
    )     :: Nothing

    # reset buffer 
    tbuff[1] .= 0.0

    # get frequency arguments
    t, vt, vtp = m.Ω[w1], m.ν[w2], m.ν[w3]

    # define integrand
    integrand!(b, v, dv) = compute_t_central!(Λ, b, v, dv, t, vt, vtp, r, m, a, da_l, temp)

    # compute integral
    ref = Λ + 0.5 * t
    integrate!(integrand!, tbuff, -100.0 * ref,  -2.0 * ref, eval)
    integrate!(integrand!, tbuff,   -2.0 * ref,   2.0 * ref, eval)
    integrate!(integrand!, tbuff,    2.0 * ref, 100.0 * ref, eval)

    # parse result
    for i in eachindex(da_c.Γ)
        da_c.Γ[i].ch_t.q3[:, w1, w2, w3] .= view(tbuff[1], i, :)
    end

    return nothing
end