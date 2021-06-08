# load code
include("channel.jl")
include("vertex.jl")

abstract type Action end

# get interpolated / extrapolated self energy for general action
function get_Σ(
    w :: Float64,
    m :: Mesh,
    a :: Action
    ) :: Float64

    # init value
    val = 0.0

    # check if in bounds, otherwise extrapolate
    if abs(w) <= m.σ[end]
        p   = get_param(abs(w), m.σ)
        val = sign(w) * (p.lower_weight * a.Σ[p.lower_index] + p.upper_weight * a.Σ[p.upper_index])
    else
        val = m.σ[end] * a.Σ[end] / w
    end

    return val
end

# get interpolated vertex component for general action
function get_Γ_comp(
    comp        :: Int64,
    site        :: Int64,
    bs          :: Buffer,
    bt          :: Buffer,
    bu          :: Buffer,
    r           :: Reduced_lattice,
    a           :: Action,
    apply_flags :: Function
    ;
    ch_s        :: Bool = true,
    ch_t        :: Bool = true,
    ch_u        :: Bool = true
    )           :: Float64

    # init with bare value
    val = a.Γ[comp].bare[site]

    # add s channel
    if ch_s
        # check for site exchange
        site_s = site

        if bs.exchange_flag
            site_s = r.exchange[site_s]
        end

        # apply other flags 
        sgn_s, comp_s = apply_flags(bs, comp)

        # check for mapping to u channel and interpolate
        if bs.map_flag
            val += sgn_s * get_vertex(site_s, bs, a.Γ[comp_s], 3)
        else
            val += sgn_s * get_vertex(site_s, bs, a.Γ[comp_s], 1)
        end
    end

    # add t channel
    if ch_t
        # check for site exchange
        site_t = site

        if bt.exchange_flag
            site_t = r.exchange[site_t]
        end

        # apply other flags 
        sgn_t, comp_t = apply_flags(bt, comp)

        # interpolate
        val += sgn_t * get_vertex(site_t, bt, a.Γ[comp_t], 2)
    end

    # add u channel
    if ch_u
        # check for site exchange
        site_u = site

        if bu.exchange_flag
            site_u = r.exchange[site_u]
        end

        # apply other flags 
        sgn_u, comp_u = apply_flags(bu, comp)

        # check for mapping to s channel and interpolate
        if bu.map_flag
            val += sgn_u * get_vertex(site_u, bu, a.Γ[comp_u], 1)
        else
            val += sgn_u * get_vertex(site_u, bu, a.Γ[comp_u], 3)
        end
    end

    return val
end

# get interpolated vertex component on all lattice sites
function get_Γ_comp_avx!(
    comp        :: Int64,
    r           :: Reduced_lattice,
    bs          :: Buffer,
    bt          :: Buffer,
    bu          :: Buffer,
    a           :: Action,
    apply_flags :: Function,
    temp        :: SubArray{Float64, 1, Array{Float64, 3}}
    ;
    ch_s        :: Bool = true,
    ch_t        :: Bool = true,
    ch_u        :: Bool = true
    )           :: Nothing

    # init with bare value
    @avx temp .= a.Γ[comp].bare

    # add s channel
    if ch_s
        # apply flags 
        sgn_s, comp_s = apply_flags(bs, comp)

        # check for mapping to u channel and interpolate
        if bs.map_flag
            get_vertex_avx!(r, bs, a.Γ[comp_s], 3, temp, bs.exchange_flag, sgn_s)
        else
            get_vertex_avx!(r, bs, a.Γ[comp_s], 1, temp, bs.exchange_flag, sgn_s)
        end
    end

    # add t channel
    if ch_t
        # apply flags 
        sgn_t, comp_t = apply_flags(bt, comp)

        # interpolate
        get_vertex_avx!(r, bt, a.Γ[comp_t], 2, temp, bt.exchange_flag, sgn_t)
    end

    # add u channel
    if ch_u
        # apply flags 
        sgn_u, comp_u = apply_flags(bu, comp)

        # check for mapping to s channel and interpolate
        if bu.map_flag
            get_vertex_avx!(r, bu, a.Γ[comp_u], 1, temp, bu.exchange_flag, sgn_u)
        else
            get_vertex_avx!(r, bu, a.Γ[comp_u], 3, temp, bu.exchange_flag, sgn_u)
        end
    end

    return nothing
end

# load saving and reading for channels and vertices
include("disk.jl")





# load specialized code for different symmetries
include("action_lib/action_su2.jl")
include("checkpoint_lib/checkpoint_su2.jl")





# interface function to replace action with another action (except for bare)
function replace_with!(
    a1 :: Action,
    a2 :: Action
    )  :: Nothing

    # replace self energy
    a1.Σ .= a2.Σ

    # replace vertices
    for i in eachindex(a1.Γ)
        replace_with!(a1.Γ[i], a2.Γ[i])
    end

    return nothing
end

# interface function to replace action with another action only on the vertex level (except for bare)
function replace_with_Γ!(
    a1 :: Action,
    a2 :: Action
    )  :: Nothing

    # replace vertices
    for i in eachindex(a1.Γ)
        replace_with!(a1.Γ[i], a2.Γ[i])
    end

    return nothing
end

# interface function to multiply action with factor (except for bare)
function mult_with!(
    a   :: Action,
    fac :: Float64
    )   :: Nothing

    # multiply self energy
    a.Σ .*= fac

    # multiply vertices
    for i in eachindex(a.Γ)
        mult_with!(a.Γ[i], fac)
    end

    return nothing
end

# interface function to multiply action with factor only on the vertex level (except for bare)
function mult_with_Γ!(
    a   :: Action,
    fac :: Float64
    )   :: Nothing

    # multiply vertices
    for i in eachindex(a.Γ)
        mult_with!(a.Γ[i], fac)
    end

    return nothing
end

# interface function to reset an action to zero (except for bare)
function reset!(
    a :: Action
    ) :: Nothing

    mult_with!(a, 0.0)

    return nothing
end

# interface function to reset an action to zero only on the vertex level (except for bare)
function reset_Γ!(
    a :: Action
    ) :: Nothing

    mult_with_Γ!(a, 0.0)

    return nothing
end

# interface function to multiply action with some factor and add to other action (except for bare)
function mult_with_add_to!(
    a2  :: Action,
    fac :: Float64,
    a1  :: Action
    )   :: Nothing

    # multiply add for the self energy
    a1.Σ .+= fac .* a2.Σ

    # multiply add for the vertices
    for i in eachindex(a1.Γ)
        mult_with_add_to!(a2.Γ[i], fac, a1.Γ[i])
    end

    return nothing
end

# interface function to multiply action with some factor and add to other action only on the vertex level (except for bare)
function mult_with_add_to_Γ!(
    a2  :: Action,
    fac :: Float64,
    a1  :: Action
    )   :: Nothing

    # multiply add for the vertices
    for i in eachindex(a1.Γ)
        mult_with_add_to!(a2.Γ[i], fac, a1.Γ[i])
    end

    return nothing
end

# interface function to add two actions (except for bare)
function add_to!(
    a2 :: Action,
    a1 :: Action
    )  :: Nothing

    mult_with_add_to!(a2, 1.0, a1)

    return nothing
end

# interface function to add two actions only on the vertex level (except for bare)
function add_to_Γ!(
    a2 :: Action,
    a1 :: Action
    )  :: Nothing

    mult_with_add_to_Γ!(a2, 1.0, a1)

    return nothing
end

# interface function to subtract two actions (except for bare)
function subtract_from!(
    a2 :: Action,
    a1 :: Action
    )  :: Nothing

    mult_with_add_to!(a2, -1.0, a1)

    return nothing
end

# interface function to subtract two actions only on the vertex level (except for bare)
function subtract_from_Γ!(
    a2 :: Action,
    a1 :: Action
    )  :: Nothing

    mult_with_add_to_Γ!(a2, -1.0, a1)

    return nothing
end

"""
    get_abs_max(
        a :: Action
        ) :: Float64

Returns maximum absolute vertex value of an action.
"""
function get_abs_max(
    a :: Action
    ) :: Float64

    abs_max_Γ = zeros(Float64, length(a.Γ))

    for i in eachindex(a.Γ)
        abs_max_Γ[i] = get_abs_max(a.Γ[i])
    end

    abs_max = maximum(abs_max_Γ)

    return abs_max
end

# set asymptotic limits by scanning the boundaries of q3
function limits!(
    a :: Action
    ) :: Nothing

    for i in eachindex(a.Γ)
        limits!(a.Γ[i])
    end

    return nothing
end

# scan cut through channel, where x is assumed to be generated by the get_mesh function with linear fraction p0
# returns linear extend such that p1 <= Δ <= p2, where Δ is the relative deviation between the value at the origin and the first finite frequency
# the linear spacing (i.e. linear extend divided by number of linear frequencies) is bounded by [p3, p4]
function scan(
    x  :: Vector{Float64},
    y  :: Vector{Float64},
    p0 :: Float64,
    p1 :: Float64,
    p2 :: Float64,
    p3 :: Float64,
    p4 :: Float64
    )  :: Float64

    # determine current mesh layout
    num_lin = ceil(Int64, p0 * (length(x) - 1))
    δ       = num_lin * x[2]
    δp      = δ

    # determine relative deviation from origin to first finite frequency
    Δ = abs(y[2] - y[1]) / max(abs(y[2]), abs(y[1]))

    # determine new width if Δ is out of required bounds
    while (p1 <= Δ <= p2) == false
        # if Δ is too large decrease the width by one percent
        if Δ > p2
            δp *= 0.99
        # if Δ is too small increase the width by one percent
        elseif Δ < p1
            δp *= 1.01
        end

        # generate new reference data
        xp = get_mesh(δp, x[end], length(x) - 1, p0)
        yp = similar(y)

        for i in eachindex(yp)
            p     = get_param(xp[i], x)
            yp[i] = p.lower_weight * y[p.lower_index] + p.upper_weight * y[p.upper_index]
        end

        # recompute Δ
        Δ = abs(yp[2] - yp[1]) / max(abs(yp[2]), abs(yp[1]))
    end

    # check that linear spacing is neither too small nor too large
    δ = min(max(δp, num_lin * p3), num_lin * p4)

    return δ
end

# resample an action to new meshes via scanning and trilinear interpolation
function resample_from_to(
    Λ     :: Float64,
    p     :: NTuple{5, Float64},
    m_old :: Mesh,
    a_old :: Action,
    a_new :: Action
    )     :: Mesh

    # scan self energy
    σ_lin = 1.2 * m_old.σ[argmax(abs.(a_old.Σ))]

    # scan the s channel
    comp   = argmax([get_abs_max(a_old.Γ[i].ch_s) for i in eachindex(a_old.Γ)])
    q3     = a_old.Γ[comp].ch_s.q3
    q3_Ω   = q3[1, :, 1, 1]
    q3_ν   = q3[1, 1, :, 1] .- q3[1, 1, end, 1]
    Ωs_lin = 5.0 * Λ
    νs_lin = 5.0 * Λ

    if maximum(abs.(q3_Ω)) > 1e-3
        Ωs_lin = scan(m_old.Ωs, q3_Ω, p[1], p[2], p[3], p[4] * Λ, p[5] * Λ)
    end

    if maximum(abs.(q3_ν)) > 1e-3
        νs_lin = scan(m_old.νs, q3_ν, p[1], p[2], p[3], p[4] * Λ, p[5] * Λ)
    end

    # scan the t channel
    comp   = argmax([get_abs_max(a_old.Γ[i].ch_t) for i in eachindex(a_old.Γ)])
    q3     = a_old.Γ[comp].ch_t.q3
    q3_Ω   = q3[1, :, 1, 1]
    q3_ν   = q3[1, 1, :, 1] .- q3[1, 1, end, 1]
    Ωt_lin = 5.0 * Λ
    νt_lin = 5.0 * Λ

    if maximum(abs.(q3_Ω)) > 1e-3
        Ωt_lin = scan(m_old.Ωt, q3_Ω, p[1], p[2], p[3], p[4] * Λ, p[5] * Λ)
    end

    if maximum(abs.(q3_ν)) > 1e-3
        νt_lin = scan(m_old.νt, q3_ν, p[1], p[2], p[3], p[4] * Λ, p[5] * Λ)
    end

    # scan the u channel
    comp   = argmax([get_abs_max(a_old.Γ[i].ch_u) for i in eachindex(a_old.Γ)])
    q3     = a_old.Γ[comp].ch_u.q3
    q3_Ω   = q3[1, :, 1, 1]
    q3_ν   = q3[1, 1, :, 1] .- q3[1, 1, end, 1]
    Ωu_lin = 5.0 * Λ
    νu_lin = 5.0 * Λ

    if maximum(abs.(q3_Ω)) > 1e-3
        Ωu_lin = scan(m_old.Ωu, q3_Ω, p[1], p[2], p[3], p[4] * Λ, p[5] * Λ)
    end

    if maximum(abs.(q3_ν)) > 1e-3
        νu_lin = scan(m_old.νu, q3_ν, p[1], p[2], p[3], p[4] * Λ, p[5] * Λ)
    end

    # set reference scale for upper mesh bound
    Λ_ref = max(Λ, 0.5)

    # build new frequency meshes according to scanning results
    σ     = get_mesh(min(σ_lin, 125.0 * Λ_ref), 250.0 * Λ_ref, m_old.num_σ - 1, p[1])
    Ωs    = get_mesh(min(Ωs_lin, 75.0 * Λ_ref), 150.0 * Λ_ref, m_old.num_Ω - 1, p[1])
    νs    = get_mesh(min(νs_lin, 35.0 * Λ_ref),  75.0 * Λ_ref, m_old.num_ν - 1, p[1])
    Ωt    = get_mesh(min(Ωt_lin, 75.0 * Λ_ref), 150.0 * Λ_ref, m_old.num_Ω - 1, p[1])
    νt    = get_mesh(min(νt_lin, 35.0 * Λ_ref),  75.0 * Λ_ref, m_old.num_ν - 1, p[1])
    Ωu    = get_mesh(min(Ωu_lin, 75.0 * Λ_ref), 150.0 * Λ_ref, m_old.num_Ω - 1, p[1])
    νu    = get_mesh(min(νu_lin, 35.0 * Λ_ref),  75.0 * Λ_ref, m_old.num_ν - 1, p[1])
    m_new = Mesh(m_old.num_σ, m_old.num_Ω, m_old.num_ν, σ, Ωs, νs, Ωt, νt, Ωu, νu)

    # resample self energy
    for w in eachindex(m_new.σ)
        a_new.Σ[w] = get_Σ(m_new.σ[w], m_old, a_old)
    end

    # resample vertices
    for i in eachindex(a_new.Γ)
        resample_from_to!(m_old, a_old.Γ[i], m_new, a_new.Γ[i])
    end

    return m_new
end





# generate action dummy
function get_action_empty(
    symmetry :: String,
    r        :: Reduced_lattice,
    m        :: Mesh
    ;
    S        :: Float64 = 0.5
    )        :: Action

    if symmetry == "su2"
        return get_action_su2_empty(S, r, m)
    end
end

"""
    read_checkpoint(
        file     :: HDF5.File,
        Λ        :: Float64
        )        :: Tuple{Float64, Float64, Mesh, Action}

Read checkpoint of FRG calculation from HDF5 file.
Returns cutoff Λ, ODE stepwidth dΛ, frequency meshes (wrapped in Mesh struct) and vertices (wrapped in Action struct).
"""
function read_checkpoint(
    file     :: HDF5.File,
    Λ        :: Float64
    )        :: Tuple{Float64, Float64, Mesh, Action}

    # read symmetry group from file
    symmetry = read(file, "symmetry")

    if symmetry == "su2"
        return read_checkpoint_su2(file, Λ)
    end
end





# load tests and timers
include("test.jl")
include("timers.jl")
