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

    # check if in bounds, otherwise extrapolate as 1 / w
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
        comp_s        = comp 
        sgn_s, comp_s = apply_flags(bs, comp_s)

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
        comp_t        = comp
        sgn_t, comp_t = apply_flags(bt, comp_t)

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
        comp_u        = comp
        sgn_u, comp_u = apply_flags(bu, comp_u)

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
    @turbo temp .= a.Γ[comp].bare

    # add s channel
    if ch_s
        # apply flags 
        comp_s        = comp
        sgn_s, comp_s = apply_flags(bs, comp_s)

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
        comp_t        = comp
        sgn_t, comp_t = apply_flags(bt, comp_t)

        # interpolate
        get_vertex_avx!(r, bt, a.Γ[comp_t], 2, temp, bt.exchange_flag, sgn_t)
    end

    # add u channel
    if ch_u
        # apply flags 
        comp_u        = comp
        sgn_u, comp_u = apply_flags(bu, comp_u)

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
include("action_lib/action_su2.jl")  ; include("checkpoint_lib/checkpoint_su2.jl")
include("action_lib/action_u1_dm.jl"); include("checkpoint_lib/checkpoint_u1_dm.jl")





# interface function to replace action with another action (except for bare)
function replace_with!(
    a1 :: Action,
    a2 :: Action
    )  :: Nothing

    # replace self energy
    @turbo a1.Σ .= a2.Σ

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
    @turbo a.Σ .*= fac

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
    @turbo a1.Σ .+= fac .* a2.Σ

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

Returns maximum absolute value of an action across all vertex components.
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
# returns linear extent such that p1 <= Δ <= p2, where Δ is the relative deviation between the value at the origin and the first finite frequency
# if the value at the origin vanishes within the integration tolerance, set linear extent to p3 times position of the maximum
# the linear spacing is bounded from below by p4
function scan(
    x   :: Vector{Float64},
    y   :: Vector{Float64},
    p0  :: Float64,
    p1  :: Float64,
    p2  :: Float64,
    p3  :: Float64,
    p4  :: Float64,
    tol :: Float64
    )   :: Float64

    # determine current mesh layout
    N       = length(x)
    num_lin = ceil(Int64, p0 * (N - 1))
    δ       = num_lin * x[2]

    # set linear extent such that p1 <= Δ <= p2, where Δ is the relative deviation between the value at the origin and the first finite frequency
    if abs(y[1]) > tol
        # determine relative deviation between the value at the origin and the first finite frequency
        Δ = abs(y[2] - y[1]) / max(abs(y[2]), abs(y[1]))

        # determine new linear extent if Δ is out of required bounds
        while (p1 <= Δ <= p2) == false
            # decrease or increase linear extent
            if Δ > p2
                δ *= 0.99
            elseif Δ < p1
                δ *= 1.01
            end

            # check that linear extent is not too large
            if δ > p0 * x[end]
                δ = p0 * x[end]
                break 
            end

            # generate new reference data
            xp = get_mesh(δ, x[end], N - 1, p0)
            yp = similar(y)

            for i in eachindex(yp)
                p     = get_param(xp[i], x)
                yp[i] = p.lower_weight * y[p.lower_index] + p.upper_weight * y[p.upper_index]
            end

            # recompute Δ
            Δ = abs(yp[2] - yp[1]) / max(abs(yp[2]), abs(yp[1]))
        end
    # if the value at the origin vanishes within the integration tolerance, set linear extent to p3 times position of the maximum
    else 
        δ = p3 * x[argmax(abs.(y))]
    end

    # perform sanity check
    δ = max(num_lin * p4, δ)

    return δ
end

# auxiliary function to scan channel 
function scan_channel(
    Λ     :: Float64,
    ch    :: Channel,
    Ω     :: Vector{Float64},
    ν     :: Vector{Float64},
    p_Ω   :: NTuple{5, Float64},
    p_ν   :: NTuple{5, Float64},
    tol   :: Float64
    )     :: NTuple{2, Float64}

    # scan bosonic axis 
    Ω_lin = 0.0 

    for vp in 2 : min(4, size(ch.q3, 4))
        for v in 2 : min(4, size(ch.q3, 3))
            for i in 1 : min(3, size(ch.q3, 1))
                Ω_lin += scan(Ω, ch.q3[i, :, v, vp], p_Ω[1], p_Ω[2], p_Ω[3], p_Ω[4], p_Ω[5] * Λ, tol)
            end 
        end 
    end 

    Ω_lin /= length(2 : min(4, size(ch.q3, 4)))
    Ω_lin /= length(2 : min(4, size(ch.q3, 3)))
    Ω_lin /= length(1 : min(3, size(ch.q3, 1)))

    # scan fermionic axis 
    ν_lin = 0.0 

    for vp in 2 : min(4, size(ch.q3, 4))
        for w in 1 : min(3, size(ch.q3, 2))
            for i in 1 : min(3, size(ch.q3, 1))
                ν_lin += 0.5 * scan(ν, ch.q3[i, w, :, vp], p_ν[1], p_ν[2], p_ν[3], p_ν[4], p_ν[5] * Λ, tol)
                ν_lin += 0.5 * scan(ν, ch.q3[i, w, vp, :], p_ν[1], p_ν[2], p_ν[3], p_ν[4], p_ν[5] * Λ, tol)
            end 
        end 
    end 

    ν_lin /= length(2 : min(4, size(ch.q3, 4)))
    ν_lin /= length(1 : min(3, size(ch.q3, 2)))
    ν_lin /= length(1 : min(3, size(ch.q3, 1)))

    return Ω_lin, ν_lin 
end

# resample an action to new meshes via scanning and trilinear interpolation
function resample_from_to(
    Λ     :: Float64,
    p_σ   :: NTuple{2, Float64},
    p_Ω   :: NTuple{5, Float64},
    p_ν   :: NTuple{5, Float64},
    tol   :: Float64,
    m_old :: Mesh,
    a_old :: Action,
    a_new :: Action
    )     :: Mesh

    # scan self energy
    σ_idx = argmax(abs.(a_old.Σ))
    σ_lin = p_σ[2] * m_old.σ[σ_idx]

    # scan the s channel
    Ωs_lin = Float64[8.0 * Λ for i in eachindex(a_old.Γ)]
    νs_lin = Float64[6.0 * Λ for i in eachindex(a_old.Γ)]

    for comp in eachindex(a_old.Γ)
        if maximum(abs.(a_old.Γ[comp].ch_s.q3)) > tol
            results      = scan_channel(Λ, a_old.Γ[comp].ch_s, m_old.Ωs[comp], m_old.νs[comp], p_Ω, p_ν, tol)
            Ωs_lin[comp] = results[1]
            νs_lin[comp] = results[2]
        end
    end

    # scan the t channel
    Ωt_lin = Float64[8.0 * Λ for i in eachindex(a_old.Γ)]
    νt_lin = Float64[6.0 * Λ for i in eachindex(a_old.Γ)]

    for comp in eachindex(a_old.Γ)
        if maximum(abs.(a_old.Γ[comp].ch_t.q3)) > tol
            results      = scan_channel(Λ, a_old.Γ[comp].ch_t, m_old.Ωt[comp], m_old.νt[comp], p_Ω, p_ν, tol)
            Ωt_lin[comp] = results[1]
            νt_lin[comp] = results[2]
        end
    end
    
    # build new frequency meshes according to scanning results
    ref   = max(Λ, 0.1)
    σ     = get_mesh(min(σ_lin, p_σ[1] * 500.0 * ref), 500.0 * ref, m_old.num_σ - 1, p_σ[1])
    Ωs    = SVector(ntuple(comp -> get_mesh(min(Ωs_lin[comp], p_Ω[1] * 250.0 * ref), 250.0 * ref, m_old.num_Ω - 1, p_Ω[1]), length(Ωs_lin)))
    νs    = SVector(ntuple(comp -> get_mesh(min(νs_lin[comp], p_ν[1] * 150.0 * ref), 150.0 * ref, m_old.num_ν - 1, p_ν[1]), length(νs_lin)))
    Ωt    = SVector(ntuple(comp -> get_mesh(min(Ωt_lin[comp], p_Ω[1] * 250.0 * ref), 250.0 * ref, m_old.num_Ω - 1, p_Ω[1]), length(Ωt_lin)))
    νt    = SVector(ntuple(comp -> get_mesh(min(νt_lin[comp], p_ν[1] * 150.0 * ref), 150.0 * ref, m_old.num_ν - 1, p_ν[1]), length(νt_lin)))
    m_new = get_mesh(m_old, σ, Ωs, νs, Ωt, νt)

    # resample self energy
    for w in eachindex(m_new.σ)
        a_new.Σ[w] = get_Σ(m_new.σ[w], m_old, a_old)
    end

    # resample vertices
    for comp in eachindex(a_new.Γ)
        resample_from_to!(comp, m_old, a_old.Γ[comp], m_new, a_new.Γ[comp])
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
    elseif symmetry == "u1-dm"
        return get_action_u1_dm_empty(r, m)
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
    elseif symmetry == "u1-dm"
        return read_checkpoint_u1_dm(file, Λ)
    end
end





# load tests and timers
include("test.jl")
include("timers.jl")