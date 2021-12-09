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

# scan cut through channel, where x is assumed to be generated by the get_mesh function with linear fraction p0
# in order to optimize the frequency mesh we use the following heuristics:
#   a) the linear extent δ should be large enough to capture sharp peaks at finite frequencies, that is, δ >= 'point where channel has decayed to 10 percent of peak value'
#   b) the linear extent δ should be large enough to capture sign changes at finite frequencies, that is, δ >= zero crossing
#   c) the linear extent δ should be small enough to capture sharp peaks at the origin, that is, Δ should fulfill p1 < Δ < p2,
#      where Δ is the relative deviation between the values at the origin and the first finite frequency
#   d) the linear extent δ should be bounded to avoid overambitious shrinking or broadening of the mesh, that is, num_lin * p3 <= δ <= p4,
#      where num_lin is the number of linear frequencies
function scan(
    Λ  :: Float64,
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

    # a) check for peaks at finite frequencies and set δa as 'point where channel has decayed to 10 percent of peak value'
    δa  = Inf
    idx = argmax(abs.(y))

    if idx > 1 
        for i in idx + 1 : length(x)
            if abs(y[i] / y[idx]) <= 0.1
                δa = x[i]
                break
            end 
        end 
    end

    # b) check for sign change at finite frequencies and set δb accordingly 
    δb = Inf

    for i in 2 : length(y)
        if sign(y[i]) != sign(y[1]) 
            δb = x[i]
            break
        end 
    end

    # c) check if the value at the origin is finite. If so, set δc such that p1 < Δ < p2
    δc = num_lin * x[2]

    if abs(y[1]) / abs(y[idx]) > 1e-3
        # compute Δ
        Δ = abs(y[2] - y[1]) / max(abs(y[2]), abs(y[1]))

        # adjust δc if Δ is out of required bounds
        while (p1 < Δ < p2) == false
            # if Δ is too large decrease δc by two percent
            if Δ >= p2
                δc *= 0.98
            # if Δ is too small increase δc by two percent
            elseif Δ <= p1
                δc *= 1.02
            end

            # ensure δc is within required bounds
            if δc < num_lin * p3 
                break 
            elseif δc > p4
                break 
            end

            # generate new reference data
            xp = get_mesh(δc, x[end], length(x) - 1, p0)
            yp = similar(y)

            for i in eachindex(yp)
                p     = get_param(xp[i], x)
                yp[i] = p.lower_weight * y[p.lower_index] + p.upper_weight * y[p.upper_index]
            end

            # recompute Δ
            Δ = abs(yp[2] - yp[1]) / max(abs(yp[2]), abs(yp[1]))
        end
    end

    # d) merge results from heuristics a), b) and c) and perform sanity checks
    δ = min(max(num_lin * p3, min(δa, δb, δc)), p4)

    return δ
end

# auxiliary function to scan a single channel 
function scan_channel(
    Λ  :: Float64,
    p  :: NTuple{5, Float64},
    Ω  :: Vector{Float64},
    ν  :: Vector{Float64},
    ch :: Channel
    )  :: NTuple{2, Float64}

    # deref data
    q3 = ch.q3 

    # determine position of the maximum 
    idxs = argmax(abs.(q3))

    # get cuts through the maximum 
    q3_Ω   = q3[idxs[1], :, idxs[3], idxs[4]]
    q3_ν_1 = Float64[q3[idxs[1], idxs[2],      x,       x] - q3[idxs[1], idxs[2],     end,     end] for x in eachindex(ν)]
    q3_ν_2 = Float64[q3[idxs[1], idxs[2],      x, idxs[4]] - q3[idxs[1], idxs[2],     end, idxs[4]] for x in eachindex(ν)]
    q3_ν_3 = Float64[q3[idxs[1], idxs[2], idxs[3],      x] - q3[idxs[1], idxs[2], idxs[3],     end] for x in eachindex(ν)]

    # scan bosonic cut 
    Ω_lin = scan(Λ, Ω, q3_Ω, p[1], p[2], p[3], p[4] * Λ, p[5] * Λ)

    # scan fermionic cuts 
    ν_lin_1 = scan(Λ, ν, q3_ν_1, p[1], p[2], p[3], p[4] * Λ, p[5] * Λ)
    ν_lin_2 = scan(Λ, ν, q3_ν_2, p[1], p[2], p[3], p[4] * Λ, p[5] * Λ)
    ν_lin_3 = scan(Λ, ν, q3_ν_3, p[1], p[2], p[3], p[4] * Λ, p[5] * Λ)
    ν_lin   = min(ν_lin_1, ν_lin_2, ν_lin_3)

    return Ω_lin, ν_lin 
end

# resample an action to new meshes via scanning and trilinear interpolation
function resample_from_to(
    Λ      :: Float64,
    p_σ    :: NTuple{2, Float64},
    p_Γ    :: NTuple{5, Float64},
    p_χ    :: NTuple{5, Float64},
    lins   :: NTuple{5, Float64},
    bounds :: NTuple{5, Float64},
    m_old  :: Mesh,
    a_old  :: Action,
    a_new  :: Action,
    χ      :: Vector{Matrix{Float64}} 
    )      :: Mesh

    # scale linear bounds 
    σ_lin  = lins[2] * Λ 
    Ωs_lin = lins[3] * Λ 
    νs_lin = lins[4] * Λ
    Ωt_lin = lins[3] * Λ 
    νt_lin = lins[4] * Λ
    Ωu_lin = lins[3] * Λ 
    νu_lin = lins[4] * Λ
    χ_lin  = lins[5] * Λ

    # adjust meshes via scanning once required scale is reached 
    if Λ < lins[1]
        # scan self energy
        σ_lin = p_σ[2] * m_old.σ[argmax(abs.(a_old.Σ))]

        # scan the channels
        Ωs_lins, νs_lins = zeros(Float64, length(a_old.Γ)), zeros(length(a_old.Γ))
        Ωt_lins, νt_lins = zeros(Float64, length(a_old.Γ)), zeros(length(a_old.Γ))
        Ωu_lins, νu_lins = zeros(Float64, length(a_old.Γ)), zeros(length(a_old.Γ))
        
        for i in eachindex(a_old.Γ)
            Ωs_lins[i], νs_lins[i] = scan_channel(Λ, p_Γ, m_old.Ωs, m_old.νs, a_old.Γ[i].ch_s)
            Ωt_lins[i], νt_lins[i] = scan_channel(Λ, p_Γ, m_old.Ωt, m_old.νt, a_old.Γ[i].ch_t)
            Ωu_lins[i], νu_lins[i] = scan_channel(Λ, p_Γ, m_old.Ωu, m_old.νu, a_old.Γ[i].ch_u)
        end 

        Ωs_lin, νs_lin = minimum(Ωs_lins), minimum(νs_lins)
        Ωt_lin, νt_lin = minimum(Ωt_lins), minimum(νt_lins)
        Ωu_lin, νu_lin = minimum(Ωu_lins), minimum(νu_lins)

        # scan the correlations 
        χ_lins = zeros(Float64, length(χ))
        static = (m_old.num_χ - 1) ÷ 2 + 1

        for i in eachindex(χ_lins)
            # determine site with largest correlation
            idx = argmax(abs.(χ[i]))[1]

            # scan positive side
            χ_linp = scan(Λ, m_old.χ[static : end], χ[i][idx, static : end], p_χ[1], p_χ[2], p_χ[3], p_χ[4] * Λ, p_χ[5] * Λ)

            # scan negative side
            χ_linm = scan(Λ, -1.0 .* reverse(m_old.χ[1 : static], 1, static), χ[i][idx, 1 : static], p_χ[1], p_χ[2], p_χ[3], p_χ[4] * Λ, p_χ[5] * Λ)

            # parse result 
            χ_lins[i] = min(χ_linp, χ_linm)
        end

        χ_lin = minimum(χ_lins)
    end

    # build new frequency meshes according to scanning results
    σ     = get_mesh( σ_lin, bounds[2] * max(Λ, bounds[1]),       m_old.num_σ - 1, p_σ[1])
    Ωs    = get_mesh(Ωs_lin, bounds[3] * max(Λ, bounds[1]),       m_old.num_Ω - 1, p_Γ[1])
    νs    = get_mesh(νs_lin, bounds[4] * max(Λ, bounds[1]),       m_old.num_ν - 1, p_Γ[1])
    Ωt    = get_mesh(Ωt_lin, bounds[3] * max(Λ, bounds[1]),       m_old.num_Ω - 1, p_Γ[1])
    νt    = get_mesh(νt_lin, bounds[4] * max(Λ, bounds[1]),       m_old.num_ν - 1, p_Γ[1])
    Ωu    = get_mesh(Ωu_lin, bounds[3] * max(Λ, bounds[1]),       m_old.num_Ω - 1, p_Γ[1])
    νu    = get_mesh(νu_lin, bounds[4] * max(Λ, bounds[1]),       m_old.num_ν - 1, p_Γ[1])
    χ     = get_mesh( χ_lin, bounds[5] * max(Λ, bounds[1]), (m_old.num_χ - 1) ÷ 2, p_χ[1])
    χ     = sort(vcat(-1.0 .* χ[2 : end], χ))
    m_new = Mesh(m_old.num_σ, m_old.num_Ω, m_old.num_ν, m_old.num_χ, σ, Ωs, νs, Ωt, νt, Ωu, νu, χ)

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