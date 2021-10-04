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
# returns linear extent via FWHM instead if the data is sharply peaked away from the origin
# the linear spacing is bounded by [p3, p4]
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
    N       = length(x)
    num_lin = ceil(Int64, p0 * (N - 1))
    δ       = num_lin * x[2]

    # determine value and position of the maximum and minimum value of vertex
    max_val, max_idx = findmax(y)
    min_val, min_idx = findmin(y)

    maxabs, maxabs_idx, minabs, minabs_idx = abs(max_val)>abs(min_val) ? (abs(max_val), max_idx, abs(min_val), min_idx) : (abs(min_val), min_idx, abs(max_val), max_idx)

    # if no sign change, only one maximum present if second maximum is too small or at the edge, treat as notpresent
    if max_val*min_val >= 0 || minabs/maxabs<0.1 || minabs_idx > (p0 + 0.15) * (N - 1)

        # set linear extent to FWHM, if max not at zero
        if maxabs_idx != 1

            #find fwhm
            fwhm = 0.
            for i in maxabs_idx + 1 : length(y)
                if abs(y[i]) / maxabs < 0.5
                    fwhm = x[i]
                    break
                end
             end

             #determin δfwhm such that maximum is again mesh point and fwhm is covered
            i = max(1, floor(Int64,x[maxabs_idx]/(fwhm/num_lin)))
            δfwhm = x[maxabs_idx]/i*num_lin

            #to avoid focussing on peaks veryclose to zero, check for relative deviation from value at zero
            Δ = abs(y[2] - y[1]) /  abs(y[1])

            # determine new linear extent if Δ is out of required bounds if value at zero is non-vanishing
            while abs(y[1]) > 1e-5 && (p1 <= Δ <= p2) == false
                # decrease or increase linear extent
                if Δ > p2
                    δ *= 0.99
                elseif Δ < p1
                    δ *= 1.01
                end

                # check that linear extent is way smaller than upper bound
                if δ > 0.1 * x[end]
                    δ = 0.1 * x[end]
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
                Δ = abs(yp[2] - yp[1]) / abs(yp[1])
            end

            # determine larger of the found extents
            δ = max(δ, δfwhm)

        # else set linear extent via relative deviation between the value at the origin and the first finite frequency
        else

            # determine relative deviation between the value at the origin and the first finite frequency
            Δ = abs(y[2] - y[1]) /  abs(y[1])

            # determine new linear extent if Δ is out of required bounds
            while (p1 <= Δ <= p2) == false
                # decrease or increase linear extent
                if Δ > p2
                    δ *= 0.99
                elseif Δ < p1
                    δ *= 1.01
                end

                # check that linear extent is way smaller than upper bound
                if δ > 0.1 * x[end]
                    δ = 0.1 * x[end]
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
                Δ = abs(yp[2] - yp[1]) / abs(yp[1])
            end
        end
    # if data has two significant peaks with different signs, set to FWHM of further out one
    else

        iref = max(maxabs_idx, minabs_idx)
        fwhm = 0.
        for i in iref + 1 : length(y)
            if abs(y[i]) / abs(y[iref]) < 0.5
                fwhm = x[i]
                break
            end
        end

        #set δ such that maximum is again mesh point and fwhm is covered
        i = max(1, floor(Int64,x[maxabs_idx]/(fwhm/num_lin)))
        δ = x[maxabs_idx]/i*num_lin
    end


    # perform sanity check
    δ = min(max(num_lin * p3, δ), num_lin * p4)


    return δ
end

# interface function to scan a single channel
function scan_channel(
    Λ   :: Float64,
    p_Ω :: NTuple{5, Float64},
    p_ν :: NTuple{5, Float64},
    Ω   :: Vector{Float64},
    ν   :: Vector{Float64},
    ch  :: Channel
    )   :: NTuple{2, Float64}

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
    println("Ω scan")
    Ω_lin = scan(Ω, q3_Ω, p_Ω[1], p_Ω[2], p_Ω[3], p_Ω[4] * Λ, p_Ω[5] * Λ)

    # scan fermionic cuts
    println("diag scan")
    ν_lin_1 = scan(ν, q3_ν_1, p_ν[1], p_ν[2], p_ν[3], p_ν[4] * Λ, p_ν[5] * Λ)
    println("ν scan")
    ν_lin_2 = scan(ν, q3_ν_2, p_ν[1], p_ν[2], p_ν[3], p_ν[4] * Λ, p_ν[5] * Λ)
    println("ν' scan")
    ν_lin_3 = scan(ν, q3_ν_3, p_ν[1], p_ν[2], p_ν[3], p_ν[4] * Λ, p_ν[5] * Λ)
    ν_lin   = min(ν_lin_1, ν_lin_2, ν_lin_3)

    return Ω_lin, ν_lin
end

# resample an action to new meshes via scanning and trilinear interpolation
function resample_from_to(
    Λ     :: Float64,
    p_σ   :: NTuple{2, Float64},
    p_Ω   :: NTuple{5, Float64},
    p_ν   :: NTuple{5, Float64},
    m_old :: Mesh,
    a_old :: Action,
    a_new :: Action
    )     :: Mesh

    # scan self energy
    σ_idx = argmax(abs.(a_old.Σ))
    σ_lin = p_σ[2] * m_old.σ[σ_idx]

    # scan the s channel
    Ωs_lin = Float64[4.0 * Λ for i in eachindex(a_old.Γ)]
    νs_lin = Float64[4.0 * Λ for i in eachindex(a_old.Γ)]

    if Λ <= 2.0
        for comp in eachindex(a_old.Γ)
            println("s-channel scanning, component: $comp")
            res          = scan_channel(Λ, p_Ω, p_ν, m_old.Ωs[comp], m_old.νs[comp], a_old.Γ[comp].ch_s)
            Ωs_lin[comp] = res[1]
            νs_lin[comp] = res[2]
        end
    end

    # scan the t channel
    Ωt_lin = Float64[4.0 * Λ for i in eachindex(a_old.Γ)]
    νt_lin = Float64[4.0 * Λ for i in eachindex(a_old.Γ)]

    if Λ <= 2.0
        for comp in eachindex(a_old.Γ)
            println("t-channel scanning, component: $comp")
            res          = scan_channel(Λ, p_Ω, p_ν, m_old.Ωt[comp], m_old.νt[comp], a_old.Γ[comp].ch_t)
            Ωt_lin[comp] = res[1]
            νt_lin[comp] = res[2]
        end
    end

    # scan the u channel
    Ωu_lin = Float64[4.0 * Λ for i in eachindex(a_old.Γ)]
    νu_lin = Float64[4.0 * Λ for i in eachindex(a_old.Γ)]

    if Λ <= 2.0
        for comp in eachindex(a_old.Γ)
            println("u-channel scanning, component: $comp")
            res          = scan_channel(Λ, p_Ω, p_ν, m_old.Ωu[comp], m_old.νu[comp], a_old.Γ[comp].ch_u)
            Ωu_lin[comp] = res[1]
            νu_lin[comp] = res[2]
        end
    end

    # enforce equal fermionic meshes for s- and u-channel. Stabilizes the flow for some reason
    for comp in eachindex(a_old.Γ)
        νu_lin[comp] = max(νu_lin[comp],  νs_lin[comp])
        νs_lin[comp] = νu_lin[comp]
    end


    # build new frequency meshes according to scanning results
    σ     = get_mesh(min(σ_lin, 100.0 * max(Λ, 0.5)), 1000.0 * max(Λ, 0.5), m_old.num_σ - 1, p_σ[1])
    Ωs    = SVector(ntuple(comp -> get_mesh(min(Ωs_lin[comp], 75.0 * max(Λ, 0.5)), 750.0 * max(Λ, 0.5), m_old.num_Ω - 1, p_Ω[1]), length(Ωs_lin)))
    νs    = SVector(ntuple(comp -> get_mesh(min(νs_lin[comp], 50.0 * max(Λ, 0.5)), 500.0 * max(Λ, 0.5), m_old.num_ν - 1, p_ν[1]), length(νs_lin)))
    Ωt    = SVector(ntuple(comp -> get_mesh(min(Ωt_lin[comp], 75.0 * max(Λ, 0.5)), 750.0 * max(Λ, 0.5), m_old.num_Ω - 1, p_Ω[1]), length(Ωt_lin)))
    νt    = SVector(ntuple(comp -> get_mesh(min(νt_lin[comp], 50.0 * max(Λ, 0.5)), 500.0 * max(Λ, 0.5), m_old.num_ν - 1, p_ν[1]), length(νt_lin)))
    Ωu    = SVector(ntuple(comp -> get_mesh(min(Ωu_lin[comp], 75.0 * max(Λ, 0.5)), 750.0 * max(Λ, 0.5), m_old.num_Ω - 1, p_Ω[1]), length(Ωu_lin)))
    νu    = SVector(ntuple(comp -> get_mesh(min(νu_lin[comp], 50.0 * max(Λ, 0.5)), 500.0 * max(Λ, 0.5), m_old.num_ν - 1, p_ν[1]), length(νu_lin)))
    m_new = get_mesh(m_old, σ, Ωs, νs, Ωt, νt, Ωu, νu)

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
