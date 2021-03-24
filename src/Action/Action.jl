# load code
include("channel.jl")
include("vertex.jl")

abstract type action end

# load saving and reading for channels and vertices
include("disk.jl")

# load actions for different symmetries
include("action_lib/action_su2.jl")

# load checkpoints for different actions
include("checkpoint_lib/checkpoint_su2.jl")





# interface function to replace action with another action (except for bare)
function replace_with!(
    a1 :: action,
    a2 :: action
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
    a1 :: action,
    a2 :: action
    )  :: Nothing

    # replace vertices
    for i in eachindex(a1.Γ)
        replace_with!(a1.Γ[i], a2.Γ[i])
    end

    return nothing
end

# interface function to multiply action with factor (except for bare)
function mult_with!(
    a   :: action,
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
    a   :: action,
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
    a :: action
    ) :: Nothing

    mult_with!(a, 0.0)

    return nothing
end

# interface function to reset an action to zero only on the vertex level (except for bare)
function reset_Γ!(
    a :: action
    ) :: Nothing

    mult_with_Γ!(a, 0.0)

    return nothing
end

# interface function to multiply action with some factor and add to other action (except for bare)
function mult_with_add_to!(
    a2  :: action,
    fac :: Float64,
    a1  :: action
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
    a2  :: action,
    fac :: Float64,
    a1  :: action
    )   :: Nothing

    # multiply add for the vertices
    for i in eachindex(a1.Γ)
        mult_with_add_to!(a2.Γ[i], fac, a1.Γ[i])
    end

    return nothing
end

# interface function to add two actions (except for bare)
function add_to!(
    a2 :: action,
    a1 :: action
    )  :: Nothing

    mult_with_add_to!(a2, 1.0, a1)

    return nothing
end

# interface function to add two actions only on the vertex level (except for bare)
function add_to_Γ!(
    a2 :: action,
    a1 :: action
    )  :: Nothing

    mult_with_add_to_Γ!(a2, 1.0, a1)

    return nothing
end

# interface function to subtract two actions (except for bare)
function subtract_from!(
    a2 :: action,
    a1 :: action
    )  :: Nothing

    mult_with_add_to!(a2, -1.0, a1)

    return nothing
end

# interface function to subtract two actions only on the vertex level (except for bare)
function subtract_from_Γ!(
    a2 :: action,
    a1 :: action
    )  :: Nothing

    mult_with_add_to_Γ!(a2, -1.0, a1)

    return nothing
end

"""
    get_abs_max(
        a :: action
        ) :: Float64

Returns maximum absolute value of an action.
"""
function get_abs_max(
    a :: action
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
    a :: action
    ) :: Nothing

    for i in eachindex(a.Γ)
        limits!(a.Γ[i])
    end

    return nothing
end

# scans discrete data (x is assumed to: a) contain 0.0 and only positive values otherwise b) be sorted and c) be linearly spaced from 0.0 to some finite value)
# returns width for a linear discretization such that certain criteria are fulfilled:
# 1) if the absolute maximum of y is located at x = 0.0, the value at the first finite frequency should not be smaller than p1 times the maximum.
# 2) if the absolute maximum of y is located at x > 0.0, set width to p2 times position of the maximum.
# 3) ensure that the linear spacing of x is neither too small (p3) nor too large (p4).
# 4) ensure that the linear spacing of x does not shrink by more than 50 percent.
function scan(
    x  :: Vector{Float64},
    y  :: Vector{Float64},
    p1 :: Float64,
    p2 :: Float64,
    p3 :: Float64,
    p4 :: Float64
    )  :: Float64

    # find position and value of absolute maximum
    max_val, max_arg = findmax(abs.(y))

    # get current width of linear part
    num_lin = ceil(Int64, 0.4 * (length(x) - 1))
    δ       = num_lin * x[2]

    # if the absolute maximum of y is located at x = 0.0, the value at the first finite frequency should not be smaller than p1 times the maximum
    if max_arg == 1
        if abs(y[2] / y[1]) < p1
            # ensure that the linear spacing of x does not shrink by more than 50 percent
            δ *= max((p1 - 1.0) * y[1] / (y[2] - y[1]), 0.5)
        end
    # if the absolute maximum of y is located at x > 0.0, set width to p2 times position of the maximum.
    else
        # ensure that the linear spacing of x does not shrink by more than 50 percent
        δ = max(p2 * x[max_arg], 0.5 * δ)
    end

    # ensure that the linear spacing of x is neither too small (p3) nor too large (p4)
    δ = min(max(δ, num_lin * p3), num_lin * p4)

    return δ
end

# resample an action to new meshes via scanning and trilinear interpolation
function resample_from_to(
    Λ     :: Float64,
    Z     :: Float64,
    m_old :: mesh,
    a_old :: action,
    a_new :: action
    )     :: mesh

    # scan self energy
    σ_lin = min(max(1.5 * m_old.σ[argmax(abs.(a_old.Σ))], 0.5 * Λ), 10.0 * Λ)

    # determine dominant vertex component
    max_comp = argmax(Float64[get_abs_max(a_old.Γ[i]) for i in eachindex(a_old.Γ)])

    # scan the s channel (u channel related by symmetries)
    q3     = a_old.Γ[max_comp].ch_s.q3
    q3_Ω   = q3[1, :, 1, 1]
    q3_ν   = Float64[q3[1, 1, x, x] .- q3[1, 1, end, end] for x in 1 : m_old.num_ν]
    Ωs_lin = scan(m_old.Ωs, q3_Ω, 0.85, 1.5, 0.03 * Λ, 0.3 * Λ)
    νs_lin = scan(m_old.νs, q3_ν, 0.75, 1.5, 0.18 * Λ, 1.8 * Λ)

    # scan the t channel
    q3     = a_old.Γ[max_comp].ch_t.q3
    q3_Ω   = q3[1, :, 1, 1]
    q3_ν   = Float64[q3[1, 1, x, x] .- q3[1, 1, end, end] for x in 1 : m_old.num_ν]
    Ωt_lin = scan(m_old.Ωt, q3_Ω, 0.85, 1.5, 0.03 * Λ, 0.3 * Λ)
    νt_lin = scan(m_old.νt, q3_ν, 0.75, 1.5, 0.18 * Λ, 1.8 * Λ)

    # build new frequency meshes
    Λ_ref = max(Λ, 0.25 * Z)
    σ     = get_mesh(σ_lin,  350.0 * Λ_ref, m_old.num_σ - 1)
    Ωs    = get_mesh(Ωs_lin, 200.0 * Λ_ref, m_old.num_Ω - 1)
    νs    = get_mesh(νs_lin, 100.0 * Λ_ref, m_old.num_ν - 1)
    Ωt    = get_mesh(Ωt_lin, 200.0 * Λ_ref, m_old.num_Ω - 1)
    νt    = get_mesh(νt_lin, 100.0 * Λ_ref, m_old.num_ν - 1)
    m_new = mesh(m_old.num_σ, m_old.num_Ω, m_old.num_ν, σ, Ωs, νs, Ωt, νt)

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





# interface function to obtain empty action
function get_action_empty(
    symmetry :: String,
    r        :: reduced_lattice,
    m        :: mesh
    ;
    S        :: Float64 = 0.5,
    N        :: Float64 = 2.0
    )        :: action

    if symmetry == "su2"
        return get_action_su2_empty(S, N, r, m)
    end
end

"""
    read_checkpoint(
        file     :: HDF5.File,
        symmetry :: String,
        Λ        :: Float64
        )        :: Tuple{Float64, Float64, mesh, action}

Read checkpoint of FRG calculation with a certain symmetry from HDF5 file.
Returns cutoff Λ, ODE stepwidth dΛ, frequency meshes (wrapped in mesh struct) and vertices (wrapped in action struct).
"""
function read_checkpoint(
    file     :: HDF5.File,
    symmetry :: String,
    Λ        :: Float64
    )        :: Tuple{Float64, Float64, mesh, action}

    if symmetry == "su2"
        return read_checkpoint_su2(file, Λ)
    end
end
