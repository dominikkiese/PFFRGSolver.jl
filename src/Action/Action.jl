# load code 
include("channel.jl")
include("vertex.jl")

# define abstract type action 
abstract type action end

# load saving and reading for channels and vertices
include("disk.jl")

# load actions for different symmetries 
include("action_lib/action_sun.jl")

# load checkpoints for different actions 
include("checkpoint_lib/checkpoint_sun.jl")





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

# interface function to obtain maximum between vertex components
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

# function to scan low frequency part of channels
function scan(
    x  :: Vector{Float64}, 
    y  :: Vector{Float64},
    p1 :: Float64,
    p2 :: Float64,
    p3 :: Float64,
    p4 :: Float64
    )  :: Float64

    # find position and value of maximum 
    max_val, max_arg = findmax(abs.(y))

    # get current width of linear part
    δ = ceil(Int64, 0.3 * (length(x) - 1)) * x[2]

    # if maximum at zero, value at first finite frequency should not be smaller than p1 times the maximum
    if max_arg == 1 
        if abs(y[2] / y[1]) < p1
            δ *= max(0.5 * (p1 - 1.0) * y[1] / (y[2] - y[1]), 0.5)
        end 
    # if maximum at finite frequency, set width to p2 times position of the maximum
    else 
        δ = max(min(p2 * x[max_arg], δ), 0.5 * δ)
    end 

    # ensure that the width is neither too small (p3) nor too large (p4)
    δ = min(max(δ, p3), p4)

    return δ
end

# interface function to resample action from old to new frequency meshes 
function resample_from_to( 
    Λ     :: Float64,
    m_old :: mesh,
    a_old :: action,
    a_new :: action
    )     :: mesh

    # scan self energy 
    σ_lin = min(max(1.5 * m_old.σ[argmax(abs.(a_old.Σ))], 0.1 * Λ), 8.0 * Λ)

    # determine dominant vertex component
    abs_max_Γ = Float64[get_abs_max(a_old.Γ[i]) for i in eachindex(a_old.Γ)]
    max_comp  = argmax(abs_max_Γ)

    # determine dominant channel of dominant vertex component
    max_ch = argmax((get_abs_max(a_old.Γ[max_comp].ch_s), get_abs_max(a_old.Γ[max_comp].ch_t), get_abs_max(a_old.Γ[max_comp].ch_u)))
    q3     = similar(a_old.Γ[max_comp].ch_s.q3)
    
    if max_ch == 1 
        q3 .= a_old.Γ[max_comp].ch_s.q3
    elseif max_ch == 2
        q3 .= a_old.Γ[max_comp].ch_t.q3
    elseif max_ch == 3 
        q3 .= a_old.Γ[max_comp].ch_u.q3
    end
    
    # determine dominant lattice site of dominant vertex component
    max_site = argmax(abs.(a_old.Γ[max_comp].bare))

    # scan q3 in dominant channel of dominant vertex component at dominant lattice site
    Ω_lin = scan(m_old.Ω, q3[max_site, :, 1, 1], 0.85, 1.5, 0.1 * Λ, 4.0 * Λ)
    ν_lin = scan(m_old.ν, Float64[q3[max_site, 1, x, x] .- q3[max_site, 1, end, end] for x in eachindex(m_old.ν)], 0.75, 6.0, 0.1 * Λ, 6.0 * Λ)
    
    # build new frequency meshes
    σ     = get_mesh(σ_lin, 800.0 * Λ, length(m_old.σ) - 1)
    Ω     = get_mesh(Ω_lin, 300.0 * Λ, length(m_old.Ω) - 1)
    ν     = get_mesh(ν_lin, 500.0 * Λ, length(m_old.ν) - 1)
    m_new = mesh(σ, Ω, ν)

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

    if symmetry == "sun"
        return get_action_sun_empty(S, N, r, m)
    end 
end

# interface function to read checkpoint from file
function read_checkpoint(
    file     :: HDF5.File,
    symmetry :: String,
    Λ        :: Float64
    )        :: Tuple{Float64, Float64, mesh, action}

    if symmetry == "sun"
        return read_checkpoint_sun(file, Λ)
    end 
end

