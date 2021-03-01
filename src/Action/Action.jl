# load code 
include("channel.jl")
include("vertex.jl")

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

# interface function to resample action from old to new frequency meshes 
function resample_from_to( 
    Λ     :: Float64,
    m_old :: mesh,
    a_old :: action,
    a_new :: action
    )     :: mesh

    # build new frequency meshes
    σ     = get_mesh(3.5 * Λ, 350.0 * Λ, length(m_old.σ) - 1)
    Ω     = get_mesh(2.5 * Λ, 100.0 * Λ, length(m_old.Ω) - 1)
    ν     = get_mesh(2.0 * Λ,  70.0 * Λ, length(m_old.ν) - 1)
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

