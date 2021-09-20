# define abstract mesh type 
abstract type Mesh end

"""
    get_mesh(
        linear :: Float64,
        upper  :: Float64,
        num    :: Int64,
        p      :: Float64
        )      :: Vector{Float64}
        
Generate a mesh of (num + 1) linearly (0.0 to linear) and logarithmically (linear to upper) spaced frequencies.
Thereby linear and upper are explicitly included and p * num frequencies of the grid are devoted to the linear part.
"""
function get_mesh(
    linear :: Float64,
    upper  :: Float64,
    num    :: Int64,
    p      :: Float64
    )      :: Vector{Float64}

    # sanity check
    @assert linear < upper "Linear bound must be smaller than upper bound." 

    # compute number of linear and logarithmic points
    num_lin = ceil(Int64, p * num)
    num_log = num - num_lin 

    # sanity check
    @assert num_log > 0 "Number of frequencies is too small." 

    # determine linear width and logarithmic factor
    h = linear / num_lin
    ξ = (upper / linear)^(1.0 / num_log)

    # allocate list 
    mesh = zeros(Float64, num + 1)

    # compute frequencies
    for i in 1 : num_lin
        mesh[i + 1] = i * h 
    end 

    for i in 1 : num_log
        mesh[num_lin + 1 + i] = ξ^i * linear
    end

    return mesh 
end

# load meshes for different symmetries
include("mesh_lib/mesh_su2.jl")
include("mesh_lib/mesh_u1_dm.jl")

# interface function to generate mesh struct at given initial scale and distribution parameters
function get_mesh(
    symmetry :: String,
    initial  :: Float64,
    num_σ    :: Int64,
    num_Ω    :: Int64,
    num_ν    :: Int64,
    p_σ      :: Float64,
    p_Ω      :: Float64,
    p_ν      :: Float64
    )        :: Mesh

    if symmetry == "su2"
        return get_mesh_su2(initial, num_σ, num_Ω, num_ν, p_σ, p_Ω, p_ν)
    elseif symmetry == "u1-dm"
        return get_mesh_u1_dm(initial, num_σ, num_Ω, num_ν, p_σ, p_Ω, p_ν)
    end
end