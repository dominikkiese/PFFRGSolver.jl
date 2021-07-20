"""
    Mesh

Struct containing frequency meshes for the self energy and vertices.
* `num_σ :: Int64`           : total number of frequencies in the self energy mesh
* `num_Ω :: Int64`           : total number of frequencies in the bosonic meshes
* `num_ν :: Int64`           : total number of frequencies in the fermionic meshes
* `σ     :: Vector{Float64}` : self energy mesh
* `Ωs    :: Vector{Float64}` : bosonic mesh for the s channel
* `νs    :: Vector{Float64}` : fermionic mesh for the s channel
* `Ωt    :: Vector{Float64}` : bosonic mesh for the t channel 
* `νt    :: Vector{Float64}` : fermionic mesh for the t channel
* `Ωu    :: Vector{Float64}` : bosonic mesh for the u channel 
* `νu    :: Vector{Float64}` : fermionic mesh for the u channel
"""
struct Mesh 
    num_σ :: Int64 
    num_Ω :: Int64 
    num_ν :: Int64
    σ     :: Vector{Float64}
    Ωs    :: Vector{Float64}
    νs    :: Vector{Float64}
    Ωt    :: Vector{Float64}
    νt    :: Vector{Float64}
    Ωu    :: Vector{Float64}
    νu    :: Vector{Float64}
end

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