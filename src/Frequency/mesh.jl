"""
    Mesh

Struct containing frequency meshes for the self energy, vertices and susceptibilities.
* `num_σ :: Int64`           : total number of frequencies in the self energy mesh
* `num_Ω :: Int64`           : total number of frequencies in the bosonic meshes
* `num_ν :: Int64`           : total number of frequencies in the fermionic meshes
* `num_χ :: Int64`           : total number of frequencies in the susceptibility mesh
* `σ     :: Vector{Float64}` : self energy mesh
* `Ωs    :: Vector{Float64}` : bosonic mesh for the s channel
* `νs    :: Vector{Float64}` : fermionic mesh for the s channel
* `Ωt    :: Vector{Float64}` : bosonic mesh for the t channel 
* `νt    :: Vector{Float64}` : fermionic mesh for the t channel
* `Ωu    :: Vector{Float64}` : bosonic mesh for the u channel 
* `νu    :: Vector{Float64}` : fermionic mesh for the u channel
* `χ     :: Vector{Float64}` : susceptibility mesh
"""
struct Mesh 
    num_σ :: Int64 
    num_Ω :: Int64 
    num_ν :: Int64
    num_χ :: Int64
    σ     :: Vector{Float64}
    Ωs    :: Vector{Float64}
    νs    :: Vector{Float64}
    Ωt    :: Vector{Float64}
    νt    :: Vector{Float64}
    Ωu    :: Vector{Float64}
    νu    :: Vector{Float64}
    χ     :: Vector{Float64}
end

# auxiliary function to fetch frequency arguments for a specific channel and kernel 
function get_kernel_args(
    ch     :: Int64,
    kernel :: Int64,
    w1     :: Int64, 
    w2     :: Int64,
    w3     :: Int64,
    m      :: Mesh
    )      :: NTuple{3, Float64}

    # fetch arguments for s channel
    if ch == 1
        if kernel == 1
            return m.Ωs[w1], Inf, Inf 
        elseif kernel == 2 
            return m.Ωs[w1], m.νs[w2], Inf 
        elseif kernel == 3 
            return m.Ωs[w1], Inf, m.νs[w3]
        else 
            return m.Ωs[w1], m.νs[w2], m.νs[w3]
        end 
    # fetch arguments for t channel
    elseif ch == 2
        if kernel == 1
            return m.Ωt[w1], Inf, Inf 
        elseif kernel == 2 
            return m.Ωt[w1], m.νt[w2], Inf 
        elseif kernel == 3 
            return m.Ωt[w1], Inf, m.νt[w3]
        else 
            return m.Ωt[w1], m.νt[w2], m.νt[w3]
        end 
    # fetch arguments for u channel
    else
        if kernel == 1
            return m.Ωu[w1], Inf, Inf 
        elseif kernel == 2 
            return m.Ωu[w1], m.νu[w2], Inf 
        elseif kernel == 3 
            return m.Ωu[w1], Inf, m.νu[w3]
        else 
            return m.Ωu[w1], m.νu[w2], m.νu[w3]
        end 
    end 
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