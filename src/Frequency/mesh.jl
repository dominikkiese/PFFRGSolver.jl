struct mesh 
    σ :: Vector{Float64}
    Ω :: Vector{Float64}
    ν :: Vector{Float64}
end

# generate a mesh of (num + 1) linearly and logarithmically spaced frequencies (linear and upper are included, 40 percent of the grid are devoted to linearly spaced frequencies)
function get_mesh(
    linear :: Float64,
    upper  :: Float64,
    num    :: Int64
    )      :: Vector{Float64}

    @assert linear < upper "Linear bound must be smaller than upper bound." 

    # compute number of linear and logarithmic points
    num_lin = ceil(Int64, 0.4 * num)
    num_log = num - num_lin 

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