# spin kernel for the t channel
function compute_t_kernel_spin!(
    buff :: Vector{Float64},
    p    :: Float64,
    dv   :: Float64,
    v3   :: NTuple{2, Float64},
    v4   :: NTuple{2, Float64},
    S    :: Float64,
    r    :: Reduced_lattice,
    temp :: Array{Float64, 3}
    )    :: Nothing 

    v3s, v3d = v3 
    v4s, v4d = v4

    @inbounds @fastmath for i in eachindex(r.sites)
        v1s = temp[i, 1, 1]
        v2s = temp[i, 1, 2]
        val = -1.0 * v1s * v4s + v1s * v4d - 1.0 * v3s * v2s + v3d * v2s

        overlap_i = r.overlap[i]
        Range     = size(overlap_i, 1)

        @turbo for j in 1 : Range
            v1s  = temp[overlap_i[j, 1], 1, 1]
            v2s  = temp[overlap_i[j, 2], 1, 2]
            val += -2.0 * overlap_i[j, 3] * (2.0 * S) * v1s * v2s
        end

        buff[i] += -p * val * dv
    end

    return nothing 
end

# density kernel for the t channel
function compute_t_kernel_dens!(
    buff :: Vector{Float64},
    p    :: Float64,
    dv   :: Float64,
    v3   :: NTuple{2, Float64},
    v4   :: NTuple{2, Float64},
    S    :: Float64,
    r    :: Reduced_lattice,
    temp :: Array{Float64, 3}
    )    :: Nothing 

    v3s, v3d = v3 
    v4s, v4d = v4

    @inbounds @fastmath for i in eachindex(r.sites)
        v1d = temp[i, 2, 1]
        v2d = temp[i, 2, 2]
        val = 3.0 * v1d * v4s + v1d * v4d + 3.0 * v3s * v2d + v3d * v2d

        overlap_i = r.overlap[i]
        Range     = size(overlap_i, 1)

        @turbo for j in 1 : Range
            v1d  = temp[overlap_i[j, 1], 2, 1]
            v2d  = temp[overlap_i[j, 2], 2, 2]
            val += -2.0 * overlap_i[j, 3] * (2.0 * S) * v1d * v2d
        end

        buff[i] += -p * val * dv
    end

    return nothing 
end