# xx kernel for the t channel
function compute_t_kernel_xx!(
    buff :: Vector{Float64},
    p    :: Float64,
    dv   :: Float64,
    v3   :: NTuple{6, Float64},
    v4   :: NTuple{6, Float64},
    r    :: Reduced_lattice,
    temp :: Array{Float64, 3}
    )    :: Nothing 

    v3xx, v3zz, v3DM, v3dd, v3zd, v3dz = v3
    v4xx, v4zz, v4DM, v4dd, v4zd, v4dz = v4

    @inbounds @fastmath for i in eachindex(r.sites)
        v1xx = temp[i, 1, 1]; v1DM = temp[i, 3, 1]
        v2xx = temp[i, 1, 2]; v2DM = temp[i, 3, 2]
        val  = v1DM * v4dz - v1DM * v4zd + v1xx * v4dd - v1xx * v4zz + v3dd * v2xx + v3dz * v2DM - v3zd * v2DM - v3zz * v2xx

        overlap_i = r.overlap[i]
        Range     = size(overlap_i, 1)

        @turbo for j in 1 : Range
            v1xx  = temp[overlap_i[j, 1], 1, 1]; v1DM = temp[overlap_i[j, 1], 3, 1]
            v2xx  = temp[overlap_i[j, 2], 1, 2]; v2DM = temp[overlap_i[j, 2], 3, 2]
            val  += -2.0 * overlap_i[j, 3] * (v1xx * v2xx - v1DM * v2DM)
        end

        buff[i] += -p * val * dv
    end

    return nothing 
end

# zz kernel for the t channel
function compute_t_kernel_zz!(
    buff :: Vector{Float64},
    p    :: Float64,
    dv   :: Float64,
    v3   :: NTuple{6, Float64},
    v4   :: NTuple{6, Float64},
    r    :: Reduced_lattice,
    temp :: Array{Float64, 3}
    )    :: Nothing 

    v3xx, v3zz, v3DM, v3dd, v3zd, v3dz = v3
    v4xx, v4zz, v4DM, v4dd, v4zd, v4dz = v4

    @inbounds @fastmath for i in eachindex(r.sites)
        v1zz = temp[i, 2, 1]; v1zd = temp[i, 5, 1]
        v2zz = temp[i, 2, 2]; v2dz = temp[i, 6, 2]
        val  = v1zz * v4dd + 2.0 * v1zd * v4DM - v1zd * v4zd + v1zz * v4zz - 2.0 * v1zz * v4xx - v1zd * v4dz - 2.0 * v3DM * v2dz - v3dz * v2dz + v3dd * v2zz - 2.0 * v3xx * v2zz + v3zz * v2zz - v3zd * v2dz

        overlap_i = r.overlap[i]
        Range     = size(overlap_i, 1)

        @turbo for j in 1 : Range
            v1zz  = temp[overlap_i[j, 1], 2, 1]; v1zd = temp[overlap_i[j, 1], 5, 1]
            v2zz  = temp[overlap_i[j, 2], 2, 2]; v2dz = temp[overlap_i[j, 2], 6, 2]
            val  += -2.0 * overlap_i[j, 3] * (v1zz * v2zz - v1zd * v2dz)
        end

        buff[i] += -p * val * dv
    end

    return nothing 
end

# DM kernel for the t channel
function compute_t_kernel_DM!(
    buff :: Vector{Float64},
    p    :: Float64,
    dv   :: Float64,
    v3   :: NTuple{6, Float64},
    v4   :: NTuple{6, Float64},
    r    :: Reduced_lattice,
    temp :: Array{Float64, 3}
    )    :: Nothing 

    v3xx, v3zz, v3DM, v3dd, v3zd, v3dz = v3
    v4xx, v4zz, v4DM, v4dd, v4zd, v4dz = v4

    @inbounds @fastmath for i in eachindex(r.sites)
        v1xx = temp[i, 1, 1]; v1DM = temp[i, 3, 1]
        v2xx = temp[i, 1, 2]; v2DM = temp[i, 3, 2]
        val  = v1DM * v4dd - v1xx * v4dz + v1xx * v4zd - v1DM * v4zz - v3dz * v2xx + v3dd * v2DM + v3zd * v2xx - v3zz * v2DM

        overlap_i = r.overlap[i]
        Range     = size(overlap_i, 1)

        @turbo for j in 1 : Range
            v1xx  = temp[overlap_i[j, 1], 1, 1]; v1DM = temp[overlap_i[j, 1], 3, 1]
            v2xx  = temp[overlap_i[j, 2], 1, 2]; v2DM = temp[overlap_i[j, 2], 3, 2]
            val  += -2.0 * overlap_i[j, 3] * (v1DM * v2xx + v1xx * v2DM)
        end

        buff[i] += -p * val * dv
    end

    return nothing 
end

# dd kernel for the t channel
function compute_t_kernel_dd!(
    buff :: Vector{Float64},
    p    :: Float64,
    dv   :: Float64,
    v3   :: NTuple{6, Float64},
    v4   :: NTuple{6, Float64},
    r    :: Reduced_lattice,
    temp :: Array{Float64, 3}
    )    :: Nothing 

    v3xx, v3zz, v3DM, v3dd, v3zd, v3dz = v3
    v4xx, v4zz, v4DM, v4dd, v4zd, v4dz = v4

    @inbounds @fastmath for i in eachindex(r.sites)
        v1dd = temp[i, 4, 1]; v1dz = temp[i, 6, 1]
        v2dd = temp[i, 4, 2]; v2zd = temp[i, 5, 2]
        val  = -v1dz * v4dz + 2.0 * v1dd * v4xx + v1dd * v4zz - v1dz * v4zd - 2.0 * v1dz * v4DM + v1dd * v4dd + 2.0 * v3DM * v2zd + 2.0 * v3xx * v2dd + v3zz * v2dd - v3dz * v2zd + v3dd * v2dd - v3zd * v2zd

        overlap_i = r.overlap[i]
        Range     = size(overlap_i, 1)

        @turbo for j in 1 : Range
            v1dd  = temp[overlap_i[j, 1], 4, 1]; v1dz = temp[overlap_i[j, 1], 6, 1]
            v2dd  = temp[overlap_i[j, 2], 4, 2]; v2zd = temp[overlap_i[j, 2], 5, 2]
            val  += -2.0 * overlap_i[j, 3] * (v1dd * v2dd - v1dz * v2zd)
        end

        buff[i] += -p * val * dv
    end

    return nothing 
end

# zd kernel for the t channel
function compute_t_kernel_zd!(
    buff :: Vector{Float64},
    p    :: Float64,
    dv   :: Float64,
    v3   :: NTuple{6, Float64},
    v4   :: NTuple{6, Float64},
    r    :: Reduced_lattice,
    temp :: Array{Float64, 3}
    )    :: Nothing 

    v3xx, v3zz, v3DM, v3dd, v3zd, v3dz = v3
    v4xx, v4zz, v4DM, v4dd, v4zd, v4dz = v4

    @inbounds @fastmath for i in eachindex(r.sites)
        v1zz = temp[i, 2, 1]; v1zd = temp[i, 5, 1]
        v2dd = temp[i, 4, 2]; v2zd = temp[i, 5, 2]
        val  = v1zd * v4zz + v1zd * v4dd + 2.0 * v1zd * v4xx + v1zz * v4zd + 2.0 * v1zz * v4DM + v1zz * v4dz + v3zd * v2dd + 2.0 * v3DM * v2dd + v3dd * v2zd + v3zz * v2zd - 2.0 * v3xx * v2zd + v3dz * v2dd

        overlap_i = r.overlap[i]
        Range     = size(overlap_i, 1)

        @turbo for j in 1 : Range
            v1zz  = temp[overlap_i[j, 1], 2, 1]; v1zd = temp[overlap_i[j, 1], 5, 1]
            v2dd  = temp[overlap_i[j, 2], 4, 2]; v2zd = temp[overlap_i[j, 2], 5, 2]
            val  += -2.0 * overlap_i[j, 3] * (v1zd * v2dd + v1zz * v2zd)
        end

        buff[i] += -p * val * dv
    end

    return nothing 
end

# dz kernel for the t channel
function compute_t_kernel_dz!(
    buff :: Vector{Float64},
    p    :: Float64,
    dv   :: Float64,
    v3   :: NTuple{6, Float64},
    v4   :: NTuple{6, Float64},
    r    :: Reduced_lattice,
    temp :: Array{Float64, 3}
    )    :: Nothing 

    v3xx, v3zz, v3DM, v3dd, v3zd, v3dz = v3
    v4xx, v4zz, v4DM, v4dd, v4zd, v4dz = v4

    @inbounds @fastmath for i in eachindex(r.sites)
        v1dd = temp[i, 4, 1]; v1dz = temp[i, 6, 1]
        v2zz = temp[i, 2, 2]; v2dz = temp[i, 6, 2]
        val  = -2.0 * v1dd * v4DM + v1dd * v4zd + v1dd * v4dz + v1dz * v4zz + v1dz * v4dd - 2.0 * v1dz * v4xx + v3zd * v2zz + 2.0 * v3xx * v2dz + v3dd * v2dz + v3dz * v2zz - 2.0 * v3DM * v2zz + v3zz * v2dz

        overlap_i = r.overlap[i]
        Range     = size(overlap_i, 1)

        @turbo for j in 1 : Range
            v1dd  = temp[overlap_i[j, 1], 4, 1]; v1dz = temp[overlap_i[j, 1], 6, 1]
            v2zz  = temp[overlap_i[j, 2], 2, 2]; v2dz = temp[overlap_i[j, 2], 6, 2]
            val  += -2.0 * overlap_i[j, 3] * (v1dd * v2dz + v1dz * v2zz)
        end

        buff[i] += -p * val * dv
    end

    return nothing 
end