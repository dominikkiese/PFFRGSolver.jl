# xx kernel for the s channel
function compute_s_kernel_xx!(
    buff :: Vector{Float64},
    p    :: Float64,
    dv   :: Float64,
    r    :: Reduced_lattice,
    temp :: Array{Float64, 3}
    )    :: Nothing 

    @turbo unroll = 1 for i in eachindex(r.sites)
        v1xx = temp[i, 1, 1]; v1zz = temp[i, 2, 1]; v1DM = temp[i, 3, 1]
        v1dd = temp[i, 4, 1]; v1zd = temp[i, 5, 1]; v1dz = temp[i, 6, 1]
        v2xx = temp[i, 1, 2]; v2zz = temp[i, 2, 2]; v2DM = temp[i, 3, 2]
        v2dd = temp[i, 4, 2]; v2zd = temp[i, 5, 2]; v2dz = temp[i, 6, 2]

        buff[i] += -p * (v1DM * v2dz - v1DM * v2zd + v1xx * v2dd + v1dd * v2xx - v1xx * v2zz - v1dz * v2DM + v1zd * v2DM - v1zz * v2xx) * dv    
    end

    return nothing 
end

# zz kernel for the s channel
function compute_s_kernel_zz!(
    buff :: Vector{Float64},
    p    :: Float64,
    dv   :: Float64,
    r    :: Reduced_lattice,
    temp :: Array{Float64, 3}
    )    :: Nothing 

    @turbo unroll = 1 for i in eachindex(r.sites)
        v1xx = temp[i, 1, 1]; v1zz = temp[i, 2, 1]; v1DM = temp[i, 3, 1]
        v1dd = temp[i, 4, 1]; v1zd = temp[i, 5, 1]; v1dz = temp[i, 6, 1]
        v2xx = temp[i, 1, 2]; v2zz = temp[i, 2, 2]; v2DM = temp[i, 3, 2]
        v2dd = temp[i, 4, 2]; v2zd = temp[i, 5, 2]; v2dz = temp[i, 6, 2]

        buff[i] += -p * (v1zz * v2dd + v1dd * v2zz - v1dz * v2zd - 2.0 * v1xx * v2xx - 2.0 * v1DM * v2DM - v1zd * v2dz) * dv    
    end

    return nothing 
end

# DM kernel for the s channel
function compute_s_kernel_DM!(
    buff :: Vector{Float64},
    p    :: Float64,
    dv   :: Float64,
    r    :: Reduced_lattice,
    temp :: Array{Float64, 3}
    )    :: Nothing 

    @turbo unroll = 1 for i in eachindex(r.sites)
        v1xx = temp[i, 1, 1]; v1zz = temp[i, 2, 1]; v1DM = temp[i, 3, 1]
        v1dd = temp[i, 4, 1]; v1zd = temp[i, 5, 1]; v1dz = temp[i, 6, 1]
        v2xx = temp[i, 1, 2]; v2zz = temp[i, 2, 2]; v2DM = temp[i, 3, 2]
        v2dd = temp[i, 4, 2]; v2zd = temp[i, 5, 2]; v2dz = temp[i, 6, 2]

        buff[i] += -p * (v1dd * v2DM + v1DM * v2dd - v1zd * v2xx - v1xx * v2dz - v1zz * v2DM + v1xx * v2zd - v1DM * v2zz + v1dz * v2xx) * dv    
    end

    return nothing 
end

# dd kernel for the s channel
function compute_s_kernel_dd!(
    buff :: Vector{Float64},
    p    :: Float64,
    dv   :: Float64,
    r    :: Reduced_lattice,
    temp :: Array{Float64, 3}
    )    :: Nothing 

    @turbo unroll = 1 for i in eachindex(r.sites)
        v1xx = temp[i, 1, 1]; v1zz = temp[i, 2, 1]; v1DM = temp[i, 3, 1]
        v1dd = temp[i, 4, 1]; v1zd = temp[i, 5, 1]; v1dz = temp[i, 6, 1]
        v2xx = temp[i, 1, 2]; v2zz = temp[i, 2, 2]; v2DM = temp[i, 3, 2]
        v2dd = temp[i, 4, 2]; v2zd = temp[i, 5, 2]; v2dz = temp[i, 6, 2]

        buff[i] += -p * (-v1dz * v2dz + 2.0 * v1xx * v2xx + v1dd * v2dd + 2.0 * v1DM * v2DM - v1zd * v2zd + v1zz * v2zz) * dv    
    end

    return nothing 
end

# zd kernel for the s channel
function compute_s_kernel_zd!(
    buff :: Vector{Float64},
    p    :: Float64,
    dv   :: Float64,
    r    :: Reduced_lattice,
    temp :: Array{Float64, 3}
    )    :: Nothing 

    @turbo unroll = 1 for i in eachindex(r.sites)
        v1xx = temp[i, 1, 1]; v1zz = temp[i, 2, 1]; v1DM = temp[i, 3, 1]
        v1dd = temp[i, 4, 1]; v1zd = temp[i, 5, 1]; v1dz = temp[i, 6, 1]
        v2xx = temp[i, 1, 2]; v2zz = temp[i, 2, 2]; v2DM = temp[i, 3, 2]
        v2dd = temp[i, 4, 2]; v2zd = temp[i, 5, 2]; v2dz = temp[i, 6, 2]

        buff[i] += -p * (v1zd * v2dd + v1dd * v2zd - 2.0 * v1DM * v2xx + v1dz * v2zz + v1zz * v2dz + 2.0 * v1xx * v2DM) * dv
    end

    return nothing 
end

# dz kernel for the s channel
function compute_s_kernel_dz!(
    buff :: Vector{Float64},
    p    :: Float64,
    dv   :: Float64,
    r    :: Reduced_lattice,
    temp :: Array{Float64, 3}
    )    :: Nothing 

    @turbo unroll = 1 for i in eachindex(r.sites)
        v1xx = temp[i, 1, 1]; v1zz = temp[i, 2, 1]; v1DM = temp[i, 3, 1]
        v1dd = temp[i, 4, 1]; v1zd = temp[i, 5, 1]; v1dz = temp[i, 6, 1]
        v2xx = temp[i, 1, 2]; v2zz = temp[i, 2, 2]; v2DM = temp[i, 3, 2]
        v2dd = temp[i, 4, 2]; v2zd = temp[i, 5, 2]; v2dz = temp[i, 6, 2]

        buff[i] += -p * (v1zd * v2zz + v1dd * v2dz + v1zz * v2zd + 2.0 * v1DM * v2xx + v1dz * v2dd - 2.0 * v1xx * v2DM) * dv
    end

    return nothing 
end