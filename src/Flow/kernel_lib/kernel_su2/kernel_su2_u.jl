# spin kernel for the u channel
function compute_u_kernel_spin!(
    buff :: Vector{Float64},
    p    :: Float64,
    dv   :: Float64,
    r    :: Reduced_lattice,
    temp :: Array{Float64, 3}
    )    :: Nothing 

    @turbo for i in eachindex(r.sites)
        v1s = temp[i, 1, 1]; v1d = temp[i, 2, 1]
        v2s = temp[i, 1, 2]; v2d = temp[i, 2, 2]

        buff[i] += -p * (2.0 * v1s * v2s + v1s * v2d + v1d * v2s) * dv
    end

    return nothing 
end

# density kernel for the u channel
function compute_u_kernel_dens!(
    buff :: Vector{Float64},
    p    :: Float64,
    dv   :: Float64,
    r    :: Reduced_lattice,
    temp :: Array{Float64, 3}
    )    :: Nothing 

    @turbo for i in eachindex(r.sites)
        v1s = temp[i, 1, 1]; v1d = temp[i, 2, 1]
        v2s = temp[i, 1, 2]; v2d = temp[i, 2, 2]

        buff[i] += -p * (3.0 * v1s * v2s + v1d * v2d) * dv
    end

    return nothing 
end