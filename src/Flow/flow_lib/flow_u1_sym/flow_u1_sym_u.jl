# Katanin kernel
function compute_u_kat!(
    Λ    :: Float64,
    comp :: Int64,
    buff :: Vector{Float64},
    v    :: Float64,
    dv   :: Float64,
    u    :: Float64,
    vu   :: Float64,
    vup  :: Float64,
    r    :: Reduced_lattice,
    m    :: Mesh,
    a    :: Action_u1_sym,
    da   :: Action_u1_sym,
    temp :: Array{Float64, 3}
    )    :: Nothing

    # get propagator
    p = get_propagator_kat(Λ, v - 0.5 * u, v + 0.5 * u, m, a, da) + get_propagator_kat(Λ, v + 0.5 * u, v - 0.5 * u, m, a, da)

    # get buffers for left vertex
    bs1 = ntuple(comp -> get_u1_sym_buffer_s(comp, v + vu, 0.5 * (u - v + vu), 0.5 * (-u - v + vu), m), 6)
    bt1 = ntuple(comp -> get_u1_sym_buffer_t(comp, v - vu, 0.5 * (u + v + vu), 0.5 * (-u + v + vu), m), 6)
    bu1 = ntuple(comp -> get_u1_sym_buffer_u(comp, u, vu, v, m), 6)

    # get buffers for right vertex
    bs2 = ntuple(comp -> get_u1_sym_buffer_s(comp, v + vup, 0.5 * (u + v - vup), 0.5 * (-u + v - vup), m), 6)
    bt2 = ntuple(comp -> get_u1_sym_buffer_t(comp, -v + vup, 0.5 * (u + v + vup), 0.5 * (-u + v + vup), m), 6)
    bu2 = ntuple(comp -> get_u1_sym_buffer_u(comp, u, v, vup, m), 6)

    # cache vertex values for all lattice sites in temporary buffer
    get_Γ_avx!(r, bs1, bt1, bu1, a, temp, 1)
    get_Γ_avx!(r, bs2, bt2, bu2, a, temp, 2)

    # compute xx contributions for all lattice sites 
    if comp == 1
        @turbo unroll = 1 for i in eachindex(r.sites)
            # read cached values for site i
            v1xx = temp[i, 1, 1]
            v1zz = temp[i, 2, 1]
            v1DM = temp[i, 3, 1]
            v1dd = temp[i, 4, 1]
            v1zd = temp[i, 5, 1]
            v1dz = temp[i, 6, 1]
    
            v2xx = temp[i, 1, 2]
            v2zz = temp[i, 2, 2]
            v2DM = temp[i, 3, 2]
            v2dd = temp[i, 4, 2]
            v2zd = temp[i, 5, 2]
            v2dz = temp[i, 6, 2]       

            # compute contribution at site i
            buff[i] += -p * (-v1DM * v2dz - v1DM * v2zd + v1xx * v2dd + v1dd * v2xx + v1xx * v2zz + v1dz * v2DM + v1zd * v2DM + v1zz * v2xx) * dv  
        end
    # compute zz contributions for all lattice sites
    elseif comp == 2
        @turbo unroll = 1 for i in eachindex(r.sites)
            # read cached values for site i
            v1xx = temp[i, 1, 1]
            v1zz = temp[i, 2, 1]
            v1DM = temp[i, 3, 1]
            v1dd = temp[i, 4, 1]
            v1zd = temp[i, 5, 1]
            v1dz = temp[i, 6, 1]
    
            v2xx = temp[i, 1, 2]
            v2zz = temp[i, 2, 2]
            v2DM = temp[i, 3, 2]
            v2dd = temp[i, 4, 2]
            v2zd = temp[i, 5, 2]
            v2dz = temp[i, 6, 2]       

            # compute contribution at site i
            buff[i] += -p * (v1zz * v2dd + v1dd * v2zz - v1dz * v2zd + 2.0 * v1xx * v2xx + 2.0 * v1DM * v2DM - v1zd * v2dz) * dv  
        end
    # compute DM contributions for all lattice sites
    elseif comp == 3
        @turbo unroll = 1 for i in eachindex(r.sites)
            # read cached values for site i
            v1xx = temp[i, 1, 1]
            v1zz = temp[i, 2, 1]
            v1DM = temp[i, 3, 1]
            v1dd = temp[i, 4, 1]
            v1zd = temp[i, 5, 1]
            v1dz = temp[i, 6, 1]
    
            v2xx = temp[i, 1, 2]
            v2zz = temp[i, 2, 2]
            v2DM = temp[i, 3, 2]
            v2dd = temp[i, 4, 2]
            v2zd = temp[i, 5, 2]
            v2dz = temp[i, 6, 2]       

            # compute contribution at site i
            buff[i] += -p * (v1dd * v2DM + v1DM * v2dd - v1zd * v2xx + v1xx * v2dz + v1zz * v2DM + v1xx * v2zd + v1DM * v2zz - v1dz * v2xx) * dv  
        end
    # compute dd contributions for all lattice sites
    elseif comp == 4
        @turbo unroll = 1 for i in eachindex(r.sites)
            # read cached values for site i
            v1xx = temp[i, 1, 1]
            v1zz = temp[i, 2, 1]
            v1DM = temp[i, 3, 1]
            v1dd = temp[i, 4, 1]
            v1zd = temp[i, 5, 1]
            v1dz = temp[i, 6, 1]
    
            v2xx = temp[i, 1, 2]
            v2zz = temp[i, 2, 2]
            v2DM = temp[i, 3, 2]
            v2dd = temp[i, 4, 2]
            v2zd = temp[i, 5, 2]
            v2dz = temp[i, 6, 2]       

            # compute contribution at site i
            buff[i] += -p * (-v1dz * v2dz + 2.0 * v1xx * v2xx + v1dd * v2dd + 2.0 * v1DM * v2DM - v1zd * v2zd + v1zz * v2zz) * dv  
        end
    # compute zd contributions for all lattice sites 
    elseif comp == 5
        @turbo unroll = 1 for i in eachindex(r.sites)
            # read cached values for site i
            v1xx = temp[i, 1, 1]
            v1zz = temp[i, 2, 1]
            v1DM = temp[i, 3, 1]
            v1dd = temp[i, 4, 1]
            v1zd = temp[i, 5, 1]
            v1dz = temp[i, 6, 1]
    
            v2xx = temp[i, 1, 2]
            v2zz = temp[i, 2, 2]
            v2DM = temp[i, 3, 2]
            v2dd = temp[i, 4, 2]
            v2zd = temp[i, 5, 2]
            v2dz = temp[i, 6, 2]       

            # compute contribution at site i
            buff[i] += -p * (v1zd * v2dd + v1dd * v2zd - 2.0 * v1DM * v2xx + v1dz * v2zz + v1zz * v2dz + 2.0 * v1xx * v2DM) * dv  
        end
    # compute dz contributions for all lattice sites 
    else
        @turbo unroll = 1 for i in eachindex(r.sites)
            # read cached values for site i
            v1xx = temp[i, 1, 1]
            v1zz = temp[i, 2, 1]
            v1DM = temp[i, 3, 1]
            v1dd = temp[i, 4, 1]
            v1zd = temp[i, 5, 1]
            v1dz = temp[i, 6, 1]
    
            v2xx = temp[i, 1, 2]
            v2zz = temp[i, 2, 2]
            v2DM = temp[i, 3, 2]
            v2dd = temp[i, 4, 2]
            v2zd = temp[i, 5, 2]
            v2dz = temp[i, 6, 2]       

            # compute contribution at site i
            buff[i] += -p * (v1zd * v2zz + v1dd * v2dz + v1zz * v2zd - 2.0 * v1DM * v2xx + v1dz * v2dd + 2.0 * v1xx * v2DM) * dv  
        end
    end

    return nothing
end