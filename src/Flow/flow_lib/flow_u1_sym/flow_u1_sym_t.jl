# Katanin kernel (chalice)
function compute_t_chalice_kat!(
    Λ    :: Float64,
    comp :: Int64,
    buff :: Vector{Float64},
    v    :: Float64,
    dv   :: Float64,
    t    :: Float64,
    vt   :: Float64,
    vtp  :: Float64,
    r    :: Reduced_lattice,
    m    :: Mesh,
    a    :: Action_u1_sym,
    da   :: Action_u1_sym,
    temp :: Array{Float64, 3}
    )    :: Nothing

    # get propagator
    p = get_propagator_kat(Λ, v + 0.5 * t, v - 0.5 * t, m, a, da) + get_propagator_kat(Λ, v - 0.5 * t, v + 0.5 * t, m, a, da)

    # get buffers for left non-local vertex
    bs1 = ntuple(comp -> get_u1_sym_buffer_s(comp, v + vt, 0.5 * (-t - v + vt), 0.5 * (-t + v - vt), m), 6)
    bt1 = ntuple(comp -> get_u1_sym_buffer_t(comp, t, vt, v, m), 6)
    bu1 = ntuple(comp -> get_u1_sym_buffer_u(comp, -v + vt, 0.5 * (-t + v + vt), 0.5 * (t + v + vt), m), 6)

    # get buffers for right non-local vertex
    bs2 = ntuple(comp -> get_u1_sym_buffer_s(comp, v + vtp, 0.5 * (-t + v - vtp), 0.5 * (-t - v + vtp), m), 6)
    bt2 = ntuple(comp -> get_u1_sym_buffer_t(comp, t, v, vtp, m), 6)
    bu2 = ntuple(comp -> get_u1_sym_buffer_u(comp, v - vtp, 0.5 * (-t + v + vtp), 0.5 * (t + v + vtp), m), 6)

    # get buffers for local left vertex
    bs3 = ntuple(comp -> get_u1_sym_buffer_s(comp, v + vt, 0.5 * (-t - v + vt), 0.5 * (t - v + vt), m), 6)
    bt3 = ntuple(comp -> get_u1_sym_buffer_t(comp, v - vt, 0.5 * (-t + v + vt), 0.5 * (t + v + vt), m), 6)
    bu3 = ntuple(comp -> get_u1_sym_buffer_u(comp, -t, vt, v, m), 6)

    # get buffers for local right vertex
    bs4 = ntuple(comp -> get_u1_sym_buffer_s(comp, v + vtp, 0.5 * (-t + v - vtp), 0.5 * (t + v - vtp), m), 6)
    bt4 = ntuple(comp -> get_u1_sym_buffer_t(comp, -v + vtp, 0.5 * (-t + v + vtp), 0.5 * (t + v + vtp), m), 6)
    bu4 = ntuple(comp -> get_u1_sym_buffer_u(comp, -t, v, vtp, m), 6)

    # cache local vertex values
    v3xx, v3zz, v3DM, v3dd, v3zd, v3dz = get_Γ(1, bs3, bt3, bu3, r, a)
    v4xx, v4zz, v4DM, v4dd, v4zd, v4dz = get_Γ(1, bs4, bt4, bu4, r, a)

    # compute xx contributions for all lattice sites 
    if comp == 1 
        # cache vertex values for all lattice sites in temporary buffer
        get_Γ_avx!(r, bs1, bt1, bu1, a, temp, 1, comps = 1 : 1)
        get_Γ_avx!(r, bs1, bt1, bu1, a, temp, 1, comps = 3 : 3)

        get_Γ_avx!(r, bs2, bt2, bu2, a, temp, 2, comps = 1 : 1)
        get_Γ_avx!(r, bs2, bt2, bu2, a, temp, 2, comps = 3 : 3)

        @turbo unroll = 1 for i in eachindex(r.sites)
            # read cached values for site i
            v1xx = temp[i, 1, 1]
            v1DM = temp[i, 3, 1]

            v2xx = temp[i, 1, 2]
            v2DM = temp[i, 3, 2]
            
            # compute contribution at site i
            buff[i] += -p * (v1DM * v4dz - v1DM * v4zd + v1xx * v4dd - v1xx * v4zz + 
                             v3dd * v2xx + v3dz * v2DM - v3zd * v2DM - v3zz * v2xx) * dv
        end 
    # compute zz contributions for all lattice sites
    elseif comp == 2
        # cache vertex values for all lattice sites in temporary buffer
        get_Γ_avx!(r, bs1, bt1, bu1, a, temp, 1, comps = 2 : 2)
        get_Γ_avx!(r, bs1, bt1, bu1, a, temp, 1, comps = 5 : 5)

        get_Γ_avx!(r, bs2, bt2, bu2, a, temp, 2, comps = 2 : 2)
        get_Γ_avx!(r, bs2, bt2, bu2, a, temp, 2, comps = 6 : 6)

        @turbo unroll = 1 for i in eachindex(r.sites)
            # read cached values for site i
            v1zz = temp[i, 2, 1]
            v1zd = temp[i, 5, 1]

            v2zz = temp[i, 2, 2]
            v2dz = temp[i, 6, 2]
            
            # compute contribution at site i
            buff[i] += -p * (v1zz * v4dd + 2.0 * v1zd * v4DM - v1zd * v4zd + v1zz * v4zz - 2.0 * v1zz * v4xx - v1zd * v4dz -
                             2.0 * v3DM * v2dz - v3dz * v2dz + v3dd * v2zz - 2.0 * v3xx * v2zz + v3zz * v2zz - v3zd * v2dz) * dv
        end 
    # compute DM contributions for all lattice sites
    elseif comp == 3
        # cache vertex values for all lattice sites in temporary buffer
        get_Γ_avx!(r, bs1, bt1, bu1, a, temp, 1, comps = 1 : 1)
        get_Γ_avx!(r, bs1, bt1, bu1, a, temp, 1, comps = 3 : 3)

        get_Γ_avx!(r, bs2, bt2, bu2, a, temp, 2, comps = 1 : 1)
        get_Γ_avx!(r, bs2, bt2, bu2, a, temp, 2, comps = 3 : 3)

        @turbo unroll = 1 for i in eachindex(r.sites)
            # read cached values for site i
            v1xx = temp[i, 1, 1]
            v1DM = temp[i, 3, 1]
            
            v2xx = temp[i, 1, 2]
            v2DM = temp[i, 3, 2]        
            
            # compute contribution at site i
            buff[i] += -p * (v1DM * v4dd - v1xx * v4dz + v1xx * v4zd - v1DM * v4zz -
                             v3dz * v2xx + v3dd * v2DM + v3zd * v2xx - v3zz * v2DM) * dv
        end
    # compute dd contributions for all lattice sites
    elseif comp == 4
        # cache vertex values for all lattice sites in temporary buffer
        get_Γ_avx!(r, bs1, bt1, bu1, a, temp, 1, comps = 4 : 4)
        get_Γ_avx!(r, bs1, bt1, bu1, a, temp, 1, comps = 6 : 6)

        get_Γ_avx!(r, bs2, bt2, bu2, a, temp, 2, comps = 4 : 4)
        get_Γ_avx!(r, bs2, bt2, bu2, a, temp, 2, comps = 5 : 5)

        @turbo unroll = 1 for i in eachindex(r.sites)
            # read cached values for site i
            v1dd = temp[i, 4, 1]
            v1dz = temp[i, 6, 1]
    
            v2dd = temp[i, 4, 2]
            v2zd = temp[i, 5, 2]

            # compute contribution at site i
            buff[i] += -p * (-v1dz * v4dz + 2.0 * v1dd * v4xx + v1dd * v4zz - v1dz * v4zd - 2.0 * v1dz * v4DM + v1dd * v4dd +
                             2.0 * v3DM * v2zd + 2.0 * v3xx * v2dd + v3zz * v2dd - v3dz * v2zd + v3dd * v2dd - v3zd * v2zd) * dv
        end
    # compute zd contributions for all lattice sites 
    elseif comp == 5
        # cache vertex values for all lattice sites in temporary buffer
        get_Γ_avx!(r, bs1, bt1, bu1, a, temp, 1, comps = 2 : 2)
        get_Γ_avx!(r, bs1, bt1, bu1, a, temp, 1, comps = 5 : 5)

        get_Γ_avx!(r, bs2, bt2, bu2, a, temp, 2, comps = 4 : 4)
        get_Γ_avx!(r, bs2, bt2, bu2, a, temp, 2, comps = 5 : 5)

        @turbo unroll = 1 for i in eachindex(r.sites)
            # read cached values for site i
            v1zz = temp[i, 2, 1]
            v1zd = temp[i, 5, 1]
        
            v2dd = temp[i, 4, 2]
            v2zd = temp[i, 5, 2]
            
            # compute contribution at site i
            buff[i] += -p * (v1zd * v4zz + v1zd * v4dd + 2.0 * v1zd * v4xx + v1zz * v4zd + 2.0 * v1zz * v4DM + v1zz * v4dz +
                             v3zd * v2dd + 2.0 * v3DM * v2dd + v3dd * v2zd + v3zz * v2zd - 2.0 * v3xx * v2zd + v3dz * v2dd) * dv
        end
    # compute dz contributions for all lattice sites 
    else
        # cache vertex values for all lattice sites in temporary buffer
        get_Γ_avx!(r, bs1, bt1, bu1, a, temp, 1, comps = 4 : 4)
        get_Γ_avx!(r, bs1, bt1, bu1, a, temp, 1, comps = 6 : 6)

        get_Γ_avx!(r, bs2, bt2, bu2, a, temp, 2, comps = 2 : 2)
        get_Γ_avx!(r, bs2, bt2, bu2, a, temp, 2, comps = 6 : 6)

        @turbo unroll = 1 for i in eachindex(r.sites)
            # read cached values for site i
            v1dd = temp[i, 4, 1]
            v1dz = temp[i, 6, 1]
    
            v2zz = temp[i, 2, 2]
            v2dz = temp[i, 6, 2]
            
            # compute contribution at site i
            buff[i] += -p * (-2.0 * v1dd * v4DM + v1dd * v4zd + v1dd * v4dz + v1dz * v4zz + v1dz * v4dd - 2.0 * v1dz * v4xx +
                             v3zd * v2zz + 2.0 * v3xx * v2dz + v3dd * v2dz + v3dz * v2zz - 2.0 * v3DM * v2zz + v3zz * v2dz) * dv
        end
    end

    return nothing
end

# Katanin kernel (RPA)
function compute_t_RPA_kat!(
    Λ    :: Float64,
    comp :: Int64,
    buff :: Vector{Float64},
    v    :: Float64,
    dv   :: Float64,
    t    :: Float64,
    vt   :: Float64,
    vtp  :: Float64,
    r    :: Reduced_lattice,
    m    :: Mesh,
    a    :: Action_u1_sym,
    da   :: Action_u1_sym,
    temp :: Array{Float64, 3}
    )    :: Nothing

    # get propagator and overlap
    p       = get_propagator_kat(Λ, v + 0.5 * t, v - 0.5 * t, m, a, da) + get_propagator_kat(Λ, v - 0.5 * t, v + 0.5 * t, m, a, da)
    overlap = r.overlap

    # get buffers for left non-local vertex
    bs1 = ntuple(comp -> get_u1_sym_buffer_s(comp, v + vt, 0.5 * (-t - v + vt), 0.5 * (-t + v - vt), m), 6)
    bt1 = ntuple(comp -> get_u1_sym_buffer_t(comp, t, vt, v, m), 6)
    bu1 = ntuple(comp -> get_u1_sym_buffer_u(comp, -v + vt, 0.5 * (-t + v + vt), 0.5 * (t + v + vt), m), 6)

    # get buffers for right non-local vertex
    bs2 = ntuple(comp -> get_u1_sym_buffer_s(comp, v + vtp, 0.5 * (-t + v - vtp), 0.5 * (-t - v + vtp), m), 6)
    bt2 = ntuple(comp -> get_u1_sym_buffer_t(comp, t, v, vtp, m), 6)
    bu2 = ntuple(comp -> get_u1_sym_buffer_u(comp, v - vtp, 0.5 * (-t + v + vtp), 0.5 * (t + v + vtp), m), 6)

    # compute xx contributions for all lattice sites 
    if comp == 1 
        # cache vertex values for all lattice sites in temporary buffer
        get_Γ_avx!(r, bs1, bt1, bu1, a, temp, 1, comps = 1 : 1)
        get_Γ_avx!(r, bs1, bt1, bu1, a, temp, 1, comps = 3 : 3)

        get_Γ_avx!(r, bs2, bt2, bu2, a, temp, 2, comps = 1 : 1)
        get_Γ_avx!(r, bs2, bt2, bu2, a, temp, 2, comps = 3 : 3)

        for i in eachindex(r.sites)
            # determine overlap for site i
            overlap_i = overlap[i]

            # determine range for inner sum
            Range = size(overlap_i, 1)

            # initialize spin contribution
            Γxx = 0.0

            # compute inner sum
            @turbo unroll = 1 for j in 1 : Range
                # read cached values for inner site
                v1xx = temp[overlap_i[j, 1], 1, 1]
                v1DM = temp[overlap_i[j, 1], 3, 1]
        
                v2xx = temp[overlap_i[j, 2], 1, 2]
                v2DM = temp[overlap_i[j, 2], 3, 2]

                # compute contribution at inner site
                Γxx += overlap_i[j, 3] * (v1xx * v2xx - v1DM * v2DM)
            end 

            # parse result 
            buff[i] += -p * (-2.0) * Γxx * dv
        end 
    # compute zz contributions for all lattice sites
    elseif comp == 2
        # cache vertex values for all lattice sites in temporary buffer
        get_Γ_avx!(r, bs1, bt1, bu1, a, temp, 1, comps = 2 : 2)
        get_Γ_avx!(r, bs1, bt1, bu1, a, temp, 1, comps = 5 : 5)

        get_Γ_avx!(r, bs2, bt2, bu2, a, temp, 2, comps = 2 : 2)
        get_Γ_avx!(r, bs2, bt2, bu2, a, temp, 2, comps = 6 : 6)

        for i in eachindex(r.sites)
            # determine overlap for site i
            overlap_i = overlap[i]

            # determine range for inner sum
            Range = size(overlap_i, 1)

            # initialize spin contribution
            Γzz = 0.0

            # compute inner sum
            @turbo unroll = 1 for j in 1 : Range
                # read cached values for inner site
                v1zz = temp[overlap_i[j, 1], 2, 1]
                v1zd = temp[overlap_i[j, 1], 5, 1]
        
                v2zz = temp[overlap_i[j, 2], 2, 2]
                v2dz = temp[overlap_i[j, 2], 6, 2]

                # compute contribution at inner site
                Γzz += overlap_i[j, 3] * (v1zz * v2zz - v1zd * v2dz)
            end 

            # parse result 
            buff[i] += -p * (-2.0) * Γzz * dv
        end 
    # compute DM contributions for all lattice sites
    elseif comp == 3
        # cache vertex values for all lattice sites in temporary buffer
        get_Γ_avx!(r, bs1, bt1, bu1, a, temp, 1, comps = 1 : 1)
        get_Γ_avx!(r, bs1, bt1, bu1, a, temp, 1, comps = 3 : 3)

        get_Γ_avx!(r, bs2, bt2, bu2, a, temp, 2, comps = 1 : 1)
        get_Γ_avx!(r, bs2, bt2, bu2, a, temp, 2, comps = 3 : 3)

        for i in eachindex(r.sites)
            # determine overlap for site i
            overlap_i = overlap[i]

            # determine range for inner sum
            Range = size(overlap_i, 1)

            # initialize spin contribution
            ΓDM = 0.0

            # compute inner sum
            @turbo unroll = 1 for j in 1 : Range
                # read cached values for inner site
                v1xx = temp[overlap_i[j, 1], 1, 1]
                v1DM = temp[overlap_i[j, 1], 3, 1]
        
                v2xx = temp[overlap_i[j, 2], 1, 2]
                v2DM = temp[overlap_i[j, 2], 3, 2]

                # compute contribution at inner site
                ΓDM += overlap_i[j, 3] * (v1DM * v2xx + v1xx * v2DM)
            end 

            # parse result 
            buff[i] += -p * (-2.0) * ΓDM * dv
        end 
    # compute dd contributions for all lattice sites
    elseif comp == 4
        # cache vertex values for all lattice sites in temporary buffer
        get_Γ_avx!(r, bs1, bt1, bu1, a, temp, 1, comps = 4 : 4)
        get_Γ_avx!(r, bs1, bt1, bu1, a, temp, 1, comps = 6 : 6)

        get_Γ_avx!(r, bs2, bt2, bu2, a, temp, 2, comps = 4 : 4)
        get_Γ_avx!(r, bs2, bt2, bu2, a, temp, 2, comps = 5 : 5)

        for i in eachindex(r.sites)
            # determine overlap for site i
            overlap_i = overlap[i]

            # determine range for inner sum
            Range = size(overlap_i, 1)

            # initialize spin contribution
            Γdd = 0.0

            # compute inner sum
            @turbo unroll = 1 for j in 1 : Range
                # read cached values for inner site
                v1dd = temp[overlap_i[j, 1], 4, 1]
                v1dz = temp[overlap_i[j, 1], 6, 1]

                v2dd = temp[overlap_i[j, 2], 4, 2]
                v2zd = temp[overlap_i[j, 2], 5, 2]

                # compute contribution at inner site
                Γdd += overlap_i[j, 3] * (v1dd * v2dd - v1dz * v2zd)
            end 

            # parse result 
            buff[i] += -p * (-2.0) * Γdd * dv
        end
    # compute zd contributions for all lattice sites 
    elseif comp == 5
        # cache vertex values for all lattice sites in temporary buffer
        get_Γ_avx!(r, bs1, bt1, bu1, a, temp, 1, comps = 2 : 2)
        get_Γ_avx!(r, bs1, bt1, bu1, a, temp, 1, comps = 5 : 5)

        get_Γ_avx!(r, bs2, bt2, bu2, a, temp, 2, comps = 4 : 4)
        get_Γ_avx!(r, bs2, bt2, bu2, a, temp, 2, comps = 5 : 5)

        for i in eachindex(r.sites)
            # determine overlap for site i
            overlap_i = overlap[i]

            # determine range for inner sum
            Range = size(overlap_i, 1)

            # initialize spin contribution
            Γzd = 0.0

            # compute inner sum
            @turbo unroll = 1 for j in 1 : Range
                # read cached values for inner site
                v1zz = temp[overlap_i[j, 1], 2, 1]
                v1zd = temp[overlap_i[j, 1], 5, 1]
        
                v2dd = temp[overlap_i[j, 2], 4, 2]
                v2zd = temp[overlap_i[j, 2], 5, 2]

                # compute contribution at inner site
                Γzd += overlap_i[j, 3] * (v1zd * v2dd + v1zz * v2zd)
            end 

            # parse result 
            buff[i] += -p * (-2.0) * Γzd * dv
        end
    # compute dz contributions for all lattice sites 
    else
        # cache vertex values for all lattice sites in temporary buffer
        get_Γ_avx!(r, bs1, bt1, bu1, a, temp, 1, comps = 4 : 4)
        get_Γ_avx!(r, bs1, bt1, bu1, a, temp, 1, comps = 6 : 6)

        get_Γ_avx!(r, bs2, bt2, bu2, a, temp, 2, comps = 2 : 2)
        get_Γ_avx!(r, bs2, bt2, bu2, a, temp, 2, comps = 6 : 6)

        for i in eachindex(r.sites)
            # determine overlap for site i
            overlap_i = overlap[i]

            # determine range for inner sum
            Range = size(overlap_i, 1)

            # initialize spin contribution
            Γdz = 0.0

            # compute inner sum
            @turbo unroll = 1 for j in 1 : Range
                # read cached values for inner site
                v1dd = temp[overlap_i[j, 1], 4, 1]
                v1dz = temp[overlap_i[j, 1], 6, 1]
        
                v2zz = temp[overlap_i[j, 2], 2, 2]
                v2dz = temp[overlap_i[j, 2], 6, 2]

                # compute contribution at inner site
                Γdz += overlap_i[j, 3] * (v1dd * v2dz + v1dz * v2zz)
            end 

            # parse result 
            buff[i] += -p * (-2.0) * Γdz * dv
        end
    end

    return nothing
end