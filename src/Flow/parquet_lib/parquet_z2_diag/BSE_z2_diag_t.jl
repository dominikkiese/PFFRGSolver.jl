# BSE kernel for the t channel
function compute_t_BSE!(
    Λ    :: Float64,
    buff :: Matrix{Float64},
    v    :: Float64,
    dv   :: Float64,
    t    :: Float64,
    vt   :: Float64,
    vtp  :: Float64,
    r    :: Reduced_lattice,
    m    :: Mesh,
    a    :: Action_z2_diag,
    temp :: Array{Float64, 3}
    )    :: Nothing

    # get propagator and overlap
    p       = -get_propagator(Λ, v + 0.5 * t, v - 0.5 * t, m, a)
    overlap = r.overlap

    # get buffers for left non-local vertex
    bs1 = get_buffer_s( v + vt, 0.5 * (-t - v + vt), 0.5 * (-t + v - vt), m)
    bt1 = get_buffer_empty()
    bu1 = get_buffer_u(-v + vt, 0.5 * (-t + v + vt), 0.5 * ( t + v + vt), m)

    # get buffers for right non-local vertex
    bs2 = get_buffer_s(v + vtp, 0.5 * (-t + v - vtp), 0.5 * (-t - v + vtp), m)
    bt2 = get_buffer_t(t, v, vtp, m)
    bu2 = get_buffer_u(v - vtp, 0.5 * (-t + v + vtp), 0.5 * ( t + v + vtp), m)

    # get buffers for local left vertex
    bs3 = get_buffer_s( v + vt, 0.5 * (t + v - vt), 0.5 * (-t + v - vt), m)
    bt3 = get_buffer_t(-v + vt, 0.5 * (t + v + vt), 0.5 * (-t + v + vt), m)
    bu3 = get_buffer_empty()

    # get buffers for local right vertex
    bs4 = get_buffer_s(v + vtp, 0.5 * (t - v + vtp), 0.5 * (-t - v + vtp), m)
    bt4 = get_buffer_t(v - vtp, 0.5 * (t + v + vtp), 0.5 * (-t + v + vtp), m)
    bu4 = get_buffer_u(t, vtp, v, m)

    # cache local vertex values
    v3xx, v3yy, v3zz, v3dd = get_Γ(1, bs3, bt3, bu3, r, a, ch_u = false)
    v4xx, v4yy, v4zz, v4dd = get_Γ(1, bs4, bt4, bu4, r, a)

    # cache vertex values for all lattice sites in temporary buffer
    get_Γ_avx!(r, bs1, bt1, bu1, a, temp, 1, ch_t = false)
    get_Γ_avx!(r, bs2, bt2, bu2, a, temp, 2)

    # compute contributions for all lattice sites
    for i in eachindex(r.sites)
        # read cached values for site i
        v1xx = temp[i, 1, 1]
        v1yy = temp[i, 2, 1]
        v1zz = temp[i, 3, 1]
        v1dd = temp[i, 4, 1]

        v2xx = temp[i, 1, 2]
        v2yy = temp[i, 2, 2]
        v2zz = temp[i, 3, 2]
        v2dd = temp[i, 4, 2]

        # compute contribution at site i
        Γxx = -p * (v1xx * v4dd + v1xx * v4xx - v1xx * v4zz - v1xx * v4yy +
                    v3dd * v2xx + v3xx * v2xx - v3yy * v2xx - v3zz * v2xx)

        Γyy = -p * (v1yy * v4dd - v1yy * v4xx + v1yy * v4yy - v1yy * v4zz + 
                    v3dd * v2yy - v3xx * v2yy + v3yy * v2yy - v3zz * v2yy)   

        Γzz = -p * (v1zz * v4dd + v1zz * v4zz - v1zz * v4xx - v1zz * v4yy +
                    v3dd * v2zz - v3xx * v2zz + v3zz * v2zz - v3yy * v2zz)
        
        Γdd = -p * (v1dd * v4xx + v1dd * v4yy + v1dd * v4zz + v1dd * v4dd + 
                    v3zz * v2dd + v3xx * v2dd + v3yy * v2dd + v3dd * v2dd)

        # determine overlap for site i
        overlap_i = overlap[i]

        # determine range for inner sum
        Range = size(overlap_i, 1)

        # compute inner sum
        @turbo unroll = 1 for j in 1 : Range
            # read cached values for inner site
            v1xx = temp[overlap_i[j, 1], 1, 1]
            v1yy = temp[overlap_i[j, 1], 2, 1]
            v1zz = temp[overlap_i[j, 1], 3, 1]
            v1dd = temp[overlap_i[j, 1], 4, 1]
    
            v2xx = temp[overlap_i[j, 2], 1, 2]
            v2yy = temp[overlap_i[j, 2], 2, 2]
            v2zz = temp[overlap_i[j, 2], 3, 2]
            v2dd = temp[overlap_i[j, 2], 4, 2]

            # compute contribution at inner site
            Γxx += -p * (-2.0) * overlap_i[j, 3] * v1xx * v2xx
            Γyy += -p * (-2.0) * overlap_i[j, 3] * v1yy * v2yy
            Γzz += -p * (-2.0) * overlap_i[j, 3] * v1zz * v2zz
            Γdd += -p * (-2.0) * overlap_i[j, 3] * v1dd * v2dd
        end

        # parse result to output buffer
        buff[1, i] += dv * Γxx
        buff[2, i] += dv * Γyy
        buff[3, i] += dv * Γzz
        buff[4, i] += dv * Γdd
    end

    return nothing
end