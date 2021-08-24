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
    a    :: Action_u1_sym,
    temp :: Array{Float64, 3}
    )    :: Nothing

    # get propagator and overlap
    p       = -get_propagator(Λ, v + 0.5 * t, v - 0.5 * t, m, a)
    overlap = r.overlap

    # get buffers for left non-local vertex
    bs1 = get_buffer_s(v + vt, 0.5 * (-t - v + vt), 0.5 * (-t + v - vt), m)
    bt1 = get_buffer_empty()
    bu1 = get_buffer_u(-v + vt, 0.5 * (-t + v + vt), 0.5 * (t + v + vt), m)

    # get buffers for right non-local vertex
    bs2 = get_buffer_s(v + vtp, 0.5 * (-t + v - vtp), 0.5 * (-t - v + vtp), m)
    bt2 = get_buffer_t(t, v, vtp, m)
    bu2 = get_buffer_u(v - vtp, 0.5 * (-t + v + vtp), 0.5 * (t + v + vtp), m)

    # get buffers for local left vertex
    bs3 = get_buffer_s(v + vt, 0.5 * (-t - v + vt), 0.5 * (t - v + vt), m)
    bt3 = get_buffer_t(v - vt, 0.5 * (-t + v + vt), 0.5 * (t + v + vt), m)
    bu3 = get_buffer_empty()

    # get buffers for local right vertex
    bs4 = get_buffer_s(v + vtp, 0.5 * (-t + v - vtp), 0.5 * (t + v - vtp), m)
    bt4 = get_buffer_t(-v + vtp, 0.5 * (-t + v + vtp), 0.5 * (t + v + vtp), m)
    bu4 = get_buffer_u(-t, v, vtp, m)

    # cache local vertex values
    v3xx, v3zz, v3DM, v3dd, v3zd, v3dz = get_Γ(1, bs3, bt3, bu3, r, a, ch_u = false)
    v4xx, v4zz, v4DM, v4dd, v4zd, v4dz = get_Γ(1, bs4, bt4, bu4, r, a)

    # cache vertex values for all lattice sites in temporary buffer
    get_Γ_avx!(r, bs1, bt1, bu1, a, temp, 1, ch_t = false)
    get_Γ_avx!(r, bs2, bt2, bu2, a, temp, 2)

    # compute contributions for all lattice sites
    for i in eachindex(r.sites)
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
        Γxx = -p * (v1DM * v4dz - v1DM * v4zd + v1xx * v4dd - v1xx * v4zz +
                    v3dd * v2xx + v3dz * v2DM - v3zd * v2DM - v3zz * v2xx)

        Γzz = -p * (v1zz * v4dd + 2.0 * v1zd * v4DM - v1zd * v4zd + v1zz * v4zz - 2.0 * v1zz * v4xx - v1zd * v4dz -
                    2.0 * v3DM * v2dz - v3dz * v2dz + v3dd * v2zz - 2.0 * v3xx * v2zz + v3zz * v2zz - v3zd * v2dz)

        ΓDM = -p * (v1DM * v4dd - v1xx * v4dz + v1xx * v4zd - v1DM * v4zz -
                    v3dz * v2xx + v3dd * v2DM + v3zd * v2xx - v3zz * v2DM)
        
        Γdd = -p * (-v1dz * v4dz + 2.0 * v1dd * v4xx + v1dd * v4zz - v1dz * v4zd - 2.0 * v1dz * v4DM + v1dd * v4dd +
                    2.0 * v3DM * v2zd + 2.0 * v3xx * v2dd + v3zz * v2dd - v3dz * v2zd + v3dd * v2dd - v3zd * v2zd)

        Γzd = -p * (v1zd * v4zz + v1zd * v4dd + 2.0 * v1zd * v4xx + v1zz * v4zd + 2.0 * v1zz * v4DM + v1zz * v4dz +
                    v3zd * v2dd + 2.0 * v3DM * v2dd + v3dd * v2zd + v3zz * v2zd - 2.0 * v3xx * v2zd + v3dz * v2dd)

        Γdz = -p * (-2.0 * v1dd * v4DM + v1dd * v4zd + v1dd * v4dz + v1dz * v4zz + v1dz * v4dd - 2.0 * v1dz * v4xx +
                    v3zd * v2zz + 2.0 * v3xx * v2dz + v3dd * v2dz + v3dz * v2zz - 2.0 * v3DM * v2zz + v3zz * v2dz)

        # determine overlap for site i
        overlap_i = overlap[i]

        # determine range for inner sum
        Range = size(overlap_i, 1)

        # compute inner sum
        @turbo unroll = 1 for j in 1 : Range
            # read cached values for inner site
            v1xx = temp[overlap_i[j, 1], 1, 1]
            v1zz = temp[overlap_i[j, 1], 2, 1]
            v1DM = temp[overlap_i[j, 1], 3, 1]
            v1dd = temp[overlap_i[j, 1], 4, 1]
            v1zd = temp[overlap_i[j, 1], 5, 1]
            v1dz = temp[overlap_i[j, 1], 6, 1]
    
            v2xx = temp[overlap_i[j, 2], 1, 2]
            v2zz = temp[overlap_i[j, 2], 2, 2]
            v2DM = temp[overlap_i[j, 2], 3, 2]
            v2dd = temp[overlap_i[j, 2], 4, 2]
            v2zd = temp[overlap_i[j, 2], 5, 2]
            v2dz = temp[overlap_i[j, 2], 6, 2]

            # compute contribution at inner site
            Γxx += -p * (-2.0) * overlap_i[j, 3] * (v1xx * v2xx - v1DM * v2DM)
            Γzz += -p * (-2.0) * overlap_i[j, 3] * (v1zz * v2zz - v1zd * v2dz)
            ΓDM += -p * (-2.0) * overlap_i[j, 3] * (v1DM * v2xx + v1xx * v2DM)
            Γdd += -p * (-2.0) * overlap_i[j, 3] * (v1dd * v2dd - v1dz * v2zd)
            Γzd += -p * (-2.0) * overlap_i[j, 3] * (v1zd * v2dd + v1zz * v2zd)
            Γdz += -p * (-2.0) * overlap_i[j, 3] * (v1dd * v2dz + v1dz * v2zz)
        end

        # parse result to output buffer
        buff[1, i] += dv * Γxx
        buff[2, i] += dv * Γzz
        buff[3, i] += dv * ΓDM
        buff[4, i] += dv * Γdd
        buff[5, i] += dv * Γzd
        buff[6, i] += dv * Γdz
    end

    return nothing
end