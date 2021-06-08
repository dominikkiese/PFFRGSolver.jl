# Katanin kernel
function compute_s_kat!(
    Λ    :: Float64,
    buff :: Matrix{Float64},
    v    :: Float64,
    dv   :: Float64,
    s    :: Float64,
    vs   :: Float64,
    vsp  :: Float64,
    r    :: Reduced_lattice,
    m    :: Mesh,
    a    :: Action_u1_sym,
    da   :: Action_u1_sym,
    temp :: Array{Float64, 3}
    )    :: Nothing

    # get propagator
    p = get_propagator_kat(Λ, v + 0.5 * s, 0.5 * s - v, m, a, da) + get_propagator_kat(Λ, 0.5 * s - v, v + 0.5 * s, m, a, da)

    # get buffers for left vertex
    bs1 = get_buffer_s(s, vs, -v, m)
    bt1 = get_buffer_t(v - vs, 0.5 * (s + v + vs), 0.5 * (s - v - vs), m)
    bu1 = get_buffer_u(v + vs, 0.5 * (s - v + vs), 0.5 * (s + v - vs), m)

    # get buffers for right vertex
    bs2 = get_buffer_s(s, v, vsp, m)
    bt2 = get_buffer_t(-v - vsp, 0.5 * (s + v - vsp), 0.5 * (s - v + vsp), m)
    bu2 = get_buffer_u(v - vsp, 0.5 * (s + v + vsp), 0.5 * (s - v - vsp), m)

    # cache vertex values for all lattice sites in temporary buffer
    get_Γ_avx!(r, bs1, bt1, bu1, a, temp, 1)
    get_Γ_avx!(r, bs2, bt2, bu2, a, temp, 2)

    # compute contributions for all lattice sites
    @avx unroll = 1 for i in eachindex(r.sites)
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
        Γxx = -p * (v1DM * v2dz - v1DM * v2zd + v1xx * v2dd + v1dd * v2xx - v1xx * v2zz - v1dz * v2DM + v1zd * v2DM - v1zz * v2xx)       
        Γzz = -p * (v1zz * v2dd + v1dd * v2zz - v1dz * v2zd - 2.0 * v1xx * v2xx - 2.0 * v1DM * v2DM - v1zd * v2dz)
        ΓDM = -p * (v1dd * v2DM + v1DM * v2dd - v1zd * v2xx - v1xx * v2dz - v1zz * v2DM + v1xx * v2zd - v1DM * v2zz + v1dz * v2xx)
        Γdd = -p * (-v1dz * v2dz + 2.0 * v1xx * v2xx + v1dd * v2dd + 2.0 * v1DM * v2DM - v1zd * v2zd + v1zz * v2zz)
        Γzd = -p * (v1zd * v2dd + v1dd * v2zd - 2.0 * v1DM * v2xx + v1dz * v2zz + v1zz * v2dz + 2.0 * v1xx * v2DM)
        Γdz = -p * (v1zd * v2zz + v1dd * v2dz + v1zz * v2zd + 2.0 * v1DM * v2xx + v1dz * v2dd - 2.0 * v1xx * v2DM)

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