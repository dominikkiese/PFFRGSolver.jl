# Katanin kernel
function compute_t_kat!(
    Λ    :: Float64,
    buff :: Matrix{Float64},
    v    :: Float64,
    dv   :: Float64,
    t    :: Float64,
    vt   :: Float64,
    vtp  :: Float64,
    r    :: reduced_lattice,
    m    :: mesh,
    a    :: action_su2,
    da   :: action_su2,
    temp :: Array{Float64, 3}
    )    :: Nothing

    # get propagator and prefactors
    p       = get_propagator_kat(Λ, v + 0.5 * t, v - 0.5 * t, m, a, da) + get_propagator_kat(Λ, v - 0.5 * t, v + 0.5 * t, m, a, da)
    overlap = r.overlap

    # get buffers for left non-local vertex
    bs1 = get_buffer_su2_s(v + vt, 0.5 * (-t - v + vt), 0.5 * (-t + v - vt), m)
    bt1 = get_buffer_su2_t(t, vt, v, m)
    bu1 = get_buffer_su2_u(-v + vt, 0.5 * (-t + v + vt), 0.5 * (t + v + vt), m)

    # get buffers for right non-local vertex
    bs2 = get_buffer_su2_s(v + vtp, 0.5 * (-t + v - vtp), 0.5 * (-t - v + vtp), m)
    bt2 = get_buffer_su2_t(t, v, vtp, m)
    bu2 = get_buffer_su2_u(v - vtp, 0.5 * (-t + v + vtp), 0.5 * (t + v + vtp), m)

    # get buffers for local left vertex
    bs3 = get_buffer_su2_s(v + vt, 0.5 * (-t - v + vt), 0.5 * (t - v + vt), m)
    bt3 = get_buffer_su2_t(v - vt, 0.5 * (-t + v + vt), 0.5 * (t + v + vt), m)
    bu3 = get_buffer_su2_u(-t, vt, v, m)

    # get buffers for local right vertex
    bs4 = get_buffer_su2_s(v + vtp, 0.5 * (-t + v - vtp), 0.5 * (t + v - vtp), m)
    bt4 = get_buffer_su2_t(-v + vtp, 0.5 * (-t + v + vtp), 0.5 * (t + v + vtp), m)
    bu4 = get_buffer_su2_u(-t, v, vtp, m)

    # cache local vertex values
    v3s, v3d = get_Γ(1, bs3, bt3, bu3, r, a)
    v4s, v4d = get_Γ(1, bs4, bt4, bu4, r, a)

    # cache vertex values for all lattice sites in temporary buffer
    get_Γ_avx!(r, bs1, bt1, bu1, a, temp, 1)
    get_Γ_avx!(r, bs2, bt2, bu2, a, temp, 2)

    # compute contributions for all lattice sites
    for i in eachindex(r.sites)
        # read cached values for site i
        v1s = temp[i, 1, 1]; v1d = temp[i, 2, 1]
        v2s = temp[i, 1, 2]; v2d = temp[i, 2, 2]

        # compute contribution at site i
        Γs = -p * (-1.0 * v1s * v4s + v1s * v4d - 1.0 * v3s * v2s + v3d * v2s)
        Γd = -p * (3.0 * v1d * v4s + v1d * v4d + 3.0 * v3s * v2d + v3d * v2d)

        # determine range for inner sum
        Range = size(overlap[i], 1)

        # compute inner sum
        @avx unroll = 1 for j in 1 : Range
            # determine overlap for site i
            overlap_i = overlap[i]

            # read cached values for inner site
            v1s = temp[overlap_i[j, 1], 1, 1]; v1d = temp[overlap_i[j, 1], 2, 1]
            v2s = temp[overlap_i[j, 2], 1, 2]; v2d = temp[overlap_i[j, 2], 2, 2]

            # compute contribution at inner site
            Γs += -p * (-2.0) * overlap_i[j, 3] * (2.0 * a.S) * v1s * v2s
            Γd += -p * (-2.0) * overlap_i[j, 3] * (2.0 * a.S) * v1d * v2d
        end

        # parse result to output buffer
        buff[1, i] += dv * Γs
        buff[2, i] += dv * Γd
    end

    return nothing
end





# left kernel (right part obtained by symmetries)
function compute_t_left!(
    Λ    :: Float64,
    buff :: Matrix{Float64},
    v    :: Float64,
    dv   :: Float64,
    t    :: Float64,
    vt   :: Float64,
    vtp  :: Float64,
    r    :: reduced_lattice,
    m    :: mesh,
    a    :: action_su2,
    da   :: action_su2,
    temp :: Array{Float64, 3}
    )    :: Nothing

    # get propagator and prefactors
    p       = -get_propagator(Λ, v + 0.5 * t, v - 0.5 * t, m, a)
    overlap = r.overlap

    # get buffers for left non-local vertex
    bs1 = get_buffer_su2_s(v + vt, 0.5 * (-t - v + vt), 0.5 * (-t + v - vt), m)
    bt1 = get_buffer_su2_empty()
    bu1 = get_buffer_su2_u(-v + vt, 0.5 * (-t + v + vt), 0.5 * (t + v + vt), m)

    # get buffers for right non-local vertex
    bs2 = get_buffer_su2_s(v + vtp, 0.5 * (-t + v - vtp), 0.5 * (-t - v + vtp), m)
    bt2 = get_buffer_su2_t(t, v, vtp, m)
    bu2 = get_buffer_su2_u(v - vtp, 0.5 * (-t + v + vtp), 0.5 * (t + v + vtp), m)

    # get buffers for local left vertex
    bs3 = get_buffer_su2_s(v + vt, 0.5 * (-t - v + vt), 0.5 * (t - v + vt), m)
    bt3 = get_buffer_su2_t(v - vt, 0.5 * (-t + v + vt), 0.5 * (t + v + vt), m)
    bu3 = get_buffer_su2_empty()

    # get buffers for local right vertex
    bs4 = get_buffer_su2_s(v + vtp, 0.5 * (-t + v - vtp), 0.5 * (t + v - vtp), m)
    bt4 = get_buffer_su2_t(-v + vtp, 0.5 * (-t + v + vtp), 0.5 * (t + v + vtp), m)
    bu4 = get_buffer_su2_u(-t, v, vtp, m)

    # cache local vertex values
    v3s_st, v3d_st = get_Γ(1, bs3, bt3, bu3, r, da, ch_u = false)
    v4s, v4d       = get_Γ(1, bs4, bt4, bu4, r,  a)

    # cache vertex values for all lattice sites in temporary buffer
    get_Γ_avx!(r, bs1, bt1, bu1, da, temp, 1, ch_t = false)
    get_Γ_avx!(r, bs2, bt2, bu2,  a, temp, 2)

    # compute contributions for all lattice sites
    for i in eachindex(r.sites)
        # read cached values for site i
        v1s_su = temp[i, 1, 1]; v1d_su = temp[i, 2, 1]
        v2s    = temp[i, 1, 2]; v2d    = temp[i, 2, 2]

        # compute contribution at site i
        Γs = -p * (-1.0 * v1s_su * v4s + v1s_su * v4d - 1.0 * v3s_st * v2s + v3d_st * v2s)
        Γd = -p * (3.0 * v1d_su * v4s + v1d_su * v4d + 3.0 * v3s_st * v2d + v3d_st * v2d)

        # determine range for inner sum
        Range = size(overlap[i], 1)

        # compute inner sum
        @avx unroll = 1 for j in 1 : Range
            # determine overlap for site i
            overlap_i = overlap[i]

            # read cached values for inner site
            v1s_su = temp[overlap_i[j, 1], 1, 1]; v1d_su = temp[overlap_i[j, 1], 2, 1]
            v2s    = temp[overlap_i[j, 2], 1, 2]; v2d    = temp[overlap_i[j, 2], 2, 2]

            # compute contribution at inner site
            Γs += -p * (-2.0) * overlap_i[j, 3] * (2.0 * a.S) * v1s_su * v2s
            Γd += -p * (-2.0) * overlap_i[j, 3] * (2.0 * a.S) * v1d_su * v2d
        end

        # parse result to output buffer
        buff[1, i] += dv * Γs
        buff[2, i] += dv * Γd
    end

    return nothing
end





# central kernel
function compute_t_central!(
    Λ    :: Float64,
    buff :: Matrix{Float64},
    v    :: Float64,
    dv   :: Float64,
    t    :: Float64,
    vt   :: Float64,
    vtp  :: Float64,
    r    :: reduced_lattice,
    m    :: mesh,
    a    :: action_su2,
    da_l :: action_su2,
    temp :: Array{Float64, 3}
    )    :: Nothing

    # get propagator and prefactors
    p       = -get_propagator(Λ, v + 0.5 * t, v - 0.5 * t, m, a)
    overlap = r.overlap

    # get buffers for left non-local vertex
    bs1 = get_buffer_su2_s(v + vt, 0.5 * (-t - v + vt), 0.5 * (-t + v - vt), m)
    bt1 = get_buffer_su2_t(t, vt, v, m)
    bu1 = get_buffer_su2_u(-v + vt, 0.5 * (-t + v + vt), 0.5 * (t + v + vt), m)

    # get buffers for right non-local vertex
    bs2 = get_buffer_su2_empty()
    bt2 = get_buffer_su2_t(t, v, vtp, m)
    bu2 = get_buffer_su2_empty()

    # get buffers for local left vertex
    bs3 = get_buffer_su2_s(v + vt, 0.5 * (-t - v + vt), 0.5 * (t - v + vt), m)
    bt3 = get_buffer_su2_t(v - vt, 0.5 * (-t + v + vt), 0.5 * (t + v + vt), m)
    bu3 = get_buffer_su2_u(-t, vt, v, m)

    # get buffers for local right vertex
    bs4 = get_buffer_su2_empty()
    bt4 = get_buffer_su2_empty()
    bu4 = get_buffer_su2_u(-t, v, vtp, m)

    # cache local vertex values
    v3s, v3d     = get_Γ(1, bs3, bt3, bu3, r, a)
    v4s_u, v4d_u = get_Γ(1, bs4, bt4, bu4, r, da_l, ch_s = false, ch_t = false)

    # cache vertex values for all lattice sites in temporary buffer
    get_Γ_avx!(r, bs1, bt1, bu1,    a, temp, 1)
    get_Γ_avx!(r, bs2, bt2, bu2, da_l, temp, 2, ch_s = false, ch_u = false)

    # compute contributions for all lattice sites
    for i in eachindex(r.sites)
        # read cached values for site i
        v1s   = temp[i, 1, 1]; v1d   = temp[i, 2, 1]
        v2s_t = temp[i, 1, 2]; v2d_t = temp[i, 2, 2]

        # compute contribution at site i
        Γs = -p * (-1.0 * v1s * v4s_u + v1s * v4d_u - 1.0 * v3s * v2s_t + v3d * v2s_t)
        Γd = -p * (3.0 * v1d * v4s_u + v1d * v4d_u + 3.0 * v3s * v2d_t + v3d * v2d_t)

        # determine range for inner sum
        Range = size(overlap[i], 1)

        # compute inner sum
        @avx unroll = 1 for j in 1 : Range
            # determine overlap for site i
            overlap_i = overlap[i]

            # read cached values for inner site
            v1s   = temp[overlap_i[j, 1], 1, 1]; v1d   = temp[overlap_i[j, 1], 2, 1]
            v2s_t = temp[overlap_i[j, 2], 1, 2]; v2d_t = temp[overlap_i[j, 2], 2, 2]

            # compute contribution at inner site
            Γs += -p * (-2.0) * overlap_i[j, 3] * (2.0 * a.S) * v1s * v2s_t
            Γd += -p * (-2.0) * overlap_i[j, 3] * (2.0 * a.S) * v1d * v2d_t
        end

        # parse result to output buffer
        buff[1, i] += dv * Γs
        buff[2, i] += dv * Γd
    end

    return nothing
end
