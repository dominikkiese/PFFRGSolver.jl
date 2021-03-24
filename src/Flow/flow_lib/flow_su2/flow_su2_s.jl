# Katanin kernel
function compute_s_kat!(
    Λ    :: Float64,
    buff :: Matrix{Float64},
    v    :: Float64,
    dv   :: Float64,
    s    :: Float64,
    vs   :: Float64,
    vsp  :: Float64,
    r    :: reduced_lattice,
    m    :: mesh,
    a    :: action_su2,
    da   :: action_su2,
    temp :: Array{Float64, 3}
    )    :: Nothing

    # get propagator and prefactors
    p    = get_propagator_kat(Λ, v + 0.5 * s, 0.5 * s - v, m, a, da) + get_propagator_kat(Λ, 0.5 * s - v, v + 0.5 * s, m, a, da)
    pre1 = -2.0
    pre2 = 3.0

    # get buffers for left vertex
    bs1 = get_buffer_su2_s(s, vs, -v, m)
    bt1 = get_buffer_su2_t(v - vs, 0.5 * (s + v + vs), 0.5 * (s - v - vs), m)
    bu1 = get_buffer_su2_u(v + vs, 0.5 * (s - v + vs), 0.5 * (s + v - vs), m)

    # get buffers for right vertex
    bs2 = get_buffer_su2_s(s, v, vsp, m)
    bt2 = get_buffer_su2_t(-v - vsp, 0.5 * (s + v - vsp), 0.5 * (s - v + vsp), m)
    bu2 = get_buffer_su2_u(v - vsp, 0.5 * (s + v + vsp), 0.5 * (s - v - vsp), m)

    # cache vertex values for all lattice sites in temporary buffer
    get_Γ_avx!(r, bs1, bt1, bu1, a, temp, 1)
    get_Γ_avx!(r, bs2, bt2, bu2, a, temp, 2)

    # compute contributions for all lattice sites
    for i in eachindex(r.sites)
        # read cached values for site i
        v1s = temp[i, 1, 1]; v1d = temp[i, 2, 1]
        v2s = temp[i, 1, 2]; v2d = temp[i, 2, 2]

        # compute contribution at site i
        Γs = -p * (pre1 * v1s * v2s + v1s * v2d + v1d * v2s)
        Γd = -p * (pre2 * v1s * v2s + v1d * v2d)

        # parse result to output buffer
        buff[1, i] += dv * Γs
        buff[2, i] += dv * Γd
    end

    return nothing
end





# left kernel (right part obtained by symmetries)
function compute_s_left!(
    Λ    :: Float64,
    buff :: Matrix{Float64},
    v    :: Float64,
    dv   :: Float64,
    s    :: Float64,
    vs   :: Float64,
    vsp  :: Float64,
    r    :: reduced_lattice,
    m    :: mesh,
    a    :: action_su2,
    da   :: action_su2,
    temp :: Array{Float64, 3}
    )    :: Nothing

    # get propagator and prefactors
    p    = -get_propagator(Λ, v + 0.5 * s, 0.5 * s - v, m, a)
    pre1 = -2.0
    pre2 = 3.0

    # get buffers for left vertex
    bs1 = get_buffer_su2_s(s, vs, -v, m)
    bt1 = get_buffer_su2_t(v - vs, 0.5 * (s + v + vs), 0.5 * (s - v - vs), m)
    bu1 = get_buffer_su2_u(v + vs, 0.5 * (s - v + vs), 0.5 * (s + v - vs), m)

    # get buffers for right vertex
    bs2 = get_buffer_su2_s(s, v, vsp, m)
    bt2 = get_buffer_su2_t(-v - vsp, 0.5 * (s + v - vsp), 0.5 * (s - v + vsp), m)
    bu2 = get_buffer_su2_u(v - vsp, 0.5 * (s + v + vsp), 0.5 * (s - v - vsp), m)

    # cache vertex values for all lattice sites in temporary buffer
    get_Γ_avx!(r, bs1, bt1, bu1, da, temp, 1, ch_s = false)
    get_Γ_avx!(r, bs2, bt2, bu2,  a, temp, 2)

    # compute contributions for all lattice sites
    for i in eachindex(r.sites)
        # read cached values for site i
        v1s_tu = temp[i, 1, 1]; v1d_tu = temp[i, 2, 1]
        v2s    = temp[i, 1, 2]; v2d    = temp[i, 2, 2]

        # compute contribution at site i
        Γs = -p * (pre1 * v1s_tu * v2s + v1s_tu * v2d + v1d_tu * v2s)
        Γd = -p * (pre2 * v1s_tu * v2s + v1d_tu * v2d)

        # parse result to output buffer
        buff[1, i] += dv * Γs
        buff[2, i] += dv * Γd
    end

    return nothing
end





# central kernel
function compute_s_central!(
    Λ    :: Float64,
    buff :: Matrix{Float64},
    v    :: Float64,
    dv   :: Float64,
    s    :: Float64,
    vs   :: Float64,
    vsp  :: Float64,
    r    :: reduced_lattice,
    m    :: mesh,
    a    :: action_su2,
    da_l :: action_su2,
    temp :: Array{Float64, 3}
    )    :: Nothing

    # get propagator and prefactors
    p    = -get_propagator(Λ, v + 0.5 * s, 0.5 * s - v, m, a)
    pre1 = -2.0
    pre2 = 3.0

    # get buffers for left vertex
    bs1 = get_buffer_su2_s(s, vs, -v, m)
    bt1 = get_buffer_su2_t(v - vs, 0.5 * (s + v + vs), 0.5 * (s - v - vs), m)
    bu1 = get_buffer_su2_u(v + vs, 0.5 * (s - v + vs), 0.5 * (s + v - vs), m)

    # get buffers for right vertex
    bs2 = get_buffer_su2_s(s, v, vsp, m)
    bt2 = get_buffer_su2_t(-v - vsp, 0.5 * (s + v - vsp), 0.5 * (s - v + vsp), m)
    bu2 = get_buffer_su2_u(v - vsp, 0.5 * (s + v + vsp), 0.5 * (s - v - vsp), m)

    # cache vertex values for all lattice sites in temporary buffer
    get_Γ_avx!(r, bs1, bt1, bu1,    a, temp, 1)
    get_Γ_avx!(r, bs2, bt2, bu2, da_l, temp, 2, ch_t = false, ch_u = false)

    # compute contributions for all lattice sites
    for i in eachindex(r.sites)
        # read cached values for site i
        v1s   = temp[i, 1, 1]; v1d   = temp[i, 2, 1]
        v2s_s = temp[i, 1, 2]; v2d_s = temp[i, 2, 2]

        # compute contribution at site i
        Γs = -p * (pre1 * v1s * v2s_s + v1s * v2d_s + v1d * v2s_s)
        Γd = -p * (pre2 * v1s * v2s_s + v1d * v2d_s)

        # parse result to output buffer
        buff[1, i] += dv * Γs
        buff[2, i] += dv * Γd
    end

    return nothing
end
