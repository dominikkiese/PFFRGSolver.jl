# BSE kernel for the u channel
function compute_u_BSE!(
    Λ    :: Float64,
    buff :: Matrix{Float64},
    v    :: Float64,
    dv   :: Float64,
    u    :: Float64,
    vu   :: Float64,
    vup  :: Float64,
    r    :: reduced_lattice,
    m    :: Mesh,
    a    :: Action_su2,
    temp :: Array{Float64, 3}
    )    :: Nothing

    # get propagator
    p    = -get_propagator(Λ, v - 0.5 * u, v + 0.5 * u, m, a)

    # get buffers for left vertex
    bs1 = get_buffer_su2_s(v + vu, 0.5 * (u - v + vu), 0.5 * (-u - v + vu), m)
    bt1 = get_buffer_su2_t(v - vu, 0.5 * (u + v + vu), 0.5 * (-u + v + vu), m)
    bu1 = get_buffer_su2_empty()

    # get buffers for right vertex
    bs2 = get_buffer_su2_s(v + vup, 0.5 * (u + v - vup), 0.5 * (-u + v - vup), m)
    bt2 = get_buffer_su2_t(-v + vup, 0.5 * (u + v + vup), 0.5 * (-u + v + vup), m)
    bu2 = get_buffer_su2_u(u, v, vup, m)

    # cache vertex values for all lattice sites in temporary buffer
    get_Γ_avx!(r, bs1, bt1, bu1, a, temp, 1, ch_u = false)
    get_Γ_avx!(r, bs2, bt2, bu2, a, temp, 2)

    # compute contributions for all lattice sites
    for i in eachindex(r.sites)
        # read cached values for site i
        v1s_st = temp[i, 1, 1]; v1d_st = temp[i, 2, 1]
        v2s    = temp[i, 1, 2]; v2d    = temp[i, 2, 2]

        # compute contribution at site i
        Γs = -p * (2.0 * v1s_st * v2s + v1s_st * v2d + v1d_st * v2s)
        Γd = -p * (3.0 * v1s_st * v2s + v1d_st * v2d)

        # parse result to output buffer
        buff[1, i] += dv * Γs
        buff[2, i] += dv * Γd
    end

    return nothing
end
