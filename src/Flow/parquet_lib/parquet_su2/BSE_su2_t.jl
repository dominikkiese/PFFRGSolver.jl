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
    a    :: Action_su2,
    temp :: Array{Float64, 3}
    )    :: Nothing

    # get propagator and overlap
    p       = -get_propagator(Λ, v + 0.5 * t, v - 0.5 * t, m, a)
    overlap = r.overlap

    # get buffers for left non-local vertex
    bs1 = ntuple(comp -> get_buffer_s(comp, v + vt, 0.5 * (-t - v + vt), 0.5 * (-t + v - vt), m), 2)
    bt1 = ntuple(comp -> get_buffer_empty(comp), 2)
    bu1 = ntuple(comp -> get_buffer_u(comp, -v + vt, 0.5 * (-t + v + vt), 0.5 * (t + v + vt), m), 2)

    # get buffers for right non-local vertex
    bs2 = ntuple(comp -> get_buffer_s(comp, v + vtp, 0.5 * (-t + v - vtp), 0.5 * (-t - v + vtp), m), 2)
    bt2 = ntuple(comp -> get_buffer_t(comp, t, v, vtp, m), 2)
    bu2 = ntuple(comp -> get_buffer_u(comp, v - vtp, 0.5 * (-t + v + vtp), 0.5 * (t + v + vtp), m), 2)

    # get buffers for local left vertex
    bs3 = ntuple(comp -> get_buffer_s(comp, v + vt, 0.5 * (-t - v + vt), 0.5 * (t - v + vt), m), 2)
    bt3 = ntuple(comp -> get_buffer_t(comp, v - vt, 0.5 * (-t + v + vt), 0.5 * (t + v + vt), m), 2)
    bu3 = ntuple(comp -> get_buffer_empty(comp), 2)

    # get buffers for local right vertex
    bs4 = ntuple(comp -> get_buffer_s(comp, v + vtp, 0.5 * (-t + v - vtp), 0.5 * (t + v - vtp), m), 2)
    bt4 = ntuple(comp -> get_buffer_t(comp, -v + vtp, 0.5 * (-t + v + vtp), 0.5 * (t + v + vtp), m), 2)
    bu4 = ntuple(comp -> get_buffer_u(comp, -t, v, vtp, m), 2)

    # cache local vertex values
    v3s_st, v3d_st = get_Γ(1, bs3, bt3, bu3, r, a, ch_u = false)
    v4s, v4d       = get_Γ(1, bs4, bt4, bu4, r, a)

    # cache vertex values for all lattice sites in temporary buffer
    get_Γ_avx!(r, bs1, bt1, bu1, a, temp, 1, ch_t = false)
    get_Γ_avx!(r, bs2, bt2, bu2, a, temp, 2)

    # compute contributions for all lattice sites
    for i in eachindex(r.sites)
        # read cached values for site i
        v1s_su = temp[i, 1, 1]; v1d_su = temp[i, 2, 1]
        v2s    = temp[i, 1, 2]; v2d    = temp[i, 2, 2]

        # compute contribution at site i
        Γs = -p * (-1.0 * v1s_su * v4s + v1s_su * v4d -1.0 * v3s_st * v2s + v3d_st * v2s)
        Γd = -p * (3.0 * v1d_su * v4s + v1d_su * v4d + 3.0 * v3s_st * v2d + v3d_st * v2d)

        # determine overlap for site i
        overlap_i = overlap[i]

        # determine range for inner sum
        Range = size(overlap_i, 1)

        # compute inner sum
        @turbo unroll = 1 for j in 1 : Range
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
