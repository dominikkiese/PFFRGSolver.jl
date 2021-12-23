# reduced kernel for the s channel
function compute_s_reduced!(
    Λ    :: Float64,
    buff :: Matrix{Float64},
    v    :: Float64,
    dv   :: Float64,
    s    :: Float64,
    vsp  :: Float64,
    r    :: Reduced_lattice,
    m    :: Mesh,
    a    :: Action_u1_dm,
    temp :: Array{Float64, 3}
    )    :: Nothing

    # get propagator
    p = -get_propagator(Λ, v + 0.5 * s, 0.5 * s - v, m, a)

    # get buffers for right vertex (left vertex is given by bare)
    bs2 = get_buffer_s(s, v, vsp, m)
    bt2 = get_buffer_t(-v - vsp, 0.5 * (s + v - vsp), 0.5 * (s - v + vsp), m)
    bu2 = get_buffer_u( v - vsp, 0.5 * (s + v + vsp), 0.5 * (s - v - vsp), m)

    # cache vertex values for all lattice sites in temporary buffer
    get_Γ_avx!(r, bs2, bt2, bu2, a, temp, 2)

    # compute contributions for all lattice sites
    @turbo unroll = 1 for i in eachindex(r.sites)
        # read cached values for site i
        v1xx = a.Γ[1].bare[i]
        v1zz = a.Γ[2].bare[i]
        v1DM = a.Γ[3].bare[i]
        v1dd = a.Γ[4].bare[i]
        v1zd = a.Γ[5].bare[i]
        v1dz = a.Γ[6].bare[i]

        v2xx = temp[i, 1, 2]
        v2zz = temp[i, 2, 2]
        v2DM = temp[i, 3, 2]
        v2dd = temp[i, 4, 2]
        v2zd = temp[i, 5, 2]
        v2dz = temp[i, 6, 2]

        # compute contribution at site i
        Γxx = -p * (v1DM * v2dz - v1DM * v2zd + v1xx * v2dd + v1dd * v2xx - v1xx * v2zz - v1dz * v2DM + v1zd * v2DM - v1zz * v2xx)       
        Γzz = -p * (v1zz * v2dd + v1dd * v2zz - v1dz * v2zd - 2.0 * v1xx * v2xx - 2.0 * v1DM * v2DM - v1zd * v2dz)
        Γdd = -p * (-v1dz * v2dz + 2.0 * v1xx * v2xx + v1dd * v2dd + 2.0 * v1DM * v2DM - v1zd * v2zd + v1zz * v2zz)

        # parse result to output buffer
        buff[1, i] += dv * Γxx
        buff[2, i] += dv * Γzz
        buff[4, i] += dv * Γdd
    end

    return nothing
end

# integration kernel for loop function
function compute_Σ_kernel(
    Λ  :: Float64,
    w  :: Float64,
    v  :: Float64,
    r  :: Reduced_lattice,
    m  :: Mesh,
    a1 :: Action_u1_dm,
    a2 :: Action_u1_dm,
    )  :: Float64

    # get buffers for non-local vertex
    b1s = get_buffer_s(v + w, Inf, 0.5 * (v - w), m)
    b1t = get_buffer_empty()
    b1u = get_buffer_empty()

    # get buffers for local vertex
    b2s = get_buffer_s(v + w, Inf, 0.5 * (-v + w), m)
    b2t = get_buffer_empty()
    b2u = get_buffer_empty()

    # compute local contributions
    val = 2.0 * get_Γ_comp(1, 1, b2s, b2t, b2u, r, a2, apply_flags_u1_dm, ch_t = false, ch_u = false) + 
                get_Γ_comp(2, 1, b2s, b2t, b2u, r, a2, apply_flags_u1_dm, ch_t = false, ch_u = false) + 
                get_Γ_comp(4, 1, b2s, b2t, b2u, r, a2, apply_flags_u1_dm, ch_t = false, ch_u = false)

    # compute contributions for all lattice sites
    for j in eachindex(r.sites)
        val -= 2.0 * r.mult[j] * get_Γ_comp(4, j, b1s, b1t, b1u, r, a2, apply_flags_u1_dm, ch_t = false, ch_u = false)
    end

    # multiply with full propagator
    val *= -get_G(Λ, v, m, a1) / (2.0 * pi)

    return val
end