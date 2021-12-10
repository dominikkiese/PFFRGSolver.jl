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
    a    :: Action_su2,
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
        v1s = a.Γ[1].bare[i]; v1d = a.Γ[2].bare[i]
        v2s =  temp[i, 1, 2]; v2d = temp[i, 2, 2]

        # compute contribution at site i
        Γs = -p * (-2.0 * v1s * v2s + v1s * v2d + v1d * v2s)
        Γd = -p * ( 3.0 * v1s * v2s + v1d * v2d)

        # parse result to output buffer
        buff[1, i] += dv * Γs
        buff[2, i] += dv * Γd
    end

    return nothing
end

# integration kernel for loop function
function compute_Σ_kernel(
    Λ     :: Float64,
    w     :: Float64,
    v     :: Float64,
    r     :: Reduced_lattice,
    m     :: Mesh,
    a1    :: Action_su2,
    a2    :: Action_su2,
    Σ_tol :: NTuple{2, Float64}
    )     :: Float64

    # get buffers for non-local vertex
    b1s = get_buffer_s(v + w, Inf, 0.5 * (v - w), m)
    b1t = get_buffer_empty()
    b1u = get_buffer_empty()

    # get buffers for local vertex
    b2s = get_buffer_s(v + w, Inf, 0.5 * (-v + w), m)
    b2t = get_buffer_empty()
    b2u = get_buffer_empty()

    # compute local contributions
    val = 3.0 * get_Γ_comp(1, 1, b2s, b2t, b2u, r, a2, apply_flags_su2, ch_t = false, ch_u = false) + 
                get_Γ_comp(2, 1, b2s, b2t, b2u, r, a2, apply_flags_su2, ch_t = false, ch_u = false)

    # compute contributions for all lattice sites
    for j in eachindex(r.sites)
        val -= 2.0 * r.mult[j] * (2.0 * a2.S) * get_Γ_comp(2, j, b1s, b1t, b1u, r, a2, apply_flags_su2, ch_t = false, ch_u = false)
    end

    # multiply with full propagator
    val *= -get_G(Λ, v, m, a1) / (2.0 * pi)

    return val
end