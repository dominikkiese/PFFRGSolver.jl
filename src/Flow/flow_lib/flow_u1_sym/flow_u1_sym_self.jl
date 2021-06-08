# integration kernel for loop function
function compute_dΣ_kernel(
    Λ :: Float64,
    w :: Float64,
    v :: Float64,
    r :: Reduced_lattice,
    m :: Mesh,
    a :: Action_u1_sym
    ) :: Float64

    # get buffers for non-local vertex
    b1s = get_buffer_s(v + w, 0.5 * (-v + w), 0.5 * (v - w), m)
    b1t = get_buffer_t(0.0, w, v, m)
    b1u = get_buffer_u(-v + w, 0.5 * (v + w), 0.5 * (v + w), m)

    # get buffers for local vertex
    b2s = get_buffer_s(v + w, 0.5 * (-v + w), 0.5 * (-v + w), m)
    b2t = get_buffer_t(v - w, 0.5 * (v + w), 0.5 * (v + w), m)
    b2u = get_buffer_u(0.0, w, v, m)

    # compute local contributions
    val = 2.0 * get_Γ_comp(1, 1, b2s, b2t, b2u, r, a, apply_flags_u1_sym) +
                get_Γ_comp(2, 1, b2s, b2t, b2u, r, a, apply_flags_u1_sym) + 
                get_Γ_comp(4, 1, b2s, b2t, b2u, r, a, apply_flags_u1_sym)

    # compute contributions for all lattice sites
    for j in eachindex(r.sites)
        val -= 2.0 * r.mult[j] * get_Γ_comp(4, j, b1s, b1t, b1u, r, a, apply_flags_u1_sym)
    end

    # multiply with single scale propagator
    val *= get_S(Λ, v, m, a) / (2.0 * pi)

    return val
end