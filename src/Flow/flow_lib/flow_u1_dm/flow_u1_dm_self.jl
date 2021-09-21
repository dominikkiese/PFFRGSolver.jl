# integration kernel for loop function
function compute_dΣ_kernel(
    Λ :: Float64,
    w :: Float64,
    v :: Float64,
    r :: Reduced_lattice,
    m :: Mesh,
    a :: Action_u1_dm
    ) :: Float64

    # get buffers for non-local vertex
    b1s = get_buffer_s(4,  v + w, 0.5 * (-v + w), 0.5 * (v - w), m)
    b1t = get_buffer_t(4, 0.0, w, v, m)
    b1u = get_buffer_u(4, -v + w, 0.5 * ( v + w), 0.5 * (v + w), m)

    # get buffers for local vertex
    b2s1 = get_buffer_s(1, v + w, 0.5 * (-v + w), 0.5 * (-v + w), m)
    b2t1 = get_buffer_t(1, v - w, 0.5 * ( v + w), 0.5 * ( v + w), m)
    b2u1 = get_buffer_u(1, 0.0, w, v, m)

    b2s2 = get_buffer_s(2, v + w, 0.5 * (-v + w), 0.5 * (-v + w), m)
    b2t2 = get_buffer_t(2, v - w, 0.5 * ( v + w), 0.5 * ( v + w), m)
    b2u2 = get_buffer_u(2, 0.0, w, v, m)

    b2s4 = get_buffer_s(4, v + w, 0.5 * (-v + w), 0.5 * (-v + w), m)
    b2t4 = get_buffer_t(4, v - w, 0.5 * ( v + w), 0.5 * ( v + w), m)
    b2u4 = get_buffer_u(4, 0.0, w, v, m)

    # compute local contributions
    val = 2.0 * get_Γ_comp(1, 1, b2s1, b2t1, b2u1, r, a, apply_flags_u1_dm) +
                get_Γ_comp(2, 1, b2s2, b2t2, b2u2, r, a, apply_flags_u1_dm) + 
                get_Γ_comp(4, 1, b2s4, b2t4, b2u4, r, a, apply_flags_u1_dm)

    # compute contributions for all lattice sites
    for j in eachindex(r.sites)
        val -= 2.0 * r.mult[j] * get_Γ_comp(4, j, b1s, b1t, b1u, r, a, apply_flags_u1_dm)
    end

    # multiply with single scale propagator
    val *= get_S(Λ, v, m, a) / (2.0 * pi)

    return val
end

# first integration kernel for loop function for self energy corrections
function compute_dΣ_kernel_corr1(
    Λ    :: Float64,
    w    :: Float64,
    v    :: Float64,
    r    :: Reduced_lattice,
    m    :: Mesh,
    a    :: Action_u1_dm,
    da_Σ :: Action_u1_dm
    )    :: Float64

    # get buffers for non-local vertex
    b1s = get_buffer_s(4,  v + w, 0.5 * (-v + w), 0.5 * (v - w), m)
    b1t = get_buffer_t(4, 0.0, w, v, m)
    b1u = get_buffer_u(4, -v + w, 0.5 * ( v + w), 0.5 * (v + w), m)

    # get buffers for local vertex
    b2s1 = get_buffer_s(1, v + w, 0.5 * (-v + w), 0.5 * (-v + w), m)
    b2t1 = get_buffer_t(1, v - w, 0.5 * ( v + w), 0.5 * ( v + w), m)
    b2u1 = get_buffer_u(1, 0.0, w, v, m)

    b2s2 = get_buffer_s(2, v + w, 0.5 * (-v + w), 0.5 * (-v + w), m)
    b2t2 = get_buffer_t(2, v - w, 0.5 * ( v + w), 0.5 * ( v + w), m)
    b2u2 = get_buffer_u(2, 0.0, w, v, m)

    b2s4 = get_buffer_s(4, v + w, 0.5 * (-v + w), 0.5 * (-v + w), m)
    b2t4 = get_buffer_t(4, v - w, 0.5 * ( v + w), 0.5 * ( v + w), m)
    b2u4 = get_buffer_u(4, 0.0, w, v, m)

    # compute local contributions
    val = 2.0 * get_Γ_comp(1, 1, b2s1, b2t1, b2u1, r, da_Σ, apply_flags_u1_dm, ch_u = false) +
                get_Γ_comp(2, 1, b2s2, b2t2, b2u2, r, da_Σ, apply_flags_u1_dm, ch_u = false) + 
                get_Γ_comp(4, 1, b2s4, b2t4, b2u4, r, da_Σ, apply_flags_u1_dm, ch_u = false)

    # compute contributions for all lattice sites
    for j in eachindex(r.sites)
        val -= 2.0 * r.mult[j] * get_Γ_comp(4, j, b1s, b1t, b1u, r, da_Σ, apply_flags_u1_dm, ch_t = false)
    end

    # multiply with full propagator
    val *= -get_G(Λ, v, m, a) / (2.0 * pi)

    return val
end

# second integration kernel for loop function for self energy corrections
function compute_dΣ_kernel_corr2(
    Λ    :: Float64,
    w    :: Float64,
    v    :: Float64,
    r    :: Reduced_lattice,
    m    :: Mesh,
    a    :: Action_u1_dm,
    da_Σ :: Action_u1_dm
    )    :: Float64

    # get buffers for non-local vertex
    b1s = get_buffer_s(4,  v + w, 0.5 * (-v + w), 0.5 * (v - w), m)
    b1t = get_buffer_t(4, 0.0, w, v, m)
    b1u = get_buffer_u(4, -v + w, 0.5 * ( v + w), 0.5 * (v + w), m)

    # get buffers for local vertex
    b2s1 = get_buffer_s(1, v + w, 0.5 * (-v + w), 0.5 * (-v + w), m)
    b2t1 = get_buffer_t(1, v - w, 0.5 * ( v + w), 0.5 * ( v + w), m)
    b2u1 = get_buffer_u(1, 0.0, w, v, m)

    b2s2 = get_buffer_s(2, v + w, 0.5 * (-v + w), 0.5 * (-v + w), m)
    b2t2 = get_buffer_t(2, v - w, 0.5 * ( v + w), 0.5 * ( v + w), m)
    b2u2 = get_buffer_u(2, 0.0, w, v, m)

    b2s4 = get_buffer_s(4, v + w, 0.5 * (-v + w), 0.5 * (-v + w), m)
    b2t4 = get_buffer_t(4, v - w, 0.5 * ( v + w), 0.5 * ( v + w), m)
    b2u4 = get_buffer_u(4, 0.0, w, v, m)

    # compute local contributions
    val = 2.0 * get_Γ_comp(1, 1, b2s1, b2t1, b2u1, r, a, apply_flags_u1_dm) +
                get_Γ_comp(2, 1, b2s2, b2t2, b2u2, r, a, apply_flags_u1_dm) + 
                get_Γ_comp(4, 1, b2s4, b2t4, b2u4, r, a, apply_flags_u1_dm)

    # compute contributions for all lattice sites
    for j in eachindex(r.sites)
        val -= 2.0 * r.mult[j] * get_Γ_comp(4, j, b1s, b1t, b1u, r, a, apply_flags_u1_dm)
    end

    # multiply with full propagator
    val *= get_G(Λ, v, m, a)^2 * get_Σ(v, m, da_Σ) / (2.0 * pi)

    return val
end