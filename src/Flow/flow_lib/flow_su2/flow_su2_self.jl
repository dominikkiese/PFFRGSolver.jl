# integration kernel for loop function
function compute_dΣ_kernel(
    Λ :: Float64,
    w :: Float64,
    v :: Float64,
    r :: Reduced_lattice,
    m :: Mesh,
    a :: Action_su2
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
    val = 3.0 * get_spin(1, b2s, b2t, b2u, r, a) + get_dens(1, b2s, b2t, b2u, r, a)

    # compute contributions for all lattice sites
    for j in eachindex(r.sites)
        val -= 2.0 * r.mult[j] * (2.0 * a.S)  * get_dens(j, b1s, b1t, b1u, r, a)
    end

    # multiply with single scale propagator
    val *= get_S(Λ, v, m, a) / (2.0 * pi)

    return val
end

# compute self energy derivative
function compute_dΣ!(
    Λ  :: Float64,
    r  :: Reduced_lattice,
    m  :: Mesh,
    a  :: Action_su2,
    da :: Action_su2
    )  :: Nothing

    # compute self energy derivative for all frequencies
    @sync for i in 2 : length(m.σ)
        Threads.@spawn begin
            integrand = v -> compute_dΣ_kernel(Λ, m.σ[i], v, r, m, a)
            da.Σ[i]   = quadgk(integrand, -Inf, -2.0 * Λ, 2.0 * Λ, Inf, atol = 1e-8, rtol = 1e-3)[1]
        end
    end

    return nothing
end





# first integration kernel for loop function for self energy corrections
function compute_dΣ_kernel_corr1(
    Λ    :: Float64,
    w    :: Float64,
    v    :: Float64,
    r    :: Reduced_lattice,
    m    :: Mesh,
    a    :: Action_su2,
    da_Σ :: Action_su2
    )    :: Float64

    # get buffers for non-local vertex
    b1s = get_buffer_s(v + w, 0.5 * (-v + w), 0.5 * (v - w), m)
    b1t = get_buffer_t(0.0, w, v, m)
    b1u = get_buffer_u(-v + w, 0.5 * (v + w), 0.5 * (v + w), m)

    # get buffers for local vertex
    b2s = get_buffer_s(v + w, 0.5 * (-v + w), 0.5 * (-v + w), m)
    b2t = get_buffer_t(v - w, 0.5 * (v + w), 0.5 * (v + w), m)
    b2u = get_buffer_u(0.0, w, v, m)

    # compute local contributions
    val = 3.0 * get_spin(1, b2s, b2t, b2u, r, da_Σ, ch_u = false) + get_dens(1, b2s, b2t, b2u, r, da_Σ, ch_u = false)

    # compute contributions for all lattice sites
    for j in eachindex(r.sites)
        val -= 2.0 * r.mult[j] * (2.0 * a.S) * get_dens(j, b1s, b1t, b1u, r, da_Σ, ch_t = false)
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
    a    :: Action_su2,
    da_Σ :: Action_su2
    )    :: Float64

    # get buffers for non-local vertex
    b1s = get_buffer_s(v + w, 0.5 * (-v + w), 0.5 * (v - w), m)
    b1t = get_buffer_t(0.0, w, v, m)
    b1u = get_buffer_u(-v + w, 0.5 * (v + w), 0.5 * (v + w), m)

    # get buffers for local vertex
    b2s = get_buffer_s(v + w, 0.5 * (-v + w), 0.5 * (-v + w), m)
    b2t = get_buffer_t(v - w, 0.5 * (v + w), 0.5 * (v + w), m)
    b2u = get_buffer_u(0.0, w, v, m)

    # compute local contributions
    val = 3.0 * get_spin(1, b2s, b2t, b2u, r, a) + get_dens(1, b2s, b2t, b2u, r, a)

    # compute contributions for all lattice sites
    for j in eachindex(r.sites)
        val -= 2.0 * r.mult[j] * (2.0 * a.S) * get_dens(j, b1s, b1t, b1u, r, a)
    end

    # multiply with full propagator
    val *= get_G(Λ, v, m, a)^2 * get_Σ(v, m, da_Σ) / (2.0 * pi)

    return val
end

# compute corrections to self energy derivative
function compute_dΣ_corr!(
    Λ    :: Float64,
    r    :: Reduced_lattice,
    m    :: Mesh,
    a    :: Action_su2,
    da   :: Action_su2,
    da_Σ :: Action_su2
    )    :: Nothing

    # compute first correction
    @sync for i in 2 : length(m.σ)
        Threads.@spawn begin
            integrand = v -> compute_dΣ_kernel_corr1(Λ, m.σ[i], v, r, m, a, da_Σ)
            da_Σ.Σ[i] = quadgk(integrand, -Inf, -2.0 * Λ, 2.0 * Λ, Inf, atol = 1e-8, rtol = 1e-3)[1]
        end
    end

    # compute second correction and parse to da
    @sync for i in 2 : length(m.σ)
        Threads.@spawn begin
            integrand  = v -> compute_dΣ_kernel_corr2(Λ, m.σ[i], v, r, m, a, da_Σ)
            da.Σ[i]   += da_Σ.Σ[i]
            da.Σ[i]   += quadgk(integrand, -Inf, -2.0 * Λ, 2.0 * Λ, Inf, atol = 1e-8, rtol = 1e-3)[1]
        end
    end

    return nothing
end
