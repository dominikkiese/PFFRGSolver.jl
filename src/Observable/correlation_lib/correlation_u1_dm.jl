# inner kernel for double integral (xx correlation)
function inner_kernel_xx(
    Λ    :: Float64,
    site :: Int64,
    v    :: Float64,
    vp   :: Float64,
    r    :: Reduced_lattice,
    m    :: Mesh_u1_dm,
    a    :: Action_u1_dm
    )    :: Float64

    # get buffers for non-local term
    b1s = get_buffer_s(1, v + vp, 0.5 * (v - vp), 0.5 * (-v + vp), m)
    b1t = get_buffer_t(1, 0.0, v, vp, m)
    b1u = get_buffer_u(1, v - vp, 0.5 * (v + vp), 0.5 * ( v + vp), m)

    # get buffers for local term
    b2s2 = get_buffer_s(2, v + vp, 0.5 * (-v + vp), 0.5 * (-v + vp), m)
    b2t2 = get_buffer_t(2, v - vp, 0.5 * ( v + vp), 0.5 * ( v + vp), m)
    b2u2 = get_buffer_u(2, 0.0, vp, v, m)

    b2s4 = get_buffer_s(4, v + vp, 0.5 * (-v + vp), 0.5 * (-v + vp), m)
    b2t4 = get_buffer_t(4, v - vp, 0.5 * ( v + vp), 0.5 * ( v + vp), m)
    b2u4 = get_buffer_u(4, 0.0, vp, v, m)

    # compute value
    inner = get_Γ_comp(1, site, b1s, b1t, b1u, r, a, apply_flags_u1_dm) / (2.0 * pi)^2

    if site == 1
        vzz    = get_Γ_comp(2, site, b2s2, b2t2, b2u2, r, a, apply_flags_u1_dm)
        vdd    = get_Γ_comp(4, site, b2s4, b2t4, b2u4, r, a, apply_flags_u1_dm)
        inner += (vzz - vdd) / (2.0 * (2.0 * pi)^2)
    end

    inner *= get_G(Λ, v, m, a)^2 * get_G(Λ, vp, m, a)^2

    return inner
end

# outer kernel for double integral (xx correlation)
function outer_kernel_xx(
    Λ     :: Float64,
    site  :: Int64,
    v     :: Float64,
    r     :: Reduced_lattice,
    m     :: Mesh_u1_dm,
    a     :: Action_u1_dm,
    χ_tol :: NTuple{2, Float64}
    )     :: Float64

    # define integrand
    integrand = vp -> inner_kernel_xx(Λ, site, v, vp, r, m, a)

    # compute value
    outer = -quadgk(integrand, -Inf, -2.0 * Λ, 2.0 * Λ, Inf, atol = χ_tol[1], rtol = χ_tol[2])[1]

    if site == 1
        outer += get_G(Λ, v, m, a)^2 / (4.0 * pi)
    end

    return outer
end





# inner kernel for double integral (zz correlation)
function inner_kernel_zz(
    Λ    :: Float64,
    site :: Int64,
    v    :: Float64,
    vp   :: Float64,
    r    :: Reduced_lattice,
    m    :: Mesh_u1_dm,
    a    :: Action_u1_dm
    )    :: Float64

    # get buffers for non-local term
    b1s = get_buffer_s(2, v + vp, 0.5 * (v - vp), 0.5 * (-v + vp), m)
    b1t = get_buffer_t(2, 0.0, v, vp, m)
    b1u = get_buffer_u(2, v - vp, 0.5 * (v + vp), 0.5 * ( v + vp), m)

    # get buffers for local term
    b2s1 = get_buffer_s(1, v + vp, 0.5 * (-v + vp), 0.5 * (-v + vp), m)
    b2t1 = get_buffer_t(1, v - vp, 0.5 * ( v + vp), 0.5 * ( v + vp), m)
    b2u1 = get_buffer_u(1, 0.0, vp, v, m)

    b2s2 = get_buffer_s(2, v + vp, 0.5 * (-v + vp), 0.5 * (-v + vp), m)
    b2t2 = get_buffer_t(2, v - vp, 0.5 * ( v + vp), 0.5 * ( v + vp), m)
    b2u2 = get_buffer_u(2, 0.0, vp, v, m)

    b2s4 = get_buffer_s(4, v + vp, 0.5 * (-v + vp), 0.5 * (-v + vp), m)
    b2t4 = get_buffer_t(4, v - vp, 0.5 * ( v + vp), 0.5 * ( v + vp), m)
    b2u4 = get_buffer_u(4, 0.0, vp, v, m)

    # compute value
    inner = get_Γ_comp(2, site, b1s, b1t, b1u, r, a, apply_flags_u1_dm) / (2.0 * pi)^2

    if site == 1
        vxx    = get_Γ_comp(1, site, b2s1, b2t1, b2u1, r, a, apply_flags_u1_dm)
        vzz    = get_Γ_comp(2, site, b2s2, b2t2, b2u2, r, a, apply_flags_u1_dm)
        vdd    = get_Γ_comp(4, site, b2s4, b2t4, b2u4, r, a, apply_flags_u1_dm)
        inner += (2.0 * vxx - vzz - vdd) / (2.0 * (2.0 * pi)^2)
    end

    inner *= get_G(Λ, v, m, a)^2 * get_G(Λ, vp, m, a)^2

    return inner
end

# outer kernel for double integral (zz correlation)
function outer_kernel_zz(
    Λ     :: Float64,
    site  :: Int64,
    v     :: Float64,
    r     :: Reduced_lattice,
    m     :: Mesh_u1_dm,
    a     :: Action_u1_dm,
    χ_tol :: NTuple{2, Float64}
    )     :: Float64

    # define integrand
    integrand = vp -> inner_kernel_zz(Λ, site, v, vp, r, m, a)

    # compute value
    outer = -quadgk(integrand, -Inf, -2.0 * Λ, 2.0 * Λ, Inf, atol = χ_tol[1], rtol = χ_tol[2])[1]

    if site == 1
        outer += get_G(Λ, v, m, a)^2 / (4.0 * pi)
    end

    return outer
end





# inner kernel for double integral (xy correlation)
function inner_kernel_xy(
    Λ    :: Float64,
    site :: Int64,
    v    :: Float64,
    vp   :: Float64,
    r    :: Reduced_lattice,
    m    :: Mesh_u1_dm,
    a    :: Action_u1_dm
    )    :: Float64

    # get buffers for non-local term
    b1s = get_buffer_s(3, v + vp, 0.5 * (v - vp), 0.5 * (-v + vp), m)
    b1t = get_buffer_t(3, 0.0, v, vp, m)
    b1u = get_buffer_u(3, v - vp, 0.5 * (v + vp), 0.5 * ( v + vp), m)

    # get buffers for local term
    b2s5 = get_buffer_s(5, v + vp, 0.5 * (-v + vp), 0.5 * (-v + vp), m)
    b2t5 = get_buffer_t(5, v - vp, 0.5 * ( v + vp), 0.5 * ( v + vp), m)
    b2u5 = get_buffer_u(5, 0.0, vp, v, m)

    b2s6 = get_buffer_s(6, v + vp, 0.5 * (-v + vp), 0.5 * (-v + vp), m)
    b2t6 = get_buffer_t(6, v - vp, 0.5 * ( v + vp), 0.5 * ( v + vp), m)
    b2u6 = get_buffer_u(6, 0.0, vp, v, m)

    # compute value
    inner = get_Γ_comp(3, site, b1s, b1t, b1u, r, a, apply_flags_u1_dm) / (2.0 * pi)^2

    if site == 1
        vzd    = get_Γ_comp(5, site, b2s5, b2t5, b2u5, r, a, apply_flags_u1_dm)
        vdz    = get_Γ_comp(6, site, b2s6, b2t6, b2u6, r, a, apply_flags_u1_dm)
        inner += (vzd - vdz) / (2.0 * (2.0 * pi)^2)
    end

    inner *= get_G(Λ, v, m, a)^2 * get_G(Λ, vp, m, a)^2

    return inner
end

# outer kernel for double integral (xy correlation)
function outer_kernel_xy(
    Λ     :: Float64,
    site  :: Int64,
    v     :: Float64,
    r     :: Reduced_lattice,
    m     :: Mesh_u1_dm,
    a     :: Action_u1_dm,
    χ_tol :: NTuple{2, Float64}
    )     :: Float64

    # define integrand
    integrand = vp -> inner_kernel_xy(Λ, site, v, vp, r, m, a)

    # compute value
    outer = -quadgk(integrand, -Inf, -2.0 * Λ, 2.0 * Λ, Inf, atol = χ_tol[1], rtol = χ_tol[2])[1]

    if site == 1
        outer += get_G(Λ, v, m, a)^2 / (4.0 * pi)
    end

    return outer
end





# compute correlations in real space
function compute_χ(
    Λ     :: Float64,
    r     :: Reduced_lattice,
    m     :: Mesh_u1_dm,
    a     :: Action_u1_dm,
    χ_tol :: NTuple{2, Float64}
    )     :: Vector{Vector{Float64}}

    # allocate output
    χ_xx = zeros(Float64, length(r.sites))
    χ_zz = zeros(Float64, length(r.sites))
    χ_xy = zeros(Float64, length(r.sites))

    @sync for i in eachindex(r.sites)
        Threads.@spawn begin
            # compute xx correlation
            integrand = v -> outer_kernel_xx(Λ, i, v, r, m, a, χ_tol)
            χ_xx[i]   = quadgk(integrand, -Inf, -2.0 * Λ, 2.0 * Λ, Inf, atol = χ_tol[1], rtol = χ_tol[2])[1]

            # compute zz correlation
            integrand = v -> outer_kernel_zz(Λ, i, v, r, m, a, χ_tol)
            χ_zz[i]   = quadgk(integrand, -Inf, -2.0 * Λ, 2.0 * Λ, Inf, atol = χ_tol[1], rtol = χ_tol[2])[1]

            # compute xy correlation
            integrand = v -> outer_kernel_xy(Λ, i, v, r, m, a, χ_tol)
            χ_xy[i]   = quadgk(integrand, -Inf, -2.0 * Λ, 2.0 * Λ, Inf, atol = χ_tol[1], rtol = χ_tol[2])[1]
        end
    end

    # wrap in array for generalization
    wrap = Vector{Float64}[χ_xx, χ_zz, χ_xy]

    return wrap
end