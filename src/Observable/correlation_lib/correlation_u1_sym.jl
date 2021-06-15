# inner kernel for double integral (xx correlation)
function inner_kernel_xx(
    Λ    :: Float64,
    site :: Int64,
    v    :: Float64,
    vp   :: Float64,
    r    :: Reduced_lattice,
    m    :: Mesh,
    a    :: Action_u1_sym
    )    :: Float64

    # get buffers for non-local term
    bs1 = get_buffer_s(v + vp, 0.5 * (v - vp), 0.5 * (-v + vp), m)
    bt1 = get_buffer_t(0.0, v, vp, m)
    bu1 = get_buffer_u(v - vp, 0.5 * (v + vp), 0.5 * (v + vp), m)

    # get buffers for local term
    bs2 = get_buffer_s(v + vp, 0.5 * (-v + vp), 0.5 * (-v + vp), m)
    bt2 = get_buffer_t(v - vp, 0.5 * (v + vp), 0.5 * (v + vp), m)
    bu2 = get_buffer_u(0.0, vp, v, m)

    # compute value
    inner = get_Γ_comp(1, site, bs1, bt1, bu1, r, a, apply_flags_u1_sym) / (2.0 * pi)^2

    if site == 1
        vzz    = get_Γ_comp(2, site, bs2, bt2, bu2, r, a, apply_flags_u1_sym)
        vdd    = get_Γ_comp(4, site, bs2, bt2, bu2, r, a, apply_flags_u1_sym)
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
    m     :: Mesh,
    a     :: Action_u1_sym,
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
    m    :: Mesh,
    a    :: Action_u1_sym
    )    :: Float64

    # get buffers for non-local term
    bs1 = get_buffer_s(v + vp, 0.5 * (v - vp), 0.5 * (-v + vp), m)
    bt1 = get_buffer_t(0.0, v, vp, m)
    bu1 = get_buffer_u(v - vp, 0.5 * (v + vp), 0.5 * (v + vp), m)

    # get buffers for local term
    bs2 = get_buffer_s(v + vp, 0.5 * (-v + vp), 0.5 * (-v + vp), m)
    bt2 = get_buffer_t(v - vp, 0.5 * (v + vp), 0.5 * (v + vp), m)
    bu2 = get_buffer_u(0.0, vp, v, m)

    # compute value
    inner = get_Γ_comp(2, site, bs1, bt1, bu1, r, a, apply_flags_u1_sym) / (2.0 * pi)^2

    if site == 1
        vxx    = get_Γ_comp(1, site, bs2, bt2, bu2, r, a, apply_flags_u1_sym)
        vzz    = get_Γ_comp(2, site, bs2, bt2, bu2, r, a, apply_flags_u1_sym)
        vdd    = get_Γ_comp(4, site, bs2, bt2, bu2, r, a, apply_flags_u1_sym)
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
    m     :: Mesh,
    a     :: Action_u1_sym,
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
    m    :: Mesh,
    a    :: Action_u1_sym
    )    :: Float64

    # get buffers for non-local term
    bs1 = get_buffer_s(v + vp, 0.5 * (v - vp), 0.5 * (-v + vp), m)
    bt1 = get_buffer_t(0.0, v, vp, m)
    bu1 = get_buffer_u(v - vp, 0.5 * (v + vp), 0.5 * (v + vp), m)

    # get buffers for local term
    bs2 = get_buffer_s(v + vp, 0.5 * (-v + vp), 0.5 * (-v + vp), m)
    bt2 = get_buffer_t(v - vp, 0.5 * (v + vp), 0.5 * (v + vp), m)
    bu2 = get_buffer_u(0.0, vp, v, m)

    # compute value
    inner = get_Γ_comp(3, site, bs1, bt1, bu1, r, a, apply_flags_u1_sym) / (2.0 * pi)^2

    if site == 1
        vzd    = get_Γ_comp(5, site, bs2, bt2, bu2, r, a, apply_flags_u1_sym)
        vdz    = get_Γ_comp(6, site, bs2, bt2, bu2, r, a, apply_flags_u1_sym)
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
    m     :: Mesh,
    a     :: Action_u1_sym,
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
    m     :: Mesh,
    a     :: Action_u1_sym,
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