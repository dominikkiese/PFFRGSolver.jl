# inner kernel for double integral
function inner_kernel(
    Λ    :: Float64,
    site :: Int64,
    v    :: Float64,
    vp   :: Float64,
    r    :: Reduced_lattice,
    m    :: Mesh,
    a    :: Action_su2
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
    inner = (2.0 * a.S)^2 * get_Γ_comp(1, site, bs1, bt1, bu1, r, a, apply_flags_su2) / (2.0 * pi)^2

    if site == 1
        vs, vd  = get_Γ(site, bs2, bt2, bu2, r, a)
        inner  += (2.0 * a.S) * (vs - vd) / (2.0 * (2.0 * pi)^2)
    end

    inner *= get_G(Λ, v, m, a)^2 * get_G(Λ, vp, m, a)^2

    return inner
end

# outer kernel for double integral
function outer_kernel(
    Λ    :: Float64,
    site :: Int64,
    v    :: Float64,
    r    :: Reduced_lattice,
    m    :: Mesh,
    a    :: Action_su2
    )    :: Float64

    # define integrand
    integrand = vp -> inner_kernel(Λ, site, v, vp, r, m, a)

    # compute value
    outer = -quadgk(integrand, -Inf, -2.0 * Λ, 2.0 * Λ, Inf, atol = 1e-8, rtol = 1e-3)[1]

    if site == 1
        outer += (2.0 * a.S) * get_G(Λ, v, m, a)^2 / (4.0 * pi)
    end

    return outer
end

# compute isotropic spin-spin correlation in real space
function compute_χ(
    Λ   :: Float64,
    r   :: Reduced_lattice,
    m   :: Mesh,
    a   :: Action_su2
    )   :: Vector{Vector{Float64}}

    # allocate output
    χ = zeros(Float64, length(r.sites))

    @sync for i in eachindex(χ)
        Threads.@spawn begin
            integrand = v -> outer_kernel(Λ, i, v, r, m, a)
            χ[i]      = quadgk(integrand, -Inf, -2.0 * Λ, 2.0 * Λ, Inf, atol = 1e-8, rtol = 1e-3)[1]
        end
    end

    # wrap in array for generalization
    wrap = Vector{Float64}[χ]

    return wrap
end
