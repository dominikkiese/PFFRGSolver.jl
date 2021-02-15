# inner kernel for double integral
function inner_kernel(
    Λ    :: Float64,
    site :: Int64,
    v    :: Float64,
    vp   :: Float64,
    r    :: reduced_lattice,
    m    :: mesh,
    a    :: action_sun
    )    :: Float64

    # get buffers for non-local term
    bs1 = get_buffer_sun_s(v + vp, 0.5 * (v - vp), 0.5 * (-v + vp), m)
    bt1 = get_buffer_sun_t(0.0, v, vp, m)
    bu1 = get_buffer_sun_u(v - vp, 0.5 * (v + vp), 0.5 * (v + vp), m)

    # get buffers for local term
    bs2 = get_buffer_sun_s(v + vp, 0.5 * (-v + vp), 0.5 * (-v + vp), m)
    bt2 = get_buffer_sun_t(v - vp, 0.5 * (v + vp), 0.5 * (v + vp), m)
    bu2 = get_buffer_sun_u(0.0, vp, v, m)

    # compute value
    inner = a.S^2 * get_spin(site, bs1, bt1, bu1, r, a) / (2.0 * pi)^2

    if site == 1
        vs, vd  = get_Γ(site, bs2, bt2, bu2, r, a)
        inner  += a.S * (vs / (2.0 * a.N) - vd) / (2.0 * pi)^2
    end

    inner *= get_G(Λ, v, m, a)^2 * get_G(Λ, vp, m, a)^2

    return inner
end

# outer kernel for double integral
function outer_kernel(
    Λ    :: Float64,
    site :: Int64,
    v    :: Float64,
    r    :: reduced_lattice,
    m    :: mesh,
    a    :: action_sun
    )    :: Float64

    # define integrand
    integrand = vp -> inner_kernel(Λ, site, v, vp, r, m, a)

    # compute value
    outer = -quadgk(integrand, -Inf, Inf, atol = 1e-10, rtol = 1e-3)[1]

    if site == 1
        outer += a.S * get_G(Λ, v, m, a)^2 / (2.0 * pi) 
    end

    return outer
end

# compute isotropic spin-spin correlation in real space
function compute_χ(
    Λ   :: Float64,
    r   :: reduced_lattice,
    m   :: mesh,
    a   :: action_sun
    )   :: Vector{Vector{Float64}}

    # allocate output
    χ = zeros(Float64, length(r.sites))

    @sync for i in eachindex(χ)
        Threads.@spawn begin
            integrand = v -> outer_kernel(Λ, i, v, r, m, a)
            χ[i]      = quadgk(integrand, -Inf, Inf, atol = 1e-10, rtol = 1e-3)[1]
        end
    end

    # wrap in array for generalization 
    wrap = Vector{Float64}[χ]

    return wrap
end