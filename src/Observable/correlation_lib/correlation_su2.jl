# inner kernel for double integral
function inner_kernel(
    Λ    :: Float64,
    site :: Int64,
    v    :: Float64,
    vp   :: Float64,
    r    :: Reduced_lattice,
    m    :: Mesh_su2,
    a    :: Action_su2
    )    :: Float64

    # get buffers for non-local term
    b1s = get_buffer_s(1, v + vp, 0.5 * (v - vp), 0.5 * (-v + vp), m)
    b1t = get_buffer_t(1, 0.0, v, vp, m)
    b1u = get_buffer_u(1, v - vp, 0.5 * (v + vp), 0.5 * ( v + vp), m)

    # get buffers for local term
    b2s = get_buffers_s(v + vp, 0.5 * (-v + vp), 0.5 * (-v + vp), m)
    b2t = get_buffers_t(v - vp, 0.5 * ( v + vp), 0.5 * ( v + vp), m)
    b2u = get_buffers_u(0.0, vp, v, m)

    # compute value
    inner = (2.0 * a.S)^2 * get_Γ_comp(1, site, b1s, b1t, b1u, r, a, apply_flags_su2) / (2.0 * pi)^2

    if site == 1
        vs, vd  = get_Γ(site, b2s, b2t, b2u, r, a)
        inner  += (2.0 * a.S) * (vs - vd) / (2.0 * (2.0 * pi)^2)
    end

    inner *= get_G(Λ, v, m, a)^2 * get_G(Λ, vp, m, a)^2

    return inner
end

# outer kernel for double integral
function outer_kernel(
    Λ     :: Float64,
    site  :: Int64,
    v     :: Float64,
    r     :: Reduced_lattice,
    m     :: Mesh_su2,
    a     :: Action_su2,
    χ_tol :: NTuple{2, Float64}
    )     :: Float64

    # define integrand
    integrand = vp -> inner_kernel(Λ, site, v, vp, r, m, a)

    # compute value
    outer = -quadgk(integrand, -Inf, -2.0 * Λ, 2.0 * Λ, Inf, atol = χ_tol[1], rtol = χ_tol[2])[1]

    if site == 1
        outer += (2.0 * a.S) * get_G(Λ, v, m, a)^2 / (4.0 * pi)
    end

    return outer
end

# compute isotropic spin-spin correlation in real space
function compute_χ(
    Λ     :: Float64,
    r     :: Reduced_lattice,
    m     :: Mesh_su2,
    a     :: Action_su2,
    χ_tol :: NTuple{2, Float64}
    )     :: Vector{Vector{Float64}}

    # allocate output
    χ = zeros(Float64, length(r.sites))

    @sync for i in eachindex(χ)
        Threads.@spawn begin
            integrand = v -> outer_kernel(Λ, i, v, r, m, a, χ_tol)
            χ[i]      = quadgk(integrand, -Inf, -2.0 * Λ, 2.0 * Λ, Inf, atol = χ_tol[1], rtol = χ_tol[2])[1]
        end
    end

    # wrap in array for generalization
    wrap = Vector{Float64}[χ]

    return wrap
end