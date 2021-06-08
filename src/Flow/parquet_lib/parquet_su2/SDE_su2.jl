# spin integration kernel of reduced s bubble
function compute_spin_kernel(
    Λ    :: Float64,
    v    :: Float64,
    site :: Int64,
    s    :: Float64,
    vs   :: Float64,
    vsp  :: Float64,
    r    :: Reduced_lattice,
    m    :: Mesh,
    a    :: Action_su2
    )    :: Float64

    # get propagator
    p = -get_propagator(Λ, v + 0.5 * s, 0.5 * s - v, m, a)

    # get buffers for right vertex (left vertex is given by bare)
    bs = get_buffer_s(s, v, vsp, m)
    bt = get_buffer_t(-v - vsp, 0.5 * (s + v - vsp), 0.5 * (s - v + vsp), m)
    bu = get_buffer_u(v - vsp, 0.5 * (s + v + vsp), 0.5 * (s - v - vsp), m)

    # get left vertex
    v1s = a.Γ[1].bare[site]
    v1d = a.Γ[2].bare[site]

    # get right vertex
    v2s, v2d = get_Γ(site, bs, bt, bu, r, a)

    # compute spin
    Γs = -p * (-2.0 * v1s * v2s + v1s * v2d + v1d * v2s)

    return Γs
end

# density integration kernel of reduced s bubble
function compute_dens_kernel(
    Λ    :: Float64,
    v    :: Float64,
    site :: Int64,
    s    :: Float64,
    vs   :: Float64,
    vsp  :: Float64,
    r    :: Reduced_lattice,
    m    :: Mesh,
    a    :: Action_su2
    )    :: Float64

    # get propagator
    p = -get_propagator(Λ, v + 0.5 * s, 0.5 * s - v, m, a)

    # get buffers for right vertex (left vertex is given by bare)
    bs = get_buffer_s(s, v, vsp, m)
    bt = get_buffer_t(-v - vsp, 0.5 * (s + v - vsp), 0.5 * (s - v + vsp), m)
    bu = get_buffer_u(v - vsp, 0.5 * (s + v + vsp), 0.5 * (s - v - vsp), m)

    # get left vertex
    v1s = a.Γ[1].bare[site]
    v1d = a.Γ[2].bare[site]

    # get right vertex
    v2s, v2d = get_Γ(site, bs, bt, bu, r, a)

    # compute density
    Γd = -p * (3.0 * v1s * v2s + v1d * v2d)

    return Γd
end





# compute spin component of reduced s bubble
function compute_reduced_bubble_spin(
    Λ    :: Float64,
    site :: Int64,
    s    :: Float64,
    vs   :: Float64,
    vsp  :: Float64,
    r    :: Reduced_lattice,
    m    :: Mesh,
    a    :: Action_su2
    )    :: Float64

    # define integrand
    integrand = v -> compute_spin_kernel(Λ, v, site, s, vs, vsp, r, m, a)

    # compute reduced bubble
    ref = Λ + 0.5 * s
    res = quadgk(integrand, -Inf, -2.0 * ref, 2.0 * ref, Inf, atol = 1e-8, rtol = 1e-5)[1]

    return res
end

# compute density component of reduced s bubble
function compute_reduced_bubble_dens(
    Λ    :: Float64,
    site :: Int64,
    s    :: Float64,
    vs   :: Float64,
    vsp  :: Float64,
    r    :: Reduced_lattice,
    m    :: Mesh,
    a    :: Action_su2
    )    :: Float64

    # define integrand
    integrand = v -> compute_dens_kernel(Λ, v, site, s, vs, vsp, r, m, a)

    # compute reduced bubble
    ref = Λ + 0.5 * s
    res = quadgk(integrand, -Inf, -2.0 * ref, 2.0 * ref, Inf, atol = 1e-8, rtol = 1e-5)[1]

    return res
end





# integration kernel for loop function
function compute_Σ_kernel(
    Λ  :: Float64,
    v  :: Float64,
    w  :: Float64,
    r  :: Reduced_lattice,
    m  :: Mesh,
    a  :: Action_su2
    )  :: Float64

    # compute local vertices
    vs = a.Γ[1].bare[1] + compute_reduced_bubble_spin(Λ, 1, v + w, 0.5 * (-v + w), 0.5 * (-v + w), r, m, a)
    vd = a.Γ[2].bare[1] + compute_reduced_bubble_dens(Λ, 1, v + w, 0.5 * (-v + w), 0.5 * (-v + w), r, m, a)

    # compute local contributions
    val = 3.0 * vs + vd

    for j in eachindex(r.sites)
        # compute non-local vertices
        vd = a.Γ[2].bare[j] + compute_reduced_bubble_dens(Λ, j, v + w, 0.5 * (-v + w), 0.5 * (v - w), r, m, a)

        # compute non-local contributions
        val -= 2.0 * r.mult[j] * (2.0 * a.S) * vd
    end

    # multiply with full propagator
    val *= -get_G(Λ, v, m, a) / (2.0 * pi)

    return val
end
