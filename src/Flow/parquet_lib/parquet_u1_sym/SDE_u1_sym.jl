# xx integration kernel of reduced s bubble
function compute_xx_kernel(
    Λ    :: Float64,
    v    :: Float64,
    site :: Int64,
    s    :: Float64,
    vs   :: Float64,
    vsp  :: Float64,
    r    :: Reduced_lattice,
    m    :: Mesh,
    a    :: Action_u1_sym
    )    :: Float64

    # get propagator
    p = -get_propagator(Λ, v + 0.5 * s, 0.5 * s - v, m, a)

    # get buffers for right vertex (left vertex is given by bare)
    bs = get_buffer_s(s, v, vsp, m)
    bt = get_buffer_t(-v - vsp, 0.5 * (s + v - vsp), 0.5 * (s - v + vsp), m)
    bu = get_buffer_u(v - vsp, 0.5 * (s + v + vsp), 0.5 * (s - v - vsp), m)

    # get left vertex
    v1xx = a.Γ[1].bare[site]
    v1zz = a.Γ[2].bare[site]
    v1DM = a.Γ[3].bare[site]
    v1dd = a.Γ[4].bare[site]
    v1zd = a.Γ[5].bare[site]
    v1dz = a.Γ[6].bare[site]

    # get right vertex
    v2xx, v2zz, v2DM, v2dd, v2zd, v2dz = get_Γ(site, bs, bt, bu, r, a)

    # compute xx
    Γxx = -p * (v1DM * v2dz - v1DM * v2zd + v1xx * v2dd + v1dd * v2xx - v1xx * v2zz - v1dz * v2DM + v1zd * v2DM - v1zz * v2xx)       

    return Γxx
end

# zz integration kernel of reduced s bubble
function compute_zz_kernel(
    Λ    :: Float64,
    v    :: Float64,
    site :: Int64,
    s    :: Float64,
    vs   :: Float64,
    vsp  :: Float64,
    r    :: Reduced_lattice,
    m    :: Mesh,
    a    :: Action_u1_sym
    )    :: Float64

    # get propagator
    p = -get_propagator(Λ, v + 0.5 * s, 0.5 * s - v, m, a)

    # get buffers for right vertex (left vertex is given by bare)
    bs = get_buffer_s(s, v, vsp, m)
    bt = get_buffer_t(-v - vsp, 0.5 * (s + v - vsp), 0.5 * (s - v + vsp), m)
    bu = get_buffer_u(v - vsp, 0.5 * (s + v + vsp), 0.5 * (s - v - vsp), m)

    # get left vertex
    v1xx = a.Γ[1].bare[site]
    v1zz = a.Γ[2].bare[site]
    v1DM = a.Γ[3].bare[site]
    v1dd = a.Γ[4].bare[site]
    v1zd = a.Γ[5].bare[site]
    v1dz = a.Γ[6].bare[site]

    # get right vertex
    v2xx, v2zz, v2DM, v2dd, v2zd, v2dz = get_Γ(site, bs, bt, bu, r, a)

    # compute zz
    Γzz = -p * (v1zz * v2dd + v1dd * v2zz - v1dz * v2zd - 2.0 * v1xx * v2xx - 2.0 * v1DM * v2DM - v1zd * v2dz)

    return Γzz
end

# dd integration kernel of reduced s bubble
function compute_dd_kernel(
    Λ    :: Float64,
    v    :: Float64,
    site :: Int64,
    s    :: Float64,
    vs   :: Float64,
    vsp  :: Float64,
    r    :: Reduced_lattice,
    m    :: Mesh,
    a    :: Action_u1_sym
    )    :: Float64

    # get propagator
    p = -get_propagator(Λ, v + 0.5 * s, 0.5 * s - v, m, a)

    # get buffers for right vertex (left vertex is given by bare)
    bs = get_buffer_s(s, v, vsp, m)
    bt = get_buffer_t(-v - vsp, 0.5 * (s + v - vsp), 0.5 * (s - v + vsp), m)
    bu = get_buffer_u(v - vsp, 0.5 * (s + v + vsp), 0.5 * (s - v - vsp), m)

    # get left vertex
    v1xx = a.Γ[1].bare[site]
    v1zz = a.Γ[2].bare[site]
    v1DM = a.Γ[3].bare[site]
    v1dd = a.Γ[4].bare[site]
    v1zd = a.Γ[5].bare[site]
    v1dz = a.Γ[6].bare[site]

    # get right vertex
    v2xx, v2zz, v2DM, v2dd, v2zd, v2dz = get_Γ(site, bs, bt, bu, r, a)

    # compute dd
    Γdd = -p * (-v1dz * v2dz + 2.0 * v1xx * v2xx + v1dd * v2dd + 2.0 * v1DM * v2DM - v1zd * v2zd + v1zz * v2zz)

    return Γdd
end





# compute xx component of reduced s bubble
function compute_reduced_bubble_xx(
    Λ     :: Float64,
    site  :: Int64,
    s     :: Float64,
    vs    :: Float64,
    vsp   :: Float64,
    r     :: Reduced_lattice,
    m     :: Mesh,
    a     :: Action_u1_sym,
    Σ_tol :: NTuple{2, Float64}
    )     :: Float64

    # define integrand
    integrand = v -> compute_xx_kernel(Λ, v, site, s, vs, vsp, r, m, a)

    # compute reduced bubble
    ref = Λ + 0.5 * s
    res = quadgk(integrand, -Inf, -2.0 * ref, 2.0 * ref, Inf, atol = Σ_tol[1], rtol = Σ_tol[2])[1]

    return res
end

# compute zz component of reduced s bubble
function compute_reduced_bubble_zz(
    Λ     :: Float64,
    site  :: Int64,
    s     :: Float64,
    vs    :: Float64,
    vsp   :: Float64,
    r     :: Reduced_lattice,
    m     :: Mesh,
    a     :: Action_u1_sym,
    Σ_tol :: NTuple{2, Float64}
    )     :: Float64

    # define integrand
    integrand = v -> compute_zz_kernel(Λ, v, site, s, vs, vsp, r, m, a)

    # compute reduced bubble
    ref = Λ + 0.5 * s
    res = quadgk(integrand, -Inf, -2.0 * ref, 2.0 * ref, Inf, atol = Σ_tol[1], rtol = Σ_tol[2])[1]

    return res
end

# compute dd component of reduced s bubble
function compute_reduced_bubble_dd(
    Λ     :: Float64,
    site  :: Int64,
    s     :: Float64,
    vs    :: Float64,
    vsp   :: Float64,
    r     :: Reduced_lattice,
    m     :: Mesh,
    a     :: Action_u1_sym,
    Σ_tol :: NTuple{2, Float64}
    )     :: Float64

    # define integrand
    integrand = v -> compute_dd_kernel(Λ, v, site, s, vs, vsp, r, m, a)

    # compute reduced bubble
    ref = Λ + 0.5 * s
    res = quadgk(integrand, -Inf, -2.0 * ref, 2.0 * ref, Inf, atol = Σ_tol[1], rtol = Σ_tol[2])[1]

    return res
end





# integration kernel for loop function
function compute_Σ_kernel(
    Λ     :: Float64,
    v     :: Float64,
    w     :: Float64,
    r     :: Reduced_lattice,
    m     :: Mesh,
    a     :: Action_u1_sym,
    Σ_tol :: NTuple{2, Float64}
    )     :: Float64

    # compute local vertices
    vxx = a.Γ[1].bare[1] + compute_reduced_bubble_xx(Λ, 1, v + w, 0.5 * (-v + w), 0.5 * (-v + w), r, m, a, Σ_tol)
    vzz = a.Γ[2].bare[1] + compute_reduced_bubble_zz(Λ, 1, v + w, 0.5 * (-v + w), 0.5 * (-v + w), r, m, a, Σ_tol)
    vdd = a.Γ[4].bare[1] + compute_reduced_bubble_dd(Λ, 1, v + w, 0.5 * (-v + w), 0.5 * (-v + w), r, m, a, Σ_tol)

    # compute local contributions
    val = 2.0 * vxx + vzz + vdd

    for j in eachindex(r.sites)
        # compute non-local vertices
        vdd = a.Γ[4].bare[j] + compute_reduced_bubble_dd(Λ, j, v + w, 0.5 * (-v + w), 0.5 * (v - w), r, m, a, Σ_tol)

        # compute non-local contributions
        val -= 2.0 * r.mult[j] * vdd
    end

    # multiply with full propagator
    val *= -get_G(Λ, v, m, a) / (2.0 * pi)

    return val
end