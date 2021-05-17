"""
    Action_su2 <: Action

Struct containing self energy and vertex components for SU(2) symmetric models.
* `S :: Float64`         : total spin quantum number
* `Σ :: Vector{Float64}` : negative imaginary part of the self energy
* `Γ :: Vector{vertex}`  : spin and density component of the full vertex
"""
struct Action_su2 <: Action
    S :: Float64
    Σ :: Vector{Float64}
    Γ :: Vector{vertex}
end

# generate action_su2 dummy
function get_action_su2_empty(
    S :: Float64,
    r :: reduced_lattice,
    m :: mesh,
    ) :: Action_su2

    # init self energy
    Σ = zeros(Float64, length(m.σ))

    # init vertices
    Γ = vertex[get_vertex_empty(r, m), get_vertex_empty(r, m)]

    # build action
    a = Action_su2(S, Σ, Γ)

    return a
end

# init action for su2 symmetry
function init_action!(
    l :: lattice,
    r :: reduced_lattice,
    a :: Action_su2
    ) :: Nothing

    # init bare action for spin component
    ref_int = Int64[0, 0, 0, 1]
    ref     = site(ref_int, get_vec(ref_int, l.uc))

    for i in eachindex(r.sites)
        # get bond from lattice
        b = get_bond(ref, r.sites[i], l)

        # set bare according to spin exchange, normalize with 2S
        a.Γ[1].bare[i] = b.exchange[1, 1] / 4.0 / (2.0 * a.S)
    end

    return nothing
end





# get interpolated / extrapolated self energy
function get_Σ(
    w :: Float64,
    m :: mesh,
    a :: Action_su2
    ) :: Float64

    # init value
    val = 0.0

    # check if in bounds, otherwise extrapolate
    if abs(w) <= m.σ[end]
        p   = get_param(abs(w), m.σ)
        val = sign(w) * (p.lower_weight * a.Σ[p.lower_index] + p.upper_weight * a.Σ[p.upper_index])
    else
        val = m.σ[end] * a.Σ[end] / w
    end

    return val
end

# get interpolated spin component
function get_spin(
    site :: Int64,
    bs   :: buffer_su2,
    bt   :: buffer_su2,
    bu   :: buffer_su2,
    r    :: reduced_lattice,
    a    :: Action_su2
    ;
    ch_s :: Bool = true,
    ch_t :: Bool = true,
    ch_u :: Bool = true
    )    :: Float64

    # init with bare value
    val = a.Γ[1].bare[site]

    # add s channel
    if ch_s
        # check for site exchange
        site_s = site

        if bs.exchange_flag
            site_s = r.exchange[site_s]
        end

        # check for mapping to u channel
        if bs.map_flag
            val += get_vertex(site_s, bs, a.Γ[1], 3)
        else
            val += get_vertex(site_s, bs, a.Γ[1], 1)
        end
    end

    # add t channel
    if ch_t
        # check for site exchange
        site_t = site

        if bt.exchange_flag
            site_t = r.exchange[site_t]
        end

        val += get_vertex(site_t, bt, a.Γ[1], 2)
    end

    # add u channel
    if ch_u
        # check for site exchange
        site_u = site

        if bu.exchange_flag
            site_u = r.exchange[site_u]
        end

        # check for mapping to s channel
        if bu.map_flag
            val += get_vertex(site_u, bu, a.Γ[1], 1)
        else
            val += get_vertex(site_u, bu, a.Γ[1], 3)
        end
    end

    return val
end

# get interpolated spin component on all lattice sites
function get_spin_avx!(
    r    :: reduced_lattice,
    bs   :: buffer_su2,
    bt   :: buffer_su2,
    bu   :: buffer_su2,
    a    :: Action_su2,
    temp :: SubArray{Float64, 1, Array{Float64, 3}}
    ;
    ch_s :: Bool = true,
    ch_t :: Bool = true,
    ch_u :: Bool = true
    )    :: Nothing

    # init with bare value
    @avx temp .= a.Γ[1].bare

    # add s channel
    if ch_s
        # check for mapping to u channel
        if bs.map_flag
            get_vertex_avx!(r, bs, a.Γ[1], 3, temp, exchange = bs.exchange_flag)
        else
            get_vertex_avx!(r, bs, a.Γ[1], 1, temp, exchange = bs.exchange_flag)
        end
    end

    # add t channel
    if ch_t
        get_vertex_avx!(r, bt, a.Γ[1], 2, temp, exchange = bt.exchange_flag)
    end

    # add u channel
    if ch_u
        # check for mapping to s channel
        if bu.map_flag
            get_vertex_avx!(r, bu, a.Γ[1], 1, temp, exchange = bu.exchange_flag)
        else
            get_vertex_avx!(r, bu, a.Γ[1], 3, temp, exchange = bu.exchange_flag)
        end
    end

    return nothing
end

# get interpolated density component
function get_dens(
    site :: Int64,
    bs   :: buffer_su2,
    bt   :: buffer_su2,
    bu   :: buffer_su2,
    r    :: reduced_lattice,
    a    :: Action_su2
    ;
    ch_s :: Bool = true,
    ch_t :: Bool = true,
    ch_u :: Bool = true
    )    :: Float64

    # init with bare value
    val = a.Γ[2].bare[site]

    # add s channel
    if ch_s
        # check for site exchange
        site_s = site

        if bs.exchange_flag
            site_s = r.exchange[site_s]
        end

        # check for mapping to u channel
        if bs.map_flag
            val -= get_vertex(site_s, bs, a.Γ[2], 3)
        else
            val += get_vertex(site_s, bs, a.Γ[2], 1)
        end
    end

    # add t channel
    if ch_t
        # check for site exchange
        site_t = site

        if bt.exchange_flag
            site_t = r.exchange[site_t]
        end

        # check for sign
        if bt.map_flag
            val -= get_vertex(site_t, bt, a.Γ[2], 2)
        else
            val += get_vertex(site_t, bt, a.Γ[2], 2)
        end
    end

    # add u channel if wanted
    if ch_u
        # check for site exchange
        site_u = site

        if bu.exchange_flag
            site_u = r.exchange[site_u]
        end

        # check for mapping to s channel
        if bu.map_flag
            val -= get_vertex(site_u, bu, a.Γ[2], 1)
        else
            val += get_vertex(site_u, bu, a.Γ[2], 3)
        end
    end

    return val
end

# get interpolated density component on all lattice sites
function get_dens_avx!(
    r    :: reduced_lattice,
    bs   :: buffer_su2,
    bt   :: buffer_su2,
    bu   :: buffer_su2,
    a    :: Action_su2,
    temp :: SubArray{Float64, 1, Array{Float64, 3}}
    ;
    ch_s :: Bool = true,
    ch_t :: Bool = true,
    ch_u :: Bool = true
    )    :: Nothing

    # init with bare value
    @avx temp .= a.Γ[2].bare

    # add s channel
    if ch_s
        # check for mapping to u channel
        if bs.map_flag
            get_vertex_avx!(r, bs, a.Γ[2], 3, temp, exchange = bs.exchange_flag, sgn = -1.0)
        else
            get_vertex_avx!(r, bs, a.Γ[2], 1, temp, exchange = bs.exchange_flag)
        end
    end

    # add t channel
    if ch_t
        # check for sign
        if bt.map_flag
            get_vertex_avx!(r, bt, a.Γ[2], 2, temp, exchange = bt.exchange_flag, sgn = -1.0)
        else
            get_vertex_avx!(r, bt, a.Γ[2], 2, temp, exchange = bt.exchange_flag)
        end
    end

    # add u channel if wanted
    if ch_u
        # check for mapping to s channel
        if bu.map_flag
            get_vertex_avx!(r, bu, a.Γ[2], 1, temp, exchange = bu.exchange_flag, sgn = -1.0)
        else
            get_vertex_avx!(r, bu, a.Γ[2], 3, temp, exchange = bu.exchange_flag)
        end
    end

    return nothing
end

# get interpolated vertex components
function get_Γ(
    site :: Int64,
    bs   :: buffer_su2,
    bt   :: buffer_su2,
    bu   :: buffer_su2,
    r    :: reduced_lattice,
    a    :: Action_su2
    ;
    ch_s :: Bool = true,
    ch_t :: Bool = true,
    ch_u :: Bool = true
    )    :: NTuple{2, Float64}

    spin = get_spin(site, bs, bt, bu, r, a, ch_s = ch_s, ch_t = ch_t, ch_u = ch_u)
    dens = get_dens(site, bs, bt, bu, r, a, ch_s = ch_s, ch_t = ch_t, ch_u = ch_u)

    return spin, dens
end

# get interpolated vertex components on all lattice sites
function get_Γ_avx!(
    r     :: reduced_lattice,
    bs    :: buffer_su2,
    bt    :: buffer_su2,
    bu    :: buffer_su2,
    a     :: Action_su2,
    temp  :: Array{Float64, 3},
    index :: Int64
    ;
    ch_s :: Bool = true,
    ch_t :: Bool = true,
    ch_u :: Bool = true
    )    :: Nothing

    get_spin_avx!(r, bs, bt, bu, a, view(temp, :, 1, index), ch_s = ch_s, ch_t = ch_t, ch_u = ch_u)
    get_dens_avx!(r, bs, bt, bu, a, view(temp, :, 2, index), ch_s = ch_s, ch_t = ch_t, ch_u = ch_u)

    return nothing
end





# symmetrize full loop contribution and central part
function symmetrize!(
    r :: reduced_lattice,
    a :: Action_su2
    ) :: Nothing

    # get dimensions
    num_sites = size(a.Γ[1].ch_s.q2_1, 1)
    num_Ω     = size(a.Γ[1].ch_s.q2_1, 2)
    num_ν     = size(a.Γ[1].ch_s.q2_1, 3)

    # computation for q3
    for v in 1 : num_ν
        for vp in v + 1 : num_ν
            for w in 1 : num_Ω
                for i in 1 : num_sites
                    # get upper triangular matrix for (v, v') plane for s channel
                    a.Γ[1].ch_s.q3[i, w, v, vp] = a.Γ[1].ch_s.q3[r.exchange[i], w, vp, v]
                    a.Γ[2].ch_s.q3[i, w, v, vp] = a.Γ[2].ch_s.q3[r.exchange[i], w, vp, v]

                    # get upper triangular matrix for (v, v') plane for t channel
                    a.Γ[1].ch_t.q3[i, w, v, vp] = a.Γ[1].ch_t.q3[r.exchange[i], w, vp, v]
                    a.Γ[2].ch_t.q3[i, w, v, vp] = a.Γ[2].ch_t.q3[r.exchange[i], w, vp, v]

                    # get upper triangular matrix for (v, v') plane for u channel
                    a.Γ[1].ch_u.q3[i, w, v, vp] = a.Γ[1].ch_u.q3[i, w, vp, v]
                    a.Γ[2].ch_u.q3[i, w, v, vp] = a.Γ[2].ch_u.q3[i, w, vp, v]
                end
            end
        end
    end

    # set asymptotic limits
    limits!(a)

    return nothing
end

# symmetrized addition for left part (right part symmetric to left part)
function symmetrize_add_to!(
    r   :: reduced_lattice,
    a_l :: Action_su2,
    a   :: Action_su2
    )   :: Nothing

    # get dimensions
    num_sites = size(a_l.Γ[1].ch_s.q2_1, 1)
    num_Ω     = size(a_l.Γ[1].ch_s.q2_1, 2)
    num_ν     = size(a_l.Γ[1].ch_s.q2_1, 3)

    # computation for q3
    for vp in 1 : num_ν
        for v in 1 : num_ν
            for w in 1 : num_Ω
                for i in 1 : num_sites
                    # add q3 to s channel (right part from v <-> v' exchange)
                    a.Γ[1].ch_s.q3[i, w, v, vp] += a_l.Γ[1].ch_s.q3[i, w, v, vp] + a_l.Γ[1].ch_s.q3[r.exchange[i], w, vp, v]
                    a.Γ[2].ch_s.q3[i, w, v, vp] += a_l.Γ[2].ch_s.q3[i, w, v, vp] + a_l.Γ[2].ch_s.q3[r.exchange[i], w, vp, v]

                    # add q3 to t channel (right part from v <-> v' exchange)
                    a.Γ[1].ch_t.q3[i, w, v, vp] += a_l.Γ[1].ch_t.q3[i, w, v, vp] + a_l.Γ[1].ch_t.q3[r.exchange[i], w, vp, v]
                    a.Γ[2].ch_t.q3[i, w, v, vp] += a_l.Γ[2].ch_t.q3[i, w, v, vp] + a_l.Γ[2].ch_t.q3[r.exchange[i], w, vp, v]

                    # add q3 to u channel (right part from v <-> v' exchange)
                    a.Γ[1].ch_u.q3[i, w, v, vp] += a_l.Γ[1].ch_u.q3[i, w, v, vp] + a_l.Γ[1].ch_u.q3[i, w, vp, v]
                    a.Γ[2].ch_u.q3[i, w, v, vp] += a_l.Γ[2].ch_u.q3[i, w, v, vp] + a_l.Γ[2].ch_u.q3[i, w, vp, v]
                end
            end
        end
    end

    # set asymptotic limits
    limits!(a_l)
    limits!(a)

    return nothing
end