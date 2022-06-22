"""
    Action_z2_diag <: Action

Struct containing self energy and vertex components for diagonal Z2 symmetric models.
* `Σ :: Vector{Float64}` : negative imaginary part of the self energy
* `Γ :: Vector{Vertex}`  : Γxx, Γyy, Γzz, Γdd component of the full vertex
"""
struct Action_z2_diag <: Action
    Σ :: Vector{Float64}
    Γ :: Vector{Vertex}
end

# generate action_z2_diag dummy
function get_action_z2_diag_empty(
    r :: Reduced_lattice,
    m :: Mesh,
    ) :: Action_z2_diag

    # init self energy
    Σ = zeros(Float64, length(m.σ))

    # init vertices
    Γ = Vertex[get_vertex_empty(r, m) for i in 1 : 4]

    # build action
    a = Action_z2_diag(Σ, Γ)

    return a
end

# init action for diagonal z2 symmetry
function init_action!(
    l :: Lattice,
    r :: Reduced_lattice,
    a :: Action_z2_diag
    ) :: Nothing

    # init bare action for Γxx, Γyy and Γzz component
    ref_int = SVector{4, Int64}(0, 0, 0, 1)
    ref     = Site(ref_int, get_vec(ref_int, l.uc))

    for i in eachindex(r.sites)
        # get bond from lattice
        b = get_bond(ref, r.sites[i], l)

        # set Γxx bare according to spin exchange
        a.Γ[1].bare[i] = b.exchange[1, 1] / 4.0

        # set Γyy bare according to spin exchange
        a.Γ[2].bare[i] = b.exchange[2, 2] / 4.0

        # set Γzz bare according to spin exchange
        a.Γ[3].bare[i] = b.exchange[3, 3] / 4.0
    end

    return nothing
end

# add repulsion for diagonal z2 symmetry
function add_repulsion!(
    A :: Float64,
    a :: Action_z2_diag
    ) :: Nothing

    # add on-site level repulsion for Γxx
    a.Γ[1].bare[1] += A / 4.0

    # add on-site level repulsion for Γyy
    a.Γ[2].bare[1] += A / 4.0

    # add on-site level repulsion for Γzz
    a.Γ[3].bare[1] += A / 4.0

    return nothing
end





# helper function to disentangle flags during interpolation for z2_diag models
function apply_flags_z2_diag(
    b    :: Buffer,
    comp :: Int64
    )    :: Tuple{Float64, Int64}

    # clarify sign of contribution
    sgn = 1.0

    # -ξ(μ) = -1 for density component
    if comp == 4 && b.sgn_μ 
        sgn *= -1.0 
    end 

    # -ξ(ν) = -1 for density component
    if comp == 4 && b.sgn_ν
        sgn *= -1.0 
    end

    return sgn, comp 
end

# get all interpolated vertex components for z2_diag models
function get_Γ(
    site :: Int64,
    bs   :: Buffer,
    bt   :: Buffer,
    bu   :: Buffer,
    r    :: Reduced_lattice,
    a    :: Action_z2_diag
    ;
    ch_s :: Bool = true,
    ch_t :: Bool = true,
    ch_u :: Bool = true
    )    :: NTuple{4, Float64}

    Γxx = get_Γ_comp(1, site, bs, bt, bu, r, a, apply_flags_z2_diag, ch_s = ch_s, ch_t = ch_t, ch_u = ch_u)
    Γyy = get_Γ_comp(2, site, bs, bt, bu, r, a, apply_flags_z2_diag, ch_s = ch_s, ch_t = ch_t, ch_u = ch_u)
    Γzz = get_Γ_comp(3, site, bs, bt, bu, r, a, apply_flags_z2_diag, ch_s = ch_s, ch_t = ch_t, ch_u = ch_u)
    Γdd = get_Γ_comp(4, site, bs, bt, bu, r, a, apply_flags_z2_diag, ch_s = ch_s, ch_t = ch_t, ch_u = ch_u)

    return Γxx, Γyy, Γzz, Γdd
end

# get all interpolated vertex components for z2_diag models on all lattice sites
function get_Γ_avx!(
    r     :: Reduced_lattice,
    bs    :: Buffer,
    bt    :: Buffer,
    bu    :: Buffer,
    a     :: Action_z2_diag,
    temp  :: Array{Float64, 3},
    index :: Int64
    ;
    ch_s  :: Bool = true,
    ch_t  :: Bool = true,
    ch_u  :: Bool = true
    )     :: Nothing

    for comp in 1 : 4
        get_Γ_comp_avx!(comp, r, bs, bt, bu, a, apply_flags_z2_diag, view(temp, :, comp, index), ch_s = ch_s, ch_t = ch_t, ch_u = ch_u)
    end

    return nothing
end





# symmetrize full loop contribution and central part
function symmetrize!(
    r :: Reduced_lattice,
    a :: Action_z2_diag
    ) :: Nothing

    # get dimensions
    num_sites = size(a.Γ[1].ch_s.q2_1, 1)
    num_Ω     = size(a.Γ[1].ch_s.q2_1, 2)
    num_ν     = size(a.Γ[1].ch_s.q2_1, 3)

    # computation for q3
    for v in 1 : num_ν
        for vp in v + 1 : num_ν
            @turbo for w in 1 : num_Ω
                for i in 1 : num_sites
                    # get upper triangular matrix for (v, v') plane for s channel
                    a.Γ[1].ch_s.q3[i, w, v, vp] = a.Γ[1].ch_s.q3[r.exchange[i], w, vp, v]
                    a.Γ[2].ch_s.q3[i, w, v, vp] = a.Γ[2].ch_s.q3[r.exchange[i], w, vp, v]
                    a.Γ[3].ch_s.q3[i, w, v, vp] = a.Γ[3].ch_s.q3[r.exchange[i], w, vp, v]
                    a.Γ[4].ch_s.q3[i, w, v, vp] = a.Γ[4].ch_s.q3[r.exchange[i], w, vp, v]

                    # get upper triangular matrix for (v, v') plane for t channel
                    a.Γ[1].ch_t.q3[i, w, v, vp] = a.Γ[1].ch_t.q3[r.exchange[i], w, vp, v]
                    a.Γ[2].ch_t.q3[i, w, v, vp] = a.Γ[2].ch_t.q3[r.exchange[i], w, vp, v]
                    a.Γ[3].ch_t.q3[i, w, v, vp] = a.Γ[3].ch_t.q3[r.exchange[i], w, vp, v]
                    a.Γ[4].ch_t.q3[i, w, v, vp] = a.Γ[4].ch_t.q3[r.exchange[i], w, vp, v]

                    # get upper triangular matrix for (v, v') plane for u channel
                    a.Γ[1].ch_u.q3[i, w, v, vp] = a.Γ[1].ch_u.q3[i, w, vp, v]
                    a.Γ[2].ch_u.q3[i, w, v, vp] = a.Γ[2].ch_u.q3[i, w, vp, v]
                    a.Γ[3].ch_u.q3[i, w, v, vp] = a.Γ[3].ch_u.q3[i, w, vp, v]
                    a.Γ[4].ch_u.q3[i, w, v, vp] = a.Γ[4].ch_u.q3[i, w, vp, v]
                end
            end
        end
    end

    return nothing
end

# symmetrized addition for left part (right part symmetric to left part)
function symmetrize_add_to!(
    r   :: Reduced_lattice,
    a_l :: Action_z2_diag,
    a   :: Action_z2_diag
    )   :: Nothing

    # get dimensions
    num_sites = size(a_l.Γ[1].ch_s.q2_1, 1)
    num_Ω     = size(a_l.Γ[1].ch_s.q2_1, 2)
    num_ν     = size(a_l.Γ[1].ch_s.q2_1, 3)

    # computation for q1
    @turbo for w in 1 : num_Ω
        for i in 1 : num_sites
            # add q1 to s channel (right part from v <-> v' exchange)
            a.Γ[1].ch_s.q1[i, w] += a_l.Γ[1].ch_s.q1[i, w] + a_l.Γ[1].ch_s.q1[r.exchange[i], w]
            a.Γ[2].ch_s.q1[i, w] += a_l.Γ[2].ch_s.q1[i, w] + a_l.Γ[2].ch_s.q1[r.exchange[i], w]
            a.Γ[3].ch_s.q1[i, w] += a_l.Γ[3].ch_s.q1[i, w] + a_l.Γ[3].ch_s.q1[r.exchange[i], w]
            a.Γ[4].ch_s.q1[i, w] += a_l.Γ[4].ch_s.q1[i, w] + a_l.Γ[4].ch_s.q1[r.exchange[i], w]

            # add q1 to t channel (right part from v <-> v' exchange)
            a.Γ[1].ch_t.q1[i, w] += a_l.Γ[1].ch_t.q1[i, w] + a_l.Γ[1].ch_t.q1[r.exchange[i], w]
            a.Γ[2].ch_t.q1[i, w] += a_l.Γ[2].ch_t.q1[i, w] + a_l.Γ[2].ch_t.q1[r.exchange[i], w]
            a.Γ[3].ch_t.q1[i, w] += a_l.Γ[3].ch_t.q1[i, w] + a_l.Γ[3].ch_t.q1[r.exchange[i], w]
            a.Γ[4].ch_t.q1[i, w] += a_l.Γ[4].ch_t.q1[i, w] + a_l.Γ[4].ch_t.q1[r.exchange[i], w]

            # add q1 to u channel (right part from v <-> v' exchange)
            a.Γ[1].ch_u.q1[i, w] += a_l.Γ[1].ch_u.q1[i, w] + a_l.Γ[1].ch_u.q1[i, w]
            a.Γ[2].ch_u.q1[i, w] += a_l.Γ[2].ch_u.q1[i, w] + a_l.Γ[2].ch_u.q1[i, w]
            a.Γ[3].ch_u.q1[i, w] += a_l.Γ[3].ch_u.q1[i, w] + a_l.Γ[3].ch_u.q1[i, w]
            a.Γ[4].ch_u.q1[i, w] += a_l.Γ[4].ch_u.q1[i, w] + a_l.Γ[4].ch_u.q1[i, w]
        end
    end

    # computation for q2_1 and q2_2
    @turbo for v in 1 : num_ν
        for w in 1 : num_Ω
            for i in 1 : num_sites
                # add q2_1 and q2_2 to s channel (right part from v <-> v' exchange)
                a.Γ[1].ch_s.q2_1[i, w, v] += a_l.Γ[1].ch_s.q2_1[i, w, v] + a_l.Γ[1].ch_s.q2_2[r.exchange[i], w, v]
                a.Γ[2].ch_s.q2_1[i, w, v] += a_l.Γ[2].ch_s.q2_1[i, w, v] + a_l.Γ[2].ch_s.q2_2[r.exchange[i], w, v]
                a.Γ[3].ch_s.q2_1[i, w, v] += a_l.Γ[3].ch_s.q2_1[i, w, v] + a_l.Γ[3].ch_s.q2_2[r.exchange[i], w, v]
                a.Γ[4].ch_s.q2_1[i, w, v] += a_l.Γ[4].ch_s.q2_1[i, w, v] + a_l.Γ[4].ch_s.q2_2[r.exchange[i], w, v]
                a.Γ[1].ch_s.q2_2[i, w, v] += a_l.Γ[1].ch_s.q2_2[i, w, v] + a_l.Γ[1].ch_s.q2_1[r.exchange[i], w, v]
                a.Γ[2].ch_s.q2_2[i, w, v] += a_l.Γ[2].ch_s.q2_2[i, w, v] + a_l.Γ[2].ch_s.q2_1[r.exchange[i], w, v]
                a.Γ[3].ch_s.q2_2[i, w, v] += a_l.Γ[3].ch_s.q2_2[i, w, v] + a_l.Γ[3].ch_s.q2_1[r.exchange[i], w, v]
                a.Γ[4].ch_s.q2_2[i, w, v] += a_l.Γ[4].ch_s.q2_2[i, w, v] + a_l.Γ[4].ch_s.q2_1[r.exchange[i], w, v]

                # add q2_1 and q2_2 to t channel (right part from v <-> v' exchange)
                a.Γ[1].ch_t.q2_1[i, w, v] += a_l.Γ[1].ch_t.q2_1[i, w, v] + a_l.Γ[1].ch_t.q2_2[r.exchange[i], w, v]
                a.Γ[2].ch_t.q2_1[i, w, v] += a_l.Γ[2].ch_t.q2_1[i, w, v] + a_l.Γ[2].ch_t.q2_2[r.exchange[i], w, v]
                a.Γ[3].ch_t.q2_1[i, w, v] += a_l.Γ[3].ch_t.q2_1[i, w, v] + a_l.Γ[3].ch_t.q2_2[r.exchange[i], w, v]
                a.Γ[4].ch_t.q2_1[i, w, v] += a_l.Γ[4].ch_t.q2_1[i, w, v] + a_l.Γ[4].ch_t.q2_2[r.exchange[i], w, v]
                a.Γ[1].ch_t.q2_2[i, w, v] += a_l.Γ[1].ch_t.q2_2[i, w, v] + a_l.Γ[1].ch_t.q2_1[r.exchange[i], w, v]
                a.Γ[2].ch_t.q2_2[i, w, v] += a_l.Γ[2].ch_t.q2_2[i, w, v] + a_l.Γ[2].ch_t.q2_1[r.exchange[i], w, v]
                a.Γ[3].ch_t.q2_2[i, w, v] += a_l.Γ[3].ch_t.q2_2[i, w, v] + a_l.Γ[3].ch_t.q2_1[r.exchange[i], w, v]
                a.Γ[4].ch_t.q2_2[i, w, v] += a_l.Γ[4].ch_t.q2_2[i, w, v] + a_l.Γ[4].ch_t.q2_1[r.exchange[i], w, v]

                # add q2_1 and q2_2 to u channel (right part from v <-> v' exchange)
                a.Γ[1].ch_u.q2_1[i, w, v] += a_l.Γ[1].ch_u.q2_1[i, w, v] + a_l.Γ[1].ch_u.q2_2[i, w, v]
                a.Γ[2].ch_u.q2_1[i, w, v] += a_l.Γ[2].ch_u.q2_1[i, w, v] + a_l.Γ[2].ch_u.q2_2[i, w, v]
                a.Γ[3].ch_u.q2_1[i, w, v] += a_l.Γ[3].ch_u.q2_1[i, w, v] + a_l.Γ[3].ch_u.q2_2[i, w, v]
                a.Γ[4].ch_u.q2_1[i, w, v] += a_l.Γ[4].ch_u.q2_1[i, w, v] + a_l.Γ[4].ch_u.q2_2[i, w, v]
                a.Γ[1].ch_u.q2_2[i, w, v] += a_l.Γ[1].ch_u.q2_2[i, w, v] + a_l.Γ[1].ch_u.q2_1[i, w, v]
                a.Γ[2].ch_u.q2_2[i, w, v] += a_l.Γ[2].ch_u.q2_2[i, w, v] + a_l.Γ[2].ch_u.q2_1[i, w, v]
                a.Γ[3].ch_u.q2_2[i, w, v] += a_l.Γ[3].ch_u.q2_2[i, w, v] + a_l.Γ[3].ch_u.q2_1[i, w, v]
                a.Γ[4].ch_u.q2_2[i, w, v] += a_l.Γ[4].ch_u.q2_2[i, w, v] + a_l.Γ[4].ch_u.q2_1[i, w, v]
            end
        end
    end

    # computation for q3
    @turbo for vp in 1 : num_ν
        for v in 1 : num_ν
            for w in 1 : num_Ω
                for i in 1 : num_sites
                    # add q3 to s channel (right part from v <-> v' exchange)
                    a.Γ[1].ch_s.q3[i, w, v, vp] += a_l.Γ[1].ch_s.q3[i, w, v, vp] + a_l.Γ[1].ch_s.q3[r.exchange[i], w, vp, v]
                    a.Γ[2].ch_s.q3[i, w, v, vp] += a_l.Γ[2].ch_s.q3[i, w, v, vp] + a_l.Γ[2].ch_s.q3[r.exchange[i], w, vp, v]
                    a.Γ[3].ch_s.q3[i, w, v, vp] += a_l.Γ[3].ch_s.q3[i, w, v, vp] + a_l.Γ[3].ch_s.q3[r.exchange[i], w, vp, v]
                    a.Γ[4].ch_s.q3[i, w, v, vp] += a_l.Γ[4].ch_s.q3[i, w, v, vp] + a_l.Γ[4].ch_s.q3[r.exchange[i], w, vp, v]

                    # add q3 to t channel (right part from v <-> v' exchange)
                    a.Γ[1].ch_t.q3[i, w, v, vp] += a_l.Γ[1].ch_t.q3[i, w, v, vp] + a_l.Γ[1].ch_t.q3[r.exchange[i], w, vp, v]
                    a.Γ[2].ch_t.q3[i, w, v, vp] += a_l.Γ[2].ch_t.q3[i, w, v, vp] + a_l.Γ[2].ch_t.q3[r.exchange[i], w, vp, v]
                    a.Γ[3].ch_t.q3[i, w, v, vp] += a_l.Γ[3].ch_t.q3[i, w, v, vp] + a_l.Γ[3].ch_t.q3[r.exchange[i], w, vp, v]
                    a.Γ[4].ch_t.q3[i, w, v, vp] += a_l.Γ[4].ch_t.q3[i, w, v, vp] + a_l.Γ[4].ch_t.q3[r.exchange[i], w, vp, v]

                    # add q3 to u channel (right part from v <-> v' exchange)
                    a.Γ[1].ch_u.q3[i, w, v, vp] += a_l.Γ[1].ch_u.q3[i, w, v, vp] + a_l.Γ[1].ch_u.q3[i, w, vp, v]
                    a.Γ[2].ch_u.q3[i, w, v, vp] += a_l.Γ[2].ch_u.q3[i, w, v, vp] + a_l.Γ[2].ch_u.q3[i, w, vp, v]
                    a.Γ[3].ch_u.q3[i, w, v, vp] += a_l.Γ[3].ch_u.q3[i, w, v, vp] + a_l.Γ[3].ch_u.q3[i, w, vp, v]
                    a.Γ[4].ch_u.q3[i, w, v, vp] += a_l.Γ[4].ch_u.q3[i, w, v, vp] + a_l.Γ[4].ch_u.q3[i, w, vp, v]
                end
            end
        end
    end

    return nothing
end