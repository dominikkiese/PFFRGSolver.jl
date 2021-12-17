"""
    Action_su2 <: Action

Struct containing self energy and vertex components for SU(2) symmetric models.
* `S :: Float64`         : total spin quantum number
* `Σ :: Vector{Float64}` : negative imaginary part of the self energy
* `Γ :: Vector{Vertex}`  : spin and density component of the full vertex
"""
struct Action_su2 <: Action
    S :: Float64
    Σ :: Vector{Float64}
    Γ :: Vector{Vertex}
end

# generate action_su2 dummy
function get_action_su2_empty(
    S :: Float64,
    r :: Reduced_lattice,
    m :: Mesh,
    ) :: Action_su2

    # init self energy
    Σ = zeros(Float64, length(m.σ))

    # init vertices
    Γ = Vertex[get_vertex_empty(r, m) for i in 1 : 2]

    # build action
    a = Action_su2(S, Σ, Γ)

    return a
end

# init action for su2 symmetry
function init_action!(
    l :: Lattice,
    r :: Reduced_lattice,
    a :: Action_su2
    ) :: Nothing

    # init bare action for spin component
    ref_int = SVector{4, Int64}(0, 0, 0, 1)
    ref     = Site(ref_int, get_vec(ref_int, l.uc))

    for i in eachindex(r.sites)
        # get bond from lattice
        b = get_bond(ref, r.sites[i], l)

        # set bare according to spin exchange, normalize with 2S
        a.Γ[1].bare[i] = b.exchange[1, 1] / 4.0 / (2.0 * a.S)
    end

    return nothing
end

# set repulsion for su2 symmetry
function set_repulsion!(
    A :: Float64,
    a :: Action_su2
    ) ::Nothing

    # init bare action onsite with level repulsion for spin component
    a.Γ[1].bare[1] = A / 4.0 / (2.0 * a.S)

    return nothing
end



# helper function to disentangle flags during interpolation for su2 models
function apply_flags_su2(
    b    :: Buffer,
    comp :: Int64
    )    :: Tuple{Float64, Int64}

    # clarify sign of contribution
    sgn = 1.0

    # -ξ(μ) = -1 for density component
    if comp == 2 && b.sgn_μ 
        sgn *= -1.0 
    end 

    # -ξ(ν) = -1 for density component
    if comp == 2 && b.sgn_ν
        sgn *= -1.0 
    end

    return sgn, comp 
end

# get all interpolated vertex components for su2 models
function get_Γ(
    site :: Int64,
    bs   :: Buffer,
    bt   :: Buffer,
    bu   :: Buffer,
    r    :: Reduced_lattice,
    a    :: Action_su2
    ;
    ch_s :: Bool = true,
    ch_t :: Bool = true,
    ch_u :: Bool = true
    )    :: NTuple{2, Float64}

    spin = get_Γ_comp(1, site, bs, bt, bu, r, a, apply_flags_su2, ch_s = ch_s, ch_t = ch_t, ch_u = ch_u)
    dens = get_Γ_comp(2, site, bs, bt, bu, r, a, apply_flags_su2, ch_s = ch_s, ch_t = ch_t, ch_u = ch_u)

    return spin, dens
end

# get all interpolated vertex components for su2 models on all lattice sites
function get_Γ_avx!(
    r     :: Reduced_lattice,
    bs    :: Buffer,
    bt    :: Buffer,
    bu    :: Buffer,
    a     :: Action_su2,
    temp  :: Array{Float64, 3},
    index :: Int64
    ;
    ch_s  :: Bool = true,
    ch_t  :: Bool = true,
    ch_u  :: Bool = true
    )     :: Nothing

    for comp in 1 : 2
        get_Γ_comp_avx!(comp, r, bs, bt, bu, a, apply_flags_su2, view(temp, :, comp, index), ch_s = ch_s, ch_t = ch_t, ch_u = ch_u)
    end

    return nothing
end





# symmetrize full loop contribution and central part
function symmetrize!(
    r :: Reduced_lattice,
    a :: Action_su2
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

    return nothing
end

# symmetrized addition for left part (right part symmetric to left part)
function symmetrize_add_to!(
    r   :: Reduced_lattice,
    a_l :: Action_su2,
    a   :: Action_su2
    )   :: Nothing

    # get dimensions
    num_sites = size(a_l.Γ[1].ch_s.q2_1, 1)
    num_Ω     = size(a_l.Γ[1].ch_s.q2_1, 2)
    num_ν     = size(a_l.Γ[1].ch_s.q2_1, 3)

    # computation for q3
    @turbo for vp in 1 : num_ν
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

    return nothing
end