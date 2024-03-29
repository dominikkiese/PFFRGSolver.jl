"""
    Action_u1_dm <: Action

Struct containing self energy and vertex components for models with U(1) symmetric Dzyaloshinskii-Moriya interaction.
* `Σ :: Vector{Float64}` : negative imaginary part of the self energy
* `Γ :: Vector{Vertex}`  : Γxx, Γzz, ΓDM (i.e Γxy), Γdd, Γzd, Γdz component of the full vertex
"""
struct Action_u1_dm <: Action
    Σ :: Vector{Float64}
    Γ :: Vector{Vertex}
end

# generate action_u1_dm dummy
function get_action_u1_dm_empty(
    r :: Reduced_lattice,
    m :: Mesh,
    ) :: Action_u1_dm

    # init self energy
    Σ = zeros(Float64, length(m.σ))

    # init vertices
    Γ = Vertex[get_vertex_empty(r, m) for comp in 1 : 6]

    # build action
    a = Action_u1_dm(Σ, Γ)

    return a
end

# init action for symmetric u1 models
function init_action!(
    l :: Lattice,
    r :: Reduced_lattice,
    a :: Action_u1_dm
    ) :: Nothing

    # init bare action for Γxx, Γzz and ΓDM component
    ref_int = SVector{4, Int64}(0, 0, 0, 1)
    ref     = Site(ref_int, get_vec(ref_int, l.uc))

    for i in eachindex(r.sites)
        # get bond from lattice
        b = get_bond(ref, r.sites[i], l)

        # set Γxx bare according to spin exchange
        a.Γ[1].bare[i] = b.exchange[1, 1] / 4.0

        # set Γzz bare according to spin exchange
        a.Γ[2].bare[i] = b.exchange[3, 3] / 4.0

        # set ΓDM bare according to spin exchange
        a.Γ[3].bare[i] = b.exchange[1, 2] / 4.0
    end

    return nothing
end

# add repulsion for u1-dm symmetry
function add_repulsion!(
    A :: Float64,
    a :: Action_u1_dm
    ) :: Nothing

    # add on-site level repulsion for Γxx
    a.Γ[1].bare[1] += A / 4.0

    # add on-site level repulsion for Γzz
    a.Γ[2].bare[1] += A / 4.0

    return nothing
end





# helper function to disentangle flags during interpolation for u1 symmetric dm models
function apply_flags_u1_dm(
    b    :: Buffer,
    comp :: Int64
    )    :: Tuple{Float64, Int64}

    # clarify sign of contribution
    sgn = 1.0

    # ξ(μ) * ξ(ν) = -1 for Γzd & Γdz
    if comp in (5, 6) && b.sgn_μν
        sgn *= -1.0 
    end

    # -ξ(μ) = -1 for Γdd & Γdz
    if comp in (4, 6) && b.sgn_μ 
        sgn *= -1.0 
    end 

    # -ξ(ν) = -1 for Γdd & Γzd
    if comp in (4, 5) && b.sgn_ν
        sgn *= -1.0 
    end

    # clarify which component to interpolate
    if b.exchange_flag
        # ΓDM -> -ΓDM for spin exchange (since ΓDM = Γxy = -Γyx)
        if comp == 3
            sgn *= -1.0
        # Γzd -> Γdz for spin exchange
        elseif comp == 5
            comp = 6 
        # Γdz -> Γzd for spin exchange
        elseif comp == 6
            comp = 5 
        end 
    end 

    return sgn, comp 
end

# get all interpolated vertex components for u1 symmetric dm models
function get_Γ(
    site :: Int64,
    bs   :: Buffer,
    bt   :: Buffer,
    bu   :: Buffer,
    r    :: Reduced_lattice,
    a    :: Action_u1_dm
    ;
    ch_s :: Bool = true,
    ch_t :: Bool = true,
    ch_u :: Bool = true
    )    :: NTuple{6, Float64}

    Γxx = get_Γ_comp(1, site, bs, bt, bu, r, a, apply_flags_u1_dm, ch_s = ch_s, ch_t = ch_t, ch_u = ch_u)
    Γzz = get_Γ_comp(2, site, bs, bt, bu, r, a, apply_flags_u1_dm, ch_s = ch_s, ch_t = ch_t, ch_u = ch_u)
    ΓDM = get_Γ_comp(3, site, bs, bt, bu, r, a, apply_flags_u1_dm, ch_s = ch_s, ch_t = ch_t, ch_u = ch_u)
    Γdd = get_Γ_comp(4, site, bs, bt, bu, r, a, apply_flags_u1_dm, ch_s = ch_s, ch_t = ch_t, ch_u = ch_u)
    Γzd = get_Γ_comp(5, site, bs, bt, bu, r, a, apply_flags_u1_dm, ch_s = ch_s, ch_t = ch_t, ch_u = ch_u)
    Γdz = get_Γ_comp(6, site, bs, bt, bu, r, a, apply_flags_u1_dm, ch_s = ch_s, ch_t = ch_t, ch_u = ch_u)

    return Γxx, Γzz, ΓDM, Γdd, Γzd, Γdz
end

# get all interpolated vertex components for u1 symmetric dm models on all lattice sites
function get_Γ_avx!(
    r     :: Reduced_lattice,
    bs    :: Buffer,
    bt    :: Buffer,
    bu    :: Buffer,
    a     :: Action_u1_dm,
    temp  :: Array{Float64, 3},
    index :: Int64
    ;
    ch_s  :: Bool = true,
    ch_t  :: Bool = true,
    ch_u  :: Bool = true
    )     :: Nothing

    for comp in 1 : 6
        get_Γ_comp_avx!(comp, r, bs, bt, bu, a, apply_flags_u1_dm, view(temp, :, comp, index), ch_s = ch_s, ch_t = ch_t, ch_u = ch_u)
    end

    return nothing
end





# symmetrize full loop contribution and central part
function symmetrize!(
    r :: Reduced_lattice,
    a :: Action_u1_dm
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
                    a.Γ[1].ch_s.q3[i, w, v, vp] =  a.Γ[1].ch_s.q3[r.exchange[i], w, vp, v]
                    a.Γ[2].ch_s.q3[i, w, v, vp] =  a.Γ[2].ch_s.q3[r.exchange[i], w, vp, v]
                    a.Γ[3].ch_s.q3[i, w, v, vp] = -a.Γ[3].ch_s.q3[r.exchange[i], w, vp, v]
                    a.Γ[4].ch_s.q3[i, w, v, vp] =  a.Γ[4].ch_s.q3[r.exchange[i], w, vp, v]
                    a.Γ[5].ch_s.q3[i, w, v, vp] = -a.Γ[6].ch_s.q3[r.exchange[i], w, vp, v]
                    a.Γ[6].ch_s.q3[i, w, v, vp] = -a.Γ[5].ch_s.q3[r.exchange[i], w, vp, v]

                    # get upper triangular matrix for (v, v') plane for t channel
                    a.Γ[1].ch_t.q3[i, w, v, vp] =  a.Γ[1].ch_t.q3[r.exchange[i], w, vp, v]
                    a.Γ[2].ch_t.q3[i, w, v, vp] =  a.Γ[2].ch_t.q3[r.exchange[i], w, vp, v]
                    a.Γ[3].ch_t.q3[i, w, v, vp] = -a.Γ[3].ch_t.q3[r.exchange[i], w, vp, v]
                    a.Γ[4].ch_t.q3[i, w, v, vp] =  a.Γ[4].ch_t.q3[r.exchange[i], w, vp, v]
                    a.Γ[5].ch_t.q3[i, w, v, vp] = -a.Γ[6].ch_t.q3[r.exchange[i], w, vp, v]
                    a.Γ[6].ch_t.q3[i, w, v, vp] = -a.Γ[5].ch_t.q3[r.exchange[i], w, vp, v]

                    # get upper triangular matrix for (v, v') plane for u channel
                    a.Γ[1].ch_u.q3[i, w, v, vp] =  a.Γ[1].ch_u.q3[i, w, vp, v]
                    a.Γ[2].ch_u.q3[i, w, v, vp] =  a.Γ[2].ch_u.q3[i, w, vp, v]
                    a.Γ[3].ch_u.q3[i, w, v, vp] =  a.Γ[3].ch_u.q3[i, w, vp, v]
                    a.Γ[4].ch_u.q3[i, w, v, vp] =  a.Γ[4].ch_u.q3[i, w, vp, v]
                    a.Γ[5].ch_u.q3[i, w, v, vp] = -a.Γ[5].ch_u.q3[i, w, vp, v]
                    a.Γ[6].ch_u.q3[i, w, v, vp] = -a.Γ[6].ch_u.q3[i, w, vp, v]
                end
            end
        end
    end

    return nothing
end

# symmetrized addition for left part (right part symmetric to left part)
function symmetrize_add_to!(
    r   :: Reduced_lattice,
    a_l :: Action_u1_dm,
    a   :: Action_u1_dm
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
            a.Γ[3].ch_s.q1[i, w] += a_l.Γ[3].ch_s.q1[i, w] - a_l.Γ[3].ch_s.q1[r.exchange[i], w]
            a.Γ[4].ch_s.q1[i, w] += a_l.Γ[4].ch_s.q1[i, w] + a_l.Γ[4].ch_s.q1[r.exchange[i], w]
            a.Γ[5].ch_s.q1[i, w] += a_l.Γ[5].ch_s.q1[i, w] - a_l.Γ[6].ch_s.q1[r.exchange[i], w]
            a.Γ[6].ch_s.q1[i, w] += a_l.Γ[6].ch_s.q1[i, w] - a_l.Γ[5].ch_s.q1[r.exchange[i], w]

            # add q1 to t channel (right part from v <-> v' exchange)
            a.Γ[1].ch_t.q1[i, w] += a_l.Γ[1].ch_t.q1[i, w] + a_l.Γ[1].ch_t.q1[r.exchange[i], w]
            a.Γ[2].ch_t.q1[i, w] += a_l.Γ[2].ch_t.q1[i, w] + a_l.Γ[2].ch_t.q1[r.exchange[i], w]
            a.Γ[3].ch_t.q1[i, w] += a_l.Γ[3].ch_t.q1[i, w] - a_l.Γ[3].ch_t.q1[r.exchange[i], w]
            a.Γ[4].ch_t.q1[i, w] += a_l.Γ[4].ch_t.q1[i, w] + a_l.Γ[4].ch_t.q1[r.exchange[i], w]
            a.Γ[5].ch_t.q1[i, w] += a_l.Γ[5].ch_t.q1[i, w] - a_l.Γ[6].ch_t.q1[r.exchange[i], w]
            a.Γ[6].ch_t.q1[i, w] += a_l.Γ[6].ch_t.q1[i, w] - a_l.Γ[5].ch_t.q1[r.exchange[i], w]

            # add q1 to u channel (right part from v <-> v' exchange)
            a.Γ[1].ch_u.q1[i, w] += a_l.Γ[1].ch_u.q1[i, w] + a_l.Γ[1].ch_u.q1[i, w]
            a.Γ[2].ch_u.q1[i, w] += a_l.Γ[2].ch_u.q1[i, w] + a_l.Γ[2].ch_u.q1[i, w]
            a.Γ[3].ch_u.q1[i, w] += a_l.Γ[3].ch_u.q1[i, w] + a_l.Γ[3].ch_u.q1[i, w]
            a.Γ[4].ch_u.q1[i, w] += a_l.Γ[4].ch_u.q1[i, w] + a_l.Γ[4].ch_u.q1[i, w]
            a.Γ[5].ch_u.q1[i, w] += a_l.Γ[5].ch_u.q1[i, w] - a_l.Γ[5].ch_u.q1[i, w]
            a.Γ[6].ch_u.q1[i, w] += a_l.Γ[6].ch_u.q1[i, w] - a_l.Γ[6].ch_u.q1[i, w]
        end
    end

    # computation for q2_1 and q2_2
    @turbo for v in 1 : num_ν
        for w in 1 : num_Ω
            for i in 1 : num_sites
                # add q2_1 and q2_2 to s channel (right part from v <-> v' exchange)
                a.Γ[1].ch_s.q2_1[i, w, v] += a_l.Γ[1].ch_s.q2_1[i, w, v] + a_l.Γ[1].ch_s.q2_2[r.exchange[i], w, v]
                a.Γ[2].ch_s.q2_1[i, w, v] += a_l.Γ[2].ch_s.q2_1[i, w, v] + a_l.Γ[2].ch_s.q2_2[r.exchange[i], w, v]
                a.Γ[3].ch_s.q2_1[i, w, v] += a_l.Γ[3].ch_s.q2_1[i, w, v] - a_l.Γ[3].ch_s.q2_2[r.exchange[i], w, v]
                a.Γ[4].ch_s.q2_1[i, w, v] += a_l.Γ[4].ch_s.q2_1[i, w, v] + a_l.Γ[4].ch_s.q2_2[r.exchange[i], w, v]
                a.Γ[5].ch_s.q2_1[i, w, v] += a_l.Γ[5].ch_s.q2_1[i, w, v] - a_l.Γ[6].ch_s.q2_2[r.exchange[i], w, v]
                a.Γ[6].ch_s.q2_1[i, w, v] += a_l.Γ[6].ch_s.q2_1[i, w, v] - a_l.Γ[5].ch_s.q2_2[r.exchange[i], w, v]
                a.Γ[1].ch_s.q2_2[i, w, v] += a_l.Γ[1].ch_s.q2_2[i, w, v] + a_l.Γ[1].ch_s.q2_1[r.exchange[i], w, v]
                a.Γ[2].ch_s.q2_2[i, w, v] += a_l.Γ[2].ch_s.q2_2[i, w, v] + a_l.Γ[2].ch_s.q2_1[r.exchange[i], w, v]
                a.Γ[3].ch_s.q2_2[i, w, v] += a_l.Γ[3].ch_s.q2_2[i, w, v] - a_l.Γ[3].ch_s.q2_1[r.exchange[i], w, v]
                a.Γ[4].ch_s.q2_2[i, w, v] += a_l.Γ[4].ch_s.q2_2[i, w, v] + a_l.Γ[4].ch_s.q2_1[r.exchange[i], w, v]
                a.Γ[5].ch_s.q2_2[i, w, v] += a_l.Γ[5].ch_s.q2_2[i, w, v] - a_l.Γ[6].ch_s.q2_1[r.exchange[i], w, v]
                a.Γ[6].ch_s.q2_2[i, w, v] += a_l.Γ[6].ch_s.q2_2[i, w, v] - a_l.Γ[5].ch_s.q2_1[r.exchange[i], w, v]

                # add q2_1 and q2_2 to t channel (right part from v <-> v' exchange)
                a.Γ[1].ch_t.q2_1[i, w, v] += a_l.Γ[1].ch_t.q2_1[i, w, v] + a_l.Γ[1].ch_t.q2_2[r.exchange[i], w, v]
                a.Γ[2].ch_t.q2_1[i, w, v] += a_l.Γ[2].ch_t.q2_1[i, w, v] + a_l.Γ[2].ch_t.q2_2[r.exchange[i], w, v]
                a.Γ[3].ch_t.q2_1[i, w, v] += a_l.Γ[3].ch_t.q2_1[i, w, v] - a_l.Γ[3].ch_t.q2_2[r.exchange[i], w, v]
                a.Γ[4].ch_t.q2_1[i, w, v] += a_l.Γ[4].ch_t.q2_1[i, w, v] + a_l.Γ[4].ch_t.q2_2[r.exchange[i], w, v]
                a.Γ[5].ch_t.q2_1[i, w, v] += a_l.Γ[5].ch_t.q2_1[i, w, v] - a_l.Γ[6].ch_t.q2_2[r.exchange[i], w, v]
                a.Γ[6].ch_t.q2_1[i, w, v] += a_l.Γ[6].ch_t.q2_1[i, w, v] - a_l.Γ[5].ch_t.q2_2[r.exchange[i], w, v]
                a.Γ[1].ch_t.q2_2[i, w, v] += a_l.Γ[1].ch_t.q2_2[i, w, v] + a_l.Γ[1].ch_t.q2_1[r.exchange[i], w, v]
                a.Γ[2].ch_t.q2_2[i, w, v] += a_l.Γ[2].ch_t.q2_2[i, w, v] + a_l.Γ[2].ch_t.q2_1[r.exchange[i], w, v]
                a.Γ[3].ch_t.q2_2[i, w, v] += a_l.Γ[3].ch_t.q2_2[i, w, v] - a_l.Γ[3].ch_t.q2_1[r.exchange[i], w, v]
                a.Γ[4].ch_t.q2_2[i, w, v] += a_l.Γ[4].ch_t.q2_2[i, w, v] + a_l.Γ[4].ch_t.q2_1[r.exchange[i], w, v]
                a.Γ[5].ch_t.q2_2[i, w, v] += a_l.Γ[5].ch_t.q2_2[i, w, v] - a_l.Γ[6].ch_t.q2_1[r.exchange[i], w, v]
                a.Γ[6].ch_t.q2_2[i, w, v] += a_l.Γ[6].ch_t.q2_2[i, w, v] - a_l.Γ[5].ch_t.q2_1[r.exchange[i], w, v]

                # add q2_1 and q2_2 to u channel (right part from v <-> v' exchange)
                a.Γ[1].ch_u.q2_1[i, w, v] += a_l.Γ[1].ch_u.q2_1[i, w, v] + a_l.Γ[1].ch_u.q2_2[i, w, v]
                a.Γ[2].ch_u.q2_1[i, w, v] += a_l.Γ[2].ch_u.q2_1[i, w, v] + a_l.Γ[2].ch_u.q2_2[i, w, v]
                a.Γ[3].ch_u.q2_1[i, w, v] += a_l.Γ[3].ch_u.q2_1[i, w, v] + a_l.Γ[3].ch_u.q2_2[i, w, v]
                a.Γ[4].ch_u.q2_1[i, w, v] += a_l.Γ[4].ch_u.q2_1[i, w, v] + a_l.Γ[4].ch_u.q2_2[i, w, v]
                a.Γ[5].ch_u.q2_1[i, w, v] += a_l.Γ[5].ch_u.q2_1[i, w, v] - a_l.Γ[5].ch_u.q2_2[i, w, v]
                a.Γ[6].ch_u.q2_1[i, w, v] += a_l.Γ[6].ch_u.q2_1[i, w, v] - a_l.Γ[6].ch_u.q2_2[i, w, v]
                a.Γ[1].ch_u.q2_2[i, w, v] += a_l.Γ[1].ch_u.q2_2[i, w, v] + a_l.Γ[1].ch_u.q2_1[i, w, v]
                a.Γ[2].ch_u.q2_2[i, w, v] += a_l.Γ[2].ch_u.q2_2[i, w, v] + a_l.Γ[2].ch_u.q2_1[i, w, v]
                a.Γ[3].ch_u.q2_2[i, w, v] += a_l.Γ[3].ch_u.q2_2[i, w, v] + a_l.Γ[3].ch_u.q2_1[i, w, v]
                a.Γ[4].ch_u.q2_2[i, w, v] += a_l.Γ[4].ch_u.q2_2[i, w, v] + a_l.Γ[4].ch_u.q2_1[i, w, v]
                a.Γ[5].ch_u.q2_2[i, w, v] += a_l.Γ[5].ch_u.q2_2[i, w, v] - a_l.Γ[5].ch_u.q2_1[i, w, v]
                a.Γ[6].ch_u.q2_2[i, w, v] += a_l.Γ[6].ch_u.q2_2[i, w, v] - a_l.Γ[6].ch_u.q2_1[i, w, v]
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
                    a.Γ[3].ch_s.q3[i, w, v, vp] += a_l.Γ[3].ch_s.q3[i, w, v, vp] - a_l.Γ[3].ch_s.q3[r.exchange[i], w, vp, v]
                    a.Γ[4].ch_s.q3[i, w, v, vp] += a_l.Γ[4].ch_s.q3[i, w, v, vp] + a_l.Γ[4].ch_s.q3[r.exchange[i], w, vp, v]
                    a.Γ[5].ch_s.q3[i, w, v, vp] += a_l.Γ[5].ch_s.q3[i, w, v, vp] - a_l.Γ[6].ch_s.q3[r.exchange[i], w, vp, v]
                    a.Γ[6].ch_s.q3[i, w, v, vp] += a_l.Γ[6].ch_s.q3[i, w, v, vp] - a_l.Γ[5].ch_s.q3[r.exchange[i], w, vp, v]

                    # add q3 to t channel (right part from v <-> v' exchange)
                    a.Γ[1].ch_t.q3[i, w, v, vp] += a_l.Γ[1].ch_t.q3[i, w, v, vp] + a_l.Γ[1].ch_t.q3[r.exchange[i], w, vp, v]
                    a.Γ[2].ch_t.q3[i, w, v, vp] += a_l.Γ[2].ch_t.q3[i, w, v, vp] + a_l.Γ[2].ch_t.q3[r.exchange[i], w, vp, v]
                    a.Γ[3].ch_t.q3[i, w, v, vp] += a_l.Γ[3].ch_t.q3[i, w, v, vp] - a_l.Γ[3].ch_t.q3[r.exchange[i], w, vp, v]
                    a.Γ[4].ch_t.q3[i, w, v, vp] += a_l.Γ[4].ch_t.q3[i, w, v, vp] + a_l.Γ[4].ch_t.q3[r.exchange[i], w, vp, v]
                    a.Γ[5].ch_t.q3[i, w, v, vp] += a_l.Γ[5].ch_t.q3[i, w, v, vp] - a_l.Γ[6].ch_t.q3[r.exchange[i], w, vp, v]
                    a.Γ[6].ch_t.q3[i, w, v, vp] += a_l.Γ[6].ch_t.q3[i, w, v, vp] - a_l.Γ[5].ch_t.q3[r.exchange[i], w, vp, v]

                    # add q3 to u channel (right part from v <-> v' exchange)
                    a.Γ[1].ch_u.q3[i, w, v, vp] += a_l.Γ[1].ch_u.q3[i, w, v, vp] + a_l.Γ[1].ch_u.q3[i, w, vp, v]
                    a.Γ[2].ch_u.q3[i, w, v, vp] += a_l.Γ[2].ch_u.q3[i, w, v, vp] + a_l.Γ[2].ch_u.q3[i, w, vp, v]
                    a.Γ[3].ch_u.q3[i, w, v, vp] += a_l.Γ[3].ch_u.q3[i, w, v, vp] + a_l.Γ[3].ch_u.q3[i, w, vp, v]
                    a.Γ[4].ch_u.q3[i, w, v, vp] += a_l.Γ[4].ch_u.q3[i, w, v, vp] + a_l.Γ[4].ch_u.q3[i, w, vp, v]
                    a.Γ[5].ch_u.q3[i, w, v, vp] += a_l.Γ[5].ch_u.q3[i, w, v, vp] - a_l.Γ[5].ch_u.q3[i, w, vp, v]
                    a.Γ[6].ch_u.q3[i, w, v, vp] += a_l.Γ[6].ch_u.q3[i, w, v, vp] - a_l.Γ[6].ch_u.q3[i, w, vp, v]
                end
            end
        end
    end

    return nothing
end