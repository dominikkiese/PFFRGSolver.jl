"""
    Action_u1_dm <: Action

Struct containing self energy and vertex components for models with U(1) symmetric Dzyaloshinskii-Moriya interaction.
* `Σ :: Vector{Float64}`    : negative imaginary part of the self energy
* `Γ :: SVector{6, Vertex}` : Γxx, Γzz, ΓDM (i.e Γxy), Γdd, Γzd, Γdz component of the full vertex
"""
struct Action_u1_dm <: Action
    Σ :: Vector{Float64}
    Γ :: SVector{6, Vertex}
end

# generate action_u1_dm dummy
function get_action_u1_dm_empty(
    r :: Reduced_lattice,
    m :: Mesh_u1_dm
    ) :: Action_u1_dm

    # init self energy
    Σ = zeros(Float64, length(m.σ))

    # init vertices
    Γ = SVector(ntuple(comp -> get_vertex_empty(r, m), 6))

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





# helper function to disentangle flags during interpolation for symmetric u1 models
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

# get all interpolated vertex components for symmetric u1 models
function get_Γ(
    site :: Int64,
    bs   :: NTuple{6, Buffer},
    bt   :: NTuple{6, Buffer},
    bu   :: NTuple{6, Buffer},
    r    :: Reduced_lattice,
    a    :: Action_u1_dm
    ;
    ch_s :: Bool = true,
    ch_t :: Bool = true,
    ch_u :: Bool = true
    )    :: NTuple{6, Float64}

    vals = ntuple(comp -> get_Γ_comp(comp, site, bs[comp], bt[comp], bu[comp], r, a, apply_flags_u1_dm, ch_s = ch_s, ch_t = ch_t, ch_u = ch_u), 6)

    return vals
end

# get all interpolated vertex components for symmetric u1 models on all lattice sites
function get_Γ_avx!(
    r     :: Reduced_lattice,
    bs    :: NTuple{6, Buffer},
    bt    :: NTuple{6, Buffer},
    bu    :: NTuple{6, Buffer},
    a     :: Action_u1_dm,
    temp  :: Array{Float64, 3},
    index :: Int64
    ;
    ch_s  :: Bool = true,
    ch_t  :: Bool = true,
    ch_u  :: Bool = true
    )     :: Nothing

    for comp in 1 : 6
        get_Γ_comp_avx!(comp, r, bs[comp], bt[comp], bu[comp], a, apply_flags_u1_dm, view(temp, :, comp, index), ch_s = ch_s, ch_t = ch_t, ch_u = ch_u)
    end

    return nothing
end

# get all interpolated vertex components for symmetric u1 models on all lattice sites
function get_Γ_avx!(
    comp  :: Int64,
    r     :: Reduced_lattice,
    bs    :: NTuple{6, Buffer},
    bt    :: NTuple{6, Buffer},
    bu    :: NTuple{6, Buffer},
    a     :: Action_u1_dm,
    temp  :: Array{Float64, 3},
    index :: Int64
    ;
    ch_s  :: Bool = true,
    ch_t  :: Bool = true,
    ch_u  :: Bool = true
    )     :: Nothing

   get_Γ_comp_avx!(comp, r, bs[comp], bt[comp], bu[comp], a, apply_flags_u1_dm, view(temp, :, comp, index), ch_s = ch_s, ch_t = ch_t, ch_u = ch_u)

    return nothing
end





# symmetrize full loop contribution and central part
function symmetrize!(
    r :: Reduced_lattice,
    m :: Mesh_u1_dm,
    a :: Action_u1_dm
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
                    a.Γ[1].ch_s.q3[i, w, v, vp] =  a.Γ[1].ch_s.q3[r.exchange[i], w, vp, v]
                    a.Γ[2].ch_s.q3[i, w, v, vp] =  a.Γ[2].ch_s.q3[r.exchange[i], w, vp, v]
                    a.Γ[3].ch_s.q3[i, w, v, vp] = -a.Γ[3].ch_s.q3[r.exchange[i], w, vp, v]
                    a.Γ[4].ch_s.q3[i, w, v, vp] =  a.Γ[4].ch_s.q3[r.exchange[i], w, vp, v]

                    p1 = get_param(m.Ωs[5][w],  m.Ωs[6])
                    p2 = get_param(m.νs[5][vp], m.νs[6])
                    p3 = get_param(m.νs[5][v],  m.νs[6])
                    a.Γ[5].ch_s.q3[i, w, v, vp] = -get_q3(r.exchange[i], p1, p2, p3, a.Γ[6].ch_s)
                    
                    p1 = get_param(m.Ωs[6][w],  m.Ωs[5])
                    p2 = get_param(m.νs[6][vp], m.νs[5])
                    p3 = get_param(m.νs[6][v],  m.νs[5])
                    a.Γ[6].ch_s.q3[i, w, v, vp] = -get_q3(r.exchange[i], p1, p2, p3, a.Γ[5].ch_s) 
                    
                    # get upper triangular matrix for (v, v') plane for t channel
                    a.Γ[1].ch_t.q3[i, w, v, vp] =  a.Γ[1].ch_t.q3[r.exchange[i], w, vp, v]
                    a.Γ[2].ch_t.q3[i, w, v, vp] =  a.Γ[2].ch_t.q3[r.exchange[i], w, vp, v]
                    a.Γ[3].ch_t.q3[i, w, v, vp] = -a.Γ[3].ch_t.q3[r.exchange[i], w, vp, v]
                    a.Γ[4].ch_t.q3[i, w, v, vp] =  a.Γ[4].ch_t.q3[r.exchange[i], w, vp, v]

                    p1 = get_param(m.Ωt[5][w],  m.Ωt[6])
                    p2 = get_param(m.νt[5][vp], m.νt[6])
                    p3 = get_param(m.νt[5][v],  m.νt[6])
                    a.Γ[5].ch_t.q3[i, w, v, vp] = -get_q3(r.exchange[i], p1, p2, p3, a.Γ[6].ch_t)   

                    p1 = get_param(m.Ωt[6][w],  m.Ωt[5])
                    p2 = get_param(m.νt[6][vp], m.νt[5])
                    p3 = get_param(m.νt[6][v],  m.νt[5])
                    a.Γ[6].ch_t.q3[i, w, v, vp] = -get_q3(r.exchange[i], p1, p2, p3, a.Γ[5].ch_t)  

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

    # set asymptotic limits
    limits!(a)

    return nothing
end

# symmetrized addition for left part (right part symmetric to left part)
function symmetrize_add_to!(
    r   :: Reduced_lattice,
    m   :: Mesh_u1_dm,
    a_l :: Action_u1_dm,
    a   :: Action_u1_dm
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
                    a.Γ[3].ch_s.q3[i, w, v, vp] += a_l.Γ[3].ch_s.q3[i, w, v, vp] - a_l.Γ[3].ch_s.q3[r.exchange[i], w, vp, v]
                    a.Γ[4].ch_s.q3[i, w, v, vp] += a_l.Γ[4].ch_s.q3[i, w, v, vp] + a_l.Γ[4].ch_s.q3[r.exchange[i], w, vp, v]

                    p1 = get_param(m.Ωs[5][w],  m.Ωs[6])
                    p2 = get_param(m.νs[5][vp], m.νs[6])
                    p3 = get_param(m.νs[5][v],  m.νs[6])
                    a.Γ[5].ch_s.q3[i, w, v, vp] += a_l.Γ[5].ch_s.q3[i, w, v, vp] - get_q3(r.exchange[i], p1, p2, p3, a_l.Γ[6].ch_s)

                    p1 = get_param(m.Ωs[6][w],  m.Ωs[5])
                    p2 = get_param(m.νs[6][vp], m.νs[5])
                    p3 = get_param(m.νs[6][v],  m.νs[5])
                    a.Γ[6].ch_s.q3[i, w, v, vp] += a_l.Γ[6].ch_s.q3[i, w, v, vp] - get_q3(r.exchange[i], p1, p2, p3, a_l.Γ[5].ch_s)

                    # add q3 to t channel (right part from v <-> v' exchange)
                    a.Γ[1].ch_t.q3[i, w, v, vp] += a_l.Γ[1].ch_t.q3[i, w, v, vp] + a_l.Γ[1].ch_t.q3[r.exchange[i], w, vp, v]
                    a.Γ[2].ch_t.q3[i, w, v, vp] += a_l.Γ[2].ch_t.q3[i, w, v, vp] + a_l.Γ[2].ch_t.q3[r.exchange[i], w, vp, v]
                    a.Γ[3].ch_t.q3[i, w, v, vp] += a_l.Γ[3].ch_t.q3[i, w, v, vp] - a_l.Γ[3].ch_t.q3[r.exchange[i], w, vp, v]
                    a.Γ[4].ch_t.q3[i, w, v, vp] += a_l.Γ[4].ch_t.q3[i, w, v, vp] + a_l.Γ[4].ch_t.q3[r.exchange[i], w, vp, v]

                    p1 = get_param(m.Ωt[5][w],  m.Ωt[6])
                    p2 = get_param(m.νt[5][vp], m.νt[6])
                    p3 = get_param(m.νt[5][v],  m.νt[6])
                    a.Γ[5].ch_t.q3[i, w, v, vp] += a_l.Γ[5].ch_t.q3[i, w, v, vp] - get_q3(r.exchange[i], p1, p2, p3, a_l.Γ[6].ch_t) 

                    p1 = get_param(m.Ωt[6][w],  m.Ωt[5])
                    p2 = get_param(m.νt[6][vp], m.νt[5])
                    p3 = get_param(m.νt[6][v],  m.νt[5])
                    a.Γ[6].ch_t.q3[i, w, v, vp] += a_l.Γ[6].ch_t.q3[i, w, v, vp] - get_q3(r.exchange[i], p1, p2, p3, a_l.Γ[5].ch_t) 

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

    # set asymptotic limits
    limits!(a_l)
    limits!(a)

    return nothing
end