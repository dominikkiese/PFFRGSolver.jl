"""
    Action_su2_hkg <: Action
    
Struct containing self energy and vertex components for SU(2) symmetric models.
 * `S :: Float64`         : total spin quantum number
 * `Σ :: Vector{Float64}` : negative imaginary part of the self energy
 * `Γ :: Vector{Vertex}`  : spin and density component of the full vertex
"""

struct Action_su2_hkg <: Action
    S :: Float64
    Σ :: Vector{Float64}
    Γ :: Vector{Vertex}
end

# generate action_su2_hkg dummy
function get_action_su2_hkg_empty(
    S :: Float64,
    r :: Reduced_lattice,
    m :: Mesh,
    ) :: Action_su2_hkg

    # init self energy
    Σ = zeros(Float64, length(m.σ))

    # init vertices
    Γ = Vertex[get_vertex_empty(r, m) for i in 1 : 16]

    # build action
    a = Action_su2_hkg(S, Σ, Γ)

    return a
end

# init action for su2_hkg symmetry
function init_action!(
    l :: Lattice,
    r :: Reduced_lattice,
    a :: Action_su2_hkg
    ) :: Nothing

    # init bare action for spin component
    ref_int = SVector{4, Int64}(0, 0, 0, 1)
    ref     = Site(ref_int, get_vec(ref_int, l.uc))

    for i in eachindex(r.sites)
        # get bond from lattice
        b = get_bond(ref, r.sites[i], l)

        #set Γxx bare acording to spin exchange 
        a.Γ[1].bare[i] = b.exchange[1, 1] / 4.0 / (2.0 * a.S)

        #set Γyy bare acording to spin exchange
        a.Γ[2].bare[i] = b.exchange[2, 2] / 4.0 / (2.0 * a.S)

        #set Γzz bare acording to spin exchange
        a.Γ[3].bare[i] = b.exchange[3, 3] / 4.0 / (2.0 * a.S)

        #set Γxy bare acording to spin exchange
        a.Γ[4].bare[i] = b.exchange[1, 2] / 4.0 / (2.0 * a.S)

        #set Γxz bare acording to spin exchange
        a.Γ[5].bare[i] = b.exchange[1, 3] / 4.0 / (2.0 * a.S)

        #set Γyz bare acording to spin exchange
        a.Γ[6].bare[i] = b.exchange[2, 3] / 4.0 / (2.0 * a.S)

        #set Γyx bare acording to spin exchange
        a.Γ[7].bare[i] = b.exchange[2, 1] / 4.0 / (2.0 * a.S)

        #set Γzx bare acording to spin exchange
        a.Γ[8].bare[i] = b.exchange[3, 1] / 4.0 / (2.0 * a.S)

        #set Γzy bare acording to spin exchange
        a.Γ[9].bare[i] = b.exchange[3, 2] / 4.0 / (2.0 * a.S)
        
    end

    return nothing
end


# add repulsion for su2_hkg symmetry
function add_repulsion!(                                 
    A :: Float64,
    a :: Action_su2_hkg
    ) :: Nothing

    # add on-site level repulsion for Γxx 
    a.Γ[1].bare[1] += A / 4.0 / (2.0 * a.S)

    # add on-site level repulsion for Γyy 
    a.Γ[2].bare[1] += A / 4.0 / (2.0 * a.S)

    # add on-site level repulsion for Γzz
    a.Γ[3].bare[1] += A / 4.0 / (2.0 * a.S)

    return nothing
end


# helper function to disentangle flags during interpolation for su2_hkg models
function apply_flags_su2_hkg(
    b    :: Buffer,
    comp :: Int64                                         
    )    :: Tuple{Float64, Int64}

    # clarify sign of contribution
    sgn = 1.0

    # ξ(μ) * ξ(ν) = -1 for Γxd, Γdx, Γyd, Γdy, Γzd & Γdz
    if comp in (11, 12, 13, 14, 15, 16) && b.sgn_μν
        sgn *= -1.0
    end

    # -ξ(μ) = -1 for Γdd, Γdx, Γdy & Γdz
    if comp in (10, 14, 15, 16) && b.sgn_μ
        sgn *= -1.0
    end

    # -ξ(ν) = -1 for Γdd, Γxd, Γyd & Γzd
    if comp in (10, 11, 12, 13) && b.sgn_ν
        sgn *= -1.0
    end
    
    #clarify which component to interpolate
    if b.exchange_flag
        # Γxy <-> Γyx for spin exchange
        if comp == 4
            comp = 7
        elseif comp == 7
            comp = 4
        # Γxz <-> Γzx for spin exchange
        elseif comp == 5 
            comp = 8
        elseif comp == 8  
            comp = 5 
        # Γyz <-> Γzy for spin exchange
        elseif comp == 6
            comp = 9
        elseif comp == 9 
            comp = 6
        # Γxd <-> Γdx for spin exchange
        elseif comp == 11
            comp = 14
        elseif comp == 14 
            comp = 11
        # Γyd <-> Γdy for spin exchange
        elseif comp == 12
            comp = 15
        elseif comp == 15 
            comp = 12
        # Γzd <-> Γdz for spin exchange
        elseif comp == 13
            comp = 16
        elseif comp == 16 
            comp = 13
        end
    end


    return sgn, comp
end

# get all interpolated vertex components for su2_hkg symmetric models
function get_Γ(
    site :: Int64,
    bs   :: Buffer,
    bt   :: Buffer,
    bu   :: Buffer,
    r    :: Reduced_lattice,
    a    :: Action_su2_hkg
    ;
    ch_s :: Bool = true,
    ch_t :: Bool = true,
    ch_u :: Bool = true
    )    :: NTuple{16, Float64}

    Γxx = get_Γ_comp(1 , site, bs, bt, bu, r, a, apply_flags_su2_hkg, ch_s = ch_s, ch_t = ch_t, ch_u = ch_u)
    Γyy = get_Γ_comp(2 , site, bs, bt, bu, r, a, apply_flags_su2_hkg, ch_s = ch_s, ch_t = ch_t, ch_u = ch_u)
    Γzz = get_Γ_comp(3 , site, bs, bt, bu, r, a, apply_flags_su2_hkg, ch_s = ch_s, ch_t = ch_t, ch_u = ch_u)
    Γxy = get_Γ_comp(4 , site, bs, bt, bu, r, a, apply_flags_su2_hkg, ch_s = ch_s, ch_t = ch_t, ch_u = ch_u)
    Γxz = get_Γ_comp(5 , site, bs, bt, bu, r, a, apply_flags_su2_hkg, ch_s = ch_s, ch_t = ch_t, ch_u = ch_u)
    Γyz = get_Γ_comp(6 , site, bs, bt, bu, r, a, apply_flags_su2_hkg, ch_s = ch_s, ch_t = ch_t, ch_u = ch_u)
    Γyx = get_Γ_comp(7 , site, bs, bt, bu, r, a, apply_flags_su2_hkg, ch_s = ch_s, ch_t = ch_t, ch_u = ch_u)
    Γzx = get_Γ_comp(8 , site, bs, bt, bu, r, a, apply_flags_su2_hkg, ch_s = ch_s, ch_t = ch_t, ch_u = ch_u)
    Γzy = get_Γ_comp(9 , site, bs, bt, bu, r, a, apply_flags_su2_hkg, ch_s = ch_s, ch_t = ch_t, ch_u = ch_u)
    Γdd = get_Γ_comp(10, site, bs, bt, bu, r, a, apply_flags_su2_hkg, ch_s = ch_s, ch_t = ch_t, ch_u = ch_u)
    Γxd = get_Γ_comp(11, site, bs, bt, bu, r, a, apply_flags_su2_hkg, ch_s = ch_s, ch_t = ch_t, ch_u = ch_u)
    Γyd = get_Γ_comp(12, site, bs, bt, bu, r, a, apply_flags_su2_hkg, ch_s = ch_s, ch_t = ch_t, ch_u = ch_u)
    Γzd = get_Γ_comp(13, site, bs, bt, bu, r, a, apply_flags_su2_hkg, ch_s = ch_s, ch_t = ch_t, ch_u = ch_u)
    Γdx = get_Γ_comp(14, site, bs, bt, bu, r, a, apply_flags_su2_hkg, ch_s = ch_s, ch_t = ch_t, ch_u = ch_u)
    Γdy = get_Γ_comp(15, site, bs, bt, bu, r, a, apply_flags_su2_hkg, ch_s = ch_s, ch_t = ch_t, ch_u = ch_u)
    Γdz = get_Γ_comp(16, site, bs, bt, bu, r, a, apply_flags_su2_hkg, ch_s = ch_s, ch_t = ch_t, ch_u = ch_u)

    return Γxx, Γyy, Γzz, Γxy, Γxz, Γyz, Γyx, Γzx, Γzy, Γdd, Γxd, Γyd, Γzd, Γdx, Γdy, Γdz
end

# get all interpolated vertex components for su2_hkg symetric models on all lattice sites 
function get_Γ_avx!(
    r     :: Reduced_lattice,
    bs    :: Buffer,
    bt    :: Buffer,
    bu    :: Buffer,
    a     :: Action_su2_hkg,
    temp  :: Array{Float64, 3},
    index :: Int64
    ;
    ch_s  :: Bool = true,
    ch_t  :: Bool = true,
    ch_u  :: Bool = true
    )     :: Nothing

    for comp in 1 : 16 
        get_Γ_comp_avx!(comp, r, bs, bt, bu, a, apply_flags_su2_hkg, view(temp, :, comp, index), ch_s = ch_s, ch_t = ch_t, ch_u = ch_u)
    end

    return nothing
end





# symmetrize full loop contribution and central part
function symmetrize!(
    r :: Reduced_lattice,
    a :: Action_su2_hkg
    ) :: Nothing

    # get dimensions
    num_sites = size(a.Γ[1].ch_s.q2_1, 1)
    num_Ω     = size(a.Γ[1].ch_s.q2_1, 2)           
    num_ν     = size(a.Γ[1].ch_s.q2_1, 3) 

    # computation for q3 
    for v in 1 : num_ν
        for vp in v + 1 : num_ν
            #@turbo
            for w in 1 : num_Ω
                for i in 1 : num_sites
                    i_exchange = r.exchange[i].site
                    comps_exchange = r.exchange[i].components
                    signs_exchange = r.exchange[i].signs

                    #get upper triangular matrix for (v, v') plane for s channel
                    a.Γ[1].ch_s.q3[i, w, v, vp]  =    signs_exchange[1] * a.Γ[comps_exchange[1]].ch_s.q3[i_exchange, w, vp, v] 
                    a.Γ[2].ch_s.q3[i, w, v, vp]  =    signs_exchange[2] * a.Γ[comps_exchange[2]].ch_s.q3[i_exchange, w, vp, v]
                    a.Γ[3].ch_s.q3[i, w, v, vp]  =    signs_exchange[3] * a.Γ[comps_exchange[3]].ch_s.q3[i_exchange, w, vp, v]
                    a.Γ[4].ch_s.q3[i, w, v, vp]  =    signs_exchange[7] * a.Γ[comps_exchange[7]].ch_s.q3[i_exchange, w, vp, v]
                    a.Γ[5].ch_s.q3[i, w, v, vp]  =    signs_exchange[8] * a.Γ[comps_exchange[8]].ch_s.q3[i_exchange, w, vp, v]
                    a.Γ[6].ch_s.q3[i, w, v, vp]  =    signs_exchange[9] * a.Γ[comps_exchange[9]].ch_s.q3[i_exchange, w, vp, v]
                    a.Γ[7].ch_s.q3[i, w, v, vp]  =    signs_exchange[4] * a.Γ[comps_exchange[4]].ch_s.q3[i_exchange, w, vp, v]
                    a.Γ[8].ch_s.q3[i, w, v, vp]  =    signs_exchange[5] * a.Γ[comps_exchange[5]].ch_s.q3[i_exchange, w, vp, v]
                    a.Γ[9].ch_s.q3[i, w, v, vp]  =    signs_exchange[6] * a.Γ[comps_exchange[6]].ch_s.q3[i_exchange, w, vp, v]
                    a.Γ[10].ch_s.q3[i, w, v, vp] =   signs_exchange[10] * a.Γ[comps_exchange[10]].ch_s.q3[i_exchange, w, vp, v]
                    a.Γ[11].ch_s.q3[i, w, v, vp] =  -signs_exchange[14] * a.Γ[comps_exchange[14]].ch_s.q3[i_exchange, w, vp, v]
                    a.Γ[12].ch_s.q3[i, w, v, vp] =  -signs_exchange[15] * a.Γ[comps_exchange[15]].ch_s.q3[i_exchange, w, vp, v]
                    a.Γ[13].ch_s.q3[i, w, v, vp] =  -signs_exchange[16] * a.Γ[comps_exchange[16]].ch_s.q3[i_exchange, w, vp, v]
                    a.Γ[14].ch_s.q3[i, w, v, vp] =  -signs_exchange[11] * a.Γ[comps_exchange[11]].ch_s.q3[i_exchange, w, vp, v]
                    a.Γ[15].ch_s.q3[i, w, v, vp] =  -signs_exchange[12] * a.Γ[comps_exchange[12]].ch_s.q3[i_exchange, w, vp, v]
                    a.Γ[16].ch_s.q3[i, w, v, vp] =  -signs_exchange[13] * a.Γ[comps_exchange[13]].ch_s.q3[i_exchange, w, vp, v]
                    
                    # get upper triangular matrix for (v, v') plane for t channel
                    a.Γ[1].ch_t.q3[i, w, v, vp]  =    signs_exchange[1] * a.Γ[comps_exchange[1]].ch_t.q3[i_exchange, w, vp, v] 
                    a.Γ[2].ch_t.q3[i, w, v, vp]  =    signs_exchange[2] * a.Γ[comps_exchange[2]].ch_t.q3[i_exchange, w, vp, v]
                    a.Γ[3].ch_t.q3[i, w, v, vp]  =    signs_exchange[3] * a.Γ[comps_exchange[3]].ch_t.q3[i_exchange, w, vp, v]
                    a.Γ[4].ch_t.q3[i, w, v, vp]  =    signs_exchange[7] * a.Γ[comps_exchange[7]].ch_t.q3[i_exchange, w, vp, v]
                    a.Γ[5].ch_t.q3[i, w, v, vp]  =    signs_exchange[8] * a.Γ[comps_exchange[8]].ch_t.q3[i_exchange, w, vp, v]
                    a.Γ[6].ch_t.q3[i, w, v, vp]  =    signs_exchange[9] * a.Γ[comps_exchange[9]].ch_t.q3[i_exchange, w, vp, v]
                    a.Γ[7].ch_t.q3[i, w, v, vp]  =    signs_exchange[4] * a.Γ[comps_exchange[4]].ch_t.q3[i_exchange, w, vp, v]
                    a.Γ[8].ch_t.q3[i, w, v, vp]  =    signs_exchange[5] * a.Γ[comps_exchange[5]].ch_t.q3[i_exchange, w, vp, v]
                    a.Γ[9].ch_t.q3[i, w, v, vp]  =    signs_exchange[6] * a.Γ[comps_exchange[6]].ch_t.q3[i_exchange, w, vp, v]
                    a.Γ[10].ch_t.q3[i, w, v, vp] =   signs_exchange[10] * a.Γ[comps_exchange[10]].ch_t.q3[i_exchange, w, vp, v]
                    a.Γ[11].ch_t.q3[i, w, v, vp] =  -signs_exchange[14] * a.Γ[comps_exchange[14]].ch_t.q3[i_exchange, w, vp, v]
                    a.Γ[12].ch_t.q3[i, w, v, vp] =  -signs_exchange[15] * a.Γ[comps_exchange[15]].ch_t.q3[i_exchange, w, vp, v]
                    a.Γ[13].ch_t.q3[i, w, v, vp] =  -signs_exchange[16] * a.Γ[comps_exchange[16]].ch_t.q3[i_exchange, w, vp, v]
                    a.Γ[14].ch_t.q3[i, w, v, vp] =  -signs_exchange[11] * a.Γ[comps_exchange[11]].ch_t.q3[i_exchange, w, vp, v]
                    a.Γ[15].ch_t.q3[i, w, v, vp] =  -signs_exchange[12] * a.Γ[comps_exchange[12]].ch_t.q3[i_exchange, w, vp, v]
                    a.Γ[16].ch_t.q3[i, w, v, vp] =  -signs_exchange[13] * a.Γ[comps_exchange[13]].ch_t.q3[i_exchange, w, vp, v]

                    # get upper triangular matrix for (v, v') plane for u channel 
                    a.Γ[1].ch_u.q3[i, w, v, vp]  =    a.Γ[1].ch_u.q3[i, w, vp, v]
                    a.Γ[2].ch_u.q3[i, w, v, vp]  =    a.Γ[2].ch_u.q3[i, w, vp, v]
                    a.Γ[3].ch_u.q3[i, w, v, vp]  =    a.Γ[3].ch_u.q3[i, w, vp, v]
                    a.Γ[4].ch_u.q3[i, w, v, vp]  =    a.Γ[4].ch_u.q3[i, w, vp, v]
                    a.Γ[5].ch_u.q3[i, w, v, vp]  =    a.Γ[5].ch_u.q3[i, w, vp, v]
                    a.Γ[6].ch_u.q3[i, w, v, vp]  =    a.Γ[6].ch_u.q3[i, w, vp, v]
                    a.Γ[7].ch_u.q3[i, w, v, vp]  =    a.Γ[7].ch_u.q3[i, w, vp, v]
                    a.Γ[8].ch_u.q3[i, w, v, vp]  =    a.Γ[8].ch_u.q3[i, w, vp, v]
                    a.Γ[9].ch_u.q3[i, w, v, vp]  =    a.Γ[9].ch_u.q3[i, w, vp, v]
                    a.Γ[10].ch_u.q3[i, w, v, vp] =   a.Γ[10].ch_u.q3[i, w, vp, v]
                    a.Γ[11].ch_u.q3[i, w, v, vp] =  -a.Γ[11].ch_u.q3[i, w, vp, v]
                    a.Γ[12].ch_u.q3[i, w, v, vp] =  -a.Γ[12].ch_u.q3[i, w, vp, v]
                    a.Γ[13].ch_u.q3[i, w, v, vp] =  -a.Γ[13].ch_u.q3[i, w, vp, v]
                    a.Γ[14].ch_u.q3[i, w, v, vp] =  -a.Γ[14].ch_u.q3[i, w, vp, v]
                    a.Γ[15].ch_u.q3[i, w, v, vp] =  -a.Γ[15].ch_u.q3[i, w, vp, v]
                    a.Γ[16].ch_u.q3[i, w, v, vp] =  -a.Γ[16].ch_u.q3[i, w, vp, v]
                end
            end
        end
    end

    #Get new action for site exchange
    symmetrize_site_exchange!(r, a)

    return nothing
end

#NOT YET DONE WITH SPIN SYMMETRIES
# syymetrized addition for left part (right part symmetric to left part)
function symmetrize_add_to!(
    r   :: Reduced_lattice,
    a_l :: Action_su2_hkg,
    a   :: Action_su2_hkg
    )   :: Nothing

    error("Multiloop nor yet implemented. symmetrize_add_to! needs to be corrected (and possibly more).")

    # get dimensions
    num_sites = size(a_l.Γ[1].ch_s.q2_1, 1)
    num_Ω     = size(a_l.Γ[1].ch_s.q2_1, 2)
    num_ν     = size(a_l.Γ[1].ch_s.q2_1, 3)

    # computation for q1
    @turbo for w in 1 : num_Ω
        for i in 1 : num_sites
            # add q1 to s channel (right part from v <-> v' exchange)
            a.Γ[1].ch_s.q1[i, w]  += a_l.Γ[1].ch_s.q1[i, w]  +  a_l.Γ[1].ch_s.q1[r.exchange[i], w]
            a.Γ[2].ch_s.q1[i, w]  += a_l.Γ[2].ch_s.q1[i, w]  +  a_l.Γ[2].ch_s.q1[r.exchange[i], w]
            a.Γ[3].ch_s.q1[i, w]  += a_l.Γ[3].ch_s.q1[i, w]  +  a_l.Γ[3].ch_s.q1[r.exchange[i], w]
            a.Γ[4].ch_s.q1[i, w]  += a_l.Γ[4].ch_s.q1[i, w]  +  a_l.Γ[4].ch_s.q1[r.exchange[i], w]
            a.Γ[5].ch_s.q1[i, w]  += a_l.Γ[5].ch_s.q1[i, w]  +  a_l.Γ[5].ch_s.q1[r.exchange[i], w]
            a.Γ[6].ch_s.q1[i, w]  += a_l.Γ[6].ch_s.q1[i, w]  +  a_l.Γ[6].ch_s.q1[r.exchange[i], w]
            a.Γ[7].ch_s.q1[i, w]  += a_l.Γ[7].ch_s.q1[i, w]  +  a_l.Γ[7].ch_s.q1[r.exchange[i], w]
            a.Γ[8].ch_s.q1[i, w]  += a_l.Γ[8].ch_s.q1[i, w]  +  a_l.Γ[8].ch_s.q1[r.exchange[i], w]
            a.Γ[9].ch_s.q1[i, w]  += a_l.Γ[9].ch_s.q1[i, w]  +  a_l.Γ[9].ch_s.q1[r.exchange[i], w]
            a.Γ[10].ch_s.q1[i, w] += a_l.Γ[10].ch_s.q1[i, w] + a_l.Γ[10].ch_s.q1[r.exchange[i], w]
            a.Γ[11].ch_s.q1[i, w] += a_l.Γ[11].ch_s.q1[i, w] - a_l.Γ[11].ch_s.q1[r.exchange[i], w]
            a.Γ[12].ch_s.q1[i, w] += a_l.Γ[12].ch_s.q1[i, w] - a_l.Γ[12].ch_s.q1[r.exchange[i], w]
            a.Γ[13].ch_s.q1[i, w] += a_l.Γ[13].ch_s.q1[i, w] - a_l.Γ[13].ch_s.q1[r.exchange[i], w]
            a.Γ[14].ch_s.q1[i, w] += a_l.Γ[14].ch_s.q1[i, w] - a_l.Γ[14].ch_s.q1[r.exchange[i], w]
            a.Γ[15].ch_s.q1[i, w] += a_l.Γ[15].ch_s.q1[i, w] - a_l.Γ[15].ch_s.q1[r.exchange[i], w]
            a.Γ[16].ch_s.q1[i, w] += a_l.Γ[16].ch_s.q1[i, w] - a_l.Γ[16].ch_s.q1[r.exchange[i], w]

            # add q1 to t channel (right part from v <-> v' exchange)
            a.Γ[1].ch_t.q1[i, w]  += a_l.Γ[1].ch_t.q1[i, w]  +  a_l.Γ[1].ch_t.q1[r.exchange[i], w]
            a.Γ[2].ch_t.q1[i, w]  += a_l.Γ[2].ch_t.q1[i, w]  +  a_l.Γ[2].ch_t.q1[r.exchange[i], w]
            a.Γ[3].ch_t.q1[i, w]  += a_l.Γ[3].ch_t.q1[i, w]  +  a_l.Γ[3].ch_t.q1[r.exchange[i], w]
            a.Γ[4].ch_t.q1[i, w]  += a_l.Γ[4].ch_t.q1[i, w]  +  a_l.Γ[4].ch_t.q1[r.exchange[i], w]
            a.Γ[5].ch_t.q1[i, w]  += a_l.Γ[5].ch_t.q1[i, w]  +  a_l.Γ[5].ch_t.q1[r.exchange[i], w]
            a.Γ[6].ch_t.q1[i, w]  += a_l.Γ[6].ch_t.q1[i, w]  +  a_l.Γ[6].ch_t.q1[r.exchange[i], w]
            a.Γ[7].ch_t.q1[i, w]  += a_l.Γ[7].ch_t.q1[i, w]  +  a_l.Γ[7].ch_t.q1[r.exchange[i], w]
            a.Γ[8].ch_t.q1[i, w]  += a_l.Γ[8].ch_t.q1[i, w]  +  a_l.Γ[8].ch_t.q1[r.exchange[i], w]
            a.Γ[9].ch_t.q1[i, w]  += a_l.Γ[9].ch_t.q1[i, w]  +  a_l.Γ[9].ch_t.q1[r.exchange[i], w]
            a.Γ[10].ch_t.q1[i, w] += a_l.Γ[10].ch_t.q1[i, w] + a_l.Γ[10].ch_t.q1[r.exchange[i], w]
            a.Γ[11].ch_t.q1[i, w] += a_l.Γ[11].ch_t.q1[i, w] - a_l.Γ[11].ch_t.q1[r.exchange[i], w]
            a.Γ[12].ch_t.q1[i, w] += a_l.Γ[12].ch_t.q1[i, w] - a_l.Γ[12].ch_t.q1[r.exchange[i], w]
            a.Γ[13].ch_t.q1[i, w] += a_l.Γ[13].ch_t.q1[i, w] - a_l.Γ[13].ch_t.q1[r.exchange[i], w]
            a.Γ[14].ch_t.q1[i, w] += a_l.Γ[14].ch_t.q1[i, w] - a_l.Γ[14].ch_t.q1[r.exchange[i], w]
            a.Γ[15].ch_t.q1[i, w] += a_l.Γ[15].ch_t.q1[i, w] - a_l.Γ[15].ch_t.q1[r.exchange[i], w]
            a.Γ[16].ch_t.q1[i, w] += a_l.Γ[16].ch_t.q1[i, w] - a_l.Γ[16].ch_t.q1[r.exchange[i], w]

            # add q1 to u channel (right part from v <-> v' exchange)
            a.Γ[1].ch_u.q1[i, w]  += a_l.Γ[1].ch_u.q1[i, w]  +  a_l.Γ[1].ch_u.q1[i, w]
            a.Γ[2].ch_u.q1[i, w]  += a_l.Γ[2].ch_u.q1[i, w]  +  a_l.Γ[2].ch_u.q1[i, w]
            a.Γ[3].ch_u.q1[i, w]  += a_l.Γ[3].ch_u.q1[i, w]  +  a_l.Γ[3].ch_u.q1[i, w]
            a.Γ[4].ch_u.q1[i, w]  += a_l.Γ[4].ch_u.q1[i, w]  +  a_l.Γ[4].ch_u.q1[i, w]
            a.Γ[5].ch_u.q1[i, w]  += a_l.Γ[5].ch_u.q1[i, w]  +  a_l.Γ[5].ch_u.q1[i, w]
            a.Γ[6].ch_u.q1[i, w]  += a_l.Γ[6].ch_u.q1[i, w]  +  a_l.Γ[6].ch_u.q1[i, w]
            a.Γ[7].ch_u.q1[i, w]  += a_l.Γ[7].ch_u.q1[i, w]  +  a_l.Γ[7].ch_u.q1[i, w]
            a.Γ[8].ch_u.q1[i, w]  += a_l.Γ[8].ch_u.q1[i, w]  +  a_l.Γ[8].ch_u.q1[i, w]
            a.Γ[9].ch_u.q1[i, w]  += a_l.Γ[9].ch_u.q1[i, w]  +  a_l.Γ[9].ch_u.q1[i, w]
            a.Γ[10].ch_u.q1[i, w] += a_l.Γ[10].ch_u.q1[i, w] + a_l.Γ[10].ch_u.q1[i, w]
            a.Γ[11].ch_u.q1[i, w] += a_l.Γ[11].ch_u.q1[i, w] - a_l.Γ[11].ch_u.q1[i, w]
            a.Γ[12].ch_u.q1[i, w] += a_l.Γ[12].ch_u.q1[i, w] - a_l.Γ[12].ch_u.q1[i, w]
            a.Γ[13].ch_u.q1[i, w] += a_l.Γ[13].ch_u.q1[i, w] - a_l.Γ[13].ch_u.q1[i, w]
            a.Γ[14].ch_u.q1[i, w] += a_l.Γ[14].ch_u.q1[i, w] - a_l.Γ[14].ch_u.q1[i, w]
            a.Γ[15].ch_u.q1[i, w] += a_l.Γ[15].ch_u.q1[i, w] - a_l.Γ[15].ch_u.q1[i, w]
            a.Γ[16].ch_u.q1[i, w] += a_l.Γ[16].ch_u.q1[i, w] - a_l.Γ[16].ch_u.q1[i, w]
        end 
    end

    # computation for q2_1 and q2_2
    @turbo for v in 1 : num_ν
        for w in 1 : num_Ω
            for i in 1 : num_sites
                # add q2_1 and q2_2 to s channel (right part from v <-> v' exchange)
                a.Γ[1].ch_s.q2_1[i, w, v]  += a_l.Γ[1].ch_s.q2_1[i, w, v]  +  a_l.Γ[1].ch_s.q2_2[r.exchange[i], w, v]
                a.Γ[2].ch_s.q2_1[i, w, v]  += a_l.Γ[2].ch_s.q2_1[i, w, v]  +  a_l.Γ[2].ch_s.q2_2[r.exchange[i], w, v]
                a.Γ[3].ch_s.q2_1[i, w, v]  += a_l.Γ[3].ch_s.q2_1[i, w, v]  +  a_l.Γ[3].ch_s.q2_2[r.exchange[i], w, v]
                a.Γ[4].ch_s.q2_1[i, w, v]  += a_l.Γ[4].ch_s.q2_1[i, w, v]  +  a_l.Γ[4].ch_s.q2_2[r.exchange[i], w, v]
                a.Γ[5].ch_s.q2_1[i, w, v]  += a_l.Γ[5].ch_s.q2_1[i, w, v]  +  a_l.Γ[5].ch_s.q2_2[r.exchange[i], w, v]
                a.Γ[6].ch_s.q2_1[i, w, v]  += a_l.Γ[6].ch_s.q2_1[i, w, v]  +  a_l.Γ[6].ch_s.q2_2[r.exchange[i], w, v]
                a.Γ[7].ch_s.q2_1[i, w, v]  += a_l.Γ[7].ch_s.q2_1[i, w, v]  +  a_l.Γ[7].ch_s.q2_2[r.exchange[i], w, v]
                a.Γ[8].ch_s.q2_1[i, w, v]  += a_l.Γ[8].ch_s.q2_1[i, w, v]  +  a_l.Γ[8].ch_s.q2_2[r.exchange[i], w, v]
                a.Γ[9].ch_s.q2_1[i, w, v]  += a_l.Γ[9].ch_s.q2_1[i, w, v]  +  a_l.Γ[9].ch_s.q2_2[r.exchange[i], w, v]
                a.Γ[10].ch_s.q2_1[i, w, v] += a_l.Γ[10].ch_s.q2_1[i, w, v] + a_l.Γ[10].ch_s.q2_2[r.exchange[i], w, v]
                a.Γ[11].ch_s.q2_1[i, w, v] += a_l.Γ[11].ch_s.q2_1[i, w, v] - a_l.Γ[11].ch_s.q2_2[r.exchange[i], w, v]
                a.Γ[12].ch_s.q2_1[i, w, v] += a_l.Γ[12].ch_s.q2_1[i, w, v] - a_l.Γ[12].ch_s.q2_2[r.exchange[i], w, v]
                a.Γ[13].ch_s.q2_1[i, w, v] += a_l.Γ[13].ch_s.q2_1[i, w, v] - a_l.Γ[13].ch_s.q2_2[r.exchange[i], w, v]
                a.Γ[14].ch_s.q2_1[i, w, v] += a_l.Γ[14].ch_s.q2_1[i, w, v] - a_l.Γ[14].ch_s.q2_2[r.exchange[i], w, v]
                a.Γ[15].ch_s.q2_1[i, w, v] += a_l.Γ[15].ch_s.q2_1[i, w, v] - a_l.Γ[15].ch_s.q2_2[r.exchange[i], w, v]
                a.Γ[16].ch_s.q2_1[i, w, v] += a_l.Γ[16].ch_s.q2_1[i, w, v] - a_l.Γ[16].ch_s.q2_2[r.exchange[i], w, v]
                a.Γ[1].ch_s.q2_2[i, w, v]  += a_l.Γ[1].ch_s.q2_2[i, w, v]  +  a_l.Γ[1].ch_s.q2_1[r.exchange[i], w, v]
                a.Γ[2].ch_s.q2_2[i, w, v]  += a_l.Γ[2].ch_s.q2_2[i, w, v]  +  a_l.Γ[2].ch_s.q2_1[r.exchange[i], w, v]
                a.Γ[3].ch_s.q2_2[i, w, v]  += a_l.Γ[3].ch_s.q2_2[i, w, v]  +  a_l.Γ[3].ch_s.q2_1[r.exchange[i], w, v]
                a.Γ[4].ch_s.q2_2[i, w, v]  += a_l.Γ[4].ch_s.q2_2[i, w, v]  +  a_l.Γ[4].ch_s.q2_1[r.exchange[i], w, v]
                a.Γ[5].ch_s.q2_2[i, w, v]  += a_l.Γ[5].ch_s.q2_2[i, w, v]  +  a_l.Γ[5].ch_s.q2_1[r.exchange[i], w, v]
                a.Γ[6].ch_s.q2_2[i, w, v]  += a_l.Γ[6].ch_s.q2_2[i, w, v]  +  a_l.Γ[6].ch_s.q2_1[r.exchange[i], w, v]
                a.Γ[7].ch_s.q2_2[i, w, v]  += a_l.Γ[7].ch_s.q2_2[i, w, v]  +  a_l.Γ[7].ch_s.q2_1[r.exchange[i], w, v]
                a.Γ[8].ch_s.q2_2[i, w, v]  += a_l.Γ[8].ch_s.q2_2[i, w, v]  +  a_l.Γ[8].ch_s.q2_1[r.exchange[i], w, v]
                a.Γ[9].ch_s.q2_2[i, w, v]  += a_l.Γ[9].ch_s.q2_2[i, w, v]  +  a_l.Γ[9].ch_s.q2_1[r.exchange[i], w, v]
                a.Γ[10].ch_s.q2_2[i, w, v] += a_l.Γ[10].ch_s.q2_2[i, w, v] + a_l.Γ[10].ch_s.q2_1[r.exchange[i], w, v]
                a.Γ[11].ch_s.q2_2[i, w, v] += a_l.Γ[11].ch_s.q2_2[i, w, v] - a_l.Γ[11].ch_s.q2_1[r.exchange[i], w, v]
                a.Γ[12].ch_s.q2_2[i, w, v] += a_l.Γ[12].ch_s.q2_2[i, w, v] - a_l.Γ[12].ch_s.q2_1[r.exchange[i], w, v]
                a.Γ[13].ch_s.q2_2[i, w, v] += a_l.Γ[13].ch_s.q2_2[i, w, v] - a_l.Γ[13].ch_s.q2_1[r.exchange[i], w, v]
                a.Γ[14].ch_s.q2_2[i, w, v] += a_l.Γ[14].ch_s.q2_2[i, w, v] - a_l.Γ[14].ch_s.q2_1[r.exchange[i], w, v]
                a.Γ[15].ch_s.q2_2[i, w, v] += a_l.Γ[15].ch_s.q2_2[i, w, v] - a_l.Γ[15].ch_s.q2_1[r.exchange[i], w, v]
                a.Γ[16].ch_s.q2_2[i, w, v] += a_l.Γ[16].ch_s.q2_2[i, w, v] - a_l.Γ[16].ch_s.q2_1[r.exchange[i], w, v]
                

                # add q2_1 and q2_2 to t channel (right part from v <-> v' exchange)
                a.Γ[1].ch_t.q2_1[i, w, v]  += a_l.Γ[1].ch_t.q2_1[i, w, v]  +  a_l.Γ[1].ch_t.q2_2[r.exchange[i], w, v]
                a.Γ[2].ch_t.q2_1[i, w, v]  += a_l.Γ[2].ch_t.q2_1[i, w, v]  +  a_l.Γ[2].ch_t.q2_2[r.exchange[i], w, v]
                a.Γ[3].ch_t.q2_1[i, w, v]  += a_l.Γ[3].ch_t.q2_1[i, w, v]  +  a_l.Γ[3].ch_t.q2_2[r.exchange[i], w, v]
                a.Γ[4].ch_t.q2_1[i, w, v]  += a_l.Γ[4].ch_t.q2_1[i, w, v]  +  a_l.Γ[4].ch_t.q2_2[r.exchange[i], w, v]
                a.Γ[5].ch_t.q2_1[i, w, v]  += a_l.Γ[5].ch_t.q2_1[i, w, v]  +  a_l.Γ[5].ch_t.q2_2[r.exchange[i], w, v]
                a.Γ[6].ch_t.q2_1[i, w, v]  += a_l.Γ[6].ch_t.q2_1[i, w, v]  +  a_l.Γ[6].ch_t.q2_2[r.exchange[i], w, v]
                a.Γ[7].ch_t.q2_1[i, w, v]  += a_l.Γ[7].ch_t.q2_1[i, w, v]  +  a_l.Γ[7].ch_t.q2_2[r.exchange[i], w, v]
                a.Γ[8].ch_t.q2_1[i, w, v]  += a_l.Γ[8].ch_t.q2_1[i, w, v]  +  a_l.Γ[8].ch_t.q2_2[r.exchange[i], w, v]
                a.Γ[9].ch_t.q2_1[i, w, v]  += a_l.Γ[9].ch_t.q2_1[i, w, v]  +  a_l.Γ[9].ch_t.q2_2[r.exchange[i], w, v]
                a.Γ[10].ch_t.q2_1[i, w, v] += a_l.Γ[10].ch_t.q2_1[i, w, v] + a_l.Γ[10].ch_t.q2_2[r.exchange[i], w, v]
                a.Γ[11].ch_t.q2_1[i, w, v] += a_l.Γ[11].ch_t.q2_1[i, w, v] - a_l.Γ[11].ch_t.q2_2[r.exchange[i], w, v]
                a.Γ[12].ch_t.q2_1[i, w, v] += a_l.Γ[12].ch_t.q2_1[i, w, v] - a_l.Γ[12].ch_t.q2_2[r.exchange[i], w, v]
                a.Γ[13].ch_t.q2_1[i, w, v] += a_l.Γ[13].ch_t.q2_1[i, w, v] - a_l.Γ[13].ch_t.q2_2[r.exchange[i], w, v]
                a.Γ[14].ch_t.q2_1[i, w, v] += a_l.Γ[14].ch_t.q2_1[i, w, v] - a_l.Γ[14].ch_t.q2_2[r.exchange[i], w, v]
                a.Γ[15].ch_t.q2_1[i, w, v] += a_l.Γ[15].ch_t.q2_1[i, w, v] - a_l.Γ[15].ch_t.q2_2[r.exchange[i], w, v]
                a.Γ[16].ch_t.q2_1[i, w, v] += a_l.Γ[16].ch_t.q2_1[i, w, v] - a_l.Γ[16].ch_t.q2_2[r.exchange[i], w, v]
                a.Γ[1].ch_t.q2_2[i, w, v]  += a_l.Γ[1].ch_t.q2_2[i, w, v]  +  a_l.Γ[1].ch_t.q2_1[r.exchange[i], w, v]
                a.Γ[2].ch_t.q2_2[i, w, v]  += a_l.Γ[2].ch_t.q2_2[i, w, v]  +  a_l.Γ[2].ch_t.q2_1[r.exchange[i], w, v]
                a.Γ[3].ch_t.q2_2[i, w, v]  += a_l.Γ[3].ch_t.q2_2[i, w, v]  +  a_l.Γ[3].ch_t.q2_1[r.exchange[i], w, v]
                a.Γ[4].ch_t.q2_2[i, w, v]  += a_l.Γ[4].ch_t.q2_2[i, w, v]  +  a_l.Γ[4].ch_t.q2_1[r.exchange[i], w, v]
                a.Γ[5].ch_t.q2_2[i, w, v]  += a_l.Γ[5].ch_t.q2_2[i, w, v]  +  a_l.Γ[5].ch_t.q2_1[r.exchange[i], w, v]
                a.Γ[6].ch_t.q2_2[i, w, v]  += a_l.Γ[6].ch_t.q2_2[i, w, v]  +  a_l.Γ[6].ch_t.q2_1[r.exchange[i], w, v]
                a.Γ[7].ch_t.q2_2[i, w, v]  += a_l.Γ[7].ch_t.q2_2[i, w, v]  +  a_l.Γ[7].ch_t.q2_1[r.exchange[i], w, v]
                a.Γ[8].ch_t.q2_2[i, w, v]  += a_l.Γ[8].ch_t.q2_2[i, w, v]  +  a_l.Γ[8].ch_t.q2_1[r.exchange[i], w, v]
                a.Γ[9].ch_t.q2_2[i, w, v]  += a_l.Γ[9].ch_t.q2_2[i, w, v]  +  a_l.Γ[9].ch_t.q2_1[r.exchange[i], w, v]
                a.Γ[10].ch_t.q2_2[i, w, v] += a_l.Γ[10].ch_t.q2_2[i, w, v] + a_l.Γ[10].ch_t.q2_1[r.exchange[i], w, v]
                a.Γ[11].ch_t.q2_2[i, w, v] += a_l.Γ[11].ch_t.q2_2[i, w, v] - a_l.Γ[11].ch_t.q2_1[r.exchange[i], w, v]
                a.Γ[12].ch_t.q2_2[i, w, v] += a_l.Γ[12].ch_t.q2_2[i, w, v] - a_l.Γ[12].ch_t.q2_1[r.exchange[i], w, v]
                a.Γ[13].ch_t.q2_2[i, w, v] += a_l.Γ[13].ch_t.q2_2[i, w, v] - a_l.Γ[13].ch_t.q2_1[r.exchange[i], w, v]
                a.Γ[14].ch_t.q2_2[i, w, v] += a_l.Γ[14].ch_t.q2_2[i, w, v] - a_l.Γ[14].ch_t.q2_1[r.exchange[i], w, v]
                a.Γ[15].ch_t.q2_2[i, w, v] += a_l.Γ[15].ch_t.q2_2[i, w, v] - a_l.Γ[15].ch_t.q2_1[r.exchange[i], w, v]
                a.Γ[16].ch_t.q2_2[i, w, v] += a_l.Γ[16].ch_t.q2_2[i, w, v] - a_l.Γ[16].ch_t.q2_1[r.exchange[i], w, v]
                

                # add q2_1 and q2_2 to u channel (right part from v <-> v' exchange)
                a.Γ[1].ch_u.q2_1[i, w, v]  += a_l.Γ[1].ch_u.q2_1[i, w, v]  +  a_l.Γ[1].ch_u.q2_2[i, w, v]
                a.Γ[2].ch_u.q2_1[i, w, v]  += a_l.Γ[2].ch_u.q2_1[i, w, v]  +  a_l.Γ[2].ch_u.q2_2[i, w, v]
                a.Γ[3].ch_u.q2_1[i, w, v]  += a_l.Γ[3].ch_u.q2_1[i, w, v]  +  a_l.Γ[3].ch_u.q2_2[i, w, v]
                a.Γ[4].ch_u.q2_1[i, w, v]  += a_l.Γ[4].ch_u.q2_1[i, w, v]  +  a_l.Γ[4].ch_u.q2_2[i, w, v]
                a.Γ[5].ch_u.q2_1[i, w, v]  += a_l.Γ[5].ch_u.q2_1[i, w, v]  +  a_l.Γ[5].ch_u.q2_2[i, w, v]
                a.Γ[6].ch_u.q2_1[i, w, v]  += a_l.Γ[6].ch_u.q2_1[i, w, v]  +  a_l.Γ[6].ch_u.q2_2[i, w, v]
                a.Γ[7].ch_u.q2_1[i, w, v]  += a_l.Γ[7].ch_u.q2_1[i, w, v]  +  a_l.Γ[7].ch_u.q2_2[i, w, v]
                a.Γ[8].ch_u.q2_1[i, w, v]  += a_l.Γ[8].ch_u.q2_1[i, w, v]  +  a_l.Γ[8].ch_u.q2_2[i, w, v]
                a.Γ[9].ch_u.q2_1[i, w, v]  += a_l.Γ[9].ch_u.q2_1[i, w, v]  +  a_l.Γ[9].ch_u.q2_2[i, w, v]
                a.Γ[10].ch_u.q2_1[i, w, v] += a_l.Γ[10].ch_u.q2_1[i, w, v] + a_l.Γ[10].ch_u.q2_2[i, w, v]
                a.Γ[11].ch_u.q2_1[i, w, v] += a_l.Γ[11].ch_u.q2_1[i, w, v] - a_l.Γ[11].ch_u.q2_2[i, w, v]
                a.Γ[12].ch_u.q2_1[i, w, v] += a_l.Γ[12].ch_u.q2_1[i, w, v] - a_l.Γ[12].ch_u.q2_2[i, w, v]
                a.Γ[13].ch_u.q2_1[i, w, v] += a_l.Γ[13].ch_u.q2_1[i, w, v] - a_l.Γ[13].ch_u.q2_2[i, w, v]
                a.Γ[14].ch_u.q2_1[i, w, v] += a_l.Γ[14].ch_u.q2_1[i, w, v] - a_l.Γ[14].ch_u.q2_2[i, w, v]
                a.Γ[15].ch_u.q2_1[i, w, v] += a_l.Γ[15].ch_u.q2_1[i, w, v] - a_l.Γ[15].ch_u.q2_2[i, w, v]
                a.Γ[16].ch_u.q2_1[i, w, v] += a_l.Γ[16].ch_u.q2_1[i, w, v] - a_l.Γ[16].ch_u.q2_2[i, w, v] 
                a.Γ[1].ch_u.q2_2[i, w, v]  += a_l.Γ[1].ch_u.q2_2[i, w, v]  +  a_l.Γ[1].ch_u.q2_1[i, w, v]
                a.Γ[2].ch_u.q2_2[i, w, v]  += a_l.Γ[2].ch_u.q2_2[i, w, v]  +  a_l.Γ[2].ch_u.q2_1[i, w, v]
                a.Γ[3].ch_u.q2_2[i, w, v]  += a_l.Γ[3].ch_u.q2_2[i, w, v]  +  a_l.Γ[3].ch_u.q2_1[i, w, v]
                a.Γ[4].ch_u.q2_2[i, w, v]  += a_l.Γ[4].ch_u.q2_2[i, w, v]  +  a_l.Γ[4].ch_u.q2_1[i, w, v]
                a.Γ[5].ch_u.q2_2[i, w, v]  += a_l.Γ[5].ch_u.q2_2[i, w, v]  +  a_l.Γ[5].ch_u.q2_1[i, w, v]
                a.Γ[6].ch_u.q2_2[i, w, v]  += a_l.Γ[6].ch_u.q2_2[i, w, v]  +  a_l.Γ[6].ch_u.q2_1[i, w, v]
                a.Γ[7].ch_u.q2_2[i, w, v]  += a_l.Γ[7].ch_u.q2_2[i, w, v]  +  a_l.Γ[7].ch_u.q2_1[i, w, v]
                a.Γ[8].ch_u.q2_2[i, w, v]  += a_l.Γ[8].ch_u.q2_2[i, w, v]  +  a_l.Γ[8].ch_u.q2_1[i, w, v]
                a.Γ[9].ch_u.q2_2[i, w, v]  += a_l.Γ[9].ch_u.q2_2[i, w, v]  +  a_l.Γ[9].ch_u.q2_1[i, w, v]
                a.Γ[10].ch_u.q2_2[i, w, v] += a_l.Γ[10].ch_u.q2_2[i, w, v] + a_l.Γ[10].ch_u.q2_1[i, w, v]
                a.Γ[11].ch_u.q2_2[i, w, v] += a_l.Γ[11].ch_u.q2_2[i, w, v] - a_l.Γ[11].ch_u.q2_1[i, w, v]
                a.Γ[12].ch_u.q2_2[i, w, v] += a_l.Γ[12].ch_u.q2_2[i, w, v] - a_l.Γ[12].ch_u.q2_1[i, w, v]
                a.Γ[13].ch_u.q2_2[i, w, v] += a_l.Γ[13].ch_u.q2_2[i, w, v] - a_l.Γ[13].ch_u.q2_1[i, w, v]
                a.Γ[14].ch_u.q2_2[i, w, v] += a_l.Γ[14].ch_u.q2_2[i, w, v] - a_l.Γ[14].ch_u.q2_1[i, w, v]
                a.Γ[15].ch_u.q2_2[i, w, v] += a_l.Γ[15].ch_u.q2_2[i, w, v] - a_l.Γ[15].ch_u.q2_1[i, w, v]
                a.Γ[16].ch_u.q2_2[i, w, v] += a_l.Γ[16].ch_u.q2_2[i, w, v] - a_l.Γ[16].ch_u.q2_1[i, w, v]
            end
        end
    end

    # computation for q3
    @turbo for vp in 1 : num_ν
        for v in 1 : num_ν
            for w in 1 : num_Ω
                for i in 1 : num_sites
                    # add q3 to s channel (right part from v <-> v' exchange)
                    a.Γ[1].ch_s.q3[i, w, v, vp]  += a_l.Γ[1].ch_s.q3[i, w, v, vp]  +  a_l.Γ[1].ch_s.q3[r.exchange[i], w, vp, v]
                    a.Γ[2].ch_s.q3[i, w, v, vp]  += a_l.Γ[2].ch_s.q3[i, w, v, vp]  +  a_l.Γ[2].ch_s.q3[r.exchange[i], w, vp, v]
                    a.Γ[3].ch_s.q3[i, w, v, vp]  += a_l.Γ[3].ch_s.q3[i, w, v, vp]  +  a_l.Γ[3].ch_s.q3[r.exchange[i], w, vp, v]
                    a.Γ[4].ch_s.q3[i, w, v, vp]  += a_l.Γ[4].ch_s.q3[i, w, v, vp]  +  a_l.Γ[4].ch_s.q3[r.exchange[i], w, vp, v]
                    a.Γ[5].ch_s.q3[i, w, v, vp]  += a_l.Γ[5].ch_s.q3[i, w, v, vp]  +  a_l.Γ[5].ch_s.q3[r.exchange[i], w, vp, v]
                    a.Γ[6].ch_s.q3[i, w, v, vp]  += a_l.Γ[6].ch_s.q3[i, w, v, vp]  +  a_l.Γ[6].ch_s.q3[r.exchange[i], w, vp, v]
                    a.Γ[7].ch_s.q3[i, w, v, vp]  += a_l.Γ[7].ch_s.q3[i, w, v, vp]  +  a_l.Γ[7].ch_s.q3[r.exchange[i], w, vp, v]
                    a.Γ[8].ch_s.q3[i, w, v, vp]  += a_l.Γ[8].ch_s.q3[i, w, v, vp]  +  a_l.Γ[8].ch_s.q3[r.exchange[i], w, vp, v]
                    a.Γ[9].ch_s.q3[i, w, v, vp]  += a_l.Γ[9].ch_s.q3[i, w, v, vp]  +  a_l.Γ[9].ch_s.q3[r.exchange[i], w, vp, v]
                    a.Γ[10].ch_s.q3[i, w, v, vp] += a_l.Γ[10].ch_s.q3[i, w, v, vp] + a_l.Γ[10].ch_s.q3[r.exchange[i], w, vp, v]
                    a.Γ[11].ch_s.q3[i, w, v, vp] += a_l.Γ[11].ch_s.q3[i, w, v, vp] - a_l.Γ[11].ch_s.q3[r.exchange[i], w, vp, v]
                    a.Γ[12].ch_s.q3[i, w, v, vp] += a_l.Γ[12].ch_s.q3[i, w, v, vp] - a_l.Γ[12].ch_s.q3[r.exchange[i], w, vp, v]
                    a.Γ[13].ch_s.q3[i, w, v, vp] += a_l.Γ[13].ch_s.q3[i, w, v, vp] - a_l.Γ[13].ch_s.q3[r.exchange[i], w, vp, v]
                    a.Γ[14].ch_s.q3[i, w, v, vp] += a_l.Γ[14].ch_s.q3[i, w, v, vp] - a_l.Γ[14].ch_s.q3[r.exchange[i], w, vp, v]
                    a.Γ[15].ch_s.q3[i, w, v, vp] += a_l.Γ[15].ch_s.q3[i, w, v, vp] - a_l.Γ[15].ch_s.q3[r.exchange[i], w, vp, v]
                    a.Γ[16].ch_s.q3[i, w, v, vp] += a_l.Γ[16].ch_s.q3[i, w, v, vp] - a_l.Γ[16].ch_s.q3[r.exchange[i], w, vp, v]
                    
                    #add q3 to t channel (right part from v <-> v' exchange)
                    a.Γ[1].ch_t.q3[i, w, v, vp]  += a_l.Γ[1].ch_t.q3[i, w, v, vp]  +  a_l.Γ[1].ch_t.q3[r.exchange[i], w, vp, v]
                    a.Γ[2].ch_t.q3[i, w, v, vp]  += a_l.Γ[2].ch_t.q3[i, w, v, vp]  +  a_l.Γ[2].ch_t.q3[r.exchange[i], w, vp, v]
                    a.Γ[3].ch_t.q3[i, w, v, vp]  += a_l.Γ[3].ch_t.q3[i, w, v, vp]  +  a_l.Γ[3].ch_t.q3[r.exchange[i], w, vp, v]
                    a.Γ[4].ch_t.q3[i, w, v, vp]  += a_l.Γ[4].ch_t.q3[i, w, v, vp]  +  a_l.Γ[4].ch_t.q3[r.exchange[i], w, vp, v]
                    a.Γ[5].ch_t.q3[i, w, v, vp]  += a_l.Γ[5].ch_t.q3[i, w, v, vp]  +  a_l.Γ[5].ch_t.q3[r.exchange[i], w, vp, v]
                    a.Γ[6].ch_t.q3[i, w, v, vp]  += a_l.Γ[6].ch_t.q3[i, w, v, vp]  +  a_l.Γ[6].ch_t.q3[r.exchange[i], w, vp, v]
                    a.Γ[7].ch_t.q3[i, w, v, vp]  += a_l.Γ[7].ch_t.q3[i, w, v, vp]  +  a_l.Γ[7].ch_t.q3[r.exchange[i], w, vp, v]
                    a.Γ[8].ch_t.q3[i, w, v, vp]  += a_l.Γ[8].ch_t.q3[i, w, v, vp]  +  a_l.Γ[8].ch_t.q3[r.exchange[i], w, vp, v]
                    a.Γ[9].ch_t.q3[i, w, v, vp]  += a_l.Γ[9].ch_t.q3[i, w, v, vp]  +  a_l.Γ[9].ch_t.q3[r.exchange[i], w, vp, v]
                    a.Γ[10].ch_t.q3[i, w, v, vp] += a_l.Γ[10].ch_t.q3[i, w, v, vp] + a_l.Γ[10].ch_t.q3[r.exchange[i], w, vp, v]
                    a.Γ[11].ch_t.q3[i, w, v, vp] += a_l.Γ[11].ch_t.q3[i, w, v, vp] - a_l.Γ[11].ch_t.q3[r.exchange[i], w, vp, v]
                    a.Γ[12].ch_t.q3[i, w, v, vp] += a_l.Γ[12].ch_t.q3[i, w, v, vp] - a_l.Γ[12].ch_t.q3[r.exchange[i], w, vp, v]
                    a.Γ[13].ch_t.q3[i, w, v, vp] += a_l.Γ[13].ch_t.q3[i, w, v, vp] - a_l.Γ[13].ch_t.q3[r.exchange[i], w, vp, v]
                    a.Γ[14].ch_t.q3[i, w, v, vp] += a_l.Γ[14].ch_t.q3[i, w, v, vp] - a_l.Γ[14].ch_t.q3[r.exchange[i], w, vp, v]
                    a.Γ[15].ch_t.q3[i, w, v, vp] += a_l.Γ[15].ch_t.q3[i, w, v, vp] - a_l.Γ[15].ch_t.q3[r.exchange[i], w, vp, v]
                    a.Γ[16].ch_t.q3[i, w, v, vp] += a_l.Γ[16].ch_t.q3[i, w, v, vp] - a_l.Γ[16].ch_t.q3[r.exchange[i], w, vp, v]
                    
                    #add q3 to u channel (right part from v <-> v' exchange)
                    a.Γ[1].ch_u.q3[i, w, v, vp]  += a_l.Γ[1].ch_u.q3[i, w, v, vp]  +  a_l.Γ[1].ch_u.q3[i, w, vp, v]
                    a.Γ[2].ch_u.q3[i, w, v, vp]  += a_l.Γ[2].ch_u.q3[i, w, v, vp]  +  a_l.Γ[2].ch_u.q3[i, w, vp, v]
                    a.Γ[3].ch_u.q3[i, w, v, vp]  += a_l.Γ[3].ch_u.q3[i, w, v, vp]  +  a_l.Γ[3].ch_u.q3[i, w, vp, v]
                    a.Γ[4].ch_u.q3[i, w, v, vp]  += a_l.Γ[4].ch_u.q3[i, w, v, vp]  +  a_l.Γ[4].ch_u.q3[i, w, vp, v]
                    a.Γ[5].ch_u.q3[i, w, v, vp]  += a_l.Γ[5].ch_u.q3[i, w, v, vp]  +  a_l.Γ[5].ch_u.q3[i, w, vp, v]
                    a.Γ[6].ch_u.q3[i, w, v, vp]  += a_l.Γ[6].ch_u.q3[i, w, v, vp]  +  a_l.Γ[6].ch_u.q3[i, w, vp, v]
                    a.Γ[7].ch_u.q3[i, w, v, vp]  += a_l.Γ[7].ch_u.q3[i, w, v, vp]  +  a_l.Γ[7].ch_u.q3[i, w, vp, v]
                    a.Γ[8].ch_u.q3[i, w, v, vp]  += a_l.Γ[8].ch_u.q3[i, w, v, vp]  +  a_l.Γ[8].ch_u.q3[i, w, vp, v]
                    a.Γ[9].ch_u.q3[i, w, v, vp]  += a_l.Γ[9].ch_u.q3[i, w, v, vp]  +  a_l.Γ[9].ch_u.q3[i, w, vp, v]
                    a.Γ[10].ch_u.q3[i, w, v, vp] += a_l.Γ[10].ch_u.q3[i, w, v, vp] + a_l.Γ[10].ch_u.q3[i, w, vp, v]
                    a.Γ[11].ch_u.q3[i, w, v, vp] += a_l.Γ[11].ch_u.q3[i, w, v, vp] - a_l.Γ[11].ch_u.q3[i, w, vp, v]
                    a.Γ[12].ch_u.q3[i, w, v, vp] += a_l.Γ[12].ch_u.q3[i, w, v, vp] - a_l.Γ[12].ch_u.q3[i, w, vp, v]
                    a.Γ[13].ch_u.q3[i, w, v, vp] += a_l.Γ[13].ch_u.q3[i, w, v, vp] - a_l.Γ[13].ch_u.q3[i, w, vp, v]
                    a.Γ[14].ch_u.q3[i, w, v, vp] += a_l.Γ[14].ch_u.q3[i, w, v, vp] - a_l.Γ[14].ch_u.q3[i, w, vp, v]
                    a.Γ[15].ch_u.q3[i, w, v, vp] += a_l.Γ[15].ch_u.q3[i, w, v, vp] - a_l.Γ[15].ch_u.q3[i, w, vp, v]
                    a.Γ[16].ch_u.q3[i, w, v, vp] += a_l.Γ[16].ch_u.q3[i, w, v, vp] - a_l.Γ[16].ch_u.q3[i, w, vp, v]
                end
            end
        end
    end

    return nothing
end 
