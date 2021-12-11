"""
    Vertex 

Struct containing bare values of vertex component and the respective s, t and u channel.
* `bare :: Vector{Float64}` : bare couplings on irreducible lattice sites
* `ch_s :: Channel`         : particle-particle (s) channel
* `ch_t :: Channel`         : direct particle-hole (t) channel
* `ch_u :: Channel`         : crossed particle-hole (u) channel
"""
struct Vertex 
    bare :: Vector{Float64}
    ch_s :: Channel 
    ch_t :: Channel 
    ch_u :: Channel 
end

# generate vertex dummy
function get_vertex_empty(
    r :: Reduced_lattice,
    m :: Mesh          
    ) :: Vertex

    # init bare 
    bare = zeros(Float64, length(r.sites))

    # init channels 
    ch_s = get_channel_empty(r, m)
    ch_t = get_channel_empty(r, m)
    ch_u = get_channel_empty(r, m)

    # build vertex
    Γ = Vertex(bare, ch_s, ch_t, ch_u)

    return Γ
end





# get interpolated value of vertex in certain channel for a given frequency buffer
function get_vertex(
    site :: Int64, 
    b    :: Buffer, 
    Γ    :: Vertex,
    ch   :: Int64
    )    :: Float64

    val = 0.0

    if ch == 1
        val = get_channel(site, b, Γ.ch_s)
    elseif ch == 2
        val = get_channel(site, b, Γ.ch_t)
    elseif ch == 3
        val = get_channel(site, b, Γ.ch_u)
    end 

    return val 
end

# get interpolated value of vertex in certain channel for a given frequency buffer on all lattice sites
function get_vertex_avx!(
    r        :: Reduced_lattice,
    b        :: Buffer, 
    Γ        :: Vertex,
    ch       :: Int64,
    temp     :: SubArray{Float64, 1, Array{Float64, 3}},
    exchange :: Bool,
    sgn      :: Float64
    )        :: Nothing 

    if ch == 1
        get_channel_avx!(r, b, Γ.ch_s, temp, exchange, sgn)
    elseif ch == 2
        get_channel_avx!(r, b, Γ.ch_t, temp, exchange, sgn)
    elseif ch == 3
        get_channel_avx!(r, b, Γ.ch_u, temp, exchange, sgn)
    end 

    return nothing
end





# replace vertex with another vertex (except for bare)
function replace_with!(
    Γ1 :: Vertex,
    Γ2 :: Vertex
    )  :: Nothing

    replace_with!(Γ1.ch_s, Γ2.ch_s)
    replace_with!(Γ1.ch_t, Γ2.ch_t)
    replace_with!(Γ1.ch_u, Γ2.ch_u)

    return nothing 
end 

# multiply vertex with factor (except for bare)
function mult_with!(
    Γ   :: Vertex, 
    fac :: Float64
    )   :: Nothing 

    mult_with!(Γ.ch_s, fac)
    mult_with!(Γ.ch_t, fac)
    mult_with!(Γ.ch_u, fac)

    return nothing 
end

# multiply vertex with some factor and add to other vertex (except for bare)
function mult_with_add_to!(
    Γ2  :: Vertex,
    fac :: Float64,
    Γ1  :: Vertex
    )   :: Nothing 

    mult_with_add_to!(Γ2.ch_s, fac, Γ1.ch_s)
    mult_with_add_to!(Γ2.ch_t, fac, Γ1.ch_t)
    mult_with_add_to!(Γ2.ch_u, fac, Γ1.ch_u)

    return nothing 
end

"""
    get_abs_max(
        Γ :: Vertex 
        ) :: Float64

Returns maximum absolute value of a vertex component across channels.
"""
function get_abs_max(
    Γ :: Vertex 
    ) :: Float64

    max_s = get_abs_max(Γ.ch_s)
    max_t = get_abs_max(Γ.ch_t)
    max_u = get_abs_max(Γ.ch_u)
    Γ_max = max(max_s, max_t, max_u)

    return Γ_max
end

# resample a vertex component to new meshes via trilinear interpolation
function resample_from_to!(
    m_old :: Mesh,
    Γ_old :: Vertex,
    m_new :: Mesh,
    Γ_new :: Vertex
    )     :: Nothing 

    resample_from_to!(m_old.Ωs, m_old.νs, Γ_old.ch_s, m_new.Ωs, m_new.νs, Γ_new.ch_s)
    resample_from_to!(m_old.Ωt, m_old.νt, Γ_old.ch_t, m_new.Ωt, m_new.νt, Γ_new.ch_t)
    resample_from_to!(m_old.Ωs, m_old.νs, Γ_old.ch_u, m_new.Ωs, m_new.νs, Γ_new.ch_u)

    return nothing 
end