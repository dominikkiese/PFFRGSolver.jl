# define vertex struct as composition of bare vertex and channels
struct vertex 
    bare :: Vector{Float64}
    ch_s :: channel 
    ch_t :: channel 
    ch_u :: channel 
end

# generate an empty vertex from frequency meshes and reduced lattice
function get_vertex_empty(
    r :: reduced_lattice,
    m :: mesh          
    ) :: vertex

    # init bare 
    bare = zeros(Float64, length(r.sites))

    # init channel 
    ch_s = get_channel_empty(r, m)
    ch_t = get_channel_empty(r, m)
    ch_u = get_channel_empty(r, m)

    # build vertex
    Γ = vertex(bare, ch_s, ch_t, ch_u)

    return Γ
end





# get interpolated value of vertex in certain channel for a given frequency buffer (without bare, ch_s = 1, ch_t = 2, ch_u = 3)
function get_vertex(
    site :: Int64, 
    b    :: buffer, 
    Γ    :: vertex,
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

# get interpolated value of vertex in certain channel for a given frequency buffer on all lattice sites (without bare, ch_s = 1, ch_t = 2, ch_u = 3)
function get_vertex_avx!(
    r        :: reduced_lattice,
    b        :: buffer, 
    Γ        :: vertex,
    ch       :: Int64,
    temp     :: SubArray{Float64, 1, Array{Float64, 3}}
    ;
    exchange :: Bool    = false,
    sgn      :: Float64 = 1.0
    )        :: Nothing 

    if ch == 1
        get_channel_avx!(r, b, Γ.ch_s, temp, exchange = exchange, sgn = sgn)
    elseif ch == 2
        get_channel_avx!(r, b, Γ.ch_t, temp, exchange = exchange, sgn = sgn)
    elseif ch == 3
        get_channel_avx!(r, b, Γ.ch_u, temp, exchange = exchange, sgn = sgn)
    end 

    return nothing
end





# replace vertex with another vertex (except for bare)
function replace_with!(
    Γ1 :: vertex,
    Γ2 :: vertex
    )  :: Nothing

    replace_with!(Γ1.ch_s, Γ2.ch_s)
    replace_with!(Γ1.ch_t, Γ2.ch_t)
    replace_with!(Γ1.ch_u, Γ2.ch_u)

    return nothing 
end 

# multiply vertex with factor (except for bare)
function mult_with!(
    Γ   :: vertex, 
    fac :: Float64
    )   :: Nothing 

    mult_with!(Γ.ch_s, fac)
    mult_with!(Γ.ch_t, fac)
    mult_with!(Γ.ch_u, fac)

    return nothing 
end

# multiply vertex with some factor and add to other vertex (except for bare)
function mult_with_add_to!(
    Γ2  :: vertex,
    fac :: Float64,
    Γ1  :: vertex
    )   :: Nothing 

    mult_with_add_to!(Γ2.ch_s, fac, Γ1.ch_s)
    mult_with_add_to!(Γ2.ch_t, fac, Γ1.ch_t)
    mult_with_add_to!(Γ2.ch_u, fac, Γ1.ch_u)

    return nothing 
end

# obtain maximum between channels 
function get_abs_max(
    Γ :: vertex 
    ) :: Float64

    max_s = get_abs_max(Γ.ch_s)
    max_t = get_abs_max(Γ.ch_t)
    max_u = get_abs_max(Γ.ch_u)
    Γ_max = max(max_s, max_t, max_u)

    return Γ_max
end

# set asymptotic limits by scanning the boundaries of q3
function limits!(
    Γ :: vertex
    ) :: Nothing

    limits!(Γ.ch_s)
    limits!(Γ.ch_t)
    limits!(Γ.ch_u)
    
    return nothing 
end

# resample vertex from old to new frequency meshes 
function resample_from_to!(
    m_old :: mesh,
    Γ_old :: vertex,
    m_new :: mesh,
    Γ_new :: vertex
    )     :: Nothing 

    resample_from_to!(m_old, Γ_old.ch_s, m_new, Γ_new.ch_s)
    resample_from_to!(m_old, Γ_old.ch_t, m_new, Γ_new.ch_t)
    resample_from_to!(m_old, Γ_old.ch_u, m_new, Γ_new.ch_u)

    return nothing 
end





    




