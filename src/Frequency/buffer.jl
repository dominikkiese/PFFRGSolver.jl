"""
    Buffer

Struct used for reading out vertices from Action struct. 
Contains symmetry related flags, (asymptotic) kernel specification and interpolation parameters.
* `exchange_flag :: Bool`  : flag for site [(i0, j) <-> (j, i0)] and spin [(μ, ν) <-> (ν, μ)] exchange
* `map_flag      :: Bool`  : flag for channel mapping s <-> u
* `sgn_μν        :: Bool`  : sign function for combined spin indices 
* `sgn_μ         :: Bool`  : sign function for first spin index 
* `sgn_ν         :: Bool`  : sign function for second spin index 
* `kernel        :: Int64` : specification of asymptotic kernel to be interpolated
* `p1            :: Param` : interpolation parameters for bosonic frequency argument
* `p2            :: Param` : interpolation parameters for first fermionic frequency argument
* `p3            :: Param` : interpolation parameters for second fermionic frequency argument
"""
struct Buffer
    exchange_flag :: Bool
    map_flag      :: Bool 
    sgn_μν        :: Bool
    sgn_μ         :: Bool
    sgn_ν         :: Bool
    kernel        :: Int64
    p1            :: Param
    p2            :: Param
    p3            :: Param
end

# function to invert flag
function set_flag(
    flag :: Bool
    )    :: Bool

    if flag
        flag = false
    else
        flag = true
    end

    return flag
end

# generate buffer dummy
function get_buffer_empty() :: Buffer

    b = Buffer(false, false, false, false, false, 0, get_param_empty(), get_param_empty(), get_param_empty())

    return b
end

# generate generic access buffer for Action struct given flags
function get_buffer(
    w             :: Float64,
    v             :: Float64,
    vp            :: Float64,
    Ω             :: Vector{Float64},
    ν             :: Vector{Float64},
    exchange_flag :: Bool,
    map_flag      :: Bool,
    sgn_μν        :: Bool,
    sgn_μ         :: Bool,
    sgn_ν         :: Bool
    )             :: Buffer

    if Ω[end] < abs(w)
        return get_buffer_empty()
    else
        if ν[end] < abs(v)
            if ν[end] < abs(vp)
                # interpolation for q1
                return Buffer(exchange_flag, map_flag, sgn_μν, sgn_μ, sgn_ν, 1, get_param(w, Ω), get_param_empty(), get_param_empty())
            else
                # interpolation for q2_2
                return Buffer(exchange_flag, map_flag, sgn_μν, sgn_μ, sgn_ν, 3, get_param(w, Ω), get_param_empty(), get_param(vp, ν))
            end
        else
            if ν[end] < abs(vp)
                # interpolation for q2_1
                return Buffer(exchange_flag, map_flag, sgn_μν, sgn_μ, sgn_ν, 2, get_param(w, Ω), get_param(v, ν), get_param_empty())
            else
                # interpolation for q3
                return Buffer(exchange_flag, map_flag, sgn_μν, sgn_μ, sgn_ν, 4, get_param(w, Ω), get_param(v, ν), get_param(vp, ν))
            end
        end
    end
end

# generate access buffer for s channel of Action struct
function get_buffer_s(
    comp :: Int64, 
    w    :: Float64,
    v    :: Float64,
    vp   :: Float64,
    m    :: Mesh
    )    :: Buffer

    # init flags
    exchange_flag = false
    map_flag      = false 
    sgn_μν        = false
    sgn_μ         = false
    sgn_ν         = false

    # do -w -> w
    if w < 0.0
        w             *= -1.0
        exchange_flag  = set_flag(exchange_flag)
    end

    # do -v -> v
    if v < 0.0
        v             *= -1.0
        exchange_flag  = set_flag(exchange_flag)
        map_flag       = set_flag(map_flag)
        sgn_μ          = set_flag(sgn_μ)
    end

    # do -vp -> vp
    if vp < 0.0
        vp       *= -1.0
        map_flag  = set_flag(map_flag)
        sgn_ν     = set_flag(sgn_ν)
    end

    # deref meshes for interpolation, respecting possible mapping to u channel
    Ω = m.Ωs[comp]
    ν = m.νs[comp]

    if map_flag
        Ω = m.Ωu[comp]
        ν = m.νu[comp]
    end

    return get_buffer(w, v, vp, Ω, ν, exchange_flag, map_flag, sgn_μν, sgn_μ, sgn_ν)
end

# generate access buffer for t channel of Action struct
function get_buffer_t(
    comp :: Int64, 
    w    :: Float64,
    v    :: Float64,
    vp   :: Float64,
    m    :: Mesh
    )    :: Buffer

    # init flags
    exchange_flag = false
    map_flag      = false 
    sgn_μν        = false
    sgn_μ         = false
    sgn_ν         = false

    # do -w -> w
    if w < 0.0
        w      *= -1.0
        sgn_μν  = set_flag(sgn_μν)
    end

    # do -v -> v
    if v < 0.0
        v     *= -1.0
        sgn_μ  = set_flag(sgn_μ)
    end

    # do -vp -> vp
    if vp < 0.0
        vp    *= -1.0
        sgn_ν  = set_flag(sgn_ν)
    end

    # deref meshes for interpolation
    Ω = m.Ωt[comp]
    ν = m.νt[comp]

    return get_buffer(w, v, vp, Ω, ν, exchange_flag, map_flag, sgn_μν, sgn_μ, sgn_ν)
end

# generate access buffer for u channel of Action struct
function get_buffer_u(
    comp :: Int64, 
    w    :: Float64,
    v    :: Float64,
    vp   :: Float64,
    m    :: Mesh
    )    :: Buffer

    # init flags
    exchange_flag = false
    map_flag      = false 
    sgn_μν        = false
    sgn_μ         = false
    sgn_ν         = false

    # do -w -> w
    if w < 0.0
        w             *= -1.0
        exchange_flag  = set_flag(exchange_flag)
        sgn_μν         = set_flag(sgn_μν)
    end

    # do -v -> v
    if v < 0.0
        v             *= -1.0
        exchange_flag  = set_flag(exchange_flag)
        map_flag       = set_flag(map_flag)
        sgn_ν          = set_flag(sgn_ν)
    end

    # do -vp -> vp
    if vp < 0.0
        vp       *= -1.0
        map_flag  = set_flag(map_flag)
        sgn_ν     = set_flag(sgn_ν)
    end

    # deref meshes for interpolation, respecting possible mapping to s channel
    Ω = m.Ωu[comp]
    ν = m.νu[comp]

    if map_flag
        Ω = m.Ωs[comp]
        ν = m.νs[comp]
    end

    return get_buffer(w, v, vp, Ω, ν, exchange_flag, map_flag, sgn_μν, sgn_μ, sgn_ν)
end