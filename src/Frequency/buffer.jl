"""
    Buffer

Struct used for reading out vertices from Action struct. 
Contains symmetry related flags, (asymptotic) kernel specification and interpolation parameters.
* `exchange_flag :: Bool`  : flag for site [(i0, j) -> (j, i0)] and spin [(μ, ν) -> (ν, μ)] exchange
* `sgn_flag      :: Bool`  : flag for sign change under -t -> t and -u -> u for mixed spin-density terms
* `map_flag      :: Bool`  : flag for channel mapping s -> u and sign if density first or second
* `kernel        :: Int64` : specification of asymptotic kernel to be interpolated
* `p1            :: Param` : interpolation parameters for bosonic frequency argument
* `p2            :: Param` : interpolation parameters for first fermionic frequency argument
* `p3            :: Param` : interpolation parameters for second fermionic frequency argument
"""
struct Buffer
    exchange_flag :: Bool
    sgn_flag      :: Bool
    map_flag      :: Bool
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

    b = Buffer(false, false, false, 0, get_param_empty(), get_param_empty(), get_param_empty())

    return b
end

# generate generic access buffer for Action struct given exchange_flag, sgn_flag and map_flag
function get_buffer(
    w             :: Float64,
    v             :: Float64,
    vp            :: Float64,
    Ω             :: Vector{Float64},
    ν             :: Vector{Float64},
    exchange_flag :: Bool,
    sgn_flag      :: Bool,
    map_flag      :: Bool,
    )             :: Buffer

    if Ω[end] < abs(w)
        return get_buffer_empty()
    else
        if ν[end] < abs(v)
            if ν[end] < abs(vp)
                # interpolation for q1
                return Buffer(exchange_flag, sgn_flag, map_flag, 1, get_param(w, Ω), get_param_empty(), get_param_empty())
            else
                # interpolation for q2_2
                return Buffer(exchange_flag, sgn_flag, map_flag, 3, get_param(w, Ω), get_param_empty(), get_param(vp, ν))
            end
        else
            if ν[end] < abs(vp)
                # interpolation for q2_1
                return Buffer(exchange_flag, sgn_flag, map_flag, 2, get_param(w, Ω), get_param(v, ν), get_param_empty())
            else
                # interpolation for q3
                return Buffer(exchange_flag, sgn_flag, map_flag, 4, get_param(w, Ω), get_param(v, ν), get_param(vp, ν))
            end
        end
    end
end

# generate access buffer for s channel of Action struct
function get_buffer_s(
    w  :: Float64,
    v  :: Float64,
    vp :: Float64,
    m  :: Mesh
    )  :: Buffer

    exchange_flag = false
    sgn_flag      = false
    map_flag      = false

    # do -w -> w + site exchange + spin exchange
    if w < 0.0
        w             *= -1.0
        exchange_flag  = set_flag(exchange_flag)
    end

    # do -v -> v + site exchange + spin exchange + mapping to u channel + sign if density first
    if v < 0.0
        v             *= -1.0
        exchange_flag  = set_flag(exchange_flag)
        map_flag       = set_flag(map_flag)
    end

    # do -vp -> vp + mapping to u channel + sign if density second
    if vp < 0.0
        vp       *= -1.0
        map_flag  = set_flag(map_flag)
    end

    # deref meshes for interpolation, respecting possible mapping to u channel
    Ω = m.Ωs
    ν = m.νs

    if map_flag
        Ω = m.Ωu
        ν = m.νu
    end

    return get_buffer(w, v, vp, Ω, ν, exchange_flag, sgn_flag, map_flag)
end

# generate access buffer for t channel of Action struct
function get_buffer_t(
    w  :: Float64,
    v  :: Float64,
    vp :: Float64,
    m  :: Mesh
    )  :: Buffer

    exchange_flag = false
    sgn_flag      = false
    map_flag      = false

    # do -w -> w + sign if mixed spin and density
    if w < 0.0
        w        *= -1.0
        sgn_flag  = set_flag(sgn_flag)
    end

    # do -v -> v + sign if density first
    if v < 0.0
        v        *= -1.0
        map_flag  = set_flag(map_flag)
    end

    # do -vp -> vp + sign if density second
    if vp < 0.0
        vp       *= -1.0
        map_flag  = set_flag(map_flag)
    end

    # deref meshes for interpolation
    Ω = m.Ωt
    ν = m.νt

    return get_buffer(w, v, vp, Ω, ν, exchange_flag, sgn_flag, map_flag)
end

# generate access buffer for u channel of Action struct
function get_buffer_u(
    w  :: Float64,
    v  :: Float64,
    vp :: Float64,
    m  :: Mesh
    )  :: Buffer

    exchange_flag = false
    sgn_flag      = false
    map_flag      = false

    # do -w -> w + site exchange + spin exchange + sign if mixed spin density
    if w < 0.0
        w             *= -1.0
        exchange_flag  = set_flag(exchange_flag)
        sgn_flag       = set_flag(sgn_flag)
    end

    # do -v -> v + site exchange + spin exchange + mapping to s channel + sign if density second
    if v < 0.0
        v             *= -1.0
        exchange_flag  = set_flag(exchange_flag)
        map_flag       = set_flag(map_flag)
    end

    # do -vp -> vp + mapping to s channel + sign if density second
    if vp < 0.0
        vp       *= -1.0
        map_flag  = set_flag(map_flag)
    end

    # deref meshes for interpolation, respecting possible mapping to s channel
    Ω = m.Ωu
    ν = m.νu

    if map_flag
        Ω = m.Ωs
        ν = m.νs
    end

    return get_buffer(w, v, vp, Ω, ν, exchange_flag, sgn_flag, map_flag)
end