"""
    buffer_sun <: buffer 

Struct for reading out vertices from action_sun struct. 
Contains symmetry related flags, (asymptotic) kernel specification and interpolation parameters.
"""
struct buffer_sun <: buffer  
    exchange_flag :: Bool 
    map_flag      :: Bool
    kernel        :: Int64
    p1            :: param 
    p2            :: param 
    p3            :: param 
end

"""
    get_buffer_sun_empty() :: buffer_sun

Generate buffer_sun struct with dummy fields.
"""
function get_buffer_sun_empty() :: buffer_sun 

    b = buffer_sun(false, false, 0, get_param_empty(), get_param_empty(), get_param_empty())

    return b 
end 





"""
    get_buffer_sun_s(
        w  :: Float64,
        v  :: Float64,
        vp :: Float64,
        m  :: mesh
        )  :: buffer_sun

Generate access buffer for s channel of action_sun struct.
Symmetries are applied to map all frequencies onto non-negative values:
1) -w  -> w  + site exchange 
2) -v  -> v  + site exchange + mapping to u channel (sign if density)
3) -vp -> vp + mapping to u channel (sign if density)
"""
function get_buffer_sun_s(
    w  :: Float64,
    v  :: Float64,
    vp :: Float64,
    m  :: mesh
    )  :: buffer_sun

    b             = get_buffer_sun_empty()
    exchange_flag = false
    map_flag      = false

    # do -w -> w + site exchange 
    if w < 0.0 
        w             *= -1.0 
        exchange_flag  = set_flag(exchange_flag)
    end 

    # do -v -> v + site exchange + mapping to u channel (sign if density)
    if v < 0.0 
        v             *= -1.0 
        exchange_flag  = set_flag(exchange_flag)
        map_flag       = set_flag(map_flag) 
    end

    # do -vp -> vp + mapping to u channel (sign if density)
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

    # interpolation for q3
    if heavyside(Ω[end] - abs(w)) * heavyside(ν[end] - abs(v)) * heavyside(ν[end] - abs(vp)) != 0.0
        b = buffer_sun(exchange_flag, map_flag, 4, get_param(w, Ω), get_param(v, ν), get_param(vp, ν))
    # interpolation for q2_2
    elseif heavyside(Ω[end] - abs(w)) * heavyside(ν[end] - abs(vp)) != 0.0
        b = buffer_sun(exchange_flag, map_flag, 3, get_param(w, Ω), get_param_empty(), get_param(vp, ν)) 
    # interpolation for q2_1
    elseif heavyside(Ω[end] - abs(w)) * heavyside(ν[end] - abs(v)) != 0.0 
        b = buffer_sun(exchange_flag, map_flag, 2, get_param(w, Ω), get_param(v, ν), get_param_empty()) 
    # interpolation for q1
    elseif heavyside(Ω[end] - abs(w)) != 0.0
        b = buffer_sun(exchange_flag, map_flag, 1, get_param(w, Ω), get_param_empty(), get_param_empty()) 
    end

    return b
end

"""
    get_buffer_sun_t(
        w  :: Float64,
        v  :: Float64,
        vp :: Float64,
        m  :: mesh
        )  :: buffer_sun

Generate access buffer for t channel of action_sun struct.
Symmetries are applied to map all frequencies onto non-negative values:
1) -w  -> w  
2) -v  -> v  (sign if density)
3) -vp -> vp (sign if density)
"""
function get_buffer_sun_t(
    w  :: Float64,
    v  :: Float64,
    vp :: Float64,
    m  :: mesh
    )  :: buffer_sun

    b             = get_buffer_sun_empty()
    exchange_flag = false
    map_flag      = false

    # do -w -> w
    if w < 0.0
        w *= -1.0
    end
    
    # do -v -> v + sign in density
    if v < 0.0
        v        *= -1.0
        map_flag  = set_flag(map_flag)
    end
    
    # do -vp -> vp + sign in density
    if vp < 0.0
        vp       *= -1.0
        map_flag  = set_flag(map_flag)
    end

    # deref meshes for interpolation
    Ω = m.Ωt 
    ν = m.νt 

    # interpolation for q3
    if heavyside(Ω[end] - abs(w)) * heavyside(ν[end] - abs(v)) * heavyside(ν[end] - abs(vp)) != 0.0
        b = buffer_sun(exchange_flag, map_flag, 4, get_param(w, Ω), get_param(v, ν), get_param(vp, ν))
    # interpolation for q2_2
    elseif heavyside(Ω[end] - abs(w)) * heavyside(ν[end] - abs(vp)) != 0.0
        b = buffer_sun(exchange_flag, map_flag, 3, get_param(w, Ω), get_param_empty(), get_param(vp, ν)) 
    # interpolation for q2_1
    elseif heavyside(Ω[end] - abs(w)) * heavyside(ν[end] - abs(v)) != 0.0 
        b = buffer_sun(exchange_flag, map_flag, 2, get_param(w, Ω), get_param(v, ν), get_param_empty()) 
    # interpolation for q1
    elseif heavyside(Ω[end] - abs(w)) != 0.0
        b = buffer_sun(exchange_flag, map_flag, 1, get_param(w, Ω), get_param_empty(), get_param_empty()) 
    end

    return b
end

"""
    get_buffer_sun_u(
        w  :: Float64,
        v  :: Float64,
        vp :: Float64,
        m  :: mesh
        )  :: buffer_sun

Generate access buffer for u channel of action_sun struct.
Symmetries are applied to map all frequencies onto non-negative values:
1) -w  -> w  + site exchange 
2) -v  -> v  + site exchange + mapping to s channel (sign if density)
3) -vp -> vp + mapping to s channel (sign if density)
"""
function get_buffer_sun_u(
    w  :: Float64,
    v  :: Float64,
    vp :: Float64,
    m  :: mesh
    )  :: buffer_sun

    b             = get_buffer_sun_empty()
    exchange_flag = false
    map_flag      = false

    # do -w -> w + site exchange
    if w < 0.0
        w             *= -1.0
        exchange_flag  = set_flag(exchange_flag)
    end

    # do -v -> v + site exchange + mapping to s channel (sign if density)
    if v < 0.0
        v             *= -1.0
        exchange_flag  = set_flag(exchange_flag)
        map_flag       = set_flag(map_flag)
    end

    # do -vp -> vp + mapping to s channel (sign if density)
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

    # interpolation for q3
    if heavyside(Ω[end] - abs(w)) * heavyside(ν[end] - abs(v)) * heavyside(ν[end] - abs(vp)) != 0.0
        b = buffer_sun(exchange_flag, map_flag, 4, get_param(w, Ω), get_param(v, ν), get_param(vp, ν))
    # interpolation for q2_2
    elseif heavyside(Ω[end] - abs(w)) * heavyside(ν[end] - abs(vp)) != 0.0
        b = buffer_sun(exchange_flag, map_flag, 3, get_param(w, Ω), get_param_empty(), get_param(vp, ν)) 
    # interpolation for q2_1
    elseif heavyside(Ω[end] - abs(w)) * heavyside(ν[end] - abs(v)) != 0.0 
        b = buffer_sun(exchange_flag, map_flag, 2, get_param(w, Ω), get_param(v, ν), get_param_empty()) 
    # interpolation for q1
    elseif heavyside(Ω[end] - abs(w)) != 0.0
        b = buffer_sun(exchange_flag, map_flag, 1, get_param(w, Ω), get_param_empty(), get_param_empty()) 
    end

    return b
end





